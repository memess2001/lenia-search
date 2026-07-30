"""Standard Lenia simulator using PyTorch FFT convolution.

Implements the continuous cellular automaton described by Bert Chan (2019).
Update rule: A_{t+1} = clip(A_t + dt * G(K * A_t), 0, 1)
where K*A is computed via FFT convolution and G is the growth function.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Kernel functions (spatial domain, radial profile)
# ---------------------------------------------------------------------------

def _gaussian_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Classic bell-shaped kernel.  peaks controls the number of rings."""
    return torch.exp(-((r * peaks - 1.0).pow(2)) / (2 * 0.15**2))


def _polynomial_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Polynomial kernel with compact support in [0, 1]."""
    # (1 - r^alpha)^alpha  clipped to [0,1]
    alpha = max(peaks, 1.0) * 4.0
    val = (1.0 - r.pow(alpha)).clamp(min=0.0).pow(alpha)
    return val


def _mexican_hat_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Mexican-hat (Ricker wavelet) profile — positive centre, negative ring."""
    sigma = 0.3 / max(peaks, 0.5)
    val = (1.0 - (r / sigma).pow(2)) * torch.exp(-(r.pow(2)) / (2 * sigma**2))
    # Shift so minimum is 0 (Lenia kernels are non-negative by convention)
    val = val - val.min()
    return val


def _bump_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Smooth bump function with compact support."""
    # exp(-1 / (1 - r^2)) for r < 1, else 0
    r_sq = r.pow(2)
    inside = r_sq < 1.0
    safe_r_sq = torch.where(inside, r_sq, torch.zeros_like(r_sq))
    val = torch.where(
        inside,
        torch.exp(-peaks / (1.0 - safe_r_sq + 1e-10)),
        torch.zeros_like(r_sq),
    )
    return val


def _cosine_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Cosine kernel with compact support: cos(pi*r/2)^alpha."""
    alpha = max(peaks, 1.0) * 2.0
    val = torch.cos(math.pi * r / 2.0).clamp(min=0.0).pow(alpha)
    return val * (r <= 1.0).float()


def _bessel_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Bessel J0 approximation — oscillating wave pattern."""
    alpha = peaks * 8.0
    val = torch.cos(alpha * r) * torch.exp(-r.pow(2) * 2.0)
    val = val - val.min()
    return val * (r <= 1.0).float()


def _sinc_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Sinc kernel — ideal low-pass filter."""
    alpha = max(peaks, 0.5) * math.pi * 3.0
    safe_r = torch.where(r < 1e-6, torch.ones_like(r) * 1e-6, r)
    val = torch.sin(alpha * safe_r) / (alpha * safe_r)
    val = val - val.min()
    return val * (r <= 1.0).float()


def _gabor_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Gabor kernel — localized oscillation, visual cortex model."""
    sigma = 0.3
    omega = peaks * 6.0
    val = torch.exp(-r.pow(2) / (2 * sigma**2)) * torch.cos(omega * r)
    val = val - val.min()
    return val * (r <= 1.0).float()


def _step_ring_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Ring-shaped step function — closest to discrete CA."""
    n_rings = max(int(peaks), 1)
    val = torch.zeros_like(r)
    for i in range(n_rings):
        inner = i / n_rings
        outer = (i + 0.5) / n_rings
        val = val + ((r >= inner) & (r < outer)).float()
    return val


def _power_law_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Power-law decay with cutoff — long-range interaction."""
    alpha = max(peaks, 0.5) * 1.5
    safe_r = r.clamp(min=0.05)
    val = safe_r.pow(-alpha) * (r <= 1.0).float()
    val = val / val.max().clamp(min=1e-10)
    return val


def _yukawa_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Yukawa potential — screened short-range force."""
    alpha = max(peaks, 0.5) * 3.0
    safe_r = r.clamp(min=0.05)
    val = torch.exp(-alpha * r) / safe_r * (r <= 1.0).float()
    val = val / val.max().clamp(min=1e-10)
    return val


def _elliptical_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Elliptical kernel — multi-ring directional preference."""
    sigma = 0.25
    val = torch.exp(-((r - 0.5) / sigma).pow(2))
    val = val + 0.5 * torch.exp(-((r - 0.3 * peaks) / (sigma * 0.7)).pow(2))
    return val * (r <= 1.0).float()


def _spiral_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Spiral-like kernel — chiral structure."""
    omega = peaks * 4.0
    val = torch.exp(-r.pow(2) * 3.0) * (1.0 + torch.sin(omega * r))
    return val * (r <= 1.0).float()


def _fourier_series_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Fourier series kernel — first N terms."""
    n_terms = max(int(peaks), 1) + 2
    val = torch.zeros_like(r)
    for n in range(1, n_terms + 1):
        a_n = math.cos(n * peaks) / n
        val = val + a_n * torch.cos(n * math.pi * r)
    val = val - val.min()
    return val * (r <= 1.0).float()


def _rbf_mixture_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Radial basis function mixture — multiple Gaussian bumps."""
    n_bumps = max(int(peaks), 1) + 1
    val = torch.zeros_like(r)
    for i in range(n_bumps):
        center = (i + 0.5) / (n_bumps + 1)
        sigma = 0.08
        val = val + torch.exp(-((r - center) / sigma).pow(2))
    return val * (r <= 1.0).float()


# --- Phase 2 expansion: physics-inspired kernels ---

def _lennard_jones_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Lennard-Jones 12-6 potential — repulsive core + attractive tail."""
    r_m = 0.5 / max(peaks, 0.5)
    r_safe = r.clamp(min=0.05)
    ratio = r_m / r_safe
    val = ratio.pow(12) - 2.0 * ratio.pow(6)
    val = val - val.min()
    return val * (r <= 1.0).float()


def _morse_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Morse potential — tunable exponential well."""
    a = peaks * 5.0
    r_e = 0.3
    val = (1.0 - torch.exp(-a * (r - r_e))).pow(2) * torch.exp(-r * peaks)
    val = val - val.min()
    return val * (r <= 1.0).float()


def _riesz_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Riesz kernel — fractional-power singularity with compact support."""
    s = peaks * 0.5
    alpha = peaks * 2.0
    r_safe = r.clamp(min=0.05)
    val = (1.0 - r).clamp(min=0.0).pow(alpha) * r_safe.pow(-s)
    val = val / val.max().clamp(min=1e-10)
    return val * (r <= 1.0).float()


def _double_well_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Double-well — two preferred interaction distances."""
    c1 = 0.25 / max(peaks, 0.5)
    c2 = 0.65 / max(peaks, 0.5)
    s = 0.1
    val = torch.exp(-((r - c1) / s).pow(2)) + torch.exp(-((r - c2) / s).pow(2))
    return val * (r <= 1.0).float()


# --- Phase 2 expansion: signal-processing kernels ---

def _hann_ring_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Hann-window ring — ultra-smooth with zero-derivative edges."""
    c = 0.5
    w = 0.6 / max(peaks, 0.5)
    inside = (r - c).abs() <= (w / 2.0)
    val = 0.5 * (1.0 - torch.cos(2.0 * math.pi * (r - c + w / 2.0) / w))
    return val * inside.float() * (r <= 1.0).float()


def _blackman_harris_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Blackman-Harris 4-term window — ultra-low sidelobes."""
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    rp = r * peaks
    val = (a0
           - a1 * torch.cos(2.0 * math.pi * rp)
           + a2 * torch.cos(4.0 * math.pi * rp)
           - a3 * torch.cos(6.0 * math.pi * rp))
    val = val.clamp(min=0.0)
    return val * (r <= 1.0).float()


def _chirp_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Chirp — frequency increases with radius (quadratic phase)."""
    val = torch.exp(-3.0 * r.pow(2)) * torch.cos(peaks * math.pi * r.pow(2))
    val = val - val.min()
    return val * (r <= 1.0).float()


def _dog_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Difference of Gaussians — band-pass filter."""
    s1 = 0.2 / max(peaks, 0.5)
    s2 = 0.4 / max(peaks, 0.5)
    val = torch.exp(-r.pow(2) / (2.0 * s1**2)) - 0.5 * torch.exp(-r.pow(2) / (2.0 * s2**2))
    val = val - val.min()
    return val * (r <= 1.0).float()


# --- Phase 2 expansion: compactly-supported RBF kernels ---

def _wendland_c2_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Wendland C2 — minimal-degree positive-definite polynomial."""
    n = max(int(peaks), 1)
    one_minus_r = (1.0 - r).clamp(min=0.0)
    val = one_minus_r.pow(n + 4) * ((n + 4) * r + 1.0)
    return val * (r <= 1.0).float()


def _matern_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Matérn kernel (ν=1.5) — tunable roughness between exponential and Gaussian."""
    ell = 0.3 / max(peaks, 0.5)
    sqrt3_r_ell = math.sqrt(3.0) * r / ell
    val = (1.0 + sqrt3_r_ell) * torch.exp(-sqrt3_r_ell)
    return val * (r <= 1.0).float()


# --- Phase 2 expansion: mathematical/exotic kernels ---

def _log_gaussian_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Log-Gaussian — bell in log-space, heavy short-range asymmetry."""
    r_safe = r.clamp(min=0.01)
    mu_r = -0.5 * peaks
    s = 0.5
    val = torch.exp(-((torch.log(r_safe) - mu_r).pow(2)) / (2.0 * s**2)) / r_safe
    val = val / val.max().clamp(min=1e-10)
    return val * (r <= 1.0).float()


def _plateau_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Plateau / super-Gaussian — flat-top with tunable edge sharpness."""
    n = max(int(peaks), 1) + 1
    w = 0.5
    val = torch.exp(-(r / w).pow(2 * n))
    return val * (r <= 1.0).float()


def _sech_kernel(r: torch.Tensor, peaks: float = 1.0) -> torch.Tensor:
    """Hyperbolic secant — self-reciprocal Fourier, soliton connections."""
    a = peaks * 5.0
    val = 1.0 / torch.cosh(a * r)
    return val * (r <= 1.0).float()


KERNEL_REGISTRY: dict[str, callable] = {
    # Original 4
    "gaussian": _gaussian_kernel,
    "polynomial": _polynomial_kernel,
    "mexican_hat": _mexican_hat_kernel,
    "bump": _bump_kernel,
    # Phase 1b expansion (+11 = 15 total)
    "cosine": _cosine_kernel,
    "bessel": _bessel_kernel,
    "sinc": _sinc_kernel,
    "gabor": _gabor_kernel,
    "step_ring": _step_ring_kernel,
    "power_law": _power_law_kernel,
    "yukawa": _yukawa_kernel,
    "elliptical": _elliptical_kernel,
    "spiral": _spiral_kernel,
    "fourier_series": _fourier_series_kernel,
    "rbf_mixture": _rbf_mixture_kernel,
    # Phase 2 expansion (+13 = 28 total)
    "lennard_jones": _lennard_jones_kernel,
    "morse": _morse_kernel,
    "riesz": _riesz_kernel,
    "double_well": _double_well_kernel,
    "hann_ring": _hann_ring_kernel,
    "blackman_harris": _blackman_harris_kernel,
    "chirp": _chirp_kernel,
    "dog": _dog_kernel,
    "wendland_c2": _wendland_c2_kernel,
    "matern": _matern_kernel,
    "log_gaussian": _log_gaussian_kernel,
    "plateau": _plateau_kernel,
    "sech": _sech_kernel,
}


# ---------------------------------------------------------------------------
# Growth functions  G: R -> [-1, 1]
# ---------------------------------------------------------------------------

def _gaussian_bell_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """G(u) = 2 * exp(-((u - mu)/sigma)^2 / 2) - 1"""
    return 2.0 * torch.exp(-((u - mu) / sigma).pow(2) / 2.0) - 1.0


def _step_function_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Step: +1 inside [mu-sigma, mu+sigma], -1 outside."""
    inside = (u >= (mu - sigma)) & (u <= (mu + sigma))
    return torch.where(inside, torch.ones_like(u), -torch.ones_like(u))


def _asymmetric_bell_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Asymmetric Gaussian: narrower on the left side of mu."""
    sigma_left = sigma * 0.5
    sigma_right = sigma * 1.5
    s = torch.where(u < mu, sigma_left, sigma_right)
    return 2.0 * torch.exp(-((u - mu) / s).pow(2) / 2.0) - 1.0


def _bistable_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Two stable peaks producing bistable dynamics."""
    mu1 = mu - sigma * 0.5
    mu2 = mu + sigma * 0.5
    s = sigma * 0.4
    g1 = torch.exp(-((u - mu1) / s).pow(2) / 2.0)
    g2 = torch.exp(-((u - mu2) / s).pow(2) / 2.0)
    return 2.0 * torch.max(g1, g2) - 1.0


def _laplace_peak_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Laplace peak — sharper than Gaussian, more sensitive to density."""
    return 2.0 * torch.exp(-torch.abs(u - mu) / sigma) - 1.0


def _cauchy_peak_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Cauchy peak — heavy-tailed, distant responses remain non-zero."""
    return 2.0 / (1.0 + ((u - mu) / sigma).pow(2)) - 1.0


def _sigmoid_pair_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Band-pass: sigmoid(u-a) - sigmoid(u-b), active in a narrow band."""
    a = mu - sigma
    b = mu + sigma
    s = sigma * 0.3
    return torch.sigmoid((u - a) / s) - torch.sigmoid((u - b) / s) - 0.5


def _relu_like_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Piecewise linear — ReLU-style tent function."""
    val = 1.0 - torch.abs(u - mu) / sigma
    return (2.0 * val.clamp(min=0.0, max=1.0) - 1.0)


# --- Phase 2 expansion: population-dynamics growth ---

def _ricker_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Ricker / overcompensatory — growth overshoots then drops at high density."""
    u_safe = u.clamp(min=1e-6)
    ratio = u_safe / max(mu, 1e-6)
    envelope = torch.exp(-((u - mu) / max(sigma, 1e-6)).pow(2))
    val = 2.0 * ratio * torch.exp(1.0 - ratio) * envelope - 1.0
    return val.clamp(-1.0, 1.0)


def _allee_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Allee effect — dead zone at low density (population too sparse to sustain)."""
    theta = mu * 0.3  # Allee threshold
    envelope = torch.exp(-((u - mu) / max(sigma, 1e-6)).pow(2))
    val = 2.0 * ((u - theta) / max(mu, 1e-6)).pow(2) * envelope - 1.0
    return val.clamp(-1.0, 1.0)


def _gompertz_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Gompertz — double-exponential, ultra-sharp transition."""
    val = 2.0 * torch.exp(-torch.exp(-(u - mu) / max(sigma, 1e-6))) - 1.0
    return val.clamp(-1.0, 1.0)


# --- Phase 2 expansion: reaction-diffusion growth ---

def _fitzhugh_nagumo_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """FitzHugh-Nagumo cubic — excitable medium dynamics."""
    x = (u - mu) / max(sigma, 1e-6)
    val = x * (1.0 - x.pow(2) / 3.0)
    # Scale to [-1, 1]: max of x(1-x²/3) is at x=1 → val=2/3
    val = val * 1.5  # so peak ≈ 1.0
    return val.clamp(-1.0, 1.0)


def _brusselator_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Brusselator-inspired — u² nonlinearity drives Turing instability."""
    target = u.pow(2) * mu - mu
    val = 2.0 * torch.exp(-(target / max(sigma, 1e-6)).pow(2)) - 1.0
    return val.clamp(-1.0, 1.0)


def _swift_hohenberg_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Swift-Hohenberg — tristable (three-phase coexistence)."""
    val = torch.tanh(mu * u - u.pow(3) / max(sigma, 1e-6))
    return val.clamp(-1.0, 1.0)


# --- Phase 2 expansion: signal/math growth ---

def _sinc_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Sinc growth — multi-band sidelobes for multi-phase patterns."""
    x = math.pi * (u - mu) / max(sigma, 1e-6)
    # sinc(x) = sin(x)/x, handle x=0
    val = torch.where(
        x.abs() < 1e-6,
        torch.ones_like(x),
        torch.sin(x) / x,
    )
    return (2.0 * val - 1.0).clamp(-1.0, 1.0)


def _witch_growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Witch of Agnesi growth — flat-top quartic (indifferent band)."""
    x = (u - mu) / (max(sigma, 1e-6) * 0.5)
    val = 2.0 / (1.0 + x.pow(4)) - 1.0
    return val.clamp(-1.0, 1.0)


GROWTH_REGISTRY: dict[str, callable] = {
    # Original 4
    "gaussian_bell": _gaussian_bell_growth,
    "step_function": _step_function_growth,
    "asymmetric_bell": _asymmetric_bell_growth,
    "bistable": _bistable_growth,
    # Phase 1b expansion (+4 = 8 total)
    "laplace_peak": _laplace_peak_growth,
    "cauchy_peak": _cauchy_peak_growth,
    "sigmoid_pair": _sigmoid_pair_growth,
    "relu_like": _relu_like_growth,
    # Phase 2 expansion (+8 = 16 total)
    "ricker": _ricker_growth,
    "allee": _allee_growth,
    "gompertz": _gompertz_growth,
    "fitzhugh_nagumo": _fitzhugh_nagumo_growth,
    "brusselator": _brusselator_growth,
    "swift_hohenberg": _swift_hohenberg_growth,
    "sinc_growth": _sinc_growth,
    "witch_growth": _witch_growth,
}


# ---------------------------------------------------------------------------
# Lenia simulator
# ---------------------------------------------------------------------------

class Lenia:
    """Configurable Lenia simulator.

    Parameters
    ----------
    genome : dict
        Must contain at minimum: kernel_type, kernel_radius, kernel_peaks,
        growth_type, growth_mu, growth_sigma, dt.
        Optional: resolution (default 128), device (default "cpu"),
        num_channels (default 1).
    """

    def __init__(self, genome: dict, device: Optional[str] = None) -> None:
        self.genome = dict(genome)  # defensive copy

        self.resolution: int = int(genome.get("resolution", 128))
        self.dt: float = float(genome.get("dt", 0.1))
        self.device = torch.device(device or genome.get("device", "cpu"))

        # Kernel setup
        kernel_type: str = genome["kernel_type"]
        kernel_radius: float = float(genome["kernel_radius"])
        kernel_peaks: float = float(genome.get("kernel_peaks", 1.0))
        if kernel_type not in KERNEL_REGISTRY:
            raise ValueError(f"Unknown kernel type: {kernel_type}. Choose from {list(KERNEL_REGISTRY)}")
        kernel_fn = KERNEL_REGISTRY[kernel_type]

        # Growth setup
        growth_type: str = genome["growth_type"]
        if growth_type not in GROWTH_REGISTRY:
            raise ValueError(f"Unknown growth type: {growth_type}. Choose from {list(GROWTH_REGISTRY)}")
        self.growth_fn = GROWTH_REGISTRY[growth_type]
        self.growth_mu: float = float(genome["growth_mu"])
        self.growth_sigma: float = float(genome["growth_sigma"])

        # Build kernel in spatial domain, then pre-compute its FFT
        self._kernel_fft = self._build_kernel_fft(kernel_fn, kernel_radius, kernel_peaks)

        # Initialize state: random seed in the centre
        self.state = self._init_state()

    # ----- internal helpers ------------------------------------------------

    def _build_kernel_fft(
        self, kernel_fn: callable, radius: float, peaks: float
    ) -> torch.Tensor:
        """Construct the kernel in spatial domain, normalise, embed into
        full-resolution grid, and return its FFT."""
        R = self.resolution
        # Coordinate grids centred at (0, 0)
        mid = R // 2
        y = torch.arange(R, device=self.device, dtype=torch.float32) - mid
        x = torch.arange(R, device=self.device, dtype=torch.float32) - mid
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Distance normalised so that kernel_radius maps to r=1
        dist = torch.sqrt(xx**2 + yy**2) / max(radius, 1.0)

        # Evaluate radial profile (only inside r <= 1)
        raw = kernel_fn(dist, peaks)
        raw = raw * (dist <= 1.0).float()

        # Normalise to sum = 1
        total = raw.sum()
        if total > 0:
            raw = raw / total

        # Roll so that the kernel centre sits at (0,0) for correct FFT convolution
        kernel_shifted = torch.roll(raw, shifts=(-mid, -mid), dims=(0, 1))

        return torch.fft.fft2(kernel_shifted)

    def _init_state(self) -> torch.Tensor:
        """Random seed in centre quarter of the grid."""
        R = self.resolution
        state = torch.zeros(R, R, device=self.device, dtype=torch.float32)
        q = R // 4
        state[q : 3 * q, q : 3 * q] = torch.rand(
            2 * q, 2 * q, device=self.device, dtype=torch.float32
        )
        return state

    # ----- public API ------------------------------------------------------

    def step(self) -> None:
        """Advance the simulation by one timestep using FFT convolution."""
        # Convolution in frequency domain
        state_fft = torch.fft.fft2(self.state)
        potential = torch.fft.ifft2(state_fft * self._kernel_fft).real

        # Growth mapping
        growth = self.growth_fn(potential, self.growth_mu, self.growth_sigma)

        # Lenia update rule
        self.state = (self.state + self.dt * growth).clamp(0.0, 1.0)

    def run(self, n_steps: int, record_interval: int = 10) -> list[torch.Tensor]:
        """Run *n_steps* and return state snapshots every *record_interval* steps.

        Returns a list of 2-D tensors (detached, on CPU) for downstream analysis.
        """
        history: list[torch.Tensor] = []
        for i in range(n_steps):
            self.step()
            if i % record_interval == 0 or i == n_steps - 1:
                history.append(self.state.detach().cpu().clone())
        return history
