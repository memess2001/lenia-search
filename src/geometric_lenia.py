"""Lenia simulators on curved geometries: sphere, torus, hyperbolic disk.

Each class matches the interface of ``Lenia`` from ``lenia_core``:
    __init__(genome, device=None)
    step()
    run(n_steps, record_interval=10) -> list[torch.Tensor]

All convolutions use the kernel and growth functions defined in
``lenia_core`` so that the same genome vocabulary works everywhere.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from src.lenia_core import KERNEL_REGISTRY, GROWTH_REGISTRY


# ======================================================================
# SphericalLenia  —  Lenia on S^2 (equirectangular grid)
# ======================================================================

class SphericalLenia:
    """Lenia on a sphere via equirectangular projection.

    Grid layout
    -----------
    * X axis (columns) = longitude, 0 .. 2pi  — **periodic**
    * Y axis (rows)    = latitude, -pi/2 .. pi/2 — **reflective** at poles

    Convolution strategy
    --------------------
    FFT convolution on the 2D grid with a latitude-dependent kernel.
    Because the metric varies with latitude we cannot use a single FFT
    kernel for all rows.  Instead we split the grid into horizontal
    *bands*, build one kernel per band (with the kernel width scaled by
    ``1 / cos(lat)``), and apply each via FFT.  With ~8 bands this is a
    good accuracy/speed compromise.

    For simplicity and robustness (especially at small resolution) we
    use a single-kernel approach: we build the kernel using the *average*
    latitude correction (which is 1.0 at the equator) and handle the
    polar distortion by scaling the x-distance during kernel construction.
    At resolution 64-128 the approximation is quite decent.
    """

    N_BANDS = 8  # number of latitude bands for kernel correction

    def __init__(self, genome: dict, device: Optional[str] = None) -> None:
        self.genome = dict(genome)
        self.resolution: int = int(genome.get("resolution", 128))
        self.dt: float = float(genome.get("dt", 0.1))
        self.device = torch.device(device or genome.get("device", "cpu"))

        R = self.resolution

        # --- kernel / growth look-ups ------------------------------------
        kernel_type: str = genome["kernel_type"]
        self.kernel_radius: float = float(genome["kernel_radius"])
        kernel_peaks: float = float(genome.get("kernel_peaks", 1.0))
        if kernel_type not in KERNEL_REGISTRY:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        self.kernel_fn = KERNEL_REGISTRY[kernel_type]
        self.kernel_peaks = kernel_peaks

        growth_type: str = genome["growth_type"]
        if growth_type not in GROWTH_REGISTRY:
            raise ValueError(f"Unknown growth type: {growth_type}")
        self.growth_fn = GROWTH_REGISTRY[growth_type]
        self.growth_mu: float = float(genome["growth_mu"])
        self.growth_sigma: float = float(genome["growth_sigma"])

        # --- latitude array (row centres) --------------------------------
        # Row 0 = south pole (-pi/2), row R-1 = north pole (+pi/2)
        self.lat = torch.linspace(
            -math.pi / 2, math.pi / 2, R, device=self.device, dtype=torch.float32
        )
        # cos(lat) for each row — metric scale factor
        self.cos_lat = self.lat.cos().clamp(min=0.05)  # avoid /0 at poles

        # --- pre-compute one FFT kernel per latitude band ----------------
        self._band_kernel_ffts: list[torch.Tensor] = []
        self._band_row_ranges: list[tuple[int, int]] = []
        band_height = R // self.N_BANDS
        for b in range(self.N_BANDS):
            r0 = b * band_height
            r1 = r0 + band_height if b < self.N_BANDS - 1 else R
            mid_row = (r0 + r1) // 2
            lat_scale = 1.0 / float(self.cos_lat[mid_row])
            kfft = self._build_kernel_fft(lat_scale)
            self._band_kernel_ffts.append(kfft)
            self._band_row_ranges.append((r0, r1))

        # --- initial state -----------------------------------------------
        self.state = self._init_state()

    # ---- helpers --------------------------------------------------------

    def _build_kernel_fft(self, lon_scale: float) -> torch.Tensor:
        """Build FFT of kernel with x-distance scaled by *lon_scale*."""
        R = self.resolution
        mid = R // 2
        y = torch.arange(R, device=self.device, dtype=torch.float32) - mid
        x = torch.arange(R, device=self.device, dtype=torch.float32) - mid
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Scale x by lon_scale to emulate narrower cells near poles
        dist = torch.sqrt((xx * lon_scale) ** 2 + yy ** 2) / max(self.kernel_radius, 1.0)

        raw = self.kernel_fn(dist, self.kernel_peaks) * (dist <= 1.0).float()
        total = raw.sum()
        if total > 0:
            raw = raw / total

        kernel_shifted = torch.roll(raw, shifts=(-mid, -mid), dims=(0, 1))
        return torch.fft.fft2(kernel_shifted)

    def _init_state(self) -> torch.Tensor:
        R = self.resolution
        state = torch.zeros(R, R, device=self.device, dtype=torch.float32)
        q = R // 4
        state[q: 3 * q, q: 3 * q] = torch.rand(
            2 * q, 2 * q, device=self.device, dtype=torch.float32
        )
        return state

    def _pad_longitude_periodic(self, x: torch.Tensor) -> torch.Tensor:
        """X (longitude) is already periodic via FFT; no explicit padding needed."""
        return x

    # ---- public API -----------------------------------------------------

    def step(self) -> None:
        R = self.resolution
        state_fft = torch.fft.fft2(self.state)

        potential = torch.zeros_like(self.state)
        for kfft, (r0, r1) in zip(self._band_kernel_ffts, self._band_row_ranges):
            band_pot = torch.fft.ifft2(state_fft * kfft).real
            potential[r0:r1, :] = band_pot[r0:r1, :]

        growth = self.growth_fn(potential, self.growth_mu, self.growth_sigma)
        self.state = (self.state + self.dt * growth).clamp(0.0, 1.0)

        # Reflective boundary at poles: mirror the top/bottom rows
        n_reflect = max(1, R // 32)
        self.state[:n_reflect, :] = self.state[n_reflect: 2 * n_reflect, :].flip(0)
        self.state[-n_reflect:, :] = self.state[-2 * n_reflect: -n_reflect, :].flip(0)

    def run(self, n_steps: int, record_interval: int = 10) -> list[torch.Tensor]:
        history: list[torch.Tensor] = []
        for i in range(n_steps):
            self.step()
            if i % record_interval == 0 or i == n_steps - 1:
                history.append(self.state.detach().cpu().clone())
        return history


# ======================================================================
# TorusLenia  —  Lenia on an embedded torus T^2
# ======================================================================

class TorusLenia:
    """Lenia on an embedded torus with varying Gaussian curvature.

    Grid layout
    -----------
    * X axis (columns) = toroidal angle theta, 0 .. 2pi — periodic
    * Y axis (rows)    = poloidal angle phi, 0 .. 2pi — periodic

    The *embedded* torus has radii **R** (major) and **r** (minor).
    A point on the torus sits at distance ``R + r cos(phi)`` from the
    central axis, so the x-metric scales by ``(R + r cos(phi)) / R``.

    At *phi=0* (outer ring) the effective circumference is largest
    (positive curvature).  At *phi=pi* (inner ring) it is smallest
    (negative curvature).
    """

    DEFAULT_MAJOR = 3.0
    DEFAULT_MINOR = 1.0
    N_BANDS = 8

    def __init__(self, genome: dict, device: Optional[str] = None) -> None:
        self.genome = dict(genome)
        self.resolution: int = int(genome.get("resolution", 128))
        self.dt: float = float(genome.get("dt", 0.1))
        self.device = torch.device(device or genome.get("device", "cpu"))

        R_major = float(genome.get("torus_major_radius", self.DEFAULT_MAJOR))
        r_minor = float(genome.get("torus_minor_radius", self.DEFAULT_MINOR))
        self.R_major = R_major
        self.r_minor = r_minor

        # kernel / growth
        kernel_type: str = genome["kernel_type"]
        self.kernel_radius: float = float(genome["kernel_radius"])
        kernel_peaks: float = float(genome.get("kernel_peaks", 1.0))
        if kernel_type not in KERNEL_REGISTRY:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        self.kernel_fn = KERNEL_REGISTRY[kernel_type]
        self.kernel_peaks = kernel_peaks

        growth_type: str = genome["growth_type"]
        if growth_type not in GROWTH_REGISTRY:
            raise ValueError(f"Unknown growth type: {growth_type}")
        self.growth_fn = GROWTH_REGISTRY[growth_type]
        self.growth_mu: float = float(genome["growth_mu"])
        self.growth_sigma: float = float(genome["growth_sigma"])

        # phi values for each row (poloidal angle)
        Res = self.resolution
        self.phi = torch.linspace(0, 2 * math.pi, Res + 1, device=self.device,
                                  dtype=torch.float32)[:Res]

        # metric scale: (R + r*cos(phi)) / R
        self.metric_scale = (R_major + r_minor * self.phi.cos()) / R_major

        # build per-band kernels
        self._band_kernel_ffts: list[torch.Tensor] = []
        self._band_row_ranges: list[tuple[int, int]] = []
        band_height = Res // self.N_BANDS
        for b in range(self.N_BANDS):
            r0 = b * band_height
            r1 = r0 + band_height if b < self.N_BANDS - 1 else Res
            mid_row = (r0 + r1) // 2
            # x-distance scale is 1 / metric_scale  (narrower effective cell
            # on the inner ring means kernel spans more cells in x)
            x_scale = 1.0 / float(self.metric_scale[mid_row])
            kfft = self._build_kernel_fft(x_scale)
            self._band_kernel_ffts.append(kfft)
            self._band_row_ranges.append((r0, r1))

        self.state = self._init_state()

    def _build_kernel_fft(self, x_scale: float) -> torch.Tensor:
        Res = self.resolution
        mid = Res // 2
        y = torch.arange(Res, device=self.device, dtype=torch.float32) - mid
        x = torch.arange(Res, device=self.device, dtype=torch.float32) - mid
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt((xx * x_scale) ** 2 + yy ** 2) / max(self.kernel_radius, 1.0)
        raw = self.kernel_fn(dist, self.kernel_peaks) * (dist <= 1.0).float()
        total = raw.sum()
        if total > 0:
            raw = raw / total
        return torch.fft.fft2(torch.roll(raw, shifts=(-mid, -mid), dims=(0, 1)))

    def _init_state(self) -> torch.Tensor:
        Res = self.resolution
        state = torch.zeros(Res, Res, device=self.device, dtype=torch.float32)
        q = Res // 4
        state[q: 3 * q, q: 3 * q] = torch.rand(
            2 * q, 2 * q, device=self.device, dtype=torch.float32
        )
        return state

    def step(self) -> None:
        state_fft = torch.fft.fft2(self.state)
        potential = torch.zeros_like(self.state)
        for kfft, (r0, r1) in zip(self._band_kernel_ffts, self._band_row_ranges):
            band_pot = torch.fft.ifft2(state_fft * kfft).real
            potential[r0:r1, :] = band_pot[r0:r1, :]
        growth = self.growth_fn(potential, self.growth_mu, self.growth_sigma)
        self.state = (self.state + self.dt * growth).clamp(0.0, 1.0)
        # Both directions are periodic — FFT handles this automatically.

    def run(self, n_steps: int, record_interval: int = 10) -> list[torch.Tensor]:
        history: list[torch.Tensor] = []
        for i in range(n_steps):
            self.step()
            if i % record_interval == 0 or i == n_steps - 1:
                history.append(self.state.detach().cpu().clone())
        return history


# ======================================================================
# HyperbolicLenia  —  Lenia on the Poincare disk  (H^2)
# ======================================================================

class HyperbolicLenia:
    r"""Lenia on the Poincare disk model of the hyperbolic plane.

    State representation
    --------------------
    A regular Cartesian grid of resolution N x N covers the unit disk
    ``|z| < 1``.  Cells outside the disk are masked to zero.

    Convolution
    -----------
    No FFT shortcut exists because H^2 is **not** translationally
    invariant in the Euclidean embedding.  Instead we:

    1. Pre-compute coordinates of all *active* cells (inside the disk).
    2. Pre-compute a **sparse** geodesic-distance matrix: for each cell
       keep only neighbours within ``kernel_radius`` geodesic distance
       (sparse COO tensor).
    3. At each step, evaluate the kernel on the sparse distances,
       multiply by the neighbour states, and sum (sparse matmul).

    At resolution 64 (~2000 active cells) this runs in < 1s per step.
    """

    def __init__(self, genome: dict, device: Optional[str] = None) -> None:
        self.genome = dict(genome)
        # Force lower resolution for tractability
        self.resolution: int = min(int(genome.get("resolution", 64)), 128)
        self.dt: float = float(genome.get("dt", 0.1))
        self.device = torch.device(device or genome.get("device", "cpu"))

        R = self.resolution

        # kernel / growth
        kernel_type: str = genome["kernel_type"]
        self.kernel_radius: float = float(genome["kernel_radius"])
        kernel_peaks: float = float(genome.get("kernel_peaks", 1.0))
        if kernel_type not in KERNEL_REGISTRY:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        self.kernel_fn = KERNEL_REGISTRY[kernel_type]
        self.kernel_peaks = kernel_peaks

        growth_type: str = genome["growth_type"]
        if growth_type not in GROWTH_REGISTRY:
            raise ValueError(f"Unknown growth type: {growth_type}")
        self.growth_fn = GROWTH_REGISTRY[growth_type]
        self.growth_mu: float = float(genome["growth_mu"])
        self.growth_sigma: float = float(genome["growth_sigma"])

        # --- grid coordinates in the Poincare disk -----------------------
        # Map pixel (i, j) to complex coordinate z with |z| < 1
        lin = torch.linspace(-0.95, 0.95, R, device=self.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        r2 = gx ** 2 + gy ** 2
        self.disk_mask = (r2 < 0.9025).cpu()  # |z| < 0.95, keep some margin

        # Flat indices of active cells
        active_idx = self.disk_mask.flatten().nonzero(as_tuple=False).squeeze(1)
        self.n_active = active_idx.shape[0]
        self.active_idx = active_idx  # CPU indices

        # Active cell coordinates (Euclidean in the disk)
        gx_flat = gx.flatten()
        gy_flat = gy.flatten()
        self.ax = gx_flat[active_idx].to(self.device)  # (N,)
        self.ay = gy_flat[active_idx].to(self.device)

        # --- pre-compute sparse geodesic distance matrix -----------------
        self._build_sparse_kernel()

        # --- initial state -----------------------------------------------
        self.state = self._init_state()

    def _poincare_dist(
        self,
        x1: torch.Tensor, y1: torch.Tensor,
        x2: torch.Tensor, y2: torch.Tensor,
    ) -> torch.Tensor:
        """Geodesic distance in the Poincare disk between two points."""
        dx = x1 - x2
        dy = y1 - y2
        num = dx ** 2 + dy ** 2
        denom = (1.0 - (x1 ** 2 + y1 ** 2)) * (1.0 - (x2 ** 2 + y2 ** 2))
        denom = denom.clamp(min=1e-8)
        arg = 1.0 + 2.0 * num / denom
        return torch.acosh(arg.clamp(min=1.0))

    def _build_sparse_kernel(self) -> None:
        """Pre-compute sparse neighbour list and normalised kernel values.

        We process in chunks to avoid OOM on large active sets.
        """
        N = self.n_active
        # Maximum geodesic distance that maps to r <= 1 after normalisation
        # We need: geodesic_dist / max_geodesic <= 1
        # The kernel radius in geodesic units — we scale so that
        # kernel_radius (in pixel-equivalents) maps to ~1 in normalised kernel coords.
        # On the Poincare disk at centre, pixel spacing ~ 2*0.95/R = 1.9/R
        # So kernel_radius pixels ~ kernel_radius * 1.9/R geodesic distance (approx at centre).
        # Near the boundary geodesic distances grow rapidly.
        # We use a fixed geodesic cutoff and normalise within it.
        R = self.resolution
        pixel_spacing = 1.9 / R
        self.geodesic_cutoff = self.kernel_radius * pixel_spacing * 2.5  # generous cutoff

        chunk_size = 256  # process this many source cells at a time
        row_list = []
        col_list = []
        val_list = []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            # source cells: (end-start,)
            sx = self.ax[start:end].unsqueeze(1)  # (C, 1)
            sy = self.ay[start:end].unsqueeze(1)
            # target cells: (N,)
            tx = self.ax.unsqueeze(0)  # (1, N)
            ty = self.ay.unsqueeze(0)

            gdist = self._poincare_dist(sx, sy, tx, ty)  # (C, N)
            mask = gdist < self.geodesic_cutoff
            r_norm = gdist / self.geodesic_cutoff  # normalise to [0, 1]

            # Evaluate kernel
            kvals = self.kernel_fn(r_norm, self.kernel_peaks) * (r_norm <= 1.0).float()
            kvals = kvals * mask.float()

            # Extract sparse entries
            rows_c, cols_c = mask.nonzero(as_tuple=True)
            rows_c = rows_c + start
            vals_c = kvals[mask]

            row_list.append(rows_c)
            col_list.append(cols_c)
            val_list.append(vals_c)

        all_rows = torch.cat(row_list)
        all_cols = torch.cat(col_list)
        all_vals = torch.cat(val_list)

        # Build sparse COO tensor (N x N)
        indices = torch.stack([all_rows, all_cols], dim=0).to(torch.long)
        sp = torch.sparse_coo_tensor(indices, all_vals, size=(N, N),
                                     device=self.device).coalesce()

        # Row-normalise (each row sums to 1)
        row_sums = torch.sparse.sum(sp, dim=1).to_dense().clamp(min=1e-10)
        # Scale values
        norm_vals = sp.values() / row_sums[sp.indices()[0]]
        self._kernel_sparse = torch.sparse_coo_tensor(
            sp.indices(), norm_vals, size=(N, N), device=self.device
        ).coalesce()

    def _init_state(self) -> torch.Tensor:
        """Random seed in the centre of the disk."""
        R = self.resolution
        state_2d = torch.zeros(R, R, device=self.device, dtype=torch.float32)
        q = R // 4
        centre_patch = torch.rand(2 * q, 2 * q, device=self.device, dtype=torch.float32)
        state_2d[q: 3 * q, q: 3 * q] = centre_patch
        # Mask to disk
        state_2d = state_2d * self.disk_mask.to(self.device).float()
        return state_2d

    def _state_to_active(self, state_2d: torch.Tensor) -> torch.Tensor:
        """Extract active-cell values as flat vector."""
        return state_2d.flatten()[self.active_idx.to(self.device)]

    def _active_to_state(self, active_vals: torch.Tensor) -> torch.Tensor:
        """Scatter active-cell values back to 2D grid."""
        R = self.resolution
        flat = torch.zeros(R * R, device=self.device, dtype=torch.float32)
        flat[self.active_idx.to(self.device)] = active_vals
        return flat.view(R, R)

    def step(self) -> None:
        a = self._state_to_active(self.state)  # (N,)
        # Sparse matrix-vector product: potential(i) = sum_j K(i,j) * a(j)
        potential = torch.mv(self._kernel_sparse.to_dense(), a)
        # (Using .to_dense() for mv — at N~2000 this is fast and avoids
        #  sparse mv issues on MPS/CPU; for larger N switch to sparse mm)

        growth = self.growth_fn(potential, self.growth_mu, self.growth_sigma)
        a_new = (a + self.dt * growth).clamp(0.0, 1.0)
        self.state = self._active_to_state(a_new)

    def run(self, n_steps: int, record_interval: int = 10) -> list[torch.Tensor]:
        history: list[torch.Tensor] = []
        for i in range(n_steps):
            self.step()
            if i % record_interval == 0 or i == n_steps - 1:
                history.append(self.state.detach().cpu().clone())
        return history
