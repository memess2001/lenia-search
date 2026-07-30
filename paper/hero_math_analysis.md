# Mathematical Analysis of the Hero Expression: cos(4π · ∇²A)

## 1. The Full Expression (Simplified)

The hero entry's update rule can be decomposed into three functional components:

```
A_{t+1} = clip(A + dt · [G₁(u, L) · W(u, L) + G₂(u, L)], 0, 1)
```

where:
- `u = conv(A)` — standard Lenia convolution (weighted spatial average)
- `L = laplacian(A)` — spatial Laplacian (∇²A, second spatial derivative)
- `dt = 0.31` — time step

### Component 1: Laplacian-modulated growth G₁

```
G₁(u, L) = max(
    exp(-(u - μ(L))² / (2σ²)),     ← Gaussian bell with Laplacian-modulated center
    tanh(3.8 · (u + 0.15L - 0.18))  ← Hyperbolic tangent backup
)
```

where **μ(L) = 0.12 + 0.03 · cos(4π · L)** is the key innovation.

In standard Lenia: μ is a constant (e.g., μ = 0.15).
Here: μ varies spatially, driven by the local curvature of the field.

### Component 2: Windowing function W

```
W(u, L) = threshold(u + L, 0.08) · (1 - threshold(u + L, 0.32))
```

This creates a bandpass filter: growth only occurs where `0.08 < u + L < 0.32`. Regions outside this band are frozen (no growth or decay from G₁).

### Component 3: Sinusoidal modulation G₂

```
G₂(u, L) = 0.12 · sin(7π · (u - 0.5L)) · exp(-(u - 0.16)² / (2 · 0.065²))
```

A high-frequency sinusoidal oscillation modulated by a narrow Gaussian envelope. This adds fine-grained spatial texture within the active region.

---

## 2. Why cos(4π · ∇²A) Creates Discrete Stable States

The key term is **μ(L) = 0.12 + 0.03 · cos(4πL)**.

### The Mechanism

Consider cos(4πx) as a function of x ∈ ℝ:

```
cos(4πx) has zero crossings at x = 1/8, 3/8, 5/8, 7/8, ...
cos(4πx) has maxima at x = 0, 1/2, 1, 3/2, ...
cos(4πx) has minima at x = 1/4, 3/4, 5/4, ...
```

When applied to the Laplacian L = ∇²A:
- **At flat regions** (L ≈ 0): cos(0) = 1, so μ = 0.12 + 0.03 = 0.15
- **At slightly curved regions** (L ≈ 1/4): cos(π) = -1, so μ = 0.12 - 0.03 = 0.09
- **At more curved regions** (L ≈ 1/2): cos(2π) = 1, so μ = 0.15 again

This creates a **periodic modulation of the growth center as a function of local curvature**. The effect is:

1. **Flat regions** (low |∇²A|) have growth centered at μ = 0.15
2. **Moderately curved regions** have growth centered at μ = 0.09
3. **Highly curved regions** cycle back to μ = 0.15

This creates **multiple basins of attraction** in the density field:
- Density stabilizes at values where the convolution u ≈ μ(L)
- Since μ depends on L, and L depends on the spatial arrangement, this creates a self-consistent system with multiple discrete stable density levels
- The cosine's periodicity means the system doesn't just have two states (high/low) but potentially many, depending on the range of L values present

### Why This Is Structurally Different from Standard Lenia

In standard Lenia: `A += dt · G(K * A)` where G has a fixed center μ.
- Growth is purely a function of the weighted average of neighbors
- There is ONE stable density (where G(u) = 0, i.e., u = μ)
- Spatial derivatives play no role in growth decisions

In the hero expression: `A += dt · G₁(conv(A), laplacian(A)) · ...`
- Growth depends on BOTH the weighted average AND the local curvature
- The curvature-dependent μ creates MULTIPLE stable densities
- This is analogous to a "quantization" of the continuous field into discrete levels

### Physical Analogy

This mechanism resembles **phase-field models** in physics:
- cos(4π · ∇²A) acts like a free energy landscape with multiple wells
- Each well corresponds to a stable phase (density level)
- The Laplacian coupling creates spatial coherence (nearby cells prefer the same phase)
- The result is a spatial decomposition into distinct "phases" — the geometric pattern

This is fundamentally different from Lenia's Turing-instability mechanism, where patterns arise from the interplay of short-range activation and long-range inhibition with a SINGLE growth center.

---

## 3. The Structural Inexpressibility Argument

**Claim**: The term cos(4π · laplacian(A)) is structurally inexpressible in the Lenia parametric framework A_{t+1} = A_t + dt · G(K * A_t).

**Proof sketch**:

1. In the Lenia framework, G receives a single scalar argument: u = (K * A)(x,y) — the convolution of A at point (x,y).
2. The convolution K * A is a linear operation that computes a weighted spatial average. It is a zeroth-order spatial statistic (no derivatives).
3. The Laplacian ∇²A is a second-order spatial differential operator.
4. No choice of kernel K can make K * A equal to ∇²A for all fields A, because:
   - K * A is always a smoothing operation (weighted average)
   - ∇²A is a sharpening operation (second derivative)
   - Their Fourier transforms have opposite spectral characteristics: convolution suppresses high frequencies, Laplacian amplifies them
5. Even if one approximated ∇²A via a specific kernel (e.g., Laplacian-of-Gaussian), the Lenia framework applies G to the RESULT of the convolution. It cannot apply different nonlinear functions to the convolution and the Laplacian independently, because G has only one input channel.
6. The hero expression applies cos(4π · ·) to the Laplacian INSIDE the growth center μ, while simultaneously using the convolution value u as the growth argument. This requires G to be a function of TWO independent spatial statistics — impossible with a single-argument growth function.

**QED**: The hero expression's mathematical structure lies outside the expressibility of G(K * A) for any choice of K and G. □

---

## 4. Figure Suggestion for Paper

**Figure: "Spatially-Aware Growth via Laplacian Modulation"**

Panel A: Standard Lenia growth
- x-axis: convolution value u
- y-axis: growth G(u)
- Single bell curve centered at fixed μ = 0.15
- Caption: "Standard Lenia: one growth center, one stable density"

Panel B: Hero expression growth
- x-axis: convolution value u
- Multiple bell curves, each for a different Laplacian value L
- L = 0 (flat): bell at μ = 0.15
- L = 0.125 (slight curve): bell at μ = 0.12
- L = 0.25 (moderate curve): bell at μ = 0.09
- L = 0.375: bell at μ = 0.12
- L = 0.5: bell at μ = 0.15
- Caption: "Hero expression: growth center varies with local curvature, creating multiple stable densities"

Panel C: cos(4π · x) function
- x-axis: Laplacian value L
- y-axis: cos(4π · L)
- Mark the zero crossings and extrema
- Show μ(L) = 0.12 + 0.03 · cos(4πL) on secondary axis
- Caption: "Cosine modulation creates periodic growth-center variation"

Panel D: Resulting pattern (256×256 simulation snapshot)
- High-res render of the hero expression
- Color-coded by density level, showing discrete phases
- Caption: "Self-organized geometric pattern with discrete density levels"
