#!/usr/bin/env python3
"""Manual verification: does Laplacian-modulated growth alone produce geometric patterns?

Implements a minimal Lenia variant where:
  A += dt * G(K*A, ∇²A)
  G center μ = μ₀ + ε·cos(4π·∇²A)

This isolates the core mechanism of the hero expression (cos(4π·∇²A) modulating
growth center) without the LLM-generated bandpass window or harmonic modulation.
"""

import torch
import numpy as np
from PIL import Image
import os

def run_verification(
    size=128,
    steps=8000,
    R=13,
    mu0=0.12,
    eps=0.03,
    sigma=0.055,
    dt=0.31,
    seed=42,
    device="cpu",
):
    torch.manual_seed(seed)

    # --- Kernel: standard Gaussian Lenia kernel ---
    mid = size // 2
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32) - mid,
        torch.arange(size, device=device, dtype=torch.float32) - mid,
        indexing="ij",
    )
    r = torch.sqrt(xx**2 + yy**2) / R
    # Gaussian kernel shape, normalized
    K = torch.exp(-((r - 0.5) ** 2) / (2 * 0.15**2))
    K[r > 1] = 0
    K = K / K.sum()
    K_fft = torch.fft.rfft2(torch.roll(K, (-mid, -mid), dims=(0, 1)))

    # --- Laplacian kernel (3x3 discrete) ---
    lap_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device
    )
    lap_pad = torch.zeros(size, size, device=device)
    lap_pad[:3, :3] = lap_kernel
    lap_pad = torch.roll(lap_pad, (-1, -1), dims=(0, 1))
    lap_fft = torch.fft.rfft2(lap_pad)

    # --- Initial condition: random quarter-fill ---
    A = torch.zeros(size, size, device=device)
    q = size // 4
    A[q : 3 * q, q : 3 * q] = torch.rand(2 * q, 2 * q, device=device)

    # --- Simulation loop ---
    for step in range(steps):
        A_fft = torch.fft.rfft2(A)

        # Convolution K*A
        conv_A = torch.fft.irfft2(A_fft * K_fft, s=(size, size))

        # Laplacian ∇²A
        lap_A = torch.fft.irfft2(A_fft * lap_fft, s=(size, size))

        # Core mechanism: Laplacian-modulated growth center
        mu = mu0 + eps * torch.cos(4 * torch.pi * lap_A)
        growth = torch.exp(-((conv_A - mu) ** 2) / (2 * sigma**2))

        # Update (simple version: growth - 1 maps [0,1] → [-1, 0], so use 2*growth - 1)
        A = torch.clamp(A + dt * (2 * growth - 1), 0, 1)

        if step % 2000 == 0:
            alive = (A > 0.1).float().mean().item()
            comp = _lzma_complexity(A)
            print(f"  step {step:5d}: alive={alive:.3f}, complexity={comp:.3f}")

    alive = (A > 0.1).float().mean().item()
    comp = _lzma_complexity(A)
    print(f"  FINAL step {steps}: alive={alive:.3f}, complexity={comp:.3f}")

    return A.cpu().numpy(), alive, comp


def _lzma_complexity(A):
    """LZMA compression ratio as complexity proxy."""
    import lzma
    arr = (A * 255).clamp(0, 255).byte().cpu().numpy()
    raw = arr.tobytes()
    compressed = lzma.compress(raw)
    return len(compressed) / len(raw)


def save_image(arr, path):
    img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    img.save(path)


if __name__ == "__main__":
    os.makedirs("results/verification", exist_ok=True)

    print("=== Verification: Laplacian-modulated growth (core mechanism only) ===")
    print("G center μ = 0.12 + 0.03·cos(4π·∇²A), σ = 0.055, dt = 0.31")
    print()

    # Run with multiple seeds to check consistency
    for seed in [42, 123, 7]:
        print(f"--- Seed {seed} ---")
        final_state, alive, comp = run_verification(seed=seed)
        save_image(final_state, f"results/verification/laplacian_growth_seed{seed}.png")
        print(f"  Saved to results/verification/laplacian_growth_seed{seed}.png")
        print()

    print("=== Comparison: standard Lenia (fixed growth center, no Laplacian) ===")
    print("G center μ = 0.12 (fixed), σ = 0.055, dt = 0.31")
    print()

    # Control: standard Lenia with same parameters but NO Laplacian modulation
    for seed in [42]:
        print(f"--- Seed {seed} (control) ---")
        final_state, alive, comp = run_verification(seed=seed, eps=0.0)
        save_image(final_state, f"results/verification/standard_growth_seed{seed}.png")
        print(f"  Saved to results/verification/standard_growth_seed{seed}.png")
        print()

    print("Done. Compare images in results/verification/")
