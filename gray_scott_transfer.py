#!/usr/bin/env python3
"""
Gray-Scott cross-substrate transfer experiment.

Tests whether Track B expressions discovered on Lenia produce
non-trivial patterns when injected into Gray-Scott reaction-diffusion.
"""

import os
import sys
import lzma
import numpy as np
import torch
import torch.nn.functional as F_torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class GrayScott:
    def __init__(self, resolution=256, Du=0.16, Dv=0.08,
                 feed=0.035, kill=0.065, dt=1.0, dx=1.0):
        self.res = resolution
        self.Du, self.Dv = Du, Dv
        self.feed, self.kill = feed, kill
        self.dt = dt
        self.dx = dx
        # Laplacian kernel divided by dx² for correct spatial scaling
        self.lap_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32
        ).reshape(1, 1, 3, 3) / (dx * dx)

    def init_state(self):
        U = torch.ones(1, 1, self.res, self.res)
        V = torch.zeros(1, 1, self.res, self.res)
        cx, cy = self.res // 2, self.res // 2
        r = self.res // 10
        y, x = torch.meshgrid(
            torch.arange(self.res), torch.arange(self.res), indexing='ij'
        )
        mask = ((x - cx)**2 + (y - cy)**2) < r**2
        V[0, 0, mask] = 1.0
        U += torch.randn_like(U) * 0.01
        V += torch.randn_like(V) * 0.01
        return U.clamp(0, 1), V.clamp(0, 1)

    def laplacian(self, field):
        padded = F_torch.pad(field, (1, 1, 1, 1), mode='circular')
        return F_torch.conv2d(padded, self.lap_kernel)

    def step(self, U, V):
        lap_U = self.laplacian(U)
        lap_V = self.laplacian(V)
        UVV = U * V * V
        dU = self.Du * lap_U - UVV + self.feed * (1 - U)
        dV = self.Dv * lap_V + UVV - (self.feed + self.kill) * V
        return torch.clamp(U + self.dt * dU, 0, 1), torch.clamp(V + self.dt * dV, 0, 1)

    def step_modified(self, U, V, modifier_fn, strength=0.01):
        lap_U = self.laplacian(U)
        lap_V = self.laplacian(V)
        UVV = U * V * V
        modification = modifier_fn(U, V, lap_U, lap_V)
        dU = self.Du * lap_U - UVV + self.feed * (1 - U) + strength * modification
        dV = self.Dv * lap_V + UVV - (self.feed + self.kill) * V
        return torch.clamp(U + self.dt * dU, 0, 1), torch.clamp(V + self.dt * dV, 0, 1)


# Track B expressions translated to Gray-Scott modifiers
MODIFIERS = {
    "baseline_none": lambda U, V, lap_U, lap_V: torch.zeros_like(U),
    "cos_laplacian": lambda U, V, lap_U, lap_V: torch.cos(4 * 3.14159 * lap_U),
    "tanh_lap_sq": lambda U, V, lap_U, lap_V: torch.tanh(2 * lap_U ** 2),
    "pythagorean": lambda U, V, lap_U, lap_V: torch.sin(
        3.14159 * torch.sqrt(lap_U**2 + lap_V**2 + 1e-8)
    ),
    "compete": lambda U, V, lap_U, lap_V: U * (1 - V) - V * (1 - U),
    "logistic": lambda U, V, lap_U, lap_V: U * (1 - U),
}


def run_transfer_experiment(
    steps=10000, resolution=256,
    save_dir="results/gray_scott_transfer_v2",
    strengths=(0.005, 0.02),
):
    os.makedirs(save_dir, exist_ok=True)

    # Standard Gray-Scott parameters from Pearson (1993)
    # Du=2e-5, Dv=1e-5 with dx=1/resolution, dt=1
    # OR equivalently Du=0.16, Dv=0.08 with dx=2.0, dt=0.5
    # Using the rescaled version for numerical stability
    gs = GrayScott(resolution=resolution, Du=0.16, Dv=0.08,
                   feed=0.035, kill=0.065, dt=0.5, dx=2.0)
    snapshot_times = [0, 2500, 5000, 7500, steps - 1]

    all_results = {}

    for strength in strengths:
        print(f"\n=== Strength = {strength} ===")

        for name, modifier in MODIFIERS.items():
            print(f"  Running {name}...", end="", flush=True)
            U, V = gs.init_state()

            snapshots = []
            for t in range(steps):
                if name == "baseline_none":
                    U, V = gs.step(U, V)
                else:
                    U, V = gs.step_modified(U, V, modifier, strength=strength)

                if t in snapshot_times:
                    snapshots.append(V[0, 0].detach().cpu().numpy().copy())

            # Metrics
            final = snapshots[-1]
            alive = float((final > 0.01).mean())
            raw = (final * 255).astype(np.uint8).tobytes()
            complexity = len(lzma.compress(raw)) / len(raw)

            print(f" alive={alive:.3f}, complexity={complexity:.3f}")
            all_results[f"{name}_s{strength}"] = {
                "alive": alive, "complexity": complexity
            }

            # Plot
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            fig.suptitle(
                f"Gray-Scott + {name} (strength={strength})\n"
                f"alive={alive:.3f}, complexity={complexity:.3f}",
                fontsize=13, fontweight='bold'
            )
            for i, (snap, t) in enumerate(
                zip(snapshots, [0, 2500, 5000, 7500, steps])
            ):
                axes[i].imshow(snap, cmap='viridis', vmin=0, vmax=1)
                axes[i].set_title(f"t={t}", fontsize=10)
                axes[i].axis('off')
            plt.tight_layout()
            fname = f"{save_dir}/{name}_s{strength}.png"
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close()

    # Summary comparison
    print("\n=== Summary ===")
    print(f"{'Name':<25} {'Alive':>8} {'Complexity':>12}")
    print("-" * 47)
    for key, vals in all_results.items():
        print(f"{key:<25} {vals['alive']:>8.3f} {vals['complexity']:>12.3f}")


if __name__ == "__main__":
    run_transfer_experiment()
