#!/usr/bin/env python3
"""Robustness test for the hero moving soliton (bessel / gaussian_bell).

Reconstructed from `results_localized_v4/robustness_test/robustness_results.json`
(the original generating script was lost). Reuses the exact Lenia stepping and
soliton-metric machinery from `localized_search_v4_locomotion.py`:

  * Anisotropic kernel:  K <- K * (1 + aniso_strength * cos(theta - aniso_angle))
    (via AnisotropicLenia._build_kernel_fft)
  * Off-center elliptical (2:1 aspect) blob initial condition, identical shape to
    the v4 search, but seeded at one of three grid positions.
  * net_displacement / spatial_concentration / alive_fraction from
    compute_localized_metrics (centroid at halfway-step vs final step, torus-wrapped).
  * direction_deg = atan2(ddy, ddx) of that same centroid displacement, in degrees.

The blob init is fully deterministic (no per-cell randomness), so the random
`seed` only perturbs nothing that affects the trajectory: all seeds at a given
position therefore yield identical results — consistent with the saved JSON.

We test 3 positions x 3 seeds = 9 trials at 256^2 for 10000 steps.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localized_search_v4_locomotion import (
    AnisotropicLenia,
    compute_localized_metrics,
)


# ---------------------------------------------------------------------------
# Hero genome (matches results_localized_v4/best_movers.json -> bessel/gaussian_bell)
# ---------------------------------------------------------------------------

GENOME = {
    "kernel_type": "bessel",
    "growth_type": "gaussian_bell",
    "geometry": "flat_plane",
    "resolution": 256,
    "kernel_radius": 16.23129231401403,
    "kernel_peaks": 2.1017351305047023,
    "growth_mu": 0.3114231354958413,
    "growth_sigma": 0.059725031834391214,
    "dt": 0.1594991747000343,
    "aniso_strength": 0.5818840700707387,
    "aniso_angle": 2.96441452310977,
}

# Fractional grid positions for the blob centre (cx_frac, cy_frac).
POSITIONS = {
    "top_left": (0.25, 0.25),
    "center": (0.5, 0.5),
    "bottom_right": (0.75, 0.75),
}

SEEDS = [42, 123, 777]

RES = 256
STEPS = 10000
DEVICE = "mps"

OUT_DIR = "results_localized_v4/robustness_test"
OUT_JSON = f"{OUT_DIR}/robustness_results_reproduced.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_init_state(cx_frac: float, cy_frac: float, res: int, device: str) -> torch.Tensor:
    """Off-center elliptical (2:1 aspect) blob, identical shape to the v4 search
    init, placed at the given fractional position. Deterministic (no randomness)."""
    cx_init = cx_frac * res
    cy_init = cy_frac * res
    Y_t = torch.arange(res, device=device).float()
    X_t = torch.arange(res, device=device).float()
    YY, XX = torch.meshgrid(Y_t, X_t, indexing="ij")
    # sigma_x = 0.06*RES, sigma_y = 0.03*RES  (2:1 aspect, as in v4 locomotion init)
    r2 = ((XX - cx_init) / (res * 0.06)) ** 2 + ((YY - cy_init) / (res * 0.03)) ** 2
    return torch.exp(-r2) * 0.8


def centroid(s: np.ndarray, X: np.ndarray, Y: np.ndarray):
    m = s.sum()
    if m < 1e-6:
        return None
    return ((X * s).sum() / m, (Y * s).sum() / m)


def direction_deg_from_history(state_history, res: int) -> float:
    """Heading angle (degrees) of the centroid displacement from the halfway
    snapshot to the final snapshot, using the same torus-wrapped deltas as
    compute_localized_metrics' net_displacement."""
    Y, X = np.mgrid[0:res, 0:res]
    half_idx = len(state_history) // 2

    def _np(s):
        return s.cpu().numpy() if isinstance(s, torch.Tensor) else s

    c_start = centroid(_np(state_history[half_idx]), X, Y)
    c_end = centroid(_np(state_history[-1]), X, Y)
    if c_start is None or c_end is None:
        return 0.0

    # Torus-wrapped signed deltas (choose the shorter direction on the ring).
    def _wrap(d):
        if d > res / 2:
            d -= res
        elif d < -res / 2:
            d += res
        return d

    ddx = _wrap(c_end[0] - c_start[0])
    ddy = _wrap(c_end[1] - c_start[1])
    return float(np.degrees(np.arctan2(ddy, ddx)))


def run_trial(pos_name: str, cx_frac: float, cy_frac: float, seed: int):
    sim = AnisotropicLenia(GENOME, device=DEVICE)

    # Per-trial seed (does not affect the deterministic blob, kept for fidelity).
    torch.manual_seed(seed)
    np.random.seed(seed)
    sim.state = make_init_state(cx_frac, cy_frac, RES, DEVICE)

    record_interval = max(1, STEPS // 200)
    state_history = [sim.state.cpu()]
    for step in range(STEPS):
        sim.step()
        if (step + 1) % record_interval == 0 or step == STEPS - 1:
            state_history.append(sim.state.cpu())
        if (step + 1) % 2000 == 0:
            s = sim.state.cpu().numpy()
            if (s > 0.1).mean() < 0.005:
                break

    metrics = compute_localized_metrics(state_history, RES)
    direction = direction_deg_from_history(state_history, RES)

    return {
        "pos": pos_name,
        "seed": seed,
        "nd": round(metrics["net_displacement"], 2),
        "conc": round(metrics["spatial_concentration"], 4),
        "alive": round(metrics["alive_fraction"], 4),
        "direction_deg": round(direction, 1),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Robustness test: bessel / gaussian_bell moving soliton")
    print(f"  Resolution {RES}x{RES}, {STEPS} steps, device={DEVICE}")
    print(f"  Positions: {list(POSITIONS)}   Seeds: {SEEDS}")
    print("=" * 72)

    t_start = time.time()
    results = []
    for pos_name, (cx_frac, cy_frac) in POSITIONS.items():
        for seed in SEEDS:
            r = run_trial(pos_name, cx_frac, cy_frac, seed)
            results.append(r)
            print(f"  {pos_name:>12s} seed={seed:<4d}  nd={r['nd']:7.2f}  "
                  f"conc={r['conc']:.4f}  alive={r['alive']:.4f}  "
                  f"dir={r['direction_deg']:7.1f} deg  "
                  f"({time.time()-t_start:.0f}s)")

    # --- summary ---
    directions = np.array([r["direction_deg"] for r in results])
    nds = np.array([r["nd"] for r in results])
    # "Survival" = produced a surviving moving soliton (localized + actually moving).
    survived = sum(
        1 for r in results
        if r["alive"] > 0.001 and r["conc"] > 0.7 and r["nd"] > 5
    )

    print("=" * 72)
    print("  9-TRIAL TABLE")
    print(f"  {'pos':>12s} {'seed':>5s} {'nd':>8s} {'conc':>8s} {'alive':>8s} {'dir_deg':>9s}")
    for r in results:
        print(f"  {r['pos']:>12s} {r['seed']:>5d} {r['nd']:>8.2f} "
              f"{r['conc']:>8.4f} {r['alive']:>8.4f} {r['direction_deg']:>9.1f}")
    print("-" * 72)
    print(f"  Survival:        {survived}/9")
    print(f"  net_displacement mean = {nds.mean():.2f}  "
          f"[{nds.min():.2f}, {nds.max():.2f}]")
    print(f"  direction mean   = {directions.mean():.2f} deg")
    print(f"  direction std    = {np.std(directions):.4f} deg  (np.std, ddof=0)")
    print(f"  Total time:      {time.time()-t_start:.0f}s")

    save_data = {"genome": GENOME, "results": results}
    with open(OUT_JSON, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved reproduced results to {OUT_JSON}")


if __name__ == "__main__":
    main()
