#!/usr/bin/env python3
"""Robustness test for the hero expression cos(4π·laplacian(A)).

Runs the expression with 10 different random seeds at 128×128 and 256×256,
recording metrics and saving final state images.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.expression_compiler import compile_expression
from src.evaluator import evaluate as compute_metrics_from_history

HERO_EXPR = (
    "clip(A + 0.31 * (max(exp(-((conv(A) - (0.12 + 0.03 * cos(4 * pi * laplacian(A)))) "
    "/ 0.055)**2 / 2), tanh(3.8 * (conv(A) + 0.15 * laplacian(A) - 0.18))) * "
    "threshold(conv(A) + laplacian(A), 0.08) * (1 - threshold(conv(A) + laplacian(A), 0.32)) "
    "+ 0.12 * sin(7 * pi * (conv(A) - 0.5 * laplacian(A))) * "
    "exp(-((conv(A) - 0.16) / 0.065)**2 / 2)), 0, 1)"
)

SEEDS = list(range(10))
RESOLUTIONS = [128, 256]
STEPS = 8000
KERNEL_RADIUS = 13
DEVICE = "mps"

out_dir = "results/robustness_hero"
os.makedirs(out_dir, exist_ok=True)

results = []

for res in RESOLUTIONS:
    print(f"\n{'='*60}")
    print(f"  Resolution: {res}×{res}")
    print(f"{'='*60}")

    step_fn = compile_expression(HERO_EXPR, resolution=res, kernel_radius=KERNEL_RADIUS)

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Random quarter init
        state = torch.zeros(res, res, device=DEVICE)
        quarter = res // 4
        start_r = res // 2 - quarter // 2
        start_c = res // 2 - quarter // 2
        state[start_r:start_r+quarter, start_c:start_c+quarter] = torch.rand(quarter, quarter, device=DEVICE)

        t0 = time.time()

        # Save snapshots at key timesteps
        snapshots = {}
        snapshot_times = [0, 1000, 2000, 4000, 8000]

        # Collect state history for metrics (every 10 steps for tail)
        record_interval = max(1, STEPS // 300)
        state_history = [state.cpu()]

        for step in range(STEPS + 1):
            if step in snapshot_times:
                snapshots[step] = state.cpu().numpy().copy()
            if step < STEPS:
                state = step_fn(state)
                state = state.clamp(0, 1)
                if (step + 1) % record_interval == 0 or step == STEPS - 1:
                    state_history.append(state.cpu())

        elapsed = time.time() - t0
        final = state.cpu().numpy()

        # Compute metrics using evaluator
        metrics = compute_metrics_from_history(state_history)

        result = {
            "resolution": res,
            "seed": seed,
            "time_s": round(elapsed, 2),
            "alive_fraction": round(metrics["alive_fraction"], 4),
            "complexity": round(metrics["complexity"], 4),
            "num_clusters": metrics["num_clusters"],
            "spatial_entropy": round(metrics["spatial_entropy"], 4),
        }
        results.append(result)

        print(f"  Seed {seed}: alive={result['alive_fraction']:.3f} "
              f"compl={result['complexity']:.3f} clusters={result['num_clusters']} "
              f"entropy={result['spatial_entropy']:.3f} ({elapsed:.1f}s)")

        # Save final state image
        img = (final * 255).astype(np.uint8)
        Image.fromarray(img).save(f"{out_dir}/hero_res{res}_seed{seed}.png")

        # Save snapshot panel
        n_snaps = len(snapshot_times)
        panel_w = res * n_snaps + (n_snaps - 1) * 4
        panel = np.ones((res + 30, panel_w), dtype=np.uint8) * 255
        for i, t in enumerate(snapshot_times):
            x_offset = i * (res + 4)
            snap_img = (snapshots[t] * 255).astype(np.uint8)
            panel[:res, x_offset:x_offset+res] = snap_img
        Image.fromarray(panel).save(f"{out_dir}/hero_res{res}_seed{seed}_timeline.png")

# Summary statistics
print(f"\n{'='*60}")
print(f"  ROBUSTNESS SUMMARY")
print(f"{'='*60}")

for res in RESOLUTIONS:
    res_results = [r for r in results if r["resolution"] == res]
    alive = [r["alive_fraction"] for r in res_results]
    compl = [r["complexity"] for r in res_results]
    clusters = [r["num_clusters"] for r in res_results]
    entropy = [r["spatial_entropy"] for r in res_results]

    print(f"\n  {res}×{res} (n={len(res_results)}):")
    print(f"    alive_fraction: {np.mean(alive):.4f} ± {np.std(alive):.4f} [{min(alive):.3f} - {max(alive):.3f}]")
    print(f"    complexity:     {np.mean(compl):.4f} ± {np.std(compl):.4f} [{min(compl):.3f} - {max(compl):.3f}]")
    print(f"    num_clusters:   {np.mean(clusters):.1f} ± {np.std(clusters):.1f} [{min(clusters)} - {max(clusters)}]")
    print(f"    entropy:        {np.mean(entropy):.4f} ± {np.std(entropy):.4f}")

# Save results JSON
with open(f"{out_dir}/robustness_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {out_dir}/")
