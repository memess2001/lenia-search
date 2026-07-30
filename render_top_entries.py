#!/usr/bin/env python3
"""Re-render top archive entries at 256×256 for paper figures.

Renders both the hero expression (Track B) and top Track A entries
at high resolution, saving final states and timeline panels.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.geometry_factory import create_lenia
from src.evaluator import evaluate as compute_metrics_from_history

RESOLUTION = 256
STEPS = 8000
DEVICE = "mps"

out_dir = "results/highres_renders"
os.makedirs(out_dir, exist_ok=True)

# Load merged archive
with open("results_cloud/archive_merged.json") as f:
    data = json.load(f)

entries = data["entries"]
entries_sorted = sorted(entries, key=lambda e: e["fitness"], reverse=True)

# Take top 5 Track A entries (hero is already rendered by robustness test)
top_trackA = [e for e in entries_sorted if "kernel_type" in e["genome"]][:5]

print(f"Rendering {len(top_trackA)} top Track A entries at {RESOLUTION}×{RESOLUTION}")
print(f"Steps: {STEPS}, Device: {DEVICE}")
print("=" * 60)

results = []

for idx, entry in enumerate(top_trackA):
    genome = entry["genome"]
    bin_idx = entry["bin"]
    fitness = entry["fitness"]

    kt = genome["kernel_type"]
    gt = genome["growth_type"]
    geo = genome["geometry"]

    print(f"\n[{idx+1}/{len(top_trackA)}] {kt}/{gt}/{geo}")
    print(f"  bin={bin_idx}, fitness={fitness:.4f}")

    # Create simulation
    genome_full = dict(genome)
    genome_full["resolution"] = RESOLUTION

    try:
        sim = create_lenia(genome_full, device=DEVICE)
    except Exception as e:
        print(f"  ERROR creating sim: {e}")
        continue

    # Init: mixed mode (try random_quarter first, center_blob backup)
    torch.manual_seed(42)
    np.random.seed(42)

    state = torch.zeros(RESOLUTION, RESOLUTION, device=DEVICE)
    quarter = RESOLUTION // 4
    sr = RESOLUTION // 2 - quarter // 2
    sc = RESOLUTION // 2 - quarter // 2
    state[sr:sr+quarter, sc:sc+quarter] = torch.rand(quarter, quarter, device=DEVICE)

    # Set initial state
    sim.state = state

    t0 = time.time()

    # Save snapshots
    snapshots = {}
    snapshot_times = [0, 1000, 2000, 4000, 8000]

    record_interval = max(1, STEPS // 300)
    state_history = [sim.state.cpu()]

    for step in range(STEPS + 1):
        if step in snapshot_times:
            snapshots[step] = sim.state.cpu().numpy().copy()
        if step < STEPS:
            sim.step()
            if (step + 1) % record_interval == 0 or step == STEPS - 1:
                state_history.append(sim.state.cpu())

    elapsed = time.time() - t0
    final = sim.state.cpu().numpy()

    # Compute metrics
    metrics = compute_metrics_from_history(state_history)

    print(f"  alive={metrics['alive_fraction']:.3f} compl={metrics['complexity']:.3f} "
          f"clusters={metrics['num_clusters']} ({elapsed:.1f}s)")

    result = {
        "rank": idx + 1,
        "bin": bin_idx,
        "fitness": fitness,
        "kernel_type": kt,
        "growth_type": gt,
        "geometry": geo,
        "alive_fraction": round(metrics["alive_fraction"], 4),
        "complexity": round(metrics["complexity"], 4),
        "num_clusters": metrics["num_clusters"],
        "time_s": round(elapsed, 2),
    }
    results.append(result)

    # Save final state image
    img = (final * 255).astype(np.uint8)
    name = f"top{idx+1}_{kt}_{gt}_{geo}"
    Image.fromarray(img).save(f"{out_dir}/{name}_final.png")

    # Save timeline panel
    n_snaps = len(snapshot_times)
    panel_w = RESOLUTION * n_snaps + (n_snaps - 1) * 4
    panel = np.ones((RESOLUTION + 30, panel_w), dtype=np.uint8) * 255
    for i, t in enumerate(snapshot_times):
        x_offset = i * (RESOLUTION + 4)
        snap_img = (snapshots[t] * 255).astype(np.uint8)
        panel[:RESOLUTION, x_offset:x_offset+RESOLUTION] = snap_img
    Image.fromarray(panel).save(f"{out_dir}/{name}_timeline.png")

# Save results JSON
with open(f"{out_dir}/highres_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"  Rendered {len(results)} entries at {RESOLUTION}×{RESOLUTION}")
print(f"  Saved to {out_dir}/")
print(f"{'='*60}")
