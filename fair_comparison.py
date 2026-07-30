#!/usr/bin/env python3
"""Fair comparison: Hero expression vs Track A top 5, identical conditions.

All entries evaluated at 128×128, 8000 steps, same 10 random seeds.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.expression_compiler import compile_expression
from src.geometry_factory import create_lenia
from src.evaluator import evaluate as compute_metrics_from_history

RESOLUTION = 128
STEPS = 8000
DEVICE = "mps"
SEEDS = list(range(10))
KERNEL_RADIUS = 13

out_dir = "results/fair_comparison"
os.makedirs(out_dir, exist_ok=True)


def run_one(step_fn_or_sim, seed, is_expression=False):
    """Run one simulation and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state = torch.zeros(RESOLUTION, RESOLUTION, device=DEVICE)
    quarter = RESOLUTION // 4
    sr = RESOLUTION // 2 - quarter // 2
    sc = RESOLUTION // 2 - quarter // 2
    state[sr:sr+quarter, sc:sc+quarter] = torch.rand(quarter, quarter, device=DEVICE)

    if is_expression:
        step_fn = step_fn_or_sim
    else:
        sim = step_fn_or_sim
        sim.state = state

    record_interval = max(1, STEPS // 300)
    state_history = [state.cpu()]

    t0 = time.time()
    for step in range(STEPS):
        if is_expression:
            state = step_fn(state).clamp(0, 1)
        else:
            sim.step()
            state = sim.state

        if (step + 1) % record_interval == 0 or step == STEPS - 1:
            state_history.append(state.cpu() if is_expression else sim.state.cpu())

    elapsed = time.time() - t0
    metrics = compute_metrics_from_history(state_history)
    metrics["time_s"] = round(elapsed, 2)
    return metrics


# ===== Hero Expression =====
HERO_EXPR = (
    "clip(A + 0.31 * (max(exp(-((conv(A) - (0.12 + 0.03 * cos(4 * pi * laplacian(A)))) "
    "/ 0.055)**2 / 2), tanh(3.8 * (conv(A) + 0.15 * laplacian(A) - 0.18))) * "
    "threshold(conv(A) + laplacian(A), 0.08) * (1 - threshold(conv(A) + laplacian(A), 0.32)) "
    "+ 0.12 * sin(7 * pi * (conv(A) - 0.5 * laplacian(A))) * "
    "exp(-((conv(A) - 0.16) / 0.065)**2 / 2)), 0, 1)"
)

print("=" * 70)
print("  FAIR COMPARISON: 128×128, 8000 steps, 10 seeds")
print("=" * 70)

# --- Hero ---
print("\n[Hero] cos(4π·∇²A) expression")
hero_step_fn = compile_expression(HERO_EXPR, resolution=RESOLUTION, kernel_radius=KERNEL_RADIUS)
hero_results = []
for seed in SEEDS:
    m = run_one(hero_step_fn, seed, is_expression=True)
    hero_results.append(m)
    print(f"  seed {seed}: compl={m['complexity']:.4f} alive={m['alive_fraction']:.3f} ({m['time_s']}s)")

hero_compl = [r["complexity"] for r in hero_results]
hero_alive = [r["alive_fraction"] for r in hero_results]
print(f"  MEAN: compl={np.mean(hero_compl):.4f}±{np.std(hero_compl):.4f} "
      f"alive={np.mean(hero_alive):.4f}±{np.std(hero_alive):.4f}")

# --- Track A Top 5 ---
with open("results/archive_final.json") as f:
    final = json.load(f)

trackA = [e for e in final["entries"] if "kernel_type" in e.get("genome", {})]
trackA_sorted = sorted(trackA, key=lambda e: e["fitness"], reverse=True)

all_results = {
    "hero": {
        "name": "cos(4π·∇²A)",
        "track": "B",
        "seeds": hero_results,
        "mean_complexity": round(float(np.mean(hero_compl)), 4),
        "std_complexity": round(float(np.std(hero_compl)), 4),
        "mean_alive": round(float(np.mean(hero_alive)), 4),
    }
}

for idx, entry in enumerate(trackA_sorted[:5]):
    genome = entry["genome"]
    kt = genome["kernel_type"]
    gt = genome["growth_type"]
    geo = genome["geometry"]
    name = f"{kt}/{gt}/{geo}"

    print(f"\n[Track A #{idx+1}] {name}")

    genome_full = dict(genome)
    genome_full["resolution"] = RESOLUTION

    try:
        sim = create_lenia(genome_full, device=DEVICE)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    seed_results = []
    for seed in SEEDS:
        # Need to recreate sim for each seed since state is mutated
        sim = create_lenia(genome_full, device=DEVICE)
        m = run_one(sim, seed, is_expression=False)
        seed_results.append(m)
        print(f"  seed {seed}: compl={m['complexity']:.4f} alive={m['alive_fraction']:.3f} ({m['time_s']}s)")

    compls = [r["complexity"] for r in seed_results]
    alives = [r["alive_fraction"] for r in seed_results]
    print(f"  MEAN: compl={np.mean(compls):.4f}±{np.std(compls):.4f} "
          f"alive={np.mean(alives):.4f}±{np.std(alives):.4f}")

    all_results[f"trackA_{idx+1}"] = {
        "name": name,
        "track": "A",
        "original_fitness": entry["fitness"],
        "seeds": seed_results,
        "mean_complexity": round(float(np.mean(compls)), 4),
        "std_complexity": round(float(np.std(compls)), 4),
        "mean_alive": round(float(np.mean(alives)), 4),
    }

# --- Summary ---
print("\n" + "=" * 70)
print("  SUMMARY: Mean complexity (=fitness) at 128×128, 8000 steps")
print("=" * 70)

entries_ranked = sorted(all_results.items(),
                       key=lambda x: x[1]["mean_complexity"], reverse=True)
for rank, (key, data) in enumerate(entries_ranked):
    print(f"  {rank+1}. [{data['track']}] {data['name']}: "
          f"{data['mean_complexity']:.4f} ± {data['std_complexity']:.4f}")

# Save
with open(f"{out_dir}/fair_comparison.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\nSaved to {out_dir}/fair_comparison.json")
