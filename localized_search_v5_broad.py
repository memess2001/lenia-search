#!/usr/bin/env python3
"""Task A: Broad locomotion search — 14000 genomes, all 28×16 kernel/growth, anisotropic only.
Fitness = concentration × net_displacement².
"""

from __future__ import annotations
import argparse, json, math, os, random, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localized_search_v4_locomotion import AnisotropicLenia, compute_localized_metrics
from src.lenia_core import KERNEL_REGISTRY, GROWTH_REGISTRY

ALL_KERNELS = list(KERNEL_REGISTRY.keys())
ALL_GROWTHS = list(GROWTH_REGISTRY.keys())


def random_genome(res=256):
    return {
        "kernel_type": random.choice(ALL_KERNELS),
        "growth_type": random.choice(ALL_GROWTHS),
        "geometry": "flat_plane", "resolution": res,
        "kernel_radius": random.uniform(8, 25),
        "kernel_peaks": random.uniform(0.5, 3.0),
        "growth_mu": random.uniform(0.05, 0.35),
        "growth_sigma": random.uniform(0.005, 0.12),
        "dt": random.uniform(0.02, 0.3),
        "aniso_strength": random.uniform(0.1, 0.6),
        "aniso_angle": random.uniform(0, 2 * math.pi),
    }


def mutate_genome(parent):
    child = dict(parent)
    if random.random() < 0.2: child["kernel_type"] = random.choice(ALL_KERNELS)
    if random.random() < 0.2: child["growth_type"] = random.choice(ALL_GROWTHS)
    params = {"kernel_radius": (8,25), "kernel_peaks": (0.5,3), "growth_mu": (0.05,0.35),
              "growth_sigma": (0.005,0.12), "dt": (0.02,0.3), "aniso_strength": (0.1,0.6),
              "aniso_angle": (0, 2*math.pi)}
    for p in random.sample(list(params), random.randint(1, 3)):
        lo, hi = params[p]
        child[p] = max(lo, min(hi, child[p] + random.gauss(0, (hi-lo)*0.15)))
    return child


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=14000)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str, default="results_localized_v5_broad")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/gifs", exist_ok=True)

    RES, STEPS = args.resolution, args.steps
    n_bins, archive = 20, {}
    total_eval, moving_count, localized_count = 0, 0, 0

    print(f"Broad Locomotion Search: {args.n} genomes, {len(ALL_KERNELS)}×{len(ALL_GROWTHS)} combos")
    print(f"  Fitness = concentration × net_displacement²")
    print("=" * 60)
    t_start = time.time()

    for i in range(args.n):
        genome = random_genome(RES) if (i < 100 or not archive or random.random() > 0.5) \
                 else mutate_genome(random.choice(list(archive.values()))["genome"])
        genome["resolution"] = RES
        try:
            sim = AnisotropicLenia(genome, device=args.device)
        except: continue

        torch.manual_seed(42 + i)
        YY, XX = torch.meshgrid(torch.arange(RES, device=args.device).float(),
                                 torch.arange(RES, device=args.device).float(), indexing="ij")
        cx_init = random.uniform(0.25, 0.75) * RES
        cy_init = random.uniform(0.25, 0.75) * RES
        aspect = random.uniform(1.5, 3.0)
        r2 = ((XX-cx_init)/(RES*0.06))**2 + ((YY-cy_init)/(RES*0.06/aspect))**2
        sim.state = torch.exp(-r2) * 0.8

        record_interval = max(1, STEPS // 200)
        state_history = [sim.state.cpu()]
        try:
            for step in range(STEPS):
                sim.step()
                if (step+1) % record_interval == 0 or step == STEPS-1:
                    state_history.append(sim.state.cpu())
                if (step+1) % 2000 == 0 and (sim.state > 0.1).float().mean() < 0.003: break
        except: continue

        metrics = compute_localized_metrics(state_history, RES)
        total_eval += 1
        nd = metrics["net_displacement"]
        if nd > 5: moving_count += 1
        if metrics["is_localized"]: localized_count += 1

        # Fitness = concentration × nd²
        fitness = metrics["spatial_concentration"] * nd * nd

        nd_norm = min(1.0, nd / 100.0)
        mass_norm = min(1.0, metrics["mass"] / (RES*RES*0.3))
        key = (min(int(nd_norm*n_bins), n_bins-1), min(int(mass_norm*n_bins), n_bins-1))
        entry = {"genome": genome, "metrics": metrics, "fitness": fitness, "bin": list(key)}

        if key not in archive or fitness > archive[key]["fitness"]:
            archive[key] = entry
            if nd > 20 and metrics["spatial_concentration"] > 0.5:
                from PIL import Image
                final_np = state_history[-1].numpy() if isinstance(state_history[-1], torch.Tensor) else state_history[-1]
                img = (np.clip(final_np, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(img).save(
                    f"{args.output}/gifs/v5_{i:05d}_{genome['kernel_type']}_{genome['growth_type']}_nd{nd:.0f}.png")

        if (i+1) % 200 == 0 or i == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1:5d}/{args.n}] eval={total_eval} archive={len(archive)} "
                  f"moving={moving_count} localized={localized_count} ({elapsed:.0f}s)")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  Evaluated: {total_eval}, Archive: {len(archive)}, Moving: {moving_count}, Localized: {localized_count}")
    print(f"  Time: {elapsed:.0f}s")

    if archive:
        top = sorted(archive.values(), key=lambda e: e["fitness"], reverse=True)
        print(f"\n  Top 10:")
        for j, e in enumerate(top[:10]):
            m, g = e["metrics"], e["genome"]
            print(f"    {j+1}. {g['kernel_type']}/{g['growth_type']} nd={m['net_displacement']:.1f} "
                  f"conc={m['spatial_concentration']:.3f} alive={m['alive_fraction']:.3f}")
        heroes = [e for e in archive.values()
                  if e["metrics"]["net_displacement"] > 20 and e["metrics"]["spatial_concentration"] > 0.7
                  and e["metrics"]["alive_fraction"] < 0.15]
        print(f"\n  *** LOCOMOTION CANDIDATES: {len(heroes)} ***")

    with open(f"{args.output}/search_results.json", "w") as f:
        json.dump({"total": total_eval, "archive_size": len(archive), "moving": moving_count,
                   "localized": localized_count, "time_s": round(elapsed,2),
                   "archive": [{"bin": list(k), **v} for k,v in sorted(archive.items())]}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
