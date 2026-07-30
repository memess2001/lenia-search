#!/usr/bin/env python3
"""Localized creature search v4: Locomotion-targeting with symmetry breaking.

Key changes from v3:
  A) Asymmetric initial condition: off-center ellipse (0.35, 0.5), aspect 2:1
  B) Anisotropic kernel: multiply kernel by (1 + strength*cos(theta - bias_angle))
  C) Priority: FitzHugh-Nagumo + aniso kernel, also ricker/allee growth
  D) Fitness = spatial_concentration × net_displacement (from t=5000 to t=10000)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.lenia_core import Lenia, KERNEL_REGISTRY, GROWTH_REGISTRY


# =====================================================================
# Anisotropic Lenia: patches _build_kernel_fft to add angular modulation
# =====================================================================

class AnisotropicLenia(Lenia):
    """Lenia with anisotropic kernel: kernel *= (1 + strength * cos(theta - bias))."""

    def __init__(self, genome: dict, device: str | None = None):
        self.aniso_strength: float = float(genome.get("aniso_strength", 0.3))
        self.aniso_angle: float = float(genome.get("aniso_angle", 0.0))
        super().__init__(genome, device)

    def _build_kernel_fft(self, kernel_fn, radius, peaks):
        R = self.resolution
        mid = R // 2
        y = torch.arange(R, device=self.device, dtype=torch.float32) - mid
        x = torch.arange(R, device=self.device, dtype=torch.float32) - mid
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        dist = torch.sqrt(xx**2 + yy**2) / max(radius, 1.0)

        raw = kernel_fn(dist, peaks)
        raw = raw * (dist <= 1.0).float()

        # ---- Anisotropic modulation ----
        theta = torch.atan2(yy, xx)
        aniso_mask = 1.0 + self.aniso_strength * torch.cos(theta - self.aniso_angle)
        raw = raw * aniso_mask

        total = raw.sum()
        if total > 0:
            raw = raw / total

        kernel_shifted = torch.roll(raw, shifts=(-mid, -mid), dims=(0, 1))
        return torch.fft.fft2(kernel_shifted)


# =====================================================================
# Localization metrics (fixed velocity = net displacement)
# =====================================================================

def compute_localized_metrics(state_history: list[torch.Tensor],
                               resolution: int = 256) -> dict:
    res = resolution
    final = state_history[-1].cpu().numpy() if isinstance(state_history[-1], torch.Tensor) else state_history[-1]

    Y, X = np.mgrid[0:res, 0:res]
    alive_mask = final > 0.1
    alive_fraction = float(alive_mask.mean())
    mass = float(final.sum())

    if alive_fraction < 0.005 or alive_fraction > 0.95:
        return {"velocity": 0.0, "mass": mass, "alive_fraction": alive_fraction,
                "spatial_concentration": 0.0, "fitness": 0.0, "is_localized": False,
                "net_displacement": 0.0}

    total_mass = final.sum()
    if total_mass < 1e-6:
        cx, cy = res/2, res/2
    else:
        cx = (X * final).sum() / total_mass
        cy = (Y * final).sum() / total_mass

    # Spatial concentration
    dx_arr = np.minimum(np.abs(X - cx), res - np.abs(X - cx))
    dy_arr = np.minimum(np.abs(Y - cy), res - np.abs(Y - cy))
    dist = np.sqrt(dx_arr**2 + dy_arr**2)
    max_dist = res * np.sqrt(2) / 2
    mean_dist = (dist * final).sum() / total_mass if total_mass > 1e-6 else max_dist
    spatial_concentration = max(0, 1 - mean_dist / max_dist)

    # Net displacement: centroid at halfway vs end
    half_idx = len(state_history) // 2

    def _centroid(s_tensor):
        s = s_tensor.cpu().numpy() if isinstance(s_tensor, torch.Tensor) else s_tensor
        m = s.sum()
        if m < 1e-6:
            return None
        return ((X * s).sum() / m, (Y * s).sum() / m)

    c_start = _centroid(state_history[half_idx])
    c_end = _centroid(state_history[-1])

    net_displacement = 0.0
    if c_start is not None and c_end is not None:
        ddx = min(abs(c_end[0]-c_start[0]), res-abs(c_end[0]-c_start[0]))
        ddy = min(abs(c_end[1]-c_start[1]), res-abs(c_end[1]-c_start[1]))
        net_displacement = float(np.sqrt(ddx**2 + ddy**2))

    fitness = spatial_concentration * net_displacement
    is_localized = spatial_concentration > 0.7 and alive_fraction < 0.15

    return {
        "net_displacement": round(net_displacement, 2),
        "mass": round(mass, 2),
        "alive_fraction": round(alive_fraction, 4),
        "spatial_concentration": round(spatial_concentration, 4),
        "fitness": round(fitness, 4),
        "is_localized": is_localized,
    }


# =====================================================================
# Genome generation
# =====================================================================

PRIORITY_KERNELS = ["gaussian", "cosine", "bump", "polynomial", "sech",
                    "mexican_hat", "sinc", "bessel", "power_law"]
PRIORITY_GROWTHS = ["fitzhugh_nagumo", "gaussian_bell", "ricker", "allee",
                    "swift_hohenberg", "bistable", "step_function"]


def random_genome() -> dict:
    kernel = random.choice(PRIORITY_KERNELS)
    growth = random.choice(PRIORITY_GROWTHS)
    return {
        "kernel_type": kernel,
        "growth_type": growth,
        "geometry": "flat_plane",
        "resolution": 256,
        "kernel_radius": random.uniform(8, 25),
        "kernel_peaks": random.uniform(0.5, 3.0),
        "growth_mu": random.uniform(0.05, 0.35),
        "growth_sigma": random.uniform(0.005, 0.12),
        "dt": random.uniform(0.02, 0.3),
        "aniso_strength": random.uniform(0.1, 0.6),
        "aniso_angle": random.uniform(0, 2 * math.pi),
    }


def mutate_genome(parent: dict) -> dict:
    child = dict(parent)
    if random.random() < 0.25:
        child["kernel_type"] = random.choice(PRIORITY_KERNELS)
    if random.random() < 0.25:
        child["growth_type"] = random.choice(PRIORITY_GROWTHS)

    params = {
        "kernel_radius": (8, 25), "kernel_peaks": (0.5, 3.0),
        "growth_mu": (0.05, 0.35), "growth_sigma": (0.005, 0.12),
        "dt": (0.02, 0.3), "aniso_strength": (0.1, 0.6),
        "aniso_angle": (0, 2 * math.pi),
    }
    for p in random.sample(list(params.keys()), random.randint(1, 3)):
        lo, hi = params[p]
        child[p] = max(lo, min(hi, child[p] + random.gauss(0, (hi-lo)*0.15)))
    return child


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Locomotion-targeting search v4")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--output", type=str, default="results_localized_v4")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/gifs", exist_ok=True)

    RES = args.resolution
    STEPS = args.steps

    n_bins = 20
    archive: dict[tuple[int, int], dict] = {}
    total_eval = 0
    moving_count = 0  # net_displacement > 5
    localized_count = 0

    print(f"Locomotion Search v4: {args.n} genomes")
    print(f"  Resolution: {RES}, Steps: {STEPS}, Device: {args.device}")
    print(f"  Anisotropic kernel + off-center ellipse init")
    print(f"  Fitness: concentration × net_displacement")
    print("=" * 60)

    t_start = time.time()

    for i in range(args.n):
        if i < 50 or not archive:
            genome = random_genome()
        elif random.random() < 0.5 and archive:
            parent = random.choice(list(archive.values()))["genome"]
            genome = mutate_genome(parent)
        else:
            genome = random_genome()

        genome["resolution"] = RES

        # Create anisotropic sim
        try:
            sim = AnisotropicLenia(genome, device=args.device)
        except Exception:
            continue

        # Asymmetric init: off-center ellipse at (0.35, 0.5), aspect 2:1
        torch.manual_seed(42 + i)  # vary seed slightly per eval
        state = torch.zeros(RES, RES, device=args.device)
        cy_init = int(0.5 * RES)
        cx_init = int(0.35 * RES)
        Y_t = torch.arange(RES, device=args.device).float()
        X_t = torch.arange(RES, device=args.device).float()
        YY, XX = torch.meshgrid(Y_t, X_t, indexing="ij")
        # Ellipse: sigma_x = 0.06*RES, sigma_y = 0.03*RES (2:1 aspect)
        r2 = ((XX - cx_init) / (RES * 0.06))**2 + ((YY - cy_init) / (RES * 0.03))**2
        state = torch.exp(-r2) * 0.8
        sim.state = state

        # Run
        record_interval = max(1, STEPS // 200)
        state_history = [sim.state.cpu()]
        try:
            for step in range(STEPS):
                sim.step()
                if (step+1) % record_interval == 0 or step == STEPS-1:
                    state_history.append(sim.state.cpu())
                if (step+1) % 2000 == 0:
                    s = sim.state.cpu().numpy()
                    if (s > 0.1).mean() < 0.005:
                        break
        except Exception:
            continue

        metrics = compute_localized_metrics(state_history, RES)
        total_eval += 1

        if metrics["net_displacement"] > 5:
            moving_count += 1
        if metrics["is_localized"]:
            localized_count += 1

        # Archive: net_displacement × mass
        nd_norm = min(1.0, metrics["net_displacement"] / 50.0)
        mass_norm = min(1.0, metrics["mass"] / (RES*RES*0.3))
        bin_nd = min(int(nd_norm * n_bins), n_bins-1)
        bin_mass = min(int(mass_norm * n_bins), n_bins-1)
        key = (bin_nd, bin_mass)

        entry = {"genome": genome, "metrics": metrics, "fitness": metrics["fitness"], "bin": list(key)}
        existing = archive.get(key)
        if existing is None or metrics["fitness"] > existing["fitness"]:
            archive[key] = entry

            # Save image for promising entries
            if metrics["net_displacement"] > 5 and metrics["spatial_concentration"] > 0.5:
                from PIL import Image
                final_np = state_history[-1].numpy() if isinstance(state_history[-1], torch.Tensor) else state_history[-1]
                img = (np.clip(final_np, 0, 1) * 255).astype(np.uint8)
                kt = genome["kernel_type"]
                gt = genome["growth_type"]
                Image.fromarray(img).save(
                    f"{args.output}/gifs/v4_{i:04d}_{kt}_{gt}_nd{metrics['net_displacement']:.1f}.png"
                )

        if (i+1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1:4d}/{args.n}] eval={total_eval} archive={len(archive)} "
                  f"moving(>5px)={moving_count} localized={localized_count} "
                  f"({elapsed:.0f}s)")

    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  LOCOMOTION SEARCH v4 SUMMARY")
    print(f"{'='*60}")
    print(f"  Evaluated:         {total_eval}")
    print(f"  Archive bins:      {len(archive)}/{n_bins*n_bins}")
    print(f"  Moving (>5px net): {moving_count} ({100*moving_count/max(1,total_eval):.1f}%)")
    print(f"  Localized:         {localized_count}")
    print(f"  Time:              {elapsed:.0f}s")

    if archive:
        sorted_entries = sorted(archive.values(), key=lambda e: e["fitness"], reverse=True)
        print(f"\n  Top 10 by fitness (concentration × net_displacement):")
        for j, e in enumerate(sorted_entries[:10]):
            m = e["metrics"]
            g = e["genome"]
            print(f"    {j+1}. {g['kernel_type']}/{g['growth_type']} "
                  f"fitness={m['fitness']:.2f} nd={m['net_displacement']:.1f}px "
                  f"conc={m['spatial_concentration']:.3f} alive={m['alive_fraction']:.3f} "
                  f"aniso={g['aniso_strength']:.2f}")

        # Specifically check: any with nd>20 + conc>0.7 + alive<0.15?
        heroes = [e for e in archive.values()
                  if e["metrics"]["net_displacement"] > 20
                  and e["metrics"]["spatial_concentration"] > 0.7
                  and e["metrics"]["alive_fraction"] < 0.15]
        print(f"\n  *** LOCOMOTION CANDIDATES (nd>20, conc>0.7, alive<0.15): {len(heroes)} ***")
        for e in heroes:
            m = e["metrics"]
            g = e["genome"]
            print(f"    {g['kernel_type']}/{g['growth_type']} nd={m['net_displacement']:.1f}px "
                  f"conc={m['spatial_concentration']:.3f} alive={m['alive_fraction']:.3f}")

    save_data = {
        "total_evaluated": total_eval, "archive_size": len(archive),
        "moving_count": moving_count, "localized_count": localized_count,
        "time_s": round(elapsed, 2),
        "config": {"resolution": RES, "steps": STEPS, "n_genomes": args.n,
                   "anisotropic": True, "asymmetric_init": True},
        "archive": [{"bin": list(k), **v} for k, v in sorted(archive.items())],
    }
    with open(f"{args.output}/search_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
