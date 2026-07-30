#!/usr/bin/env python3
"""Localized creature search v3: Leniabreeder-style velocity×mass descriptor.

Strategy:
- Parametric framework only (28 kernels × 16 growth × flat_plane)
- Focus on FitzHugh-Nagumo growth + classic kernels (gaussian, cosine, bump)
- Behavior descriptors: velocity × mass (reward small, moving things)
- Fitness: spatial_concentration × velocity
- Init: center blob only, 256×256, 10000 steps
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.geometry_factory import create_lenia


# =====================================================================
# Localization-specific metrics
# =====================================================================

def compute_localized_metrics(state_history: list[torch.Tensor],
                               last_n: int = 1000) -> dict:
    """Compute velocity, mass, spatial_concentration from state history."""

    final = state_history[-1].cpu().numpy() if isinstance(state_history[-1], torch.Tensor) else state_history[-1]
    res = final.shape[0]

    # Basic alive check
    alive_mask = final > 0.1
    alive_fraction = alive_mask.mean()
    mass = float(final.sum())

    if alive_fraction < 0.005 or alive_fraction > 0.95:
        # Dead or fully saturated - no creature
        return {
            "velocity": 0.0,
            "mass": mass,
            "alive_fraction": float(alive_fraction),
            "spatial_concentration": 0.0,
            "fitness": 0.0,
            "centroid": (res/2, res/2),
            "is_localized": False,
        }

    # Compute centroid of alive pixels
    Y, X = np.mgrid[0:res, 0:res]
    total_mass = final.sum()

    if total_mass < 1e-6:
        cx, cy = res/2, res/2
    else:
        cx = (X * final).sum() / total_mass
        cy = (Y * final).sum() / total_mass

    # Spatial concentration: 1 - (mean distance to centroid / max possible distance)
    # Circular distance for periodic boundary
    dx = np.minimum(np.abs(X - cx), res - np.abs(X - cx))
    dy = np.minimum(np.abs(Y - cy), res - np.abs(Y - cy))
    dist = np.sqrt(dx**2 + dy**2)
    max_dist = res * np.sqrt(2) / 2  # max possible distance

    if total_mass > 1e-6:
        mean_dist = (dist * final).sum() / total_mass
        spatial_concentration = max(0, 1 - mean_dist / max_dist)
    else:
        spatial_concentration = 0.0

    # Velocity: NET centroid displacement over last 2000 steps (not path length).
    # Net displacement catches directional movement; random jitter averages to ~0.
    # Unit: pixels per 1000 simulation steps.
    velocity = 0.0
    if len(state_history) >= 2:
        # Find the state_history index corresponding to ~halfway (step 5000 of 10000)
        # state_history has one entry per record_interval steps
        half_idx = len(state_history) // 2

        def _centroid(s_tensor):
            s = s_tensor.cpu().numpy() if isinstance(s_tensor, torch.Tensor) else s_tensor
            s_mass = s.sum()
            if s_mass < 1e-6:
                return None
            return ((X * s).sum() / s_mass, (Y * s).sum() / s_mass)

        c_start = _centroid(state_history[half_idx])
        c_end = _centroid(state_history[-1])

        if c_start is not None and c_end is not None:
            dx = c_end[0] - c_start[0]
            dy = c_end[1] - c_start[1]
            # Periodic wrap
            dx = min(abs(dx), res - abs(dx))
            dy = min(abs(dy), res - abs(dy))
            net_disp = np.sqrt(dx**2 + dy**2)

            # Convert to pixels per 1000 sim steps
            # The second half covers (len(state_history) - half_idx) samples
            # Each sample spans record_interval sim steps
            # Total sim steps in second half ≈ STEPS / 2
            sim_steps_in_half = last_n  # caller passes last_n = 1000 by default
            # Better estimate: total steps * fraction of history used
            # We use half the history → half the total steps
            # But we don't know total steps here, so use the ratio
            n_samples_in_half = len(state_history) - half_idx
            # Each sample interval = total_steps / len(state_history)
            # total sim steps in second half = n_samples_in_half * (total_steps / len(state_history))
            # We approximate total_steps ≈ len(state_history) * record_interval
            # For now, assume the caller tells us via last_n how many sim steps the tail covers
            # Actually, let's just compute: second half = len(state_history)//2 samples
            # with record_interval = STEPS // 200, so sim steps = n_samples * record_interval
            # But we don't have record_interval here. Use a simpler approach:
            # Assume state_history spans the full run, second half = half the run.
            # With STEPS=10000, second half = 5000 steps.
            # velocity in px/1000steps = net_disp / 5000 * 1000 = net_disp / 5
            # More robustly: we know last_n is meant as sim steps, default 1000
            # Let's just use half the history length as the denominator with proper scaling.
            # Safest: assume total sim steps = (len(state_history) - 1) * record_interval
            # where record_interval ≈ total_steps / (len(state_history) - 1)
            # So sim steps in second half = n_samples_in_half * total_steps / (len(state_history)-1)
            # But we don't know total_steps...
            #
            # Simple fix: just report net_disp in pixels, let the search use it directly.
            velocity = net_disp  # raw pixels of net displacement over second half of sim

    # Fitness: spatial_concentration × net_displacement
    fitness = spatial_concentration * velocity

    # Is it localized? (concentration > 0.7 and small)
    is_localized = spatial_concentration > 0.7 and alive_fraction < 0.15

    return {
        "velocity": round(float(velocity), 4),
        "mass": round(float(mass), 2),
        "alive_fraction": round(float(alive_fraction), 4),
        "spatial_concentration": round(float(spatial_concentration), 4),
        "fitness": round(float(fitness), 6),
        "centroid": (round(float(cx), 1), round(float(cy), 1)),
        "is_localized": is_localized,
    }


# =====================================================================
# Genome generation (focused on FitzHugh-Nagumo + classic kernels)
# =====================================================================

# Priority kernels (theory-motivated for traveling waves)
PRIORITY_KERNELS = ["gaussian", "cosine", "bump", "polynomial", "sech",
                    "mexican_hat", "sinc", "bessel"]
ALL_KERNELS = PRIORITY_KERNELS + [
    "gabor", "chirp", "fourier_series", "lennard_jones", "morse",
    "riesz", "yukawa", "log_gaussian", "dog", "hann_ring",
    "blackman_harris", "wendland_c2", "matern", "power_law",
    "step_ring", "elliptical", "spiral", "plateau",
]

# Priority growth functions (excitable media = traveling waves)
PRIORITY_GROWTHS = ["fitzhugh_nagumo", "swift_hohenberg", "gaussian_bell",
                    "step_function", "bistable"]
ALL_GROWTHS = PRIORITY_GROWTHS + [
    "asymmetric_bell", "laplace_peak", "cauchy_peak", "relu_like",
    "ricker", "allee", "sinc_growth", "sigmoid_pair", "witch_growth",
    "gompertz", "brusselator",
]


def random_genome(focused: bool = True) -> dict:
    """Generate a random genome, optionally focused on promising region."""
    if focused and random.random() < 0.7:
        # 70% focused on FitzHugh-Nagumo + classic kernels
        kernel = random.choice(PRIORITY_KERNELS)
        growth = random.choice(PRIORITY_GROWTHS)
    else:
        kernel = random.choice(ALL_KERNELS)
        growth = random.choice(ALL_GROWTHS)

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
    }


def mutate_genome(parent: dict) -> dict:
    """Mutate a parent genome."""
    child = dict(parent)

    # 30% chance to change categorical
    if random.random() < 0.3:
        if random.random() < 0.5:
            child["kernel_type"] = random.choice(PRIORITY_KERNELS if random.random() < 0.7 else ALL_KERNELS)
        else:
            child["growth_type"] = random.choice(PRIORITY_GROWTHS if random.random() < 0.7 else ALL_GROWTHS)

    # Mutate 1-2 continuous params
    params = ["kernel_radius", "kernel_peaks", "growth_mu", "growth_sigma", "dt"]
    ranges = {
        "kernel_radius": (8, 25),
        "kernel_peaks": (0.5, 3.0),
        "growth_mu": (0.05, 0.35),
        "growth_sigma": (0.005, 0.12),
        "dt": (0.02, 0.3),
    }

    n_mutate = random.randint(1, 2)
    for p in random.sample(params, n_mutate):
        lo, hi = ranges[p]
        noise = random.gauss(0, (hi - lo) * 0.15)
        child[p] = max(lo, min(hi, child[p] + noise))

    return child


# =====================================================================
# Main search
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Localized creature search v3")
    parser.add_argument("--n", type=int, default=500, help="Number of genomes to evaluate")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--output", type=str, default="results_localized_v3")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/gifs", exist_ok=True)

    RESOLUTION = args.resolution
    STEPS = args.steps
    DEVICE = args.device

    # MAP-Elites archive: velocity × mass (20×20)
    n_bins = 20
    archive: dict[tuple[int, int], dict] = {}

    # Track stats
    total_eval = 0
    nonzero_velocity = 0
    localized_count = 0

    print(f"Localized Creature Search v3")
    print(f"  Resolution: {RESOLUTION}, Steps: {STEPS}, Device: {DEVICE}")
    print(f"  Focused on FitzHugh-Nagumo + classic kernels")
    print(f"  Descriptors: velocity × mass")
    print(f"  Fitness: spatial_concentration × velocity")
    print("=" * 60)

    t_start = time.time()

    for i in range(args.n):
        # Generate or mutate genome
        if i < 50 or not archive:
            genome = random_genome(focused=True)
        else:
            # 50% mutate archive member, 50% fresh
            if random.random() < 0.5 and archive:
                parent = random.choice(list(archive.values()))["genome"]
                genome = mutate_genome(parent)
            else:
                genome = random_genome(focused=True)

        genome["resolution"] = RESOLUTION

        # Create sim
        try:
            sim = create_lenia(genome, device=DEVICE)
        except Exception:
            continue

        # Center blob init
        torch.manual_seed(42)
        state = torch.zeros(RESOLUTION, RESOLUTION, device=DEVICE)
        cy, cx = RESOLUTION // 2, RESOLUTION // 2
        Y, X = torch.meshgrid(torch.arange(RESOLUTION, device=DEVICE),
                               torch.arange(RESOLUTION, device=DEVICE), indexing="ij")
        r2 = ((Y - cy).float()**2 + (X - cx).float()**2) / (RESOLUTION * 0.08)**2
        state = torch.exp(-r2) * 0.8
        sim.state = state

        # Run simulation, collect history
        record_interval = max(1, STEPS // 200)
        state_history = [sim.state.cpu()]

        try:
            dead = False
            for step in range(STEPS):
                sim.step()
                if (step + 1) % record_interval == 0 or step == STEPS - 1:
                    state_history.append(sim.state.cpu())

                # Early termination check
                if (step + 1) % 2000 == 0:
                    s = sim.state.cpu().numpy()
                    af = (s > 0.1).mean()
                    if af < 0.005:  # dead
                        dead = True
                        break
        except Exception:
            continue

        if dead:
            total_eval += 1
            continue

        # Compute localized metrics
        metrics = compute_localized_metrics(state_history)
        total_eval += 1

        if metrics["velocity"] > 0.01:
            nonzero_velocity += 1
        if metrics["is_localized"]:
            localized_count += 1

        # MAP-Elites insertion (velocity × mass)
        # Normalize velocity to [0, 1] (cap at 10 pixels/1000 steps)
        vel_norm = min(1.0, metrics["velocity"] / 10.0)
        # Normalize mass to [0, 1] (cap at RESOLUTION^2 * 0.5)
        mass_norm = min(1.0, metrics["mass"] / (RESOLUTION * RESOLUTION * 0.5))

        bin_vel = min(int(vel_norm * n_bins), n_bins - 1)
        bin_mass = min(int(mass_norm * n_bins), n_bins - 1)
        key = (bin_vel, bin_mass)

        fitness = metrics["fitness"]

        entry = {
            "genome": genome,
            "metrics": metrics,
            "fitness": fitness,
            "bin": list(key),
        }

        existing = archive.get(key)
        if existing is None or fitness > existing["fitness"]:
            archive[key] = entry

            # Save image for interesting entries
            if metrics["velocity"] > 0.1 or metrics["is_localized"]:
                from PIL import Image
                final_np = state_history[-1].numpy() if isinstance(state_history[-1], torch.Tensor) else state_history[-1]
                img = (final_np * 255).astype(np.uint8)
                kt = genome["kernel_type"]
                gt = genome["growth_type"]
                Image.fromarray(img).save(
                    f"{args.output}/gifs/loc_{i:04d}_{kt}_{gt}_v{metrics['velocity']:.2f}.png"
                )

        # Progress
        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1:4d}/{args.n}] eval={total_eval} archive={len(archive)} "
                  f"nonzero_v={nonzero_velocity} localized={localized_count} "
                  f"({elapsed:.0f}s)")

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  LOCALIZED SEARCH v3 SUMMARY")
    print(f"{'='*60}")
    print(f"  Evaluated:        {total_eval}")
    print(f"  Archive bins:     {len(archive)}/{n_bins*n_bins}")
    print(f"  Nonzero velocity: {nonzero_velocity} ({100*nonzero_velocity/max(1,total_eval):.1f}%)")
    print(f"  Localized:        {localized_count}")
    print(f"  Time:             {elapsed:.0f}s")

    # Top entries by fitness
    if archive:
        sorted_entries = sorted(archive.values(), key=lambda e: e["fitness"], reverse=True)
        print(f"\n  Top 10 by fitness (concentration × velocity):")
        for j, e in enumerate(sorted_entries[:10]):
            m = e["metrics"]
            g = e["genome"]
            print(f"    {j+1}. {g['kernel_type']}/{g['growth_type']} "
                  f"fitness={m['fitness']:.4f} vel={m['velocity']:.3f} "
                  f"conc={m['spatial_concentration']:.3f} alive={m['alive_fraction']:.3f}")

        # Entries with nonzero velocity
        moving = [e for e in archive.values() if e["metrics"]["velocity"] > 0.1]
        print(f"\n  Entries with velocity > 0.1: {len(moving)}")
        for e in sorted(moving, key=lambda x: x["metrics"]["velocity"], reverse=True)[:5]:
            m = e["metrics"]
            g = e["genome"]
            print(f"    {g['kernel_type']}/{g['growth_type']} vel={m['velocity']:.3f} "
                  f"conc={m['spatial_concentration']:.3f} alive={m['alive_fraction']:.3f}")

    # Save
    save_data = {
        "total_evaluated": total_eval,
        "archive_size": len(archive),
        "nonzero_velocity": nonzero_velocity,
        "localized_count": localized_count,
        "time_s": round(elapsed, 2),
        "config": {
            "resolution": RESOLUTION,
            "steps": STEPS,
            "n_genomes": args.n,
            "focused": True,
        },
        "archive": [
            {"bin": list(k), **v}
            for k, v in sorted(archive.items())
        ],
    }

    with open(f"{args.output}/search_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
