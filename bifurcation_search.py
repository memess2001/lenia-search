#!/usr/bin/env python3
"""Task B: Bifurcation search — from stationary soliton, gradually break symmetry.

Starting from the power_law/gaussian_bell stationary soliton, add increasing
anisotropy (epsilon) and track net_displacement vs epsilon to find the
stationary→locomotion bifurcation point.
"""

from __future__ import annotations
import argparse, json, math, os, random, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localized_search_v4_locomotion import AnisotropicLenia, compute_localized_metrics

# The stationary soliton genome (from v3 search)
SEED_GENOME = {
    "kernel_type": "power_law",
    "growth_type": "gaussian_bell",
    "geometry": "flat_plane",
    "resolution": 256,
    "kernel_radius": 24.3,
    "kernel_peaks": 1.39,
    "growth_mu": 0.3400,
    "growth_sigma": 0.0499,
    "dt": 0.269,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str, default="results_bifurcation")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/gifs", exist_ok=True)

    RES, STEPS = args.resolution, args.steps

    print(f"Bifurcation Search: {args.n} genomes from stationary soliton")
    print(f"  Seed: power_law/gaussian_bell (conc=0.91, alive=2.2%, static)")
    print(f"  Varying: aniso_strength (0→0.6) + small param perturbation")
    print("=" * 60)

    results = []  # list of (epsilon, theta, nd, conc, alive, genome)
    t_start = time.time()

    for i in range(args.n):
        genome = dict(SEED_GENOME)
        genome["resolution"] = RES

        # Anisotropy: sweep epsilon from 0 to 0.6
        epsilon = random.uniform(0, 0.6)
        theta_0 = random.uniform(0, 2 * math.pi)
        genome["aniso_strength"] = epsilon
        genome["aniso_angle"] = theta_0

        # Small perturbation of continuous params (5% std, not 30%)
        for param, val in [("kernel_radius", 24.3), ("kernel_peaks", 1.39),
                           ("growth_mu", 0.34), ("growth_sigma", 0.0499), ("dt", 0.269)]:
            genome[param] = max(0.001, val + random.gauss(0, val * 0.05))

        # Occasionally try different kernel/growth types too (10% chance)
        if random.random() < 0.1:
            genome["kernel_type"] = random.choice(["power_law", "sech", "bessel", "gaussian", "bump"])
        if random.random() < 0.1:
            genome["growth_type"] = random.choice(["gaussian_bell", "bistable", "step_function"])

        try:
            sim = AnisotropicLenia(genome, device=args.device)
        except:
            continue

        # Center blob init (symmetric — we want the KERNEL asymmetry to drive motion)
        torch.manual_seed(42)
        YY, XX = torch.meshgrid(torch.arange(RES, device=args.device).float(),
                                 torch.arange(RES, device=args.device).float(), indexing="ij")
        r2 = ((XX - RES/2) / (RES*0.08))**2 + ((YY - RES/2) / (RES*0.08))**2
        sim.state = torch.exp(-r2) * 0.8

        record_interval = max(1, STEPS // 200)
        state_history = [sim.state.cpu()]
        try:
            for step in range(STEPS):
                sim.step()
                if (step+1) % record_interval == 0 or step == STEPS-1:
                    state_history.append(sim.state.cpu())
                if (step+1) % 2000 == 0 and (sim.state > 0.1).float().mean() < 0.003:
                    break
        except:
            continue

        metrics = compute_localized_metrics(state_history, RES)
        nd = metrics["net_displacement"]
        conc = metrics["spatial_concentration"]
        alive = metrics["alive_fraction"]

        results.append({
            "epsilon": round(epsilon, 4),
            "theta": round(theta_0, 4),
            "nd": round(nd, 2),
            "conc": round(conc, 4),
            "alive": round(alive, 4),
            "kernel_type": genome["kernel_type"],
            "growth_type": genome["growth_type"],
        })

        # Save images for movers
        if nd > 10 and conc > 0.5:
            from PIL import Image
            final_np = state_history[-1].numpy() if isinstance(state_history[-1], torch.Tensor) else state_history[-1]
            img = (np.clip(final_np, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img).save(
                f"{args.output}/gifs/bif_{i:04d}_eps{epsilon:.2f}_nd{nd:.0f}.png")

        if (i+1) % 100 == 0 or i == 0:
            elapsed = time.time() - t_start
            movers = sum(1 for r in results if r["nd"] > 5)
            big_movers = sum(1 for r in results if r["nd"] > 20 and r["conc"] > 0.7)
            print(f"  [{i+1:4d}/{args.n}] n={len(results)} movers(>5)={movers} "
                  f"big_movers(>20,conc>0.7)={big_movers} ({elapsed:.0f}s)")

    elapsed = time.time() - t_start

    # Analysis: displacement vs epsilon
    print(f"\n{'='*60}")
    print(f"  BIFURCATION ANALYSIS")
    print(f"{'='*60}")

    # Only look at power_law/gaussian_bell entries (the seed combo)
    seed_results = [r for r in results
                    if r["kernel_type"] == "power_law" and r["growth_type"] == "gaussian_bell"]

    print(f"  Total results: {len(results)}")
    print(f"  Seed combo (power_law/gaussian_bell): {len(seed_results)}")

    # Bin by epsilon ranges
    bins = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)]
    print(f"\n  Net displacement vs anisotropy strength (power_law/gaussian_bell only):")
    print(f"  {'Epsilon range':>15s}  {'N':>4s}  {'Mean ND':>8s}  {'Max ND':>8s}  {'Alive>0':>7s}")
    for lo, hi in bins:
        in_bin = [r for r in seed_results if lo <= r["epsilon"] < hi]
        if in_bin:
            nds = [r["nd"] for r in in_bin]
            alive_pct = sum(1 for r in in_bin if r["alive"] > 0.003) / len(in_bin)
            print(f"  [{lo:.2f}, {hi:.2f})  {len(in_bin):4d}  {np.mean(nds):8.2f}  {max(nds):8.2f}  {alive_pct:6.1%}")

    # Save everything
    with open(f"{args.output}/bifurcation_results.json", "w") as f:
        json.dump({"n_total": len(results), "time_s": round(elapsed,2),
                   "seed_genome": SEED_GENOME, "results": results}, f, indent=2)

    # Generate bifurcation plot data
    print(f"\nSaved to {args.output}/")

    # Make bifurcation scatter plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epsilons = [r["epsilon"] for r in seed_results]
        nds = [r["nd"] for r in seed_results]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(epsilons, nds, s=8, alpha=0.5, c="steelblue")
        ax.set_xlabel("Anisotropy strength (ε)", fontsize=12)
        ax.set_ylabel("Net displacement (pixels)", fontsize=12)
        ax.set_title("Bifurcation: Stationary → Locomotion\npower_law/gaussian_bell soliton", fontsize=13)
        ax.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="Locomotion threshold (20px)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(f"{args.output}/bifurcation_plot.png", dpi=200)
        fig.savefig(f"{args.output}/bifurcation_plot.pdf", dpi=200)
        plt.close()
        print(f"  Bifurcation plot saved")
    except Exception as e:
        print(f"  Plot failed: {e}")


if __name__ == "__main__":
    main()
