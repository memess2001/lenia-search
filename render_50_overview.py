#!/usr/bin/env python3
"""Generate a 10x5 overview of all 50 qualifying moving creatures.

Each thumbnail: 80x80 crop around centroid at t=5000, inferno colormap, black bg.
Grouped by kernel family, labeled with index, kernel/growth, displacement, concentration.
Output: results_localized/all_50_creatures_overview.png
"""

import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localized_search_v4_locomotion import AnisotropicLenia

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RES = 256
SIM_STEPS = 5000
CROP = 80

def load_qualifying_creatures():
    """Load and filter the 50 qualifying movers from v5_broad archive."""
    with open("results_localized_v5_broad/search_results.json") as f:
        data = json.load(f)
    archive = data["archive"] if isinstance(data, dict) and "archive" in data else data

    movers = []
    for entry in archive:
        m = entry.get("metrics", {})
        nd = m.get("net_displacement", 0)
        conc = m.get("spatial_concentration", 0)
        alive = m.get("alive_fraction", 1.0)
        if nd > 20 and conc > 0.7 and alive < 0.15:
            movers.append(entry)

    # Sort by kernel family, then by displacement descending
    kernel_order = [
        "yukawa", "sinc", "riesz", "lennard_jones", "elliptical",
        "rbf_mixture", "morse", "polynomial", "power_law", "wendland_c2",
        # any others
    ]
    def sort_key(e):
        kt = e["genome"]["kernel_type"]
        idx = kernel_order.index(kt) if kt in kernel_order else 99
        return (idx, -e["metrics"]["net_displacement"])
    movers.sort(key=sort_key)
    return movers


def init_state(sim, cx_frac=0.35, cy_frac=0.5, aspect=2.0, width_frac=0.06):
    """Initialize with off-center elliptical blob."""
    YY, XX = torch.meshgrid(
        torch.arange(RES, device=sim.device, dtype=torch.float32),
        torch.arange(RES, device=sim.device, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy = RES * cx_frac, RES * cy_frac
    w = RES * width_frac
    r2 = ((XX - cx) / w) ** 2 + ((YY - cy) / (w / aspect)) ** 2
    sim.state = (torch.exp(-r2) * 0.8).to(sim.device)


def find_centroid(state):
    """Find mass centroid."""
    Y, X = np.mgrid[0:RES, 0:RES]
    total = state.sum()
    if total < 1e-6:
        return RES // 2, RES // 2
    cx = int(round((X * state).sum() / total))
    cy = int(round((Y * state).sum() / total))
    return cx, cy


def crop_periodic(state, cx, cy, size=CROP):
    """Crop a window around centroid with periodic boundaries."""
    half = size // 2
    padded = np.tile(state, (3, 3))
    px, py = cx + RES, cy + RES
    return padded[py - half:py + half, px - half:px + half]


def simulate_creature(genome, seed=42):
    """Simulate creature to t=5000 and return state snapshot."""
    torch.manual_seed(seed)
    sim = AnisotropicLenia(genome, device=DEVICE)
    init_state(sim)

    for step in range(1, SIM_STEPS + 1):
        sim.step()

    state_np = sim.state.cpu().numpy().copy()
    return state_np


def main():
    print("Loading qualifying creatures...")
    movers = load_qualifying_creatures()
    print(f"Found {len(movers)} creatures")

    # Simulate each creature and capture snapshot
    snapshots = []
    for i, entry in enumerate(movers):
        g = entry["genome"]
        m = entry["metrics"]
        kt = g["kernel_type"]
        gt = g["growth_type"]
        nd = m["net_displacement"]
        conc = m["spatial_concentration"]
        print(f"  [{i+1:2d}/50] {kt}/{gt} (disp={nd:.0f}, conc={conc:.2f})...")

        state = simulate_creature(g)
        cx, cy = find_centroid(state)
        crop = crop_periodic(state, cx, cy)
        snapshots.append((crop, entry))

    print("\nRendering overview...")

    # Create 10x5 grid (10 columns, 5 rows)
    ncols, nrows = 10, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 14),
                              facecolor="black")
    fig.patch.set_facecolor("black")

    cmap = plt.cm.inferno
    norm = Normalize(vmin=0, vmax=1)

    for idx in range(nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        ax.set_facecolor("black")

        if idx < len(snapshots):
            crop, entry = snapshots[idx]
            g = entry["genome"]
            m = entry["metrics"]
            kt = g["kernel_type"]
            gt = g["growth_type"]
            nd = m["net_displacement"]
            conc = m["spatial_concentration"]

            # Render with inferno on black
            ax.imshow(crop, cmap=cmap, norm=norm, interpolation="nearest")

            # Shorten growth type name for display
            gt_short = gt.replace("_", "\n", 1) if len(gt) > 12 else gt
            gt_short = gt.replace("_", " ")
            if len(gt_short) > 14:
                gt_short = gt_short[:13] + "."

            # Title: index + kernel/growth
            ax.set_title(f"#{idx+1} {kt}\n{gt_short}",
                         color="white", fontsize=7, fontweight="bold", pad=2)
            # Bottom annotation: displacement + concentration
            ax.text(0.5, -0.02, f"d={nd:.0f} c={conc:.2f}",
                    transform=ax.transAxes, ha="center", va="top",
                    color="#ffcc00", fontsize=6, fontweight="bold")
        else:
            ax.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Add kernel family group separators
    # Find where kernel families change
    prev_kernel = None
    group_starts = []
    for idx, (_, entry) in enumerate(snapshots):
        kt = entry["genome"]["kernel_type"]
        if kt != prev_kernel:
            group_starts.append((idx, kt))
            prev_kernel = kt

    plt.suptitle("All 50 Moving Solitons — Grouped by Kernel Family\n"
                 "t=5000, 80×80 crop, inferno colormap | d=displacement(px), c=concentration",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.subplots_adjust(hspace=0.45, wspace=0.15)

    out_path = "results_localized/all_50_creatures_overview.png"
    os.makedirs("results_localized", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nSaved: {out_path} ({size_kb:.0f} KB)")

    # Print summary grouped by kernel
    print("\n--- Creature Index by Kernel Family ---")
    for start_idx, kt in group_starts:
        count = sum(1 for _, e in snapshots if e["genome"]["kernel_type"] == kt)
        indices = [i+1 for i, (_, e) in enumerate(snapshots) if e["genome"]["kernel_type"] == kt]
        print(f"  {kt} ({count}): #{', #'.join(map(str, indices))}")


if __name__ == "__main__":
    main()
