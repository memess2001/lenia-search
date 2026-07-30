#!/usr/bin/env python3
"""Generate browsing materials for all 174 non-extended-texture creatures.

For each creature:
  - Camera-follow GIF (inferno, black bg, 80x80 crop, 50 frames)
  - 6-frame timeline strip

Plus one big overview grid.

Output: results_localized/browse/
"""

import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LightSource
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localized_search_v4_locomotion import AnisotropicLenia

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RES = 256
TOTAL_STEPS = 10000
CROP = 80
GIF_FRAMES = 50  # every 200 steps
GIF_FPS = 12
TIMELINE_STEPS = [0, 2000, 4000, 6000, 8000, 10000]

OUT_DIR = "results_localized/browse"
GIF_DIR = os.path.join(OUT_DIR, "gifs")
TIMELINE_DIR = os.path.join(OUT_DIR, "timelines")
os.makedirs(GIF_DIR, exist_ok=True)
os.makedirs(TIMELINE_DIR, exist_ok=True)


def load_creatures():
    """Load all 174 non-extended creatures."""
    with open("results_localized_v5_broad/search_results.json") as f:
        data = json.load(f)
    archive = data["archive"]

    creatures = []
    for entry in archive:
        m = entry["metrics"]
        af = m.get("alive_fraction", 1.0)
        nd = m.get("net_displacement", 0)
        # Exclude extended textures: keep alive < 15% OR displacement <= 20
        # i.e. exclude (d>20 AND alive>=15%)
        if nd > 20 and af >= 0.15:
            continue  # skip extended textures
        creatures.append(entry)

    # Sort by kernel family, then displacement descending
    kernel_order = [
        "yukawa", "sinc", "riesz", "lennard_jones", "elliptical",
        "rbf_mixture", "morse", "polynomial", "power_law", "wendland_c2",
        "matern", "sech", "double_well", "spiral", "bump",
    ]
    def sort_key(e):
        kt = e["genome"]["kernel_type"]
        idx = kernel_order.index(kt) if kt in kernel_order else 99
        return (idx, -e["metrics"]["net_displacement"])
    creatures.sort(key=sort_key)
    return creatures


def init_state(sim):
    """Initialize with off-center elliptical blob."""
    YY, XX = torch.meshgrid(
        torch.arange(RES, device=sim.device, dtype=torch.float32),
        torch.arange(RES, device=sim.device, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy = RES * 0.35, RES * 0.5
    w = RES * 0.06
    r2 = ((XX - cx) / w) ** 2 + ((YY - cy) / (w / 2.0)) ** 2
    sim.state = (torch.exp(-r2) * 0.8).to(sim.device)


def find_centroid(state):
    Y, X = np.mgrid[0:RES, 0:RES]
    total = state.sum()
    if total < 1e-6:
        return RES // 2, RES // 2
    cx = int(round((X * state).sum() / total))
    cy = int(round((Y * state).sum() / total))
    return cx, cy


def crop_periodic(state, cx, cy, size=CROP):
    half = size // 2
    padded = np.tile(state, (3, 3))
    px, py = cx + RES, cy + RES
    return padded[py - half:py + half, px - half:px + half]


def simulate_and_capture(genome, seed=42):
    """Simulate creature, capture GIF frames + timeline snapshots."""
    torch.manual_seed(seed)
    sim = AnisotropicLenia(genome, device=DEVICE)
    init_state(sim)

    gif_interval = TOTAL_STEPS // GIF_FRAMES  # 200
    gif_data = []  # (step, state_np)
    timeline_data = {}

    state_np = sim.state.cpu().numpy().copy()
    if 0 in TIMELINE_STEPS:
        timeline_data[0] = state_np.copy()
    gif_data.append((0, state_np))

    for step in range(1, TOTAL_STEPS + 1):
        sim.step()
        if step % gif_interval == 0:
            state_np = sim.state.cpu().numpy().copy()
            gif_data.append((step, state_np))
        if step in TIMELINE_STEPS and step not in timeline_data:
            if step % gif_interval != 0:
                state_np = sim.state.cpu().numpy().copy()
            timeline_data[step] = state_np.copy()

    return gif_data, timeline_data


def render_gif(gif_data, out_path, cmap_name="inferno"):
    """Render camera-follow GIF with inferno colormap on black."""
    cmap = plt.cm.get_cmap(cmap_name)
    frames = []

    for step, state in gif_data:
        cx, cy = find_centroid(state)
        crop = crop_periodic(state, cx, cy)

        # Apply colormap: values 0→black, high→yellow/white
        rgba = cmap(crop)  # returns (H, W, 4) float
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        # Force zero regions to pure black
        mask = crop < 0.01
        rgb[mask] = 0
        frames.append(Image.fromarray(rgb))

    if frames:
        # Upscale for better viewing (80→240 px, 3x)
        upscale = 3
        upscaled = [f.resize((CROP * upscale, CROP * upscale), Image.NEAREST)
                     for f in frames]
        upscaled[0].save(
            out_path, save_all=True, append_images=upscaled[1:],
            duration=int(1000 / GIF_FPS), loop=0, optimize=True,
        )


def render_timeline(timeline_data, out_path, label, cmap_name="inferno"):
    """Render 6-frame timeline strip."""
    cmap = plt.cm.get_cmap(cmap_name)
    n = len(TIMELINE_STEPS)

    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.5), facecolor="black")
    fig.patch.set_facecolor("black")

    for i, step in enumerate(TIMELINE_STEPS):
        ax = axes[i]
        ax.set_facecolor("black")
        if step in timeline_data:
            state = timeline_data[step]
            cx, cy = find_centroid(state)
            crop = crop_periodic(state, cx, cy)
            ax.imshow(crop, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
        ax.set_title(f"t={step}", color="white", fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    fig.suptitle(label, color="#ffcc00", fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()


def render_overview(creatures, snapshots_at_5000, out_path):
    """Render big overview grid of all creatures at t=5000."""
    n = len(creatures)
    ncols = 15
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.7, nrows * 2.2),
                              facecolor="black")
    fig.patch.set_facecolor("black")
    cmap = plt.cm.inferno
    norm = Normalize(vmin=0, vmax=1)

    for idx in range(nrows * ncols):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]
        ax.set_facecolor("black")

        if idx < n:
            crop = snapshots_at_5000[idx]
            entry = creatures[idx]
            g = entry["genome"]
            m = entry["metrics"]
            kt = g["kernel_type"]
            gt = g["growth_type"]
            nd = m["net_displacement"]
            conc = m["spatial_concentration"]
            af = m["alive_fraction"]

            ax.imshow(crop, cmap=cmap, norm=norm, interpolation="nearest")

            # Shorten names
            gt_disp = gt.replace("_", " ")
            if len(gt_disp) > 14:
                gt_disp = gt_disp[:13] + "."

            ax.set_title(f"#{idx+1} {kt}\n{gt_disp}",
                         color="white", fontsize=5.5, fontweight="bold", pad=1)
            color = "#00ff88" if nd > 20 and conc > 0.7 and af < 0.15 else "#ffcc00" if nd > 20 else "#888888"
            ax.text(0.5, -0.02, f"d={nd:.0f} c={conc:.2f} a={af:.2f}",
                    transform=ax.transAxes, ha="center", va="top",
                    color=color, fontsize=5, fontweight="bold")
        else:
            ax.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle(
        f"All {n} Non-Texture Creatures — Grouped by Kernel Family\n"
        "t=5000, 80x80 crop, inferno | "
        "green=paper-50, yellow=d>20, gray=d<=20",
        color="white", fontsize=12, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.subplots_adjust(hspace=0.55, wspace=0.12)
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()
    print(f"  Overview saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")


def main():
    print("=" * 60)
    print("174 Creature Browse Renderer")
    print("=" * 60)

    print("\nLoading creatures...")
    creatures = load_creatures()
    print(f"Loaded {len(creatures)} creatures")

    # Count categories
    paper_50 = sum(1 for e in creatures
                   if e["metrics"]["net_displacement"] > 20
                   and e["metrics"]["spatial_concentration"] > 0.7
                   and e["metrics"]["alive_fraction"] < 0.15)
    big_movers = sum(1 for e in creatures if e["metrics"]["net_displacement"] > 20)
    print(f"  Paper-50 (d>20,c>0.7,a<0.15): {paper_50}")
    print(f"  Big movers (d>20): {big_movers}")
    print(f"  Small/no movers (d<=20): {len(creatures) - big_movers}")

    snapshots_at_5000 = []  # for overview

    for i, entry in enumerate(creatures):
        g = entry["genome"]
        m = entry["metrics"]
        kt = g["kernel_type"]
        gt = g["growth_type"]
        nd = m["net_displacement"]
        conc = m["spatial_concentration"]
        af = m["alive_fraction"]

        tag = f"{i+1:03d}_{kt}_{gt}"
        print(f"\n[{i+1:3d}/{len(creatures)}] {kt}/{gt} "
              f"(d={nd:.0f}, c={conc:.2f}, a={af:.2f})...")

        # Simulate
        gif_data, timeline_data = simulate_and_capture(g)

        # Grab t=5000 snapshot for overview
        # Find closest GIF frame to t=5000
        best_frame = None
        for step, state in gif_data:
            if step >= 5000:
                best_frame = state
                break
        if best_frame is None and gif_data:
            best_frame = gif_data[-1][1]
        cx, cy = find_centroid(best_frame)
        crop_5k = crop_periodic(best_frame, cx, cy)
        snapshots_at_5000.append(crop_5k)

        # Render GIF
        gif_path = os.path.join(GIF_DIR, f"{tag}.gif")
        render_gif(gif_data, gif_path)
        gif_kb = os.path.getsize(gif_path) / 1024
        print(f"  GIF: {gif_kb:.0f} KB", end="")

        # Render timeline
        tl_path = os.path.join(TIMELINE_DIR, f"{tag}.png")
        label = f"#{i+1} {kt}/{gt} — d={nd:.0f}, c={conc:.2f}"
        render_timeline(timeline_data, tl_path, label)
        tl_kb = os.path.getsize(tl_path) / 1024
        print(f" | Timeline: {tl_kb:.0f} KB")

    # Render overview
    print(f"\nRendering overview grid ({len(creatures)} creatures)...")
    overview_path = os.path.join(OUT_DIR, "overview_all.png")
    render_overview(creatures, snapshots_at_5000, overview_path)

    # Summary
    total_gif_kb = sum(os.path.getsize(os.path.join(GIF_DIR, f)) / 1024
                       for f in os.listdir(GIF_DIR) if f.endswith(".gif"))
    total_tl_kb = sum(os.path.getsize(os.path.join(TIMELINE_DIR, f)) / 1024
                      for f in os.listdir(TIMELINE_DIR) if f.endswith(".png"))
    print(f"\n{'='*60}")
    print(f"Done! {len(creatures)} creatures rendered")
    print(f"  GIFs total: {total_gif_kb/1024:.1f} MB")
    print(f"  Timelines total: {total_tl_kb/1024:.1f} MB")
    print(f"  Output: {OUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
