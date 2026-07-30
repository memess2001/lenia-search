#!/usr/bin/env python3
"""3D high-quality renders of morse/relu_like and bessel/gaussian_bell creatures.

Produces:
  - paper/figures/fig_morse_3d_timeline.png   (6-frame 3D timeline)
  - paper/figures/fig_bessel_3d_timeline.png  (6-frame 3D timeline)
  - paper/figures/fig7_dual_timeline.png      (dual-panel 2D Figure 7)
  - paper/figures/morse_3d_animation.gif      (optional rotation GIF)
  - paper/figures/bessel_3d_animation.gif     (optional rotation GIF)
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localized_search_v4_locomotion import AnisotropicLenia

OUT_DIR = "paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# Creature parameters
# =====================================================================

BESSEL_GENOME = {
    "kernel_type": "bessel",
    "growth_type": "gaussian_bell",
    "geometry": "flat_plane",
    "resolution": 256,
    "kernel_radius": 19.781727907414716,
    "kernel_peaks": 2.3706531180187023,
    "growth_mu": 0.20070798643942994,
    "growth_sigma": 0.02896182105922773,
    "dt": 0.02857361828857563,
    "aniso_strength": 0.21452527791620304,
    "aniso_angle": 1.7911590897613312,
}

MORSE_GENOME = {
    "kernel_type": "morse",
    "growth_type": "relu_like",
    "geometry": "flat_plane",
    "resolution": 256,
    "kernel_radius": 20.278144951053108,
    "kernel_peaks": 0.9056801391770588,
    "growth_mu": 0.12471621858819384,
    "growth_sigma": 0.037866588257700104,
    "dt": 0.2207736733736189,
    "aniso_strength": 0.4648416110974048,
    "aniso_angle": 3.9184863096721143,
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RES = 256
TOTAL_STEPS = 10000
SNAPSHOT_STEPS = [0, 2000, 4000, 6000, 8000, 10000]
CROP_SIZE = 64  # slightly larger than 60 for safety


# =====================================================================
# Simulation
# =====================================================================

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


def simulate(genome, seed=42):
    """Run simulation and capture snapshots."""
    torch.manual_seed(seed)
    sim = AnisotropicLenia(genome, device=DEVICE)
    init_state(sim)

    snapshots = {}
    snapshots[0] = sim.state.cpu().numpy().copy()

    for step in range(1, TOTAL_STEPS + 1):
        sim.step()
        if step in SNAPSHOT_STEPS:
            snapshots[step] = sim.state.cpu().numpy().copy()

    return snapshots


def find_centroid(state):
    """Find mass centroid with periodic boundary handling."""
    Y, X = np.mgrid[0:RES, 0:RES]
    total = state.sum()
    if total < 1e-6:
        return RES // 2, RES // 2
    cx = (X * state).sum() / total
    cy = (Y * state).sum() / total
    return int(round(cx)), int(round(cy))


def crop_around_centroid(state, cx, cy, size=CROP_SIZE):
    """Crop a window around the centroid, handling periodic boundaries."""
    half = size // 2
    # Pad periodically
    padded = np.tile(state, (3, 3))
    # Shift centroid to middle tile
    px, py = cx + RES, cy + RES
    crop = padded[py - half:py + half, px - half:px + half]
    return crop


# =====================================================================
# 3D Rendering
# =====================================================================

def render_3d_frame(state_crop, ax, title="", cmap="inferno", elev=50, azim=-60):
    """Render a single cropped state as a 3D surface."""
    h, w = state_crop.shape
    X = np.arange(w)
    Y = np.arange(h)
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(
        X, Y, state_crop,
        cmap=cmap,
        antialiased=True,
        rcount=100,
        ccount=100,
        shade=True,
        alpha=0.95,
    )
    ax.set_zlim(0, 1.0)
    ax.view_init(elev=elev, azim=azim)

    # Clean appearance
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if title:
        ax.set_title(title, fontsize=11, pad=-5, fontweight="bold")


def make_3d_timeline(snapshots, name, cmap="inferno"):
    """Create a 2×3 grid of 3D surface plots."""
    fig = plt.figure(figsize=(15, 10))

    for i, step in enumerate(SNAPSHOT_STEPS):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        state = snapshots[step]
        cx, cy = find_centroid(state)
        crop = crop_around_centroid(state, cx, cy)
        render_3d_frame(crop, ax, title=f"t = {step}", cmap=cmap)

    plt.suptitle(
        f"{name.replace('_', '/')} moving soliton — 3D surface evolution",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = f"{OUT_DIR}/fig_{name}_3d_timeline.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# =====================================================================
# 2D Timeline (for Figure 7 dual-panel)
# =====================================================================

def make_2d_timeline_strip(snapshots, height=256):
    """Create a horizontal strip of grayscale frames."""
    frames = []
    for step in SNAPSHOT_STEPS:
        state = snapshots[step]
        # Full 256x256 frame
        img = (state * 255).clip(0, 255).astype(np.uint8)
        frames.append(img)

    gap = 4
    total_w = len(frames) * height + (len(frames) - 1) * gap
    strip = np.zeros((height, total_w), dtype=np.uint8)
    for j, frame in enumerate(frames):
        x_off = j * (height + gap)
        strip[:, x_off:x_off + height] = frame

    return strip


def make_figure7_dual(bessel_snapshots, morse_snapshots):
    """Create dual-panel Figure 7: bessel (left) vs morse (right)."""
    fig, axes = plt.subplots(2, 6, figsize=(16, 5.5))

    for i, step in enumerate(SNAPSHOT_STEPS):
        # Bessel (top row)
        axes[0, i].imshow(bessel_snapshots[step], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"t={step}", fontsize=8)
        axes[0, i].axis("off")

        # Morse (bottom row)
        axes[1, i].imshow(morse_snapshots[step], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"t={step}", fontsize=8)
        axes[1, i].axis("off")

    # Row labels
    axes[0, 0].set_ylabel("bessel/gaussian_bell", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("morse/relu_like", fontsize=10, fontweight="bold")

    plt.suptitle(
        "Figure 7: Moving solitons from expanded kernel families at 256×256",
        fontsize=12, fontweight="bold", y=1.0,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = f"{OUT_DIR}/fig7_dual_timeline.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# =====================================================================
# 3D Rotation GIF
# =====================================================================

def make_rotation_gif(snapshots, name, cmap="inferno", fps=15, n_rotations=120):
    """Create a rotating 3D view GIF of the final creature state."""
    # Use the state at t=8000 (well-formed creature)
    state = snapshots[8000]
    cx, cy = find_centroid(state)
    crop = crop_around_centroid(state, cx, cy)

    h, w = crop.shape
    X = np.arange(w)
    Y = np.arange(h)
    X, Y = np.meshgrid(X, Y)

    frames_pil = []
    for i in range(n_rotations):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            X, Y, crop,
            cmap=cmap,
            antialiased=True,
            rcount=80,
            ccount=80,
            shade=True,
        )
        ax.set_zlim(0, 1.0)
        azim = -60 + (i * 360 / n_rotations)
        ax.view_init(elev=45, azim=azim)

        # Clean
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("none")
        ax.yaxis.pane.set_edgecolor("none")
        ax.zaxis.pane.set_edgecolor("none")
        ax.grid(False)
        ax.set_title(f"{name.replace('_', '/')} soliton", fontsize=11)

        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)
        frames_pil.append(Image.fromarray(buf[:, :, :3]))
        plt.close()

    out_path = f"{OUT_DIR}/{name}_3d_animation.gif"
    frames_pil[0].save(
        out_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"  Saved: {out_path}")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("3D Creature Rendering Pipeline")
    print("=" * 60)

    # --- Simulate bessel ---
    print("\n[1/6] Simulating bessel/gaussian_bell (10000 steps)...")
    bessel_snaps = simulate(BESSEL_GENOME, seed=42)

    # --- Simulate morse ---
    print("[2/6] Simulating morse/relu_like (10000 steps)...")
    morse_snaps = simulate(MORSE_GENOME, seed=42)

    # --- 3D Timelines ---
    print("[3/6] Rendering bessel 3D timeline...")
    make_3d_timeline(bessel_snaps, "bessel_gaussian_bell", cmap="magma")

    print("[4/6] Rendering morse 3D timeline...")
    make_3d_timeline(morse_snaps, "morse_relu_like", cmap="inferno")

    # --- Figure 7 dual panel ---
    print("[5/6] Creating Figure 7 dual-panel timeline...")
    make_figure7_dual(bessel_snaps, morse_snaps)

    # --- Rotation GIFs ---
    print("[6/6] Generating rotation GIFs...")
    make_rotation_gif(morse_snaps, "morse_relu_like", cmap="inferno", n_rotations=90)
    make_rotation_gif(bessel_snaps, "bessel_gaussian_bell", cmap="magma", n_rotations=90)

    print("\nAll renders complete!")
