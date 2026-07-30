#!/usr/bin/env python3
"""Generate microscope-style 3D swimming GIFs for morse and bessel creatures.

Camera follows centroid, fixed viewing angle, black background, warm colormap.
Effect: a glowing translucent organism swimming in darkness.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localized_search_v4_locomotion import AnisotropicLenia

OUT_DIR = "paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

BESSEL_GENOME = {
    "kernel_type": "bessel", "growth_type": "gaussian_bell", "geometry": "flat_plane",
    "resolution": 256, "kernel_radius": 19.781727907414716, "kernel_peaks": 2.3706531180187023,
    "growth_mu": 0.20070798643942994, "growth_sigma": 0.02896182105922773,
    "dt": 0.02857361828857563, "aniso_strength": 0.21452527791620304,
    "aniso_angle": 1.7911590897613312,
}

MORSE_GENOME = {
    "kernel_type": "morse", "growth_type": "relu_like", "geometry": "flat_plane",
    "resolution": 256, "kernel_radius": 20.278144951053108, "kernel_peaks": 0.9056801391770588,
    "growth_mu": 0.12471621858819384, "growth_sigma": 0.037866588257700104,
    "dt": 0.2207736733736189, "aniso_strength": 0.4648416110974048,
    "aniso_angle": 3.9184863096721143,
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RES = 256
TOTAL_STEPS = 10000
STEP_PER_FRAME = 100
CROP = 64
FPS = 15
ELEV, AZIM = 45, -60


def init_state(sim, cx_frac=0.35, cy_frac=0.5, aspect=2.0, width_frac=0.06):
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


def render_frame(crop, fig, ax, cmap="inferno"):
    """Render a single 3D surface frame — glowing organism on black."""
    ax.clear()

    h, w = crop.shape
    X = np.arange(w)
    Y = np.arange(h)
    X, Y = np.meshgrid(X, Y)

    ls = LightSource(azdeg=315, altdeg=45)

    ax.plot_surface(
        X, Y, crop,
        cmap=cmap,
        antialiased=True,
        rcount=80,
        ccount=80,
        shade=True,
        lightsource=ls,
        alpha=0.75,
    )

    ax.set_zlim(0, 1.0)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    ax.set_facecolor("black")


def make_swimming_gif(genome, name, cmap="inferno", seed=42):
    """Generate camera-following swimming GIF."""
    print(f"  Simulating {name}...")
    torch.manual_seed(seed)
    sim = AnisotropicLenia(genome, device=DEVICE)
    init_state(sim)

    # Collect frames
    frame_data = []  # (step, state_numpy)
    state_np = sim.state.cpu().numpy().copy()
    frame_data.append((0, state_np))

    for step in range(1, TOTAL_STEPS + 1):
        sim.step()
        if step % STEP_PER_FRAME == 0:
            state_np = sim.state.cpu().numpy().copy()
            frame_data.append((step, state_np))

    print(f"  Captured {len(frame_data)} frames. Rendering 3D...")

    # Render each frame
    fig = plt.figure(figsize=(6, 5), dpi=120)
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    pil_frames = []
    for i, (step, state) in enumerate(frame_data):
        cx, cy = find_centroid(state)
        crop = crop_periodic(state, cx, cy)
        render_frame(crop, fig, ax, cmap=cmap)

        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)
        # Convert RGBA to RGB on black background
        rgb = buf[:, :, :3].copy()
        pil_frames.append(Image.fromarray(rgb))

        if (i + 1) % 20 == 0:
            print(f"    Frame {i+1}/{len(frame_data)} (step {step})")

    plt.close()

    out_path = f"{OUT_DIR}/{name}_3d_swimming.gif"
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved: {out_path} ({size_kb:.0f} KB, {len(pil_frames)} frames)")


if __name__ == "__main__":
    print("=" * 60)
    print("Swimming GIF Rendering Pipeline")
    print("=" * 60)

    print("\n[1/2] Morse/relu_like...")
    make_swimming_gif(MORSE_GENOME, "morse_relu_like", cmap="inferno")

    print("\n[2/2] Bessel/gaussian_bell...")
    make_swimming_gif(BESSEL_GENOME, "bessel_gaussian_bell", cmap="plasma")

    print("\nDone!")
