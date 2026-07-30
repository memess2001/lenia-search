#!/usr/bin/env python3
"""Generate high-resolution GIF animations for localized creature candidates.

Re-runs each candidate at 256×256, saves frames every N steps,
and produces animated GIFs.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.geometry_factory import create_lenia

RESOLUTION = 256
STEPS = 10000
DEVICE = "mps"
FRAME_INTERVAL = 20  # save a frame every N steps
FPS = 15  # GIF frame rate

out_dir = "results_localized_v3/creature_gifs"
os.makedirs(out_dir, exist_ok=True)

# Also look at the user-spotted candidates that might not be in archive
# We'll re-derive their genomes from the search log or use nearby params

CREATURES = [
    {
        "name": "power_law_gaussian_bell",
        "genome": {
            "kernel_type": "power_law",
            "growth_type": "gaussian_bell",
            "geometry": "flat_plane",
            "resolution": RESOLUTION,
            "kernel_radius": 24.3,
            "kernel_peaks": 1.39,
            "growth_mu": 0.3400,
            "growth_sigma": 0.0499,
            "dt": 0.269,
        },
        "info": "conc=0.909, alive=2.2%, vel=138 — best candidate"
    },
    {
        "name": "lennard_jones_relu_like",
        "genome": {
            "kernel_type": "lennard_jones",
            "growth_type": "relu_like",
            "geometry": "flat_plane",
            "resolution": RESOLUTION,
            "kernel_radius": 21.0,
            "kernel_peaks": 1.15,
            "growth_mu": 0.1400,
            "growth_sigma": 0.0934,
            "dt": 0.045,
        },
        "info": "conc=0.857, alive=2.4%, vel=0.45 — static blob"
    },
    {
        "name": "bump_fitzhugh_nagumo",
        "genome": {
            "kernel_type": "bump",
            "growth_type": "fitzhugh_nagumo",
            "geometry": "flat_plane",
            "resolution": RESOLUTION,
            "kernel_radius": 22.7,
            "kernel_peaks": 1.31,
            "growth_mu": 0.1693,
            "growth_sigma": 0.0351,
            "dt": 0.298,
        },
        "info": "conc=0.815, alive=2.8%, vel=4.03 — dark rings"
    },
]

# Also try the user-spotted ones with approx params
# These were from early evals, may have been replaced in archive
# We can try sinc/step_function and cosine/step_function with typical params
USER_SPOTTED = [
    {
        "name": "sinc_step_function",
        "genome": {
            "kernel_type": "sinc",
            "growth_type": "step_function",
            "geometry": "flat_plane",
            "resolution": RESOLUTION,
            "kernel_radius": 15.0,
            "kernel_peaks": 1.5,
            "growth_mu": 0.15,
            "growth_sigma": 0.03,
            "dt": 0.1,
        },
        "info": "user-spotted small dot (approx params)"
    },
    {
        "name": "cosine_step_function",
        "genome": {
            "kernel_type": "cosine",
            "growth_type": "step_function",
            "geometry": "flat_plane",
            "resolution": RESOLUTION,
            "kernel_radius": 15.0,
            "kernel_peaks": 1.5,
            "growth_mu": 0.15,
            "growth_sigma": 0.03,
            "dt": 0.1,
        },
        "info": "user-spotted small ring (approx params)"
    },
    {
        "name": "lennard_jones_sigmoid_pair",
        "genome": {
            "kernel_type": "lennard_jones",
            "growth_type": "sigmoid_pair",
            "geometry": "flat_plane",
            "resolution": RESOLUTION,
            "kernel_radius": 15.0,
            "kernel_peaks": 1.5,
            "growth_mu": 0.15,
            "growth_sigma": 0.03,
            "dt": 0.1,
        },
        "info": "user-spotted dark dot (approx params)"
    },
]

# Try to get actual params from the search for user-spotted ones
try:
    with open("results_localized_v3/search_results.json") as f:
        search_data = json.load(f)

    # Look for these combos in the full archive (any alive level)
    for us in USER_SPOTTED:
        kt = us["genome"]["kernel_type"]
        gt = us["genome"]["growth_type"]
        matches = [e for e in search_data["archive"]
                   if e["genome"]["kernel_type"] == kt and e["genome"]["growth_type"] == gt]
        # Pick the one with lowest alive_fraction (most localized)
        if matches:
            best = min(matches, key=lambda e: e["metrics"]["alive_fraction"])
            if best["metrics"]["alive_fraction"] < 0.3:
                us["genome"].update({
                    "kernel_radius": best["genome"]["kernel_radius"],
                    "kernel_peaks": best["genome"]["kernel_peaks"],
                    "growth_mu": best["genome"]["growth_mu"],
                    "growth_sigma": best["genome"]["growth_sigma"],
                    "dt": best["genome"]["dt"],
                })
                m = best["metrics"]
                us["info"] = (f"found in archive: conc={m['spatial_concentration']:.3f} "
                              f"alive={m['alive_fraction']:.3f} vel={m['velocity']:.2f}")
                print(f"  Updated {kt}/{gt} from archive: alive={m['alive_fraction']:.3f}")
            else:
                print(f"  {kt}/{gt}: best alive={best['metrics']['alive_fraction']:.3f} (not very localized)")
        else:
            print(f"  {kt}/{gt}: not in archive, using default params")
except Exception as e:
    print(f"  Could not read search results: {e}")

ALL_CREATURES = CREATURES + USER_SPOTTED

print(f"\nGenerating GIFs for {len(ALL_CREATURES)} creatures")
print(f"Resolution: {RESOLUTION}, Steps: {STEPS}, Frame interval: {FRAME_INTERVAL}")
print("=" * 60)

for creature in ALL_CREATURES:
    name = creature["name"]
    genome = creature["genome"]
    info = creature["info"]

    print(f"\n[{name}] {info}")

    try:
        sim = create_lenia(genome, device=DEVICE)
    except Exception as e:
        print(f"  ERROR creating sim: {e}")
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

    # Collect frames
    frames = []
    t0 = time.time()

    for step in range(STEPS + 1):
        if step % FRAME_INTERVAL == 0:
            s = sim.state.cpu().numpy()
            # Convert to uint8 image
            img_array = (np.clip(s, 0, 1) * 255).astype(np.uint8)
            frames.append(Image.fromarray(img_array))

        if step < STEPS:
            try:
                sim.step()
            except Exception as e:
                print(f"  Simulation error at step {step}: {e}")
                break

    elapsed = time.time() - t0
    print(f"  Collected {len(frames)} frames in {elapsed:.1f}s")

    if len(frames) < 10:
        print(f"  Too few frames, skipping")
        continue

    # Check if creature survived
    final = sim.state.cpu().numpy()
    alive_frac = (final > 0.1).mean()
    print(f"  Final alive_fraction: {alive_frac:.3f}")

    # Save GIF (use last 300 frames = last 6000 steps to show steady-state behavior)
    n_gif_frames = min(300, len(frames))
    gif_frames = frames[-n_gif_frames:]

    # Also save a shorter "highlight" GIF with fewer frames for smaller file size
    highlight_frames = gif_frames[::3]  # every 3rd frame

    # Full GIF
    gif_path = f"{out_dir}/{name}_full.gif"
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
    )
    gif_size = os.path.getsize(gif_path) / 1024
    print(f"  Full GIF: {gif_path} ({gif_size:.0f} KB, {len(gif_frames)} frames)")

    # Highlight GIF (smaller)
    highlight_path = f"{out_dir}/{name}_highlight.gif"
    highlight_frames[0].save(
        highlight_path,
        save_all=True,
        append_images=highlight_frames[1:],
        duration=int(1000 / (FPS / 3)),
        loop=0,
    )
    hl_size = os.path.getsize(highlight_path) / 1024
    print(f"  Highlight GIF: {highlight_path} ({hl_size:.0f} KB, {len(highlight_frames)} frames)")

    # Save final frame as high-res PNG
    final_path = f"{out_dir}/{name}_final_256.png"
    Image.fromarray((np.clip(final, 0, 1) * 255).astype(np.uint8)).save(final_path)
    print(f"  Final PNG: {final_path}")

    # Save timeline (t=0, 2000, 4000, 6000, 8000, 10000)
    timeline_steps = [0, 100, 200, 300, 400, 500]  # frame indices (×20 = actual steps)
    if len(frames) > max(timeline_steps):
        fig_w = RESOLUTION * len(timeline_steps) + (len(timeline_steps) - 1) * 4
        panel = np.ones((RESOLUTION, fig_w), dtype=np.uint8) * 0  # black background
        for j, idx in enumerate(timeline_steps):
            x_off = j * (RESOLUTION + 4)
            panel[:, x_off:x_off+RESOLUTION] = np.array(frames[idx])
        Image.fromarray(panel).save(f"{out_dir}/{name}_timeline.png")
        print(f"  Timeline saved")

print(f"\n{'='*60}")
print(f"All GIFs saved to {out_dir}/")
