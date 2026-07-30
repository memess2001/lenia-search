#!/usr/bin/env python3
"""Render high-res GIFs + time-snapshot panels for localized candidates.

Outputs per candidate:
  1. GIF at 256x256 (2000 steps)
  2. Snapshot panel: 6 frames at t=0, 200, 500, 1000, 1500, 2000
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFont
from src.geometry_factory import create_lenia
from src.expression_compiler import _center_blob_init


SNAPSHOT_STEPS = [0, 200, 500, 1000, 1500, 2000]
RESOLUTION = 256
TOTAL_STEPS = 2000
RECORD_INTERVAL = 1  # record every step so we can pick exact frames
CELL_SIZE = 256  # each snapshot cell size
FONT_SIZE = 16


def get_font():
    """Try to get a reasonable font."""
    for name in ["/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/SFNSMono.ttf",
                 "/Library/Fonts/Arial.ttf"]:
        if os.path.exists(name):
            try:
                return ImageFont.truetype(name, FONT_SIZE)
            except Exception:
                continue
    return ImageFont.load_default()


def render_candidate(genome_dict, label, metrics, output_dir):
    """Render one candidate: GIF + snapshot panel."""
    genome = dict(genome_dict)
    geometry = genome.get("geometry", "flat_plane")
    genome["resolution"] = RESOLUTION

    # Scale kernel radius for higher resolution
    if "kernel_radius" in genome:
        genome["kernel_radius"] = genome_dict["kernel_radius"] * (RESOLUTION / 128.0)

    print(f"\n  [{label}] Creating simulator ({geometry})...")
    sim = create_lenia(genome, device="cpu")

    # Use center_blob init for flat_plane
    if geometry == "flat_plane":
        sim.state = _center_blob_init(RESOLUTION, blob_radius=0.15, device="cpu")

    # Run and record every step
    print(f"  [{label}] Running {TOTAL_STEPS} steps...")
    all_frames = [sim.state.cpu().numpy().copy()]
    for step in range(1, TOTAL_STEPS + 1):
        sim.step()
        if step % 10 == 0 or step in SNAPSHOT_STEPS:
            all_frames.append(sim.state.cpu().numpy().copy())
        if step % 500 == 0:
            print(f"    step {step}/{TOTAL_STEPS}")

    # Map step -> frame index
    recorded_steps = [0] + [s for s in range(1, TOTAL_STEPS + 1) if s % 10 == 0 or s in SNAPSHOT_STEPS]

    # === Generate GIF (every 10th step) ===
    print(f"  [{label}] Generating GIF...")
    gif_frames = []
    for i, arr in enumerate(all_frames):
        if arr.ndim > 2:
            arr = arr.mean(axis=0)
        img_arr = (arr * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="L")
        if img.size[0] < 512:
            img = img.resize((512, 512), Image.NEAREST)
        gif_frames.append(img)

    gif_path = os.path.join(output_dir, f"{label}_256x256.gif")
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=50, loop=0)
    print(f"    -> {gif_path}")

    # === Generate snapshot panel ===
    print(f"  [{label}] Generating snapshot panel...")
    font = get_font()
    n_snapshots = len(SNAPSHOT_STEPS)
    panel_w = CELL_SIZE * n_snapshots
    panel_h = CELL_SIZE + 30  # extra space for labels
    panel = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)

    for col, target_step in enumerate(SNAPSHOT_STEPS):
        # Find closest recorded frame
        best_idx = min(range(len(recorded_steps)), key=lambda i: abs(recorded_steps[i] - target_step))
        arr = all_frames[best_idx]
        if arr.ndim > 2:
            arr = arr.mean(axis=0)
        img_arr = (arr * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="L").convert("RGB")
        img = img.resize((CELL_SIZE, CELL_SIZE), Image.NEAREST)

        x_offset = col * CELL_SIZE
        panel.paste(img, (x_offset, 0))

        # Draw label
        label_text = f"t={target_step}"
        bbox = draw.textbbox((0, 0), label_text, font=font)
        tw = bbox[2] - bbox[0]
        tx = x_offset + (CELL_SIZE - tw) // 2
        draw.text((tx, CELL_SIZE + 5), label_text, fill=(0, 0, 0), font=font)

    # Add metadata header
    m = metrics
    g = genome_dict
    title = (f"{label}: {g['kernel_type']}+{g['growth_type']}+{g['geometry']}  "
             f"| alive={m['alive_fraction']:.3f} clusters={m['num_clusters']} "
             f"fitness={m.get('fitness', 0):.4f}")

    # Create final image with title
    title_h = 35
    final = Image.new("RGB", (panel_w, panel_h + title_h), color=(255, 255, 255))
    final_draw = ImageDraw.Draw(final)
    final_draw.text((10, 8), title, fill=(0, 0, 0), font=font)
    final.paste(panel, (0, title_h))

    panel_path = os.path.join(output_dir, f"{label}_snapshots.png")
    final.save(panel_path, dpi=(150, 150))
    print(f"    -> {panel_path}")

    return gif_path, panel_path


def main():
    archive_path = "results_localized/archive.json"
    output_dir = "results_localized/gifs"
    os.makedirs(output_dir, exist_ok=True)

    with open(archive_path) as f:
        data = json.load(f)

    entries = data["entries"]
    candidates = [e for e in entries
                  if e["metrics"]["alive_fraction"] < 0.3
                  and e["metrics"]["num_clusters"] <= 5]
    candidates.sort(key=lambda e: e["fitness"], reverse=True)

    print(f"=== Rendering {len(candidates)} localized candidates ===")
    print(f"Resolution: {RESOLUTION}x{RESOLUTION}, Steps: {TOTAL_STEPS}")
    print(f"Snapshots at: {SNAPSHOT_STEPS}\n")

    all_panel_paths = []
    for i, entry in enumerate(candidates):
        label = f"local{i+1}"
        m = entry["metrics"]
        g = entry["genome"]
        print(f"\n{'='*60}")
        print(f"{label}: fit={entry['fitness']:.4f} alive={m['alive_fraction']:.3f} "
              f"clust={m['num_clusters']} | {g['kernel_type']}+{g['growth_type']}+{g['geometry']}")

        metrics_with_fitness = dict(m)
        metrics_with_fitness["fitness"] = entry["fitness"]
        gif_path, panel_path = render_candidate(
            entry["genome"], label, metrics_with_fitness, output_dir
        )
        all_panel_paths.append(panel_path)

    # === Combined panel (all 4 stacked vertically) ===
    print(f"\n{'='*60}")
    print("Creating combined panel...")
    panels = [Image.open(p) for p in all_panel_paths]
    total_h = sum(p.height for p in panels) + 10 * (len(panels) - 1)
    max_w = max(p.width for p in panels)
    combined = Image.new("RGB", (max_w, total_h), color=(255, 255, 255))
    y = 0
    for p in panels:
        combined.paste(p, (0, y))
        y += p.height + 10

    combined_path = os.path.join(output_dir, "all_localized_candidates_snapshots.png")
    combined.save(combined_path, dpi=(150, 150))
    print(f"  -> {combined_path}")

    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
