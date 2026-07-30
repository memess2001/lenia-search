#!/usr/bin/env python3
"""Render GIFs from the Track A localized search archive.

Picks interesting entries: top fitness, localized candidates, and
low-cluster-count entries. Uses geometry_factory to handle all geometries.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from src.geometry_factory import create_lenia
from src.expression_compiler import _center_blob_init


def render_gif(entry, rank_label, output_dir, resolution=128, steps=1000,
               record_interval=5, init_mode="center_blob"):
    genome = dict(entry["genome"])
    genome["resolution"] = resolution
    m = entry["metrics"]

    fname = (f"{rank_label}_fit{entry['fitness']:.3f}"
             f"_alive{m['alive_fraction']:.2f}"
             f"_clust{m['num_clusters']}"
             f"_{genome['kernel_type']}_{genome['geometry']}.gif")

    print(f"  Rendering {fname} ...")

    try:
        sim = create_lenia(genome, device="cpu")

        if init_mode == "center_blob":
            # Only works for flat geometries with 2D state
            if genome.get("geometry", "flat_plane") == "flat_plane":
                sim.state = _center_blob_init(resolution, blob_radius=0.15, device="cpu")
            # For other geometries, use default init

        history = sim.run(steps, record_interval=record_interval)

        frames = []
        for state in history:
            arr = state.cpu().numpy()
            if arr.ndim > 2:
                arr = arr.mean(axis=0)  # multi-channel -> mean
            img_arr = (arr * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr, mode="L")
            img = img.resize((256, 256), Image.NEAREST)
            frames.append(img)

        if frames:
            path = os.path.join(output_dir, fname)
            frames[0].save(path, save_all=True, append_images=frames[1:],
                           duration=50, loop=0)
            print(f"    -> {path}")
            return path
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    archive_path = "results_localized/archive.json"
    output_dir = "results_localized/gifs"
    os.makedirs(output_dir, exist_ok=True)

    with open(archive_path) as f:
        data = json.load(f)

    entries = data["entries"]
    print(f"Loaded {len(entries)} entries from {archive_path}\n")

    # === Selection ===
    selected = []

    # Top 5 by fitness
    by_fitness = sorted(entries, key=lambda e: e["fitness"], reverse=True)
    for i, e in enumerate(by_fitness[:5]):
        selected.append((f"top{i+1}", e))

    # Localized candidates: alive < 0.3, clusters <= 5
    localized = [e for e in entries
                 if e["metrics"]["alive_fraction"] < 0.3
                 and e["metrics"]["num_clusters"] <= 5]
    localized.sort(key=lambda e: e["fitness"], reverse=True)
    for i, e in enumerate(localized[:4]):
        selected.append((f"local{i+1}", e))

    # Interesting low-cluster entries (not already selected, clusters < 20, alive 0.3-0.7)
    already = {id(e) for _, e in selected}
    interesting = [e for e in entries
                   if id(e) not in already
                   and e["metrics"]["num_clusters"] < 20
                   and 0.3 < e["metrics"]["alive_fraction"] < 0.7
                   and e["fitness"] > 0.2]
    interesting.sort(key=lambda e: e["fitness"], reverse=True)
    for i, e in enumerate(interesting[:3]):
        selected.append((f"interest{i+1}", e))

    print(f"Selected {len(selected)} entries to render:\n")
    for label, e in selected:
        m = e["metrics"]
        g = e["genome"]
        print(f"  {label:12s} fit={e['fitness']:.4f} alive={m['alive_fraction']:.3f} "
              f"clust={m['num_clusters']:4d} | {g['kernel_type']}+{g['growth_type']}+{g['geometry']}")

    print(f"\nRendering GIFs (1000 steps, 128x128 -> 256x256)...\n")

    paths = []
    for label, entry in selected:
        # Use center_blob for localized candidates, random for others
        init = "center_blob" if label.startswith("local") else "default"
        p = render_gif(entry, label, output_dir, steps=1000,
                       record_interval=5, init_mode=init)
        if p:
            paths.append(p)

    print(f"\nDone! Generated {len(paths)} GIFs in {output_dir}/")


if __name__ == "__main__":
    main()
