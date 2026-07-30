#!/usr/bin/env python3
"""Quick visualization for Lenia MAP-Elites results.

Usage:
    python visualize.py                          # default archive.json
    python visualize.py --archive results/archive.json --top_n 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_archive(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def plot_heatmap(archive_data: dict, output_dir: str) -> str:
    """Plot the MAP-Elites behaviour-space heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_bins = archive_data["n_bins"]
    dims = archive_data["behavior_dims"]
    entries = archive_data["entries"]

    grid = np.full((n_bins[0], n_bins[1]), np.nan)
    for entry in entries:
        i, j = entry["bin"]
        grid[i, j] = entry["fitness"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel(dims[0])
    ax.set_ylabel(dims[1])
    ax.set_title(
        f"MAP-Elites Archive  ({len(entries)} cells / "
        f"{n_bins[0]*n_bins[1]} total,  "
        f"{archive_data.get('total_evaluated', '?')} evaluated)"
    )
    fig.colorbar(im, ax=ax, label="fitness (complexity)")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "map_elites_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {path}")
    return path


def generate_gifs(archive_data: dict, output_dir: str, top_n: int = 5) -> list[str]:
    """Simulate the top-N highest-fitness genomes and save animated GIFs."""
    from PIL import Image

    from src.lenia_core import Lenia

    entries = sorted(archive_data["entries"], key=lambda e: e["fitness"], reverse=True)
    top = entries[:top_n]

    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    for rank, entry in enumerate(top):
        genome = entry["genome"]
        genome.setdefault("resolution", 128)
        print(f"  Generating GIF {rank+1}/{top_n}  "
              f"(fitness={entry['fitness']:.4f}, bin={entry['bin']})")

        sim = Lenia(genome, device="cpu")
        history = sim.run(500, record_interval=5)

        # Convert to PIL frames
        frames: list[Image.Image] = []
        for state in history:
            arr = state.numpy()
            img_arr = (arr * 255).clip(0, 255).astype(np.uint8)
            # Scale up for visibility
            img = Image.fromarray(img_arr, mode="L")
            img = img.resize((256, 256), Image.NEAREST)
            frames.append(img)

        if frames:
            path = os.path.join(output_dir, f"top_{rank+1}.gif")
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=50,
                loop=0,
            )
            paths.append(path)
            print(f"    Saved {path}")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise Lenia MAP-Elites results.")
    parser.add_argument(
        "--archive", type=str, default="results/archive.json",
        help="Path to archive JSON.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory for images and GIFs.",
    )
    parser.add_argument(
        "--top_n", type=int, default=5,
        help="Number of top genomes to generate GIFs for.",
    )
    parser.add_argument(
        "--no_gif", action="store_true",
        help="Skip GIF generation (only produce heatmap).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.archive):
        print(f"Archive not found: {args.archive}")
        print("Run `python run_search.py` first to generate an archive.")
        sys.exit(1)

    archive_data = load_archive(args.archive)
    print(f"Loaded archive: {len(archive_data['entries'])} cells, "
          f"{archive_data.get('total_evaluated', '?')} total evaluated")

    # Heatmap
    plot_heatmap(archive_data, args.output_dir)

    # GIFs
    if not args.no_gif and archive_data["entries"]:
        print(f"\nGenerating GIFs for top {args.top_n} genomes...")
        generate_gifs(archive_data, args.output_dir, top_n=args.top_n)

    print("\nDone.")


if __name__ == "__main__":
    main()
