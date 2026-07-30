#!/usr/bin/env python3
"""Main entry point for the Lenia MAP-Elites search.

Usage:
    python run_search.py --hours 1 --batch_size 10 --resolution 128 --steps 3000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.map_elites import LeniaMapElites


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MAP-Elites search over Lenia parameter space."
    )
    parser.add_argument(
        "--hours", type=float, default=1.0,
        help="Maximum wall-clock hours to run (default: 1)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Number of genomes per MAP-Elites iteration (default: 10)."
    )
    parser.add_argument(
        "--resolution", type=int, default=128,
        help="Simulation grid resolution NxN (default: 128)."
    )
    parser.add_argument(
        "--steps", type=int, default=3000,
        help="Simulation steps per evaluation (default: 3000)."
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="PyTorch device (default: cpu)."
    )
    parser.add_argument(
        "--archive", type=str, default=None,
        help="Path to existing archive JSON to resume from."
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory for output files (default: results/)."
    )
    parser.add_argument(
        "--fitness", type=str, default="complexity",
        choices=["complexity", "localized"],
        help="Fitness function: 'complexity' (default) or 'localized' (for creature search)."
    )
    parser.add_argument(
        "--init", type=str, default="random_quarter",
        choices=["random_quarter", "center_blob", "mixed"],
        help="Init mode: 'random_quarter' (default), 'center_blob', or 'mixed'."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Project 24: Large-scale Lenia Variant Search")
    print("=" * 60)
    print(f"  Resolution : {args.resolution}x{args.resolution}")
    print(f"  Steps/eval : {args.steps}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Time limit : {args.hours} hours")
    print(f"  Device     : {args.device}")
    print(f"  Results dir: {args.results_dir}")
    print("=" * 60)

    # Initialise MAP-Elites
    me = LeniaMapElites(
        behavior_dims=["complexity", "alive_fraction"],
        behavior_ranges=[(0.0, 1.0), (0.0, 1.0)],
        n_bins=[20, 20],
        fitness_key=args.fitness,
        resolution=args.resolution,
        steps=args.steps,
        device=args.device,
        results_dir=args.results_dir,
    )
    me.init_mode = args.init

    # Resume from existing archive if provided
    if args.archive and os.path.exists(args.archive):
        print(f"Resuming from archive: {args.archive}")
        me.load_archive(args.archive)
        print(f"  Loaded {len(me.archive)} cells, {me.total_evaluated} prior evaluations.")

    # Estimate iterations from time budget
    # Start with a generous estimate; the loop will terminate on time.
    max_seconds = args.hours * 3600
    # Rough estimate: ~2s per genome on 128x128 x 3000 steps on M1 CPU
    est_time_per_iter = args.batch_size * 2.0
    est_iterations = max(10, int(max_seconds / est_time_per_iter))

    print(f"\nEstimated iterations: ~{est_iterations} (will stop at time limit)")
    print()

    t_start = time.time()
    iteration = 0
    try:
        while True:
            elapsed = time.time() - t_start
            if elapsed >= max_seconds:
                print(f"\nTime limit reached ({args.hours}h). Stopping.")
                break

            iteration += 1
            me.run(n_iterations=1, batch_size=args.batch_size, steps_per_eval=args.steps)

            if iteration % 10 == 0:
                remaining = max_seconds - (time.time() - t_start)
                print(f"  Time remaining: {remaining / 60:.1f} min")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving archive...")

    # Final save
    path = me.save_archive()
    print(f"\nSearch complete.")
    print(f"  Total evaluated : {me.total_evaluated}")
    print(f"  Archive coverage: {len(me.archive)} cells")
    print(f"  Saved to         : {path}")


if __name__ == "__main__":
    main()
