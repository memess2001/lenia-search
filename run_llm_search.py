#!/usr/bin/env python3
"""Entry point for Track B: LLM-guided Lenia life search.

Usage:
    python run_llm_search.py --iterations 50 --archive results/archive.json
    python run_llm_search.py --iterations 100 --initial 15 --resolution 64 --steps 2000

This writes to the SAME archive.json that Track A uses. After running both,
use analyze_tracks.py to compare Track A vs Track B results.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_search import LLMSearch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-guided search (Track B) for Lenia artificial life.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Quick test run
  python run_llm_search.py --iterations 10 --initial 5 --resolution 64

  # Full search writing to shared archive
  python run_llm_search.py --iterations 100 --archive results/archive.json

  # High-resolution with more initial diversity
  python run_llm_search.py --iterations 50 --initial 20 --resolution 128 --steps 3000

  # Dual-field (coupled A/B) search for reaction-diffusion patterns
  python run_llm_search.py --iterations 500 --mode dual --archive results/archive.json
""",
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Number of mutation iterations after initial generation (default: 50).",
    )
    parser.add_argument(
        "--initial", type=int, default=10,
        help="Number of initial expressions to generate (default: 10).",
    )
    parser.add_argument(
        "--resolution", type=int, default=64,
        help="Simulation grid resolution NxN (default: 64).",
    )
    parser.add_argument(
        "--steps", type=int, default=2000,
        help="Simulation steps per evaluation (default: 2000).",
    )
    parser.add_argument(
        "--kernel_radius", type=int, default=13,
        help="Convolution kernel radius (default: 13).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "mps"],
        help="PyTorch device (default: cpu).",
    )
    parser.add_argument(
        "--archive", type=str, default="results/archive.json",
        help="Path to shared archive JSON (default: results/archive.json).",
    )
    parser.add_argument(
        "--save_interval", type=int, default=5,
        help="Save archive every N iterations (default: 5).",
    )
    parser.add_argument(
        "--mode", type=str, default="single",
        choices=["single", "dual"],
        help="Search mode: 'single' for single-field, 'dual' for dual-field "
             "coupled expressions (default: single).",
    )
    parser.add_argument(
        "--fitness", type=str, default="complexity",
        choices=["complexity", "localized"],
        help="Fitness function: 'complexity' (original) or 'localized' "
             "(rewards sparse, localized creatures). Default: complexity.",
    )
    parser.add_argument(
        "--init", type=str, default="random_quarter",
        choices=["random_quarter", "center_blob", "mixed"],
        help="Init mode: 'random_quarter' (original), 'center_blob' (localized), "
             "or 'mixed' (50/50). Default: random_quarter.",
    )
    args = parser.parse_args()

    # Verify ANTHROPIC_API_KEY is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY environment variable is not set.")
        print("The LLM mutator will fail. Set it with:")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        print()
        response = input("Continue anyway (with fallback seed expressions)? [y/N] ")
        if response.lower() != "y":
            sys.exit(1)

    search = LLMSearch(
        archive_path=args.archive,
        resolution=args.resolution,
        steps_per_eval=args.steps,
        device=args.device,
        kernel_radius=args.kernel_radius,
        mode=args.mode,
        fitness_mode=args.fitness,
        init_mode=args.init,
    )

    try:
        search.run(
            n_iterations=args.iterations,
            n_initial=args.initial,
            save_interval=args.save_interval,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving archive...")
        from src.llm_search import save_archive
        save_archive(search.archive, search.archive_path)
        print(f"Archive saved to {search.archive_path}")
        print(f"Evaluated: {search.expressions_evaluated}, "
              f"Inserted: {search.expressions_inserted}")


if __name__ == "__main__":
    main()
