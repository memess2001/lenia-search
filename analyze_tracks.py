#!/usr/bin/env python3
"""Compare Track A (MAP-Elites) vs Track B (LLM-guided) results.

Loads the shared archive, splits entries by source_track, and reports
coverage, overlap, and unique discoveries for each track.

Usage:
    python analyze_tracks.py
    python analyze_tracks.py --archive results/archive.json
    python analyze_tracks.py --archive results/archive.json --detailed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_archive(path: str) -> dict:
    """Load archive from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def classify_entry(entry: dict) -> str:
    """Determine which track produced an archive entry.

    Track B entries have source_track="llm" in their genome or metrics.
    Track A entries have kernel_type/growth_type in their genome.
    """
    genome = entry.get("genome", {})
    metrics = entry.get("metrics", {})

    # Check for explicit source_track marker
    if genome.get("source_track") == "llm" or metrics.get("source_track") == "llm":
        return "llm"

    # Track A entries have kernel_type and growth_type
    if "kernel_type" in genome and "growth_type" in genome:
        return "map_elites"

    # Check if there is an expression field (Track B)
    if "expression" in genome:
        return "llm"

    # Default: assume Track A
    return "map_elites"


def analyze(archive: dict, detailed: bool = False) -> dict:
    """Analyse the archive and return a summary dict."""
    entries = archive.get("entries", [])
    n_bins = archive.get("n_bins", [20, 20])
    total_cells = 1
    for n in n_bins:
        total_cells *= n

    # Classify entries by track
    track_a_entries = []
    track_b_entries = []
    track_a_bins = set()
    track_b_bins = set()

    for entry in entries:
        track = classify_entry(entry)
        bin_key = tuple(entry["bin"])

        if track == "map_elites":
            track_a_entries.append(entry)
            track_a_bins.add(bin_key)
        else:
            track_b_entries.append(entry)
            track_b_bins.add(bin_key)

    # Compute overlap and exclusives
    overlap = track_a_bins & track_b_bins
    a_exclusive = track_a_bins - track_b_bins
    b_exclusive = track_b_bins - track_a_bins
    all_bins = track_a_bins | track_b_bins

    # Fitness statistics
    def fitness_stats(entries_list):
        if not entries_list:
            return {"count": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
        fitnesses = [e.get("fitness", 0.0) for e in entries_list]
        return {
            "count": len(fitnesses),
            "mean": sum(fitnesses) / len(fitnesses),
            "max": max(fitnesses),
            "min": min(fitnesses),
        }

    a_stats = fitness_stats(track_a_entries)
    b_stats = fitness_stats(track_b_entries)

    # In overlapping bins, which track has higher fitness?
    a_wins_overlap = 0
    b_wins_overlap = 0
    if overlap:
        # Build lookup for each track's best fitness per bin
        a_best = {}
        for e in track_a_entries:
            bk = tuple(e["bin"])
            if bk in overlap:
                if bk not in a_best or e["fitness"] > a_best[bk]:
                    a_best[bk] = e["fitness"]
        b_best = {}
        for e in track_b_entries:
            bk = tuple(e["bin"])
            if bk in overlap:
                if bk not in b_best or e["fitness"] > b_best[bk]:
                    b_best[bk] = e["fitness"]
        for bk in overlap:
            if a_best.get(bk, 0) >= b_best.get(bk, 0):
                a_wins_overlap += 1
            else:
                b_wins_overlap += 1

    # Alive fraction distribution per track
    def alive_distribution(entries_list):
        alive = [e.get("metrics", {}).get("alive_fraction", 0.0) for e in entries_list]
        if not alive:
            return {"mean": 0.0, "alive_pct": 0.0}
        alive_count = sum(1 for a in alive if a > 0.01)
        return {
            "mean": sum(alive) / len(alive),
            "alive_pct": 100.0 * alive_count / len(alive),
        }

    a_alive = alive_distribution(track_a_entries)
    b_alive = alive_distribution(track_b_entries)

    summary = {
        "total_entries": len(entries),
        "total_bins_filled": len(all_bins),
        "total_possible_bins": total_cells,
        "coverage_pct": 100.0 * len(all_bins) / total_cells,
        "track_a": {
            "entries": len(track_a_entries),
            "bins_filled": len(track_a_bins),
            "exclusive_bins": len(a_exclusive),
            "fitness": a_stats,
            "alive_stats": a_alive,
        },
        "track_b": {
            "entries": len(track_b_entries),
            "bins_filled": len(track_b_bins),
            "exclusive_bins": len(b_exclusive),
            "fitness": b_stats,
            "alive_stats": b_alive,
        },
        "overlap": {
            "bins": len(overlap),
            "a_wins": a_wins_overlap,
            "b_wins": b_wins_overlap,
        },
    }

    return summary


def print_report(summary: dict, detailed: bool = False, archive: dict = None) -> None:
    """Print a formatted analysis report."""
    print()
    print("=" * 65)
    print("  Track A vs Track B: Lenia Life Search Comparison")
    print("=" * 65)

    ta = summary["track_a"]
    tb = summary["track_b"]
    ov = summary["overlap"]

    print(f"\n  Archive: {summary['total_entries']} total entries")
    print(f"  Grid   : {summary['total_bins_filled']}/{summary['total_possible_bins']} "
          f"bins filled ({summary['coverage_pct']:.1f}%)")

    print(f"\n  {'Metric':<30s} {'Track A (MAP-Elites)':>20s} {'Track B (LLM)':>15s}")
    print(f"  {'-' * 65}")
    print(f"  {'Entries':<30s} {ta['entries']:>20d} {tb['entries']:>15d}")
    print(f"  {'Bins filled':<30s} {ta['bins_filled']:>20d} {tb['bins_filled']:>15d}")
    print(f"  {'Exclusive bins':<30s} {ta['exclusive_bins']:>20d} {tb['exclusive_bins']:>15d}")
    print(f"  {'Mean fitness':<30s} {ta['fitness']['mean']:>20.4f} {tb['fitness']['mean']:>15.4f}")
    print(f"  {'Max fitness':<30s} {ta['fitness']['max']:>20.4f} {tb['fitness']['max']:>15.4f}")
    print(f"  {'Mean alive fraction':<30s} {ta['alive_stats']['mean']:>20.4f} {tb['alive_stats']['mean']:>15.4f}")
    print(f"  {'% alive (>0.01)':<30s} {ta['alive_stats']['alive_pct']:>19.1f}% {tb['alive_stats']['alive_pct']:>14.1f}%")

    print(f"\n  Overlapping bins: {ov['bins']}")
    if ov["bins"] > 0:
        print(f"    Track A wins (higher fitness): {ov['a_wins']}")
        print(f"    Track B wins (higher fitness): {ov['b_wins']}")

    print(f"\n  Track B exclusive bins: {tb['exclusive_bins']}")
    print(f"    (These are novel discoveries that MAP-Elites did not find)")

    if detailed and archive:
        print(f"\n  --- Detailed Track B Entries ---")
        entries = archive.get("entries", [])
        b_entries = [e for e in entries if classify_entry(e) in ("llm",)]
        b_entries.sort(key=lambda x: x.get("fitness", 0), reverse=True)

        for i, entry in enumerate(b_entries[:20], 1):
            genome = entry.get("genome", {})
            metrics = entry.get("metrics", {})
            expr = genome.get("expression", "N/A")
            print(f"\n    {i}. bin={entry['bin']}, fitness={entry['fitness']:.4f}")
            print(f"       alive={metrics.get('alive_fraction', 0):.3f}, "
                  f"complexity={metrics.get('complexity', 0):.3f}, "
                  f"clusters={metrics.get('num_clusters', 0)}")
            print(f"       expr: {expr[:100]}{'...' if len(expr) > 100 else ''}")

    print()
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Track A (MAP-Elites) vs Track B (LLM) Lenia search results.",
    )
    parser.add_argument(
        "--archive", type=str, default="results/archive.json",
        help="Path to the shared archive JSON (default: results/archive.json).",
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Show detailed Track B entries with expressions.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output summary as JSON instead of formatted text.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.archive):
        print(f"Error: Archive not found at {args.archive}")
        sys.exit(1)

    archive = load_archive(args.archive)
    summary = analyze(archive, detailed=args.detailed)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_report(summary, detailed=args.detailed, archive=archive)


if __name__ == "__main__":
    main()
