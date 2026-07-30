#!/usr/bin/env python3
"""Re-evaluate all Track A entries in the archive at 8000 steps.

This script:
1. Loads the current archive
2. Re-runs each Track A entry for 8000 steps (instead of the original 2000-3000)
3. Removes entries that die within 8000 steps
4. Saves the cleaned archive

Track B entries are kept as-is (they were already evaluated at 8000 steps in Round 6).
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.geometry_factory import create_lenia
from src.evaluator import evaluate

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x


def reeval_track_a(archive_path: str, steps: int = 8000, output_path: str = None):
    with open(archive_path) as f:
        archive = json.load(f)

    entries = archive["entries"]
    print(f"Loaded archive: {len(entries)} entries")

    track_a = []
    track_b = []
    for e in entries:
        g = e.get("genome", {})
        m = e.get("metrics", {})
        if g.get("source_track") in ("llm", "llm_dual") or m.get("source_track") in ("llm", "llm_dual"):
            track_b.append(e)
        else:
            track_a.append(e)

    print(f"Track A: {len(track_a)} entries to re-evaluate at {steps} steps")
    print(f"Track B: {len(track_b)} entries (kept as-is)")

    survived = []
    died = []

    for entry in tqdm(track_a, desc="Re-evaluating Track A"):
        genome = entry.get("genome", {})
        try:
            sim = create_lenia(genome, device="cpu")
            history = sim.run(steps, record_interval=max(1, steps // 200))

            if not history:
                died.append(entry)
                continue

            metrics = evaluate(history)

            if metrics.get("is_dead", True):
                died.append(entry)
                continue

            # Update entry with new metrics
            entry["metrics"] = metrics
            entry["fitness"] = metrics.get("complexity", 0.0)
            survived.append(entry)

        except Exception as e:
            died.append(entry)

    print(f"\nResults at {steps} steps:")
    print(f"  Survived: {len(survived)}/{len(track_a)} ({100*len(survived)/max(len(track_a),1):.1f}%)")
    print(f"  Died: {len(died)}/{len(track_a)} ({100*len(died)/max(len(track_a),1):.1f}%)")

    # Rebuild archive with survivors + Track B
    all_entries = survived + track_b

    new_archive = {
        "behavior_dims": archive.get("behavior_dims", ["complexity", "alive_fraction"]),
        "n_bins": archive.get("n_bins", 20),
        "total_evaluated": archive.get("total_evaluated", 0),
        "archive_size": len(all_entries),
        "entries": all_entries,
    }

    out = output_path or archive_path.replace(".json", "_8k.json")
    with open(out, "w") as f:
        json.dump(new_archive, f, indent=2)

    print(f"\nSaved: {out}")
    print(f"  Total entries: {len(all_entries)}")
    print(f"  Track A (survived): {len(survived)}")
    print(f"  Track B (unchanged): {len(track_b)}")

    # Rebuild bins
    bins_a = set()
    bins_b = set()
    for e in survived:
        b = e.get("bin")
        if b:
            bins_a.add(tuple(b))
    for e in track_b:
        b = e.get("bin")
        if b:
            bins_b.add(tuple(b))

    total_bins = len(bins_a | bins_b)
    exclusive_b = len(bins_b - bins_a)
    print(f"\n  Bins: {total_bins}/400 ({100*total_bins/400:.1f}%)")
    print(f"  Track A bins: {len(bins_a)}")
    print(f"  Track B exclusive: {exclusive_b}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="results/archive.json")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    reeval_archive_path = args.output or args.archive.replace(".json", "_8k.json")
    reeval_track_a(args.archive, args.steps, reeval_archive_path)
