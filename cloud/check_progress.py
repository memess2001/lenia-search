#!/usr/bin/env python3
"""Check progress of all cloud workers."""
import glob
import json
import os
from collections import Counter


def main():
    worker_dirs = sorted(glob.glob("results_cloud/worker_*"))
    if not worker_dirs:
        print("No worker directories found.")
        return

    total_evaluated = 0
    all_entries = []

    for wd in worker_dirs:
        archive_path = os.path.join(wd, "archive.json")
        if os.path.exists(archive_path):
            with open(archive_path) as f:
                d = json.load(f)
            n = d["archive_size"]
            ev = d["total_evaluated"]
            total_evaluated += ev
            all_entries.extend(d["entries"])
            print(f"  {wd}: {n}/400 bins, {ev} evaluated")
        else:
            print(f"  {wd}: no archive yet")

    # Merge to count unique bins
    bins_seen = {}
    for entry in all_entries:
        key = tuple(entry["bin"])
        if key not in bins_seen or entry["fitness"] > bins_seen[key]["fitness"]:
            bins_seen[key] = entry

    print(f"\n  Total evaluated across all workers: {total_evaluated}")
    print(f"  Unique bins (merged): {len(bins_seen)}/400 ({len(bins_seen)/4:.1f}%)")

    # Kernel / growth coverage (skip Track B entries without kernel_type)
    track_a = [e for e in bins_seen.values() if "kernel_type" in e["genome"]]
    kern = Counter(e["genome"]["kernel_type"] for e in track_a)
    grow = Counter(e["genome"]["growth_type"] for e in track_a)
    print(f"\n  Track A entries: {len(track_a)}, Track B: {len(bins_seen) - len(track_a)}")
    print(f"  Kernel types represented: {len(kern)}/28")
    print(f"  Growth types represented: {len(grow)}/16")


if __name__ == "__main__":
    main()
