#!/usr/bin/env python3
"""Merge archives from all cloud workers into a single archive.

Takes the best (highest fitness) entry for each bin across all workers.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    worker_dirs = sorted(glob.glob("results_cloud/worker_*"))
    if not worker_dirs:
        print("No worker directories found.")
        return

    merged = {}  # bin_tuple -> entry
    total_evaluated = 0

    for wd in worker_dirs:
        archive_path = os.path.join(wd, "archive.json")
        if not os.path.exists(archive_path):
            continue
        with open(archive_path) as f:
            d = json.load(f)
        total_evaluated += d.get("total_evaluated", 0)

        for entry in d["entries"]:
            key = tuple(entry["bin"])
            if key not in merged or entry["fitness"] > merged[key]["fitness"]:
                merged[key] = entry

    # Write merged archive
    output = {
        "behavior_dims": ["complexity", "alive_fraction"],
        "n_bins": [20, 20],
        "total_evaluated": total_evaluated,
        "archive_size": len(merged),
        "entries": list(merged.values()),
    }

    out_path = "results_cloud/archive_merged.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Merged {len(merged)}/400 bins from {len(worker_dirs)} workers")
    print(f"Total evaluations: {total_evaluated}")
    print(f"Saved to: {out_path}")

    # Also copy to main results for continuity
    main_path = "results/archive_cloud.json"
    os.makedirs("results", exist_ok=True)
    with open(main_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Also saved to: {main_path}")


if __name__ == "__main__":
    main()
