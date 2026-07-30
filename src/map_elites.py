"""MAP-Elites quality-diversity search over the Lenia parameter space.

The archive is a 2-D grid indexed by behaviour descriptors (default:
complexity x alive_fraction).  Each cell stores the highest-fitness
genome found so far for that behaviour region.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from src.batch_lenia import BatchLeniaRunner
from src.genome import mutate, random_genome


class LeniaMapElites:
    """MAP-Elites search loop for Lenia.

    Parameters
    ----------
    behavior_dims : list[str]
        Names of metrics to use as behaviour descriptors.
    behavior_ranges : list[tuple[float, float]]
        (min, max) range for each behaviour dimension.
    n_bins : list[int]
        Number of bins along each behaviour dimension.
    fitness_key : str
        Which metric to maximise within each cell.
    resolution : int
        Simulation grid resolution.
    steps : int
        Number of simulation steps per evaluation.
    record_interval : int
        Frame recording interval.
    device : str
        PyTorch device.
    results_dir : str
        Directory to save archive snapshots.
    """

    def __init__(
        self,
        behavior_dims: Optional[list[str]] = None,
        behavior_ranges: Optional[list[tuple[float, float]]] = None,
        n_bins: Optional[list[int]] = None,
        fitness_key: str = "complexity",
        resolution: int = 128,
        steps: int = 3000,
        record_interval: int = 10,
        device: str = "cpu",
        results_dir: str = "results",
    ) -> None:
        self.behavior_dims = behavior_dims or ["complexity", "alive_fraction"]
        self.behavior_ranges = behavior_ranges or [(0.0, 1.0), (0.0, 1.0)]
        self.n_bins = n_bins or [20, 20]
        self.fitness_key = fitness_key
        self.resolution = resolution
        self.steps = steps
        self.record_interval = record_interval
        self.device = device
        self.results_dir = results_dir
        self.init_mode = "random_quarter"  # can be overridden

        assert len(self.behavior_dims) == len(self.behavior_ranges) == len(self.n_bins)

        # Archive: dict mapping (bin_i, bin_j, ...) -> {genome, metrics, fitness}
        self.archive: dict[tuple[int, ...], dict[str, Any]] = {}
        self.total_evaluated = 0

    # ----- helpers ---------------------------------------------------------

    def _behavior_to_bin(self, metrics: dict[str, Any]) -> Optional[tuple[int, ...]]:
        """Map metric values to a bin index.  Returns None if out of range."""
        indices: list[int] = []
        for dim, (lo, hi), n in zip(self.behavior_dims, self.behavior_ranges, self.n_bins):
            val = metrics.get(dim, 0.0)
            if val < lo or val > hi:
                # Clamp instead of rejecting to keep more data
                val = max(lo, min(hi, val))
            idx = int((val - lo) / (hi - lo + 1e-12) * n)
            idx = min(idx, n - 1)
            indices.append(idx)
        return tuple(indices)

    def _try_insert(self, genome: dict, metrics: dict) -> bool:
        """Insert into archive if the cell is empty or new fitness is higher."""
        bin_idx = self._behavior_to_bin(metrics)
        if bin_idx is None:
            return False

        if self.fitness_key == "localized":
            c = metrics.get("complexity", 0.0)
            a = metrics.get("alive_fraction", 0.0)
            fitness = c * a * (1.0 - a) * 4.0
        else:
            fitness = metrics.get(self.fitness_key, 0.0)
        existing = self.archive.get(bin_idx)

        if existing is None or fitness > existing["fitness"]:
            self.archive[bin_idx] = {
                "genome": genome,
                "metrics": metrics,
                "fitness": fitness,
            }
            return True
        return False

    def _archive_to_serializable(self) -> list[dict[str, Any]]:
        """Convert archive to a JSON-friendly list."""
        entries = []
        for bin_idx, cell in self.archive.items():
            entries.append({
                "bin": list(bin_idx),
                "fitness": float(cell["fitness"]),
                "genome": {k: _jsonify(v) for k, v in cell["genome"].items()},
                "metrics": {k: _jsonify(v) for k, v in cell["metrics"].items()},
            })
        return entries

    def save_archive(self, path: Optional[str] = None) -> str:
        """Save archive to JSON.  Returns the path written."""
        os.makedirs(self.results_dir, exist_ok=True)
        if path is None:
            path = os.path.join(self.results_dir, "archive.json")
        data = {
            "behavior_dims": self.behavior_dims,
            "n_bins": self.n_bins,
            "total_evaluated": self.total_evaluated,
            "archive_size": len(self.archive),
            "entries": self._archive_to_serializable(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def load_archive(self, path: str) -> None:
        """Load a previously saved archive JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        self.behavior_dims = data["behavior_dims"]
        self.n_bins = data["n_bins"]
        self.total_evaluated = data.get("total_evaluated", 0)
        self.archive = {}
        for entry in data.get("entries", []):
            bin_idx = tuple(entry["bin"])
            self.archive[bin_idx] = {
                "genome": entry["genome"],
                "metrics": entry["metrics"],
                "fitness": entry["fitness"],
            }

    # ----- main loop -------------------------------------------------------

    def run(
        self,
        n_iterations: int = 100,
        batch_size: int = 10,
        steps_per_eval: Optional[int] = None,
    ) -> None:
        """Run the MAP-Elites search.

        Parameters
        ----------
        n_iterations : int
            Number of MAP-Elites iterations (each evaluates *batch_size* genomes).
        batch_size : int
            Genomes to evaluate per iteration.
        steps_per_eval : int, optional
            Override simulation steps for evaluation.
        """
        steps = steps_per_eval or self.steps
        t_start = time.time()

        for iteration in range(1, n_iterations + 1):
            # Generate batch: half random, half mutated from archive
            genomes: list[dict] = []
            # Only mutate Track A entries (those with kernel_type)
            archive_list = [
                cell for cell in self.archive.values()
                if "kernel_type" in cell["genome"]
            ]
            for _ in range(batch_size):
                if archive_list and len(genomes) >= batch_size // 2:
                    # Mutate a random archive member
                    import random
                    parent = random.choice(archive_list)["genome"]
                    genomes.append(mutate(parent))
                else:
                    genomes.append(random_genome())

            # Evaluate
            runner = BatchLeniaRunner(
                genomes,
                resolution=self.resolution,
                steps=steps,
                record_interval=max(1, steps // 300),
                device=self.device,
                init_mode=self.init_mode,
            )
            results = runner.run()
            self.total_evaluated += len(results)

            # Insert into archive
            new_cells = 0
            for genome, metrics in results:
                if self._try_insert(genome, metrics):
                    new_cells += 1

            # Progress report
            if iteration % 10 == 0 or iteration == 1:
                elapsed = time.time() - t_start
                coverage = len(self.archive)
                total_cells = 1
                for n in self.n_bins:
                    total_cells *= n
                pct = 100.0 * coverage / total_cells
                print(
                    f"[iter {iteration:4d}] "
                    f"evaluated={self.total_evaluated}  "
                    f"archive={coverage}/{total_cells} ({pct:.1f}%)  "
                    f"new_cells={new_cells}  "
                    f"elapsed={elapsed:.1f}s"
                )

            # Save after each iteration
            self.save_archive()


def _jsonify(val: Any) -> Any:
    """Ensure a value is JSON-serializable."""
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    if isinstance(val, (list, tuple)):
        return [_jsonify(v) for v in val]
    if isinstance(val, dict):
        return {k: _jsonify(v) for k, v in val.items()}
    # Fallback: convert to string
    return str(val)
