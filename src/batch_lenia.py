"""Batch parallel runner for evaluating multiple Lenia genomes.

Phase 0 uses sequential execution — a batched tensor approach across
different kernels/growth functions is significantly more complex and
will be added in a later phase if throughput becomes the bottleneck.
"""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

import random
import torch

from src.evaluator import evaluate
from src.geometry_factory import create_lenia
from src.expression_compiler import _center_blob_init


class BatchLeniaRunner:
    """Run a list of genomes through the Lenia simulator and evaluator.

    Parameters
    ----------
    genomes : list[dict]
        Each dict is a valid Lenia genome.
    resolution : int
        Override resolution for all runs (default 128).
    steps : int
        Number of simulation steps per genome.
    record_interval : int
        How often to record frames for the state history.
    device : str
        PyTorch device string ("cpu" or "mps").
    """

    def __init__(
        self,
        genomes: list[dict[str, Any]],
        resolution: int = 128,
        steps: int = 3000,
        record_interval: int = 10,
        device: str = "cpu",
        init_mode: str = "random_quarter",
    ) -> None:
        self.genomes = genomes
        self.resolution = resolution
        self.steps = steps
        self.record_interval = record_interval
        self.device = device
        self.init_mode = init_mode

    def run(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Execute all genomes sequentially and return (genome, metrics) pairs."""
        results: list[tuple[dict[str, Any], dict[str, Any]]] = []

        for genome in tqdm(self.genomes, desc="Evaluating genomes", unit="genome"):
            # Inject resolution and device into genome for the simulator
            run_genome = dict(genome)
            run_genome["resolution"] = self.resolution
            run_genome["device"] = self.device

            try:
                sim = create_lenia(run_genome, device=self.device)
                # Override init state if using blob mode
                mode = self.init_mode
                if mode == "mixed":
                    mode = random.choice(["random_quarter", "center_blob"])
                if mode == "center_blob":
                    sim.state = _center_blob_init(
                        self.resolution, blob_radius=0.15, device=self.device
                    )
                history = sim.run(self.steps, record_interval=self.record_interval)
                metrics = evaluate(history)
            except Exception as exc:
                # Record failure gracefully so the search loop can continue
                metrics = {
                    "alive_fraction": 0.0,
                    "mass": 0.0,
                    "complexity": 0.0,
                    "spatial_entropy": 0.0,
                    "num_clusters": 0,
                    "is_dead": True,
                    "longevity": 0,
                    "error": str(exc),
                }

            results.append((genome, metrics))

        return results
