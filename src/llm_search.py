"""LLM-guided search (Track B) for Lenia artificial life.

This module implements the main search loop that uses an LLM (Claude) to
generate free-form mathematical update rules, evaluates them using the same
metrics as Track A (MAP-Elites), and inserts results into the SAME shared
archive.

Track B entries are marked with source_track="llm" so they can be
distinguished from Track A entries (source_track="map_elites").
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Optional

import torch

from src.evaluator import evaluate
from src.expression_compiler import (
    CompilationError,
    StepTimeoutError,
    compile_expression,
    run_expression,
    compile_dual_expression,
    run_dual_expression,
)
from src.llm_mutator import LLMMutator


# ---------------------------------------------------------------------------
# Archive I/O (compatible with Track A's map_elites.py format)
# ---------------------------------------------------------------------------

def load_or_create_archive(archive_path: str) -> dict:
    """Load an existing archive or create a new empty one.

    The archive format matches Track A's map_elites.py:
    {
        "behavior_dims": ["complexity", "alive_fraction"],
        "n_bins": [20, 20],
        "total_evaluated": int,
        "archive_size": int,
        "entries": [ { "bin": [i, j], "fitness": float, "genome": {...}, "metrics": {...} }, ... ]
    }
    """
    if os.path.exists(archive_path):
        with open(archive_path, "r") as f:
            data = json.load(f)
        return data
    else:
        return {
            "behavior_dims": ["complexity", "alive_fraction"],
            "n_bins": [20, 20],
            "total_evaluated": 0,
            "archive_size": 0,
            "entries": [],
        }


def save_archive(archive: dict, archive_path: str) -> None:
    """Save the archive to disk."""
    os.makedirs(os.path.dirname(archive_path) or ".", exist_ok=True)
    archive["archive_size"] = len(archive["entries"])
    with open(archive_path, "w") as f:
        json.dump(archive, f, indent=2)


def _behavior_to_bin(
    metrics: dict,
    behavior_dims: list[str],
    behavior_ranges: list[tuple[float, float]],
    n_bins: list[int],
) -> tuple[int, ...]:
    """Map metrics to a bin index. Same logic as map_elites.py."""
    indices = []
    for dim, (lo, hi), n in zip(behavior_dims, behavior_ranges, n_bins):
        val = metrics.get(dim, 0.0)
        val = max(lo, min(hi, val))  # clamp
        idx = int((val - lo) / (hi - lo + 1e-12) * n)
        idx = min(idx, n - 1)
        indices.append(idx)
    return tuple(indices)


def _try_insert_into_archive(
    archive: dict,
    genome: dict,
    metrics: dict,
    fitness: float,
) -> bool:
    """Insert into the archive if the bin is empty or new fitness is higher.

    Returns True if inserted.
    """
    behavior_dims = archive.get("behavior_dims", ["complexity", "alive_fraction"])
    n_bins = archive.get("n_bins", [20, 20])
    behavior_ranges = [(0.0, 1.0)] * len(behavior_dims)

    bin_idx = _behavior_to_bin(metrics, behavior_dims, behavior_ranges, n_bins)
    bin_list = list(bin_idx)

    # Check if this bin already has a better entry
    for entry in archive["entries"]:
        if entry["bin"] == bin_list:
            if fitness > entry["fitness"]:
                # Replace with better entry
                entry["genome"] = genome
                entry["metrics"] = {k: _jsonify(v) for k, v in metrics.items()}
                entry["fitness"] = float(fitness)
                return True
            return False

    # New bin - add entry
    archive["entries"].append({
        "bin": bin_list,
        "fitness": float(fitness),
        "genome": {k: _jsonify(v) for k, v in genome.items()},
        "metrics": {k: _jsonify(v) for k, v in metrics.items()},
    })
    return True


def _jsonify(val: Any) -> Any:
    """Ensure a value is JSON-serializable."""
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    if isinstance(val, (list, tuple)):
        return [_jsonify(v) for v in val]
    if isinstance(val, dict):
        return {k: _jsonify(v) for k, v in val.items()}
    return str(val)


# ---------------------------------------------------------------------------
# LLMSearch: Track B main class
# ---------------------------------------------------------------------------

class LLMSearch:
    """LLM-guided search for Lenia update rules (Track B).

    Generates free-form mathematical expressions using Claude, evaluates them
    using the same metrics as Track A, and inserts results into the shared
    archive.

    Parameters
    ----------
    archive_path : str
        Path to the shared archive.json file.
    resolution : int
        Simulation grid resolution (NxN).
    steps_per_eval : int
        Number of simulation steps per evaluation.
    device : str
        PyTorch device ("cpu" or "mps").
    kernel_radius : int
        Convolution kernel radius for compiled expressions.
    """

    def __init__(
        self,
        archive_path: str = "results/archive.json",
        resolution: int = 64,
        steps_per_eval: int = 2000,
        device: str = "cpu",
        kernel_radius: int = 13,
        mode: str = "single",
        fitness_mode: str = "complexity",
        init_mode: str = "random_quarter",
    ):
        """
        Parameters
        ----------
        archive_path : str
            Path to the shared archive.json file.
        resolution : int
            Simulation grid resolution (NxN).
        steps_per_eval : int
            Number of simulation steps per evaluation.
        device : str
            PyTorch device ("cpu" or "mps").
        kernel_radius : int
            Convolution kernel radius for compiled expressions.
        mode : str
            Search mode: "single" for single-field (default), "dual" for dual-field.
        fitness_mode : str
            "complexity" (original) or "localized" (rewards localized creatures).
        init_mode : str
            "random_quarter" (original) or "center_blob" (for localized creature search).
            "mixed" uses 50% random_quarter + 50% center_blob.
        """
        if mode not in ("single", "dual"):
            raise ValueError(f"Invalid mode '{mode}': must be 'single' or 'dual'")

        self.archive_path = os.path.abspath(archive_path)
        self.resolution = resolution
        self.steps_per_eval = steps_per_eval
        self.device = device
        self.kernel_radius = kernel_radius
        self.mode = mode
        self.fitness_mode = fitness_mode
        self.init_mode = init_mode

        self.mutator = LLMMutator()
        self.archive = load_or_create_archive(self.archive_path)

        # Track B statistics
        self.expressions_evaluated = 0
        self.expressions_compiled = 0
        self.expressions_inserted = 0
        self.compilation_failures = 0
        self.runtime_failures = 0

        # Keep a local log of all Track B expressions for the abstraction step
        self._expression_log: list[dict] = []

    def _compute_fitness(self, metrics: dict) -> float:
        """Compute fitness based on fitness_mode."""
        complexity = metrics.get("complexity", 0.0)
        alive = metrics.get("alive_fraction", 0.0)

        if self.fitness_mode == "localized":
            # Reward: complex + partially alive (not full, not empty)
            # Peak at alive=0.3-0.5, penalize alive>0.8 and alive<0.05
            localization = alive * (1.0 - alive) * 4.0  # peaks at 0.5, max=1.0
            return complexity * localization
        else:
            return complexity

    def _get_init_mode(self) -> str:
        """Return init mode for this evaluation, handling 'mixed' mode."""
        if self.init_mode == "mixed":
            import random
            return random.choice(["random_quarter", "center_blob"])
        return self.init_mode

    def evaluate_expression(self, expr_str: str) -> Optional[dict]:
        """Compile an expression, run the simulation, and compute metrics.

        Parameters
        ----------
        expr_str : str
            The mathematical expression to evaluate.

        Returns
        -------
        dict or None
            Evaluation result with keys: expression, metrics, fitness, source_track.
            Returns None if compilation or simulation fails.
        """
        self.expressions_evaluated += 1

        # 1. Compile
        try:
            step_fn = compile_expression(
                expr_str,
                resolution=self.resolution,
                kernel_radius=self.kernel_radius,
            )
            self.expressions_compiled += 1
        except (CompilationError, Exception) as exc:
            self.compilation_failures += 1
            print(f"    [COMPILE FAIL] {str(exc)[:100]}")
            return None

        # 2. Run simulation
        try:
            history = run_expression(
                expr_str,
                resolution=self.resolution,
                steps=self.steps_per_eval,
                record_interval=max(1, self.steps_per_eval // 200),
                kernel_radius=self.kernel_radius,
                device=self.device,
                init_mode=self._get_init_mode(),
            )
        except Exception as exc:
            self.runtime_failures += 1
            print(f"    [RUNTIME FAIL] {str(exc)[:100]}")
            return None

        if not history:
            self.runtime_failures += 1
            return None

        # 3. Evaluate using the SAME evaluator as Track A
        try:
            metrics = evaluate(history)
        except Exception as exc:
            print(f"    [EVAL FAIL] {str(exc)[:100]}")
            return None

        # 4. Build result
        fitness = self._compute_fitness(metrics)

        # Build a "genome" dict that stores the expression + metadata
        genome = {
            "expression": expr_str,
            "source_track": "llm",
            "kernel_radius": self.kernel_radius,
            "resolution": self.resolution,
        }

        # Add source_track to metrics too for easy filtering
        metrics["source_track"] = "llm"

        result = {
            "expression": expr_str,
            "genome": genome,
            "metrics": metrics,
            "fitness": fitness,
        }

        # Log for abstraction
        self._expression_log.append(result)

        return result


    def evaluate_dual_expression(
        self, expr_a: str, expr_b: str
    ) -> Optional[dict]:
        """Compile a dual expression pair, run the simulation, and compute metrics.

        Parameters
        ----------
        expr_a : str
            Update rule for field A (can reference both A and B).
        expr_b : str
            Update rule for field B (can reference both A and B).

        Returns
        -------
        dict or None
            Evaluation result with keys: expression_a, expression_b, metrics,
            fitness, source_track. Returns None if compilation or simulation fails.
        """
        self.expressions_evaluated += 1

        # 1. Compile
        try:
            step_fn = compile_dual_expression(
                expr_a,
                expr_b,
                resolution=self.resolution,
                kernel_radius=self.kernel_radius,
            )
            self.expressions_compiled += 1
        except (CompilationError, Exception) as exc:
            self.compilation_failures += 1
            print(f"    [COMPILE FAIL] {str(exc)[:100]}")
            return None

        # 2. Two-stage evaluation: quick screen then extended run
        # Stage 1: short run (2000 steps) to filter out dead/exploding expressions
        try:
            history_short = run_dual_expression(
                expr_a,
                expr_b,
                resolution=self.resolution,
                steps=2000,
                record_interval=max(1, 2000 // 100),
                kernel_radius=self.kernel_radius,
                device=self.device,
                init_mode=self._get_init_mode(),
            )
        except Exception as exc:
            self.runtime_failures += 1
            print(f"    [RUNTIME FAIL] {str(exc)[:100]}")
            return None

        if not history_short:
            self.runtime_failures += 1
            return None

        # Quick screen: is it dead or trivially full?
        quick_metrics = evaluate(history_short)
        alive = quick_metrics.get("alive_fraction", 0)
        complexity = quick_metrics.get("complexity", 0)

        if alive < 0.01 or (alive > 0.99 and complexity < 0.05):
            # Dead or trivially uniform — skip extended run
            print(f"    [SCREENED OUT] alive={alive:.3f} complexity={complexity:.3f}")
            return None

        # Stage 2: extended run (8000 more steps = 10000 total) for survivors
        try:
            history = run_dual_expression(
                expr_a,
                expr_b,
                resolution=self.resolution,
                steps=self.steps_per_eval,
                record_interval=max(1, self.steps_per_eval // 200),
                kernel_radius=self.kernel_radius,
                device=self.device,
                init_mode=self._get_init_mode(),
            )
        except Exception as exc:
            self.runtime_failures += 1
            print(f"    [RUNTIME FAIL stage 2] {str(exc)[:100]}")
            return None

        if not history:
            self.runtime_failures += 1
            return None

        # 3. Evaluate using the SAME evaluator as Track A
        try:
            metrics = evaluate(history)
        except Exception as exc:
            print(f"    [EVAL FAIL] {str(exc)[:100]}")
            return None

        # 4. Build result
        fitness = self._compute_fitness(metrics)

        genome = {
            "expression_a": expr_a,
            "expression_b": expr_b,
            "expression": f"DUAL(A={expr_a}, B={expr_b})",
            "source_track": "llm_dual",
            "kernel_radius": self.kernel_radius,
            "resolution": self.resolution,
        }

        metrics["source_track"] = "llm_dual"

        result = {
            "expression_a": expr_a,
            "expression_b": expr_b,
            "expression": genome["expression"],
            "genome": genome,
            "metrics": metrics,
            "fitness": fitness,
        }

        self._expression_log.append(result)
        return result

    def _sample_parents(self, n: int = 2) -> list[dict]:
        """Sample high-fitness parents from the archive.

        Uses tournament selection: pick 4 random entries, return the 2 best.
        """
        entries = self.archive.get("entries", [])
        if len(entries) < 2:
            return []

        # Only consider entries that have an expression (Track B) or genome
        candidates = []
        for entry in entries:
            genome = entry.get("genome", {})
            # Track B entries have "expression" in genome
            # Track A entries have kernel_type, growth_type, etc.
            candidates.append(entry)

        if len(candidates) < 2:
            return []

        # Tournament selection
        tournament_size = min(4, len(candidates))
        tournament = random.sample(candidates, tournament_size)
        tournament.sort(key=lambda x: x.get("fitness", 0.0), reverse=True)

        parents = []
        for entry in tournament[:n]:
            # Build a parent dict that the mutator can understand
            genome = entry.get("genome", {})
            parent = {
                "expression": genome.get("expression", self._genome_to_description(genome)),
                "genome": genome,
                "fitness": entry.get("fitness", 0.0),
                "metrics": entry.get("metrics", {}),
            }
            parents.append(parent)

        return parents

    @staticmethod
    def _genome_to_description(genome: dict) -> str:
        """Convert a Track A genome into a rough expression description.

        Track A genomes use predefined kernel/growth families, so we
        translate them into an expression string the LLM can understand.
        """
        kt = genome.get("kernel_type", "gaussian")
        gt = genome.get("growth_type", "gaussian_bell")
        mu = genome.get("growth_mu", 0.15)
        sigma = genome.get("growth_sigma", 0.02)
        dt = genome.get("dt", 0.1)

        # Build a rough equivalent expression
        if gt == "gaussian_bell":
            growth_expr = f"2 * exp(-((conv(A) - {mu:.3f}) / {sigma:.4f})**2 / 2) - 1"
        elif gt == "step_function":
            lo = mu - sigma
            hi = mu + sigma
            growth_expr = f"2 * threshold(conv(A), {lo:.3f}) * (1 - threshold(conv(A), {hi:.3f})) - 1"
        elif gt == "asymmetric_bell":
            growth_expr = f"2 * exp(-((conv(A) - {mu:.3f}) / {sigma:.4f})**2 / 2) - 1"
        elif gt == "bistable":
            mu1 = mu - sigma * 0.5
            mu2 = mu + sigma * 0.5
            growth_expr = f"2 * max(exp(-((conv(A) - {mu1:.3f}) / {sigma * 0.4:.4f})**2 / 2), exp(-((conv(A) - {mu2:.3f}) / {sigma * 0.4:.4f})**2 / 2)) - 1"
        else:
            growth_expr = f"2 * exp(-((conv(A) - {mu:.3f}) / {sigma:.4f})**2 / 2) - 1"

        return f"clip(A + {dt:.2f} * ({growth_expr}), 0, 1)"

    def run(
        self,
        n_iterations: int = 100,
        n_initial: int = 10,
        save_interval: int = 5,
    ) -> None:
        """Run the LLM-guided search loop.

        Parameters
        ----------
        n_iterations : int
            Total number of search iterations.
        n_initial : int
            Number of expressions to generate in the initial batch.
        save_interval : int
            Save archive every N iterations.
        """
        t_start = time.time()

        mode_label = "Dual-Field" if self.mode == "dual" else "Single-Field"
        print("=" * 60)
        print(f"  Track B: LLM-Guided Lenia Life Search ({mode_label})")
        print("=" * 60)
        print(f"  Mode           : {self.mode}")
        print(f"  Resolution     : {self.resolution}x{self.resolution}")
        print(f"  Steps/eval     : {self.steps_per_eval}")
        print(f"  Iterations     : {n_iterations}")
        print(f"  Initial batch  : {n_initial}")
        print(f"  Archive        : {self.archive_path}")
        print(f"  Existing entries: {len(self.archive.get('entries', []))}")
        print("=" * 60)
        print()

        # ------------------------------------------------------------------
        # Phase 1: Generate initial expressions
        # ------------------------------------------------------------------
        if self.mode == "dual":
            print(f"[Phase 1] Generating {n_initial} initial DUAL expression pairs...")
            try:
                initial_pairs = self.mutator.generate_dual_initial(n_initial)
            except Exception as exc:
                print(f"  LLM generation failed: {exc}")
                print("  Falling back to hand-crafted seed dual expressions.")
                initial_pairs = self._seed_dual_expressions()

            print(f"  Got {len(initial_pairs)} dual expression pairs from LLM")
            for i, (expr_a, expr_b) in enumerate(initial_pairs):
                print(f"\n  [{i + 1}/{len(initial_pairs)}] Evaluating dual:")
                print(f"    A: {expr_a[:70]}...")
                print(f"    B: {expr_b[:70]}...")
                result = self.evaluate_dual_expression(expr_a, expr_b)
                if result:
                    inserted = _try_insert_into_archive(
                        self.archive, result["genome"], result["metrics"], result["fitness"]
                    )
                    if inserted:
                        self.expressions_inserted += 1
                        print(f"    -> INSERTED (fitness={result['fitness']:.4f}, "
                              f"alive={result['metrics'].get('alive_fraction', 0):.3f})")
                    else:
                        print(f"    -> Not inserted (fitness={result['fitness']:.4f})")
                else:
                    print(f"    -> Failed")
        else:
            print(f"[Phase 1] Generating {n_initial} initial expressions...")
            try:
                initial_exprs = self.mutator.generate_initial(n_initial)
            except Exception as exc:
                print(f"  LLM generation failed: {exc}")
                print("  Falling back to hand-crafted seed expressions.")
                initial_exprs = self._seed_expressions()

            print(f"  Got {len(initial_exprs)} expressions from LLM")
            for i, expr in enumerate(initial_exprs):
                print(f"\n  [{i + 1}/{len(initial_exprs)}] Evaluating: {expr[:80]}...")
                result = self.evaluate_expression(expr)
                if result:
                    inserted = _try_insert_into_archive(
                        self.archive, result["genome"], result["metrics"], result["fitness"]
                    )
                    if inserted:
                        self.expressions_inserted += 1
                        print(f"    -> INSERTED (fitness={result['fitness']:.4f}, "
                              f"alive={result['metrics'].get('alive_fraction', 0):.3f})")
                    else:
                        print(f"    -> Not inserted (fitness={result['fitness']:.4f})")
                else:
                    print(f"    -> Failed")

        # Reload archive (Track A may have updated it)
        if os.path.exists(self.archive_path):
            self.archive = load_or_create_archive(self.archive_path)

        save_archive(self.archive, self.archive_path)
        self._print_status(t_start)

        # ------------------------------------------------------------------
        # Phase 2: Iterative mutation loop
        # ------------------------------------------------------------------
        print(f"\n[Phase 2] Starting mutation loop ({n_iterations} iterations)...")

        for iteration in range(1, n_iterations + 1):
            print(f"\n--- Iteration {iteration}/{n_iterations} ---")

            # Reload archive periodically (Track A may be running concurrently)
            if iteration % 5 == 0 and os.path.exists(self.archive_path):
                self.archive = load_or_create_archive(self.archive_path)

            if self.mode == "dual":
                # Dual-field mutation
                parents = self._sample_parents(2)
                if len(parents) >= 2:
                    print(f"  Parents: fitness={parents[0]['fitness']:.4f}, "
                          f"{parents[1]['fitness']:.4f}")
                    try:
                        new_expr_a, new_expr_b = self.mutator.mutate_dual(
                            parents[0], parents[1]
                        )
                        print(f"  New A: {new_expr_a[:70]}...")
                        print(f"  New B: {new_expr_b[:70]}...")
                    except Exception as exc:
                        print(f"  LLM dual mutation failed: {exc}")
                        new_expr_a, new_expr_b = None, None
                else:
                    print("  Not enough parents, generating fresh dual expression...")
                    try:
                        pairs = self.mutator.generate_dual_initial(1)
                        if pairs:
                            new_expr_a, new_expr_b = pairs[0]
                        else:
                            new_expr_a, new_expr_b = None, None
                    except Exception as exc:
                        print(f"  LLM dual generation failed: {exc}")
                        new_expr_a, new_expr_b = None, None

                if new_expr_a and new_expr_b:
                    result = self.evaluate_dual_expression(new_expr_a, new_expr_b)
                    if result:
                        inserted = _try_insert_into_archive(
                            self.archive, result["genome"], result["metrics"],
                            result["fitness"]
                        )
                        if inserted:
                            self.expressions_inserted += 1
                            print(f"  -> INSERTED (fitness={result['fitness']:.4f}, "
                                  f"alive={result['metrics'].get('alive_fraction', 0):.3f}, "
                                  f"complexity={result['metrics'].get('complexity', 0):.3f})")
                        else:
                            print(f"  -> Not inserted (fitness={result['fitness']:.4f})")
                    else:
                        print(f"  -> Evaluation failed")
            else:
                # Single-field mutation (original behavior)
                parents = self._sample_parents(2)
                if len(parents) >= 2:
                    print(f"  Parents: fitness={parents[0]['fitness']:.4f}, "
                          f"{parents[1]['fitness']:.4f}")
                    try:
                        new_expr = self.mutator.mutate(parents[0], parents[1])
                        print(f"  New expr: {new_expr[:80]}...")
                    except Exception as exc:
                        print(f"  LLM mutation failed: {exc}")
                        new_expr = None
                else:
                    # Not enough parents; generate a fresh expression
                    print("  Not enough parents, generating fresh expression...")
                    try:
                        exprs = self.mutator.generate_initial(1)
                        new_expr = exprs[0] if exprs else None
                    except Exception as exc:
                        print(f"  LLM generation failed: {exc}")
                        new_expr = None

                if new_expr:
                    result = self.evaluate_expression(new_expr)
                    if result:
                        inserted = _try_insert_into_archive(
                            self.archive, result["genome"], result["metrics"],
                            result["fitness"]
                        )
                        if inserted:
                            self.expressions_inserted += 1
                            print(f"  -> INSERTED (fitness={result['fitness']:.4f}, "
                                  f"alive={result['metrics'].get('alive_fraction', 0):.3f}, "
                                  f"complexity={result['metrics'].get('complexity', 0):.3f})")
                        else:
                            print(f"  -> Not inserted (fitness={result['fitness']:.4f})")
                    else:
                        print(f"  -> Evaluation failed")

            # Every 10 iterations: run abstraction
            if iteration % 10 == 0 and len(self._expression_log) >= 5:
                self._run_abstraction(iteration)

            # Save periodically
            if iteration % save_interval == 0:
                self.archive["total_evaluated"] = self.archive.get("total_evaluated", 0)
                save_archive(self.archive, self.archive_path)
                self._print_status(t_start)

        # ------------------------------------------------------------------
        # Final save
        # ------------------------------------------------------------------
        save_archive(self.archive, self.archive_path)
        elapsed = time.time() - t_start

        print("\n" + "=" * 60)
        print("  Track B Search Complete")
        print("=" * 60)
        self._print_status(t_start)
        stats = self.mutator.get_stats()
        print(f"  API calls      : {stats['api_calls']}")
        print(f"  Input tokens   : {stats['total_input_tokens']:,}")
        print(f"  Output tokens  : {stats['total_output_tokens']:,}")
        print(f"  Patterns found : {stats['discovered_patterns']}")
        print("=" * 60)

    def _run_abstraction(self, iteration: int) -> None:
        """Run the LaSR-style abstraction step."""
        print(f"\n  [Abstraction] Analysing top expressions...")

        # Get top expressions by fitness
        top = sorted(
            self._expression_log,
            key=lambda x: x.get("fitness", 0),
            reverse=True,
        )[:15]

        try:
            new_entries = self.mutator.abstract_concepts(top)
            print(f"  Got {len(new_entries)} new expressions from abstraction")
            for entry in new_entries:
                expr = entry["expression"]
                print(f"    Evaluating: {expr[:80]}...")
                result = self.evaluate_expression(expr)
                if result:
                    inserted = _try_insert_into_archive(
                        self.archive, result["genome"], result["metrics"], result["fitness"]
                    )
                    if inserted:
                        self.expressions_inserted += 1
                        print(f"      -> INSERTED (fitness={result['fitness']:.4f})")
        except Exception as exc:
            print(f"  Abstraction failed: {exc}")

    def _print_status(self, t_start: float) -> None:
        """Print current search statistics."""
        elapsed = time.time() - t_start
        llm_entries = sum(
            1 for e in self.archive.get("entries", [])
            if e.get("genome", {}).get("source_track") in ("llm", "llm_dual")
            or e.get("metrics", {}).get("source_track") in ("llm", "llm_dual")
        )
        dual_entries = sum(
            1 for e in self.archive.get("entries", [])
            if e.get("genome", {}).get("source_track") == "llm_dual"
            or e.get("metrics", {}).get("source_track") == "llm_dual"
        )
        total_entries = len(self.archive.get("entries", []))

        print(f"\n  Status @ {elapsed:.1f}s:")
        print(f"    Evaluated        : {self.expressions_evaluated}")
        print(f"    Compiled OK      : {self.expressions_compiled}")
        print(f"    Inserted (Track B): {self.expressions_inserted}")
        print(f"    Compile failures : {self.compilation_failures}")
        print(f"    Runtime failures : {self.runtime_failures}")
        print(f"    Archive total    : {total_entries} entries")
        print(f"    Track B in archive: {llm_entries} entries")
        if dual_entries > 0:
            print(f"    Dual-field entries : {dual_entries} entries")

    @staticmethod
    def _seed_expressions() -> list[str]:
        """Hand-crafted seed expressions as fallback."""
        return [
            "clip(A + 0.1 * (2 * exp(-((conv(A) - 0.15) / 0.02)**2) - 1), 0, 1)",
            "clip(A + 0.05 * (2 * exp(-((conv(A) - 0.12) / 0.03)**2) - 1), 0, 1)",
            "clip(A + 0.1 * tanh(5 * (conv(A) - 0.2)), 0, 1)",
            "clip(A + 0.02 * laplacian(A) + 0.1 * (2 * exp(-((conv(A) - 0.15) / 0.02)**2) - 1), 0, 1)",
            "clip(A + 0.15 * (exp(-((conv(A) - 0.18) / 0.025)**2) - A), 0, 1)",
            "clip(A * (1 - A) + 0.1 * conv(A) * (1 - conv(A)), 0, 1)",
            "clip(A + 0.05 * (sin(pi * conv(A)) - 0.5 * A), 0, 1)",
            "clip(A + 0.01 * laplacian(A) + 0.08 * (conv(A) * (1 - conv(A)) - 0.2 * A), 0, 1)",
        ]
    @staticmethod
    def _seed_dual_expressions() -> list[tuple[str, str]]:
        """Hand-crafted seed dual-field expression pairs as fallback."""
        return [
            (
                "clip(A + 0.1 * (2 * exp(-((conv(A) - 0.15) / 0.02)**2) - 1) - 0.05 * B, 0, 1)",
                "clip(B + 0.05 * laplacian(B) + 0.1 * A * (1 - B), 0, 1)",
            ),
            (
                "clip(A + 0.1 * (conv(A) * (1 - conv(A)) - 0.1 * A) + 0.02 * B, 0, 1)",
                "clip(B + 0.05 * laplacian(B) + 0.05 * conv(A) - 0.1 * B, 0, 1)",
            ),
            (
                "clip(A + 0.1 * tanh(5 * (conv(A) - 0.2)) - 0.05 * B * A, 0, 1)",
                "clip(B + 0.02 * laplacian(B) + 0.15 * A * A * (1 - B) - 0.05 * B, 0, 1)",
            ),
            (
                "clip(A + 0.05 * (exp(-((conv(A) - 0.15) / 0.03)**2) - A) + 0.03 * (B - A), 0, 1)",
                "clip(B + 0.1 * laplacian(B) + 0.05 * conv(A) * (1 - B) - 0.02 * B, 0, 1)",
            ),
            (
                "clip(A + 0.01 * laplacian(A) + 0.1 * (conv(A) * (1 - conv(A))) - 0.08 * B * A, 0, 1)",
                "clip(B - 0.01 * laplacian(B) + 0.12 * A * (1 - A) * (1 - B) - 0.03 * B, 0, 1)",
            ),
        ]

