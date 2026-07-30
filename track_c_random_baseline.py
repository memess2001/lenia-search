#!/usr/bin/env python3
"""Track C: Random expression baseline for comparison with LLM-guided search.

Generates random mathematical expressions (no LLM, pure random grammar)
and evaluates them using the same pipeline as Track B.

Usage:
    python track_c_random_baseline.py --n 1000 --resolution 64 --steps 2000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.expression_compiler import compile_expression
from src.evaluator import evaluate as compute_metrics_from_history


# ---------------------------------------------------------------------------
# Random expression grammar
# ---------------------------------------------------------------------------

# Terminal nodes (leaf values)
TERMINALS = ["conv(A)", "laplacian(A)", "A"]
CONSTANTS = ["0.05", "0.1", "0.12", "0.15", "0.18", "0.2", "0.25", "0.3", "0.5"]

# Unary functions
UNARY_FNS = ["sin", "cos", "tanh", "exp", "abs", "sigmoid"]

# Binary operators
BINARY_OPS = ["+", "-", "*"]


def _rand_const() -> str:
    """Random numeric constant."""
    return random.choice(CONSTANTS)


def _rand_terminal() -> str:
    """Random terminal (field or constant)."""
    if random.random() < 0.7:
        return random.choice(TERMINALS)
    return _rand_const()


def _rand_expr(depth: int = 0, max_depth: int = 4) -> str:
    """Generate a random expression tree."""
    # Base case: return terminal
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        return _rand_terminal()

    choice = random.random()

    if choice < 0.3:
        # Unary function
        fn = random.choice(UNARY_FNS)
        arg = _rand_expr(depth + 1, max_depth)
        return f"{fn}({arg})"

    elif choice < 0.7:
        # Binary operation
        left = _rand_expr(depth + 1, max_depth)
        op = random.choice(BINARY_OPS)
        right = _rand_expr(depth + 1, max_depth)
        return f"({left} {op} {right})"

    elif choice < 0.85:
        # Gaussian bell: exp(-((conv(A) - mu) / sigma)**2)
        mu = random.choice(["0.1", "0.12", "0.15", "0.18", "0.2", "0.25", "0.3"])
        sigma = random.choice(["0.02", "0.03", "0.05", "0.065", "0.08", "0.1"])
        field = random.choice(["conv(A)", "conv(A) + laplacian(A)", "conv(A) - 0.5 * laplacian(A)"])
        return f"exp(-(({field} - {mu}) / {sigma})**2 / 2)"

    else:
        # Threshold band: threshold(x, lo) * (1 - threshold(x, hi))
        field = random.choice(["conv(A)", "conv(A) + laplacian(A)"])
        lo = random.choice(["0.05", "0.08", "0.1", "0.12"])
        hi = random.choice(["0.25", "0.3", "0.35", "0.4"])
        return f"threshold({field}, {lo}) * (1 - threshold({field}, {hi}))"


def generate_random_expression() -> str:
    """Generate a complete random Lenia update expression."""
    # Structure: clip(A + dt * growth_term, 0, 1)
    dt = random.choice(["0.05", "0.08", "0.1", "0.12", "0.15", "0.2", "0.25", "0.3"])

    # Generate 1-3 growth terms
    n_terms = random.randint(1, 3)
    terms = []
    for _ in range(n_terms):
        term = _rand_expr(depth=0, max_depth=random.randint(2, 4))
        # Scale factor
        scale = random.choice(["1", "0.5", "2", "0.1", "0.3"])
        if scale != "1":
            terms.append(f"{scale} * {term}")
        else:
            terms.append(term)

    growth = " + ".join(terms)

    # Wrap: some expressions use (2*growth - 1) centering
    if random.random() < 0.5:
        growth = f"(2 * ({growth}) - 1)"

    return f"clip(A + {dt} * ({growth}), 0, 1)"


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_expression(
    expr: str,
    resolution: int = 64,
    steps: int = 2000,
    kernel_radius: int = 13,
    device: str = "mps",
    init_mode: str = "random_quarter",
) -> dict[str, Any] | None:
    """Compile and evaluate a single expression. Returns metrics or None on failure."""
    try:
        step_fn = compile_expression(expr, resolution=resolution, kernel_radius=kernel_radius)
    except Exception:
        return None

    # Init state
    state = torch.zeros(resolution, resolution, device=device)
    if init_mode == "random_quarter":
        quarter = resolution // 4
        sr = resolution // 2 - quarter // 2
        sc = resolution // 2 - quarter // 2
        state[sr:sr+quarter, sc:sc+quarter] = torch.rand(quarter, quarter, device=device)
    elif init_mode == "center_blob":
        cy, cx = resolution // 2, resolution // 2
        Y, X = torch.meshgrid(torch.arange(resolution, device=device),
                               torch.arange(resolution, device=device), indexing="ij")
        r2 = ((Y - cy).float()**2 + (X - cx).float()**2) / (resolution * 0.15)**2
        state = torch.exp(-r2) * 0.8

    # Run simulation
    record_interval = max(1, steps // 300)
    state_history = [state.cpu()]

    try:
        for step in range(steps):
            state = step_fn(state)
            state = state.clamp(0, 1)
            if (step + 1) % record_interval == 0 or step == steps - 1:
                state_history.append(state.cpu())

            # Early termination: all dead or all same
            if (step + 1) % 500 == 0:
                s = state.cpu().numpy()
                if s.max() < 0.01 or s.std() < 0.001:
                    break
    except Exception:
        return None

    # Compute metrics
    try:
        metrics = compute_metrics_from_history(state_history)
        metrics["expression"] = expr
        return metrics
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Track C: Random expression baseline")
    parser.add_argument("--n", type=int, default=1000, help="Number of random expressions to evaluate")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--kernel_radius", type=int, default=13)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--init", type=str, default="mixed", choices=["random_quarter", "center_blob", "mixed"])
    parser.add_argument("--output", type=str, default="results/track_c_baseline.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = []
    compile_ok = 0
    runtime_ok = 0
    inserted = 0

    # MAP-Elites style archive for fair comparison
    archive: dict[tuple[int, int], dict] = {}
    n_bins = 20

    print(f"Track C Random Baseline: {args.n} expressions")
    print(f"Resolution: {args.resolution}, Steps: {args.steps}, Device: {args.device}")
    print(f"Init: {args.init}")
    print("=" * 60)

    t_start = time.time()

    for i in range(args.n):
        expr = generate_random_expression()

        # Choose init mode
        if args.init == "mixed":
            init_mode = random.choice(["random_quarter", "center_blob"])
        else:
            init_mode = args.init

        metrics = evaluate_expression(
            expr,
            resolution=args.resolution,
            steps=args.steps,
            kernel_radius=args.kernel_radius,
            device=args.device,
            init_mode=init_mode,
        )

        if metrics is None:
            continue

        compile_ok += 1

        # Check if it's "alive" (not trivially dead)
        if metrics.get("is_dead", True):
            runtime_ok += 1
            continue

        runtime_ok += 1

        # Insert into archive
        af = metrics["alive_fraction"]
        cx = metrics["complexity"]
        bin_af = min(int(af * n_bins), n_bins - 1)
        bin_cx = min(int(cx * n_bins), n_bins - 1)
        key = (bin_cx, bin_af)

        fitness = cx  # same as Track A/B

        entry = {
            "expression": expr,
            "fitness": fitness,
            "alive_fraction": af,
            "complexity": cx,
            "spatial_entropy": metrics.get("spatial_entropy", 0),
            "num_clusters": metrics.get("num_clusters", 0),
        }

        existing = archive.get(key)
        if existing is None or fitness > existing["fitness"]:
            archive[key] = entry
            inserted += 1

        # Progress
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(
                f"  [{i+1:4d}/{args.n}] "
                f"compiled={compile_ok} alive={runtime_ok} "
                f"archive={len(archive)}/{n_bins*n_bins} "
                f"({elapsed:.1f}s, {rate:.1f}/s)"
            )

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRACK C RANDOM BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total expressions generated: {args.n}")
    print(f"  Compiled successfully:       {compile_ok} ({100*compile_ok/args.n:.1f}%)")
    print(f"  Runtime OK (not crashed):    {runtime_ok}")
    print(f"  Archive bins filled:         {len(archive)}/{n_bins*n_bins}")
    print(f"  Insertions:                  {inserted}")
    print(f"  Time:                        {elapsed:.1f}s ({args.n/elapsed:.1f} expr/s)")

    # Fitness distribution
    if archive:
        fitnesses = [e["fitness"] for e in archive.values()]
        print(f"\n  Fitness: mean={np.mean(fitnesses):.4f} "
              f"max={max(fitnesses):.4f} min={min(fitnesses):.4f}")

        alive_fracs = [e["alive_fraction"] for e in archive.values()]
        print(f"  Alive:   mean={np.mean(alive_fracs):.4f}")

    # Save results
    output_data = {
        "track": "C",
        "description": "Random expression baseline (no LLM)",
        "n_generated": args.n,
        "n_compiled": compile_ok,
        "n_runtime_ok": runtime_ok,
        "archive_size": len(archive),
        "n_insertions": inserted,
        "time_s": round(elapsed, 2),
        "resolution": args.resolution,
        "steps": args.steps,
        "archive": [
            {"bin": list(k), **v}
            for k, v in sorted(archive.items())
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
