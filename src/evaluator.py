"""Unified evaluation metrics for Lenia state histories.

All metrics are computed on CPU NumPy arrays for simplicity and
compatibility with scipy.  Torch tensors in the history list are
converted automatically.
"""

from __future__ import annotations

import lzma
import math
from typing import Any

import numpy as np
import torch
from scipy import ndimage


# ---------------------------------------------------------------------------
# Individual metric helpers
# ---------------------------------------------------------------------------

def _to_numpy(t: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _alive_fraction(state: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of cells with value > threshold."""
    return float((state > threshold).sum()) / state.size


def _mass(state: np.ndarray) -> float:
    """Total cell values (sum)."""
    return float(state.sum())


def _complexity_lzma(state: np.ndarray) -> float:
    """Compression ratio as a proxy for spatial complexity.

    Lower ratio = more compressible = simpler pattern.
    Higher ratio = less compressible = more complex.
    """
    raw_bytes = (state * 255).astype(np.uint8).tobytes()
    compressed = lzma.compress(raw_bytes, preset=1)
    if len(raw_bytes) == 0:
        return 0.0
    return len(compressed) / len(raw_bytes)


def _spatial_entropy(state: np.ndarray, n_blocks: int = 8) -> float:
    """Shannon entropy of block-wise mean intensities.

    Divides the state into an n_blocks x n_blocks grid and computes entropy
    over the distribution of mean values (discretised into 64 bins).
    """
    h, w = state.shape
    bh, bw = h // n_blocks, w // n_blocks
    if bh == 0 or bw == 0:
        return 0.0

    means = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            block = state[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
            means.append(block.mean())
    means_arr = np.array(means)

    # Discretise into bins and compute entropy
    n_bins = 64
    counts, _ = np.histogram(means_arr, bins=n_bins, range=(0.0, 1.0))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -float((probs * np.log2(probs)).sum())
    return entropy


def _num_clusters(state: np.ndarray, threshold: float = 0.1) -> int:
    """Number of connected components above *threshold*."""
    binary = (state > threshold).astype(np.int32)
    labelled, n_features = ndimage.label(binary)
    return int(n_features)


# ---------------------------------------------------------------------------
# Public evaluation interface
# ---------------------------------------------------------------------------

def evaluate(state_history: list[torch.Tensor | np.ndarray]) -> dict[str, Any]:
    """Compute all metrics from a state history.

    Parameters
    ----------
    state_history : list
        List of 2-D arrays/tensors recorded during a Lenia run.

    Returns
    -------
    dict with keys:
        alive_fraction, mass, complexity, spatial_entropy, num_clusters,
        is_dead, longevity
    """
    if not state_history:
        return {
            "alive_fraction": 0.0,
            "mass": 0.0,
            "complexity": 0.0,
            "spatial_entropy": 0.0,
            "num_clusters": 0,
            "is_dead": True,
            "longevity": 0,
        }

    np_history = [_to_numpy(s) for s in state_history]
    final = np_history[-1]

    alive_fraction = _alive_fraction(final)
    mass = _mass(final)
    complexity = _complexity_lzma(final)
    spatial_entropy = _spatial_entropy(final)
    num_clusters = _num_clusters(final)

    # Death detection: alive_fraction < 0.001 for the last 100 recorded frames
    tail_length = min(100, len(np_history))
    tail_alive = [_alive_fraction(s) for s in np_history[-tail_length:]]
    is_dead = all(a < 0.001 for a in tail_alive)

    # Longevity: last frame index where alive_fraction >= 0.001
    longevity = len(np_history)  # default: survived entire run
    if is_dead:
        for idx, s in enumerate(np_history):
            if _alive_fraction(s) < 0.001:
                longevity = idx
                break

    return {
        "alive_fraction": float(alive_fraction),
        "mass": float(mass),
        "complexity": float(complexity),
        "spatial_entropy": float(spatial_entropy),
        "num_clusters": int(num_clusters),
        "is_dead": bool(is_dead),
        "longevity": int(longevity),
    }
