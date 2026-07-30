"""Genome definition and mutation operators for Lenia parameter search.

A genome is a plain dict whose keys correspond to every searchable dimension
of the Lenia parameter space.  We deliberately keep this as a dict (not a
dataclass) so that it serialises to JSON trivially.
"""

from __future__ import annotations

import copy
import random
from typing import Any


# ---------------------------------------------------------------------------
# Searchable parameter space definition
# ---------------------------------------------------------------------------

LENIA_GENOME: dict[str, dict[str, Any]] = {
    "kernel_type": {
        "type": "categorical",
        "choices": [
            # Original 4
            "gaussian", "polynomial", "mexican_hat", "bump",
            # Phase 1b (+11 = 15)
            "cosine", "bessel", "sinc", "gabor", "step_ring",
            "power_law", "yukawa", "elliptical", "spiral",
            "fourier_series", "rbf_mixture",
            # Phase 2 (+13 = 28)
            "lennard_jones", "morse", "riesz", "double_well",
            "hann_ring", "blackman_harris", "chirp", "dog",
            "wendland_c2", "matern", "log_gaussian", "plateau", "sech",
        ],
    },
    "kernel_radius": {
        "type": "float",
        "low": 5.0,
        "high": 30.0,
    },
    "kernel_peaks": {
        "type": "float",
        "low": 0.5,
        "high": 4.0,
    },
    "growth_type": {
        "type": "categorical",
        "choices": [
            # Original 4
            "gaussian_bell", "step_function", "asymmetric_bell", "bistable",
            # Phase 1b (+4 = 8)
            "laplace_peak", "cauchy_peak", "sigmoid_pair", "relu_like",
            # Phase 2 (+8 = 16)
            "ricker", "allee", "gompertz",
            "fitzhugh_nagumo", "brusselator", "swift_hohenberg",
            "sinc_growth", "witch_growth",
        ],
    },
    "growth_mu": {
        "type": "float",
        "low": 0.05,
        "high": 0.50,
    },
    "growth_sigma": {
        "type": "float",
        "low": 0.005,
        "high": 0.15,
    },
    "dt": {
        "type": "float",
        "low": 0.01,
        "high": 0.5,
    },
    "geometry": {
        "type": "categorical",
        "choices": ["flat_plane", "sphere_S2", "torus_embedded", "hyperbolic_H2"],
    },
    "num_channels": {
        "type": "categorical",
        "choices": [1],  # multi-channel in later phases
    },
}


# ---------------------------------------------------------------------------
# Genome creation
# ---------------------------------------------------------------------------

def random_genome() -> dict[str, Any]:
    """Sample a genome uniformly at random from the parameter space."""
    genome: dict[str, Any] = {}
    for key, spec in LENIA_GENOME.items():
        if spec["type"] == "categorical":
            genome[key] = random.choice(spec["choices"])
        elif spec["type"] == "float":
            genome[key] = random.uniform(spec["low"], spec["high"])
        else:
            raise ValueError(f"Unknown parameter type: {spec['type']}")
    return genome


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def _mutate_field(genome: dict[str, Any], key: str) -> None:
    """Mutate a single field in-place."""
    spec = LENIA_GENOME[key]
    if spec["type"] == "categorical":
        choices = spec["choices"]
        if len(choices) > 1:
            current = genome[key]
            others = [c for c in choices if c != current]
            genome[key] = random.choice(others)
        # If only one choice, nothing to mutate.
    elif spec["type"] == "float":
        lo, hi = spec["low"], spec["high"]
        span = hi - lo
        # Gaussian perturbation (10-30% of range)
        sigma = span * random.uniform(0.10, 0.30)
        new_val = genome[key] + random.gauss(0, sigma)
        genome[key] = max(lo, min(hi, new_val))


def mutate(genome: dict[str, Any], n_mutations: int | None = None) -> dict[str, Any]:
    """Return a mutated *copy* of *genome*, changing 1-2 fields at random.

    Parameters
    ----------
    genome : dict
        Parent genome (not modified).
    n_mutations : int, optional
        Exact number of fields to mutate.  If *None*, randomly picks 1 or 2.
    """
    child = copy.deepcopy(genome)
    mutable_keys = [
        k for k, spec in LENIA_GENOME.items()
        if not (spec["type"] == "categorical" and len(spec["choices"]) <= 1)
    ]
    if not mutable_keys:
        return child

    if n_mutations is None:
        n_mutations = random.choice([1, 2])
    n_mutations = min(n_mutations, len(mutable_keys))

    keys_to_mutate = random.sample(mutable_keys, n_mutations)
    for key in keys_to_mutate:
        _mutate_field(child, key)
    return child
