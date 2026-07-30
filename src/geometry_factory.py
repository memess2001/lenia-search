"""Factory for creating Lenia simulators with different geometries.

Usage
-----
    from src.geometry_factory import create_lenia

    genome = {"kernel_type": "gaussian", ..., "geometry": "sphere_S2"}
    sim = create_lenia(genome, device="cpu")
    history = sim.run(1000)
"""

from __future__ import annotations

from typing import Any, Optional


def create_lenia(genome: dict[str, Any], device: Optional[str] = None):
    """Create the appropriate Lenia simulator based on ``genome["geometry"]``.

    Parameters
    ----------
    genome : dict
        Must contain standard Lenia keys.  The ``geometry`` field selects
        the simulator class:
          - ``"flat_plane"`` (default) -> ``Lenia`` from lenia_core
          - ``"sphere_S2"``           -> ``SphericalLenia``
          - ``"torus_embedded"``      -> ``TorusLenia``
          - ``"hyperbolic_H2"``       -> ``HyperbolicLenia``
    device : str, optional
        PyTorch device override.

    Returns
    -------
    A simulator instance with ``.step()`` and ``.run()`` methods.
    """
    geometry = genome.get("geometry", "flat_plane")

    if geometry == "flat_plane":
        from src.lenia_core import Lenia
        return Lenia(genome, device)
    elif geometry == "sphere_S2":
        from src.geometric_lenia import SphericalLenia
        return SphericalLenia(genome, device)
    elif geometry == "torus_embedded":
        from src.geometric_lenia import TorusLenia
        return TorusLenia(genome, device)
    elif geometry == "hyperbolic_H2":
        from src.geometric_lenia import HyperbolicLenia
        return HyperbolicLenia(genome, device)
    else:
        raise ValueError(
            f"Unknown geometry: {geometry!r}. "
            f"Valid options: flat_plane, sphere_S2, torus_embedded, hyperbolic_H2"
        )
