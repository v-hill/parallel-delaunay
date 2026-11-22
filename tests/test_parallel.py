"""The default prange triangulation matches the one-core kernel and SciPy."""

import numpy as np
from scipy.spatial import Delaunay

from paralleldelaunay.triangulation_core.triangulation import triangulate
from paralleldelaunay.triangulation_core.triangulation_parallel import (
    triangulate_prange,
)


def _triangles(triangulation):
    """Return interior triangles as sorted vertex-index triples."""
    org = triangulation.org
    onext = triangulation.onext
    triangles = set()
    for edge in range(triangulation.num_edges):
        if org[edge] < 0:
            continue
        second = int(onext[edge ^ 1])
        third = int(onext[second ^ 1])
        if int(onext[third ^ 1]) != edge:
            continue
        vertices = (int(org[edge]), int(org[second]), int(org[third]))
        if len(set(vertices)) == 3 and min(vertices) >= 0:
            triangles.add(tuple(sorted(vertices)))
    return triangles


def test_default_prange_matches_serial_and_scipy():
    """parallel=True is the default and agrees with the one-core kernel."""
    raw = np.random.default_rng(0).random((128, 2))
    order = np.lexsort((raw[:, 1], raw[:, 0]))
    points = raw[order]
    scipy_triangles = {
        tuple(sorted(simplex)) for simplex in Delaunay(points).simplices
    }
    parallel = triangulate(points)
    explicit = triangulate_prange(points, n_blocks=4)
    serial = triangulate(points, parallel=False)
    assert _triangles(parallel) == scipy_triangles, (
        "prange triangles differ from scipy.spatial.Delaunay"
    )
    assert _triangles(explicit) == scipy_triangles, (
        "four-block prange triangles differ from scipy.spatial.Delaunay"
    )
    assert _triangles(serial) == scipy_triangles, (
        "serial triangles differ from scipy.spatial.Delaunay"
    )
