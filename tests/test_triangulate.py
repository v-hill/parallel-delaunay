"""End-to-end check of the divide-and-conquer triangulation."""

import numpy as np
from scipy.spatial import Delaunay

from paralleldelaunay.triangulation_core.linear_algebra import (
    lexicographic_sort,
)
from paralleldelaunay.triangulation_core.triangulation import triangulate


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


def _assert_rings(triangulation):
    """Every live edge's onext and oprev walks return to that edge."""
    org = triangulation.org
    onext = triangulation.onext
    oprev = triangulation.oprev
    for edge in range(triangulation.num_edges):
        if org[edge] < 0:
            continue
        assert int(onext[int(oprev[edge])]) == edge, (
            "oprev is inverse to onext"
        )
        cursor = edge
        for _ in range(triangulation.num_edges + 1):
            cursor = int(onext[cursor])
            if cursor == edge:
                break
        else:
            raise AssertionError("onext walk did not return to the start edge")


def _assert_matches_scipy(points):
    """The kernel and scipy.spatial.Delaunay agree on this point set."""
    triangulation = triangulate(points)
    _assert_rings(triangulation)
    scipy_triangles = {
        tuple(sorted(simplex)) for simplex in Delaunay(points).simplices
    }
    assert _triangles(triangulation) == scipy_triangles, (
        "divide-and-conquer triangles differ from scipy.spatial.Delaunay"
    )


def test_triangulate_matches_scipy():
    """The full routine reproduces SciPy's triangles for one point set."""
    points = lexicographic_sort(
        [
            [0, 0],
            [4, 0],
            [1, 1],
            [3, 1.2],
            [0.2, 3],
            [2, 2.5],
            [5, 3],
            [1.5, 4],
            [4, 4.5],
            [6, 1],
            [2.2, 0.4],
            [3.5, 3.3],
        ]
    )
    _assert_matches_scipy(points)


def test_pair_then_triangle_matches_scipy():
    """A two-point primitive merged with a triangle matches SciPy."""
    points = [[0, 0], [1, 0], [2, 0], [3, 0], [2.5, 1]]
    _assert_matches_scipy(points)


def test_random_points_match_scipy():
    """Several random sets agree with SciPy, including sizes that are not multiples of 3."""
    rng = np.random.default_rng(0)
    for count in (4, 5, 7, 8, 17, 64, 200):
        raw = rng.random((count, 2))
        order = np.lexsort((raw[:, 1], raw[:, 0]))
        _assert_matches_scipy(raw[order])
