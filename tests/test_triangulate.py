"""End-to-end check of the divide-and-conquer triangulation."""

from scipy.spatial import Delaunay

from paralleldelaunay.triangulation_core.linear_algebra import (
    lexicographic_sort,
)
from paralleldelaunay.triangulation_core.triangulation import triangulate


def _left_next(edges, index):
    """Return the next edge index around this edge's left face."""
    return edges[edges[index].sym].onext


def _triangles(triangulation):
    """Return interior triangles as sorted vertex-index triples."""
    edges = triangulation.edges
    triangles = set()
    for edge in edges:
        if edge.deactivate:
            continue
        second = _left_next(edges, edge.index)
        third = _left_next(edges, second)
        if _left_next(edges, third) != edge.index:
            continue
        vertices = (edge.org, edges[second].org, edges[third].org)
        if len(set(vertices)) == 3:
            triangles.add(tuple(sorted(vertices)))
    return triangles


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
    triangulation = triangulate(points)
    scipy_triangles = {
        tuple(sorted(simplex)) for simplex in Delaunay(points).simplices
    }
    assert _triangles(triangulation) == scipy_triangles, (
        "divide-and-conquer triangles differ from scipy.spatial.Delaunay"
    )
