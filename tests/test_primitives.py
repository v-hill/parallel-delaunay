"""Tests for the two- and three-point primitives inside the array kernel."""

import pytest

from paralleldelaunay.triangulation_core.triangulation import triangulate


def test_line_primitive_is_a_symmetric_edge_pair():
    """Two points become one edge stored at indices 0 and 1."""
    line = triangulate([[0, 0], [1, 0]])
    assert line.num_edges == 2, "a segment stores an edge and its symmetric"
    assert int(line.org[0]) == 0, "first edge starts at the left point"
    assert int(line.org[1]) == 1, "symmetric edge starts at the right point"
    assert int(line.org[0 ^ 1]) == 1, "destination is the partner origin"
    assert int(line.onext[0]) == 0, "a lone edge is its own successor"
    assert int(line.onext[1]) == 1, "the symmetric edge is its own successor"


def test_triangle_primitive_connects_three_points():
    """Three non-collinear points form three undirected edges."""
    triangle = triangulate([[0, 0], [1, 0], [0, 1]])
    origins = {int(triangle.org[edge]) for edge in range(triangle.num_edges)}
    assert triangle.num_edges == 6, (
        "three undirected edges, each with a symmetric"
    )
    assert origins == {0, 1, 2}, "every input point is an edge origin"


def test_clockwise_triangle_also_closes():
    """The opposite orientation still adds the closing edge."""
    triangle = triangulate([[0, 0], [1, 0], [0, -1]])
    assert triangle.num_edges == 6, "clockwise points still form a triangle"


def test_triangle_primitive_leaves_collinear_points_open():
    """Collinear points stay a chain instead of a closed triangle."""
    triangle = triangulate([[0, 0], [1, 0], [2, 0]])
    assert triangle.num_edges == 4, "the closing edge is not added"


def test_one_point_is_rejected():
    """A triangulation needs an edge, so a single point is an error."""
    with pytest.raises(ValueError, match="at least two"):
        triangulate([[0, 0]])
