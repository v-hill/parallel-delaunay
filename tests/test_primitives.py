"""Tests for the edge and triangle primitives."""

from paralleldelaunay.triangulation_core.triangulation_primitives import (
    line_primitive,
    make_primitives,
    triangle_primitive,
)


def test_line_primitive_builds_a_symmetric_edge():
    """Two points become one edge and its symmetric edge."""
    line = line_primitive([[0, 0], [1, 0]])
    assert line.num_edges == 2, "a segment stores an edge and its symmetric"
    assert line.edges[0].org == 0, "first edge starts at the left point"
    assert line.edges[0].dest == 1, "first edge ends at the right point"
    assert line.edges[1].org == 1, "symmetric edge reverses the endpoints"
    assert line.edges[0].sym == 1, "symmetric edge is the partner index"
    assert line.inner == 0, "inner edge is the left-most edge"
    assert line.outer == 1, "outer edge is the right-most edge"


def test_triangle_primitive_connects_three_points():
    """Three non-collinear points form a closed triangle."""
    triangle = triangle_primitive([[0, 0], [1, 0], [0, 1]])
    active = [edge for edge in triangle.edges if not edge.deactivate]
    origins = {edge.org for edge in active}
    assert len(active) == 6, "three undirected edges, each with a symmetric"
    assert origins == {0, 1, 2}, "every input point is an edge origin"


def test_triangle_primitive_leaves_collinear_points_open():
    """Collinear points stay a chain instead of a closed triangle."""
    triangle = triangle_primitive([[0, 0], [1, 0], [2, 0]])
    assert triangle.num_edges == 4, "the closing edge is not added"


def test_make_primitives_builds_edges_and_triangles():
    """Pairs become edges and triples become triangles."""
    groups = [
        [[0, 0], [1, 0]],
        [[2, 0], [3, 0], [2.5, 1]],
    ]
    primitives = make_primitives(groups)
    assert len(primitives) == 2, "one primitive per point group"
    assert primitives[0].num_edges == 2, "pair is a single edge"
    assert primitives[1].num_edges == 6, "triple is a triangle"
