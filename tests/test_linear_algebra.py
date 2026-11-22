"""Tests for vector helpers and the triangulation predicates."""

import pytest

import paralleldelaunay.triangulation_core.linear_algebra as linalg


def test_vector_add_sub_and_perpendicular():
    """Addition, subtraction, and a quarter-turn stay in 2D."""
    assert linalg.vector_add([1, 2], [3, 4]) == [4, 6], "add components"
    assert linalg.vector_sub([3, 4], [1, 2]) == [2, 2], "subtract components"
    assert linalg.perpendicular([1, 2]) == [-2, 1], "rotate a quarter turn"
    assert linalg.list_equal([1, 2], [1, 2]), "identical components match"
    assert not linalg.list_equal([1, 2], [1, 3]), "one mismatch fails"
    assert linalg.list_divide([2, 4], 2) == [1, 2], "divide by a scalar"


def test_normalise_scales_to_requested_length():
    """A non-zero vector is scaled to the requested length."""
    unit = linalg.normalise([3, 4])
    doubled = linalg.normalise([3, 4], length=2)
    assert unit == pytest.approx([0.6, 0.8]), "default length is 1"
    assert doubled == pytest.approx([1.2, 1.6]), "length scales the vector"


def test_normalise_rejects_zero_vector():
    """A zero vector has no direction to scale."""
    with pytest.raises(ValueError, match="zero-length"):
        linalg.normalise([0, 0])


def test_lexicographic_sort_orders_x_then_y():
    """Points are ordered by x, then by y."""
    points = [[1, 0], [0, 2], [0, 1]]
    ordered = linalg.lexicographic_sort(points)
    assert ordered == [[0, 1], [0, 2], [1, 0]], "x is the primary key"


def test_orientation_predicates_follow_ccw_angle():
    """on_right and on_left follow the sign of ccw_angle."""
    start, end, above, below = [0, 0], [1, 0], [0, 1], [0, -1]
    assert linalg.ccw_angle(start, end, above) > 0, "above has positive angle"
    assert linalg.on_right(start, end, above), "positive angle is on_right"
    assert not linalg.on_left(start, end, above), "positive angle is not left"
    assert linalg.ccw_angle(start, end, below) < 0, "below has negative angle"
    assert linalg.on_left(start, end, below), "negative angle is on_left"
    assert linalg.ccw_angle(start, end, end) == 0, "collinear points are zero"


def test_in_circle_for_clockwise_triangle():
    """A clockwise triple reports points inside its circumcircle."""
    first, second, third = [0, 0], [0, 1], [1, 0]
    assert linalg.in_circle(first, second, third, [0.2, 0.2]), "inside"
    assert not linalg.in_circle(first, second, third, [2, 2]), "outside"
