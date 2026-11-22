"""Tests for splitting and generating point sets."""

from paralleldelaunay.triangulation_core.points_tools import (
    generate_values,
    split_list,
)
from paralleldelaunay.utilities.settings import World


def test_split_in_half_keeps_the_extra_point_on_the_left():
    """An odd-length list gives the left half the middle point."""
    left, right = split_list.split_in_half([0, 1, 2, 3, 4])
    assert left == [0, 1, 2], "left half includes the middle point"
    assert right == [3, 4], "right half is the remainder"


def test_split_in_half_even_lengths_are_equal():
    """An even-length list splits into two equal halves."""
    left, right = split_list.split_in_half([0, 1, 2, 3])
    assert left == [0, 1], "first half"
    assert right == [2, 3], "second half"


def test_groups_of_3_cover_every_remainder():
    """Groups are triples, with a leading pair when the count is not exact."""
    assert split_list.groups_of_3([0, 1, 2, 3, 4, 5]) == [
        [0, 1, 2],
        [3, 4, 5],
    ], "exact multiples stay in triples"
    assert split_list.groups_of_3([0, 1, 2, 3]) == [
        [0, 1],
        [2, 3],
    ], "remainder 1 becomes two pairs"
    assert split_list.groups_of_3([0, 1, 2, 3, 4]) == [
        [0, 1],
        [2, 3, 4],
    ], "remainder 2 keeps a leading pair"


def test_random_points_stay_inside_the_world():
    """Random coordinates lie inside the requested bounds."""
    world = World([0, 10, -5, 5])
    points = generate_values.random(20, world)
    assert len(points) == 20, "one coordinate pair per requested point"
    assert all(0 <= point[0] <= 10 for point in points), "x stays in range"
    assert all(-5 <= point[1] <= 5 for point in points), "y stays in range"


def test_lattice_fills_a_square_grid():
    """A square count of points lands on an even grid."""
    world = World([0, 2, 0, 2])
    points = generate_values.lattice(4, world)
    assert points == [[0, 0], [2, 0], [0, 2], [2, 2]], "corners of the world"
