"""The package-level Delaunay constructor accepts unsorted points."""

import numpy as np
import pytest
from scipy.spatial import Delaunay as ScipyDelaunay

from paralleldelaunay import Delaunay


def _keys(simplices):
    return {tuple(sorted(int(v) for v in row)) for row in simplices}


def test_delaunay_is_imported_from_the_package():
    """Callers do not need the triangulation_core modules."""
    assert Delaunay.__module__ == "paralleldelaunay.delaunay"


def test_unsorted_points_match_scipy():
    """Simplex indices refer to the caller's point order, not the sorted order."""
    raw = np.random.default_rng(1).random((80, 2))
    raw = raw[np.random.default_rng(2).permutation(raw.shape[0])]
    triangulation = Delaunay(raw)
    reference = ScipyDelaunay(raw)
    assert _keys(triangulation.simplices) == _keys(reference.simplices)
    assert triangulation.points.shape == raw.shape


def test_one_core_path_matches_scipy():
    """parallel=False is the same triangulation."""
    raw = np.random.default_rng(3).random((40, 2))
    triangulation = Delaunay(raw, parallel=False)
    reference = ScipyDelaunay(triangulation.points)
    assert _keys(triangulation.simplices) == _keys(reference.simplices)


def test_simplices_are_counterclockwise():
    """Each stored triple has positive area."""
    raw = np.random.default_rng(4).random((30, 2))
    triangulation = Delaunay(raw)
    points = triangulation.points
    simplices = triangulation.simplices
    first = points[simplices[:, 0]]
    second = points[simplices[:, 1]]
    third = points[simplices[:, 2]]
    area = (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1])
    area -= (second[:, 1] - first[:, 1]) * (third[:, 0] - first[:, 0])
    assert np.all(area > 0.0)


def test_two_points_have_no_triangles():
    """An edge is a triangulation, but it contains no simplex."""
    triangulation = Delaunay([[0.0, 0.0], [1.0, 0.0]])
    assert triangulation.simplices.shape == (0, 3)
    assert int(triangulation.org[0]) == 0
    assert int(triangulation.org[1]) == 1


def test_one_point_is_rejected():
    """A triangulation needs an edge."""
    with pytest.raises(ValueError, match="at least two"):
        Delaunay([[0.0, 0.0]])
