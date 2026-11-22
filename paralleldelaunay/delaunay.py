"""Delaunay triangulation of a 2D point set.

``Delaunay(points)`` sorts the coordinates, runs the quad-edge kernel, and
reports triangles as indices into the points that were passed in.
"""

import numpy as np

from paralleldelaunay.triangulation_core.triangulation import triangulate


class Delaunay:
    """Delaunay triangulation of points with shape ``(n, 2)``.

    The input does not need to be sorted. ``simplices`` indexes ``points`` in
    the order given by the caller, with each triangle listed counterclockwise.
    ``org``, ``onext``, and ``oprev`` are the quad-edge arrays; ``org[e]`` is
    a vertex of ``points``, and the symmetric edge is ``e ^ 1``.

    Parameters
    ----------
    points : array-like, shape (n, 2)
        Coordinates. ``n`` must be at least 2.
    parallel : bool, optional
        When True, the default, independent merges run together. ``False``
        keeps the merge on one core.
    """

    def __init__(self, points, parallel=True):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must have shape (n, 2)")
        if pts.shape[0] < 2:
            raise ValueError("need at least two points")
        pts = np.ascontiguousarray(pts)
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        triangulation = triangulate(pts[order], parallel=parallel)
        org = np.array(triangulation.org, dtype=np.int64, copy=True)
        live = org >= 0
        org[live] = order[org[live]]
        self.points = pts
        self.org = org
        self.onext = triangulation.onext
        self.oprev = triangulation.oprev
        self.num_edges = triangulation.num_edges
        self._simplices = None

    @property
    def simplices(self):
        """Triangle vertex indices, shape ``(m, 3)``, counterclockwise."""
        if self._simplices is None:
            self._simplices = _simplices(
                self.points, self.org, self.onext, self.num_edges
            )
        return self._simplices


def _simplices(points, org, onext, num_edges):
    """Unique counterclockwise triangles stored in the quad-edge."""
    org = org[:num_edges]
    onext = onext[:num_edges]
    edges = np.arange(num_edges, dtype=np.int64)
    live = org >= 0
    second = np.full(num_edges, -1, dtype=np.int64)
    second[live] = onext[edges[live] ^ 1]
    usable = live & (second >= 0) & (second < num_edges)
    third = np.full(num_edges, -1, dtype=np.int64)
    third[usable] = onext[second[usable] ^ 1]
    face = usable & (third >= 0) & (third < num_edges)
    sym_third = third ^ 1
    face &= (sym_third >= 0) & (sym_third < num_edges)
    returns = np.full(num_edges, -1, dtype=np.int64)
    returns[face] = onext[sym_third[face]]
    face &= returns == edges
    origin = np.full(num_edges, -1, dtype=np.int64)
    middle = np.full(num_edges, -1, dtype=np.int64)
    last = np.full(num_edges, -1, dtype=np.int64)
    origin[face] = org[edges[face]]
    middle[face] = org[second[face]]
    last[face] = org[third[face]]
    keep = (
        face
        & (origin >= 0)
        & (middle >= 0)
        & (last >= 0)
        & (origin != middle)
        & (origin != last)
        & (middle != last)
    )
    if not np.any(keep):
        return np.empty((0, 3), dtype=np.int64)
    vertices = np.column_stack((origin[keep], middle[keep], last[keep]))
    _, index = np.unique(np.sort(vertices, axis=1), axis=0, return_index=True)
    return _counterclockwise(points, vertices[index])


def _counterclockwise(points, vertices):
    """Flip each clockwise triple so its interior lies to the left."""
    first = points[vertices[:, 0]]
    second = points[vertices[:, 1]]
    third = points[vertices[:, 2]]
    area = (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1])
    area -= (second[:, 1] - first[:, 1]) * (third[:, 0] - first[:, 0])
    clockwise = area < 0.0
    if not np.any(clockwise):
        return np.ascontiguousarray(vertices)
    ordered = np.array(vertices, copy=True)
    ordered[clockwise, 1] = vertices[clockwise, 2]
    ordered[clockwise, 2] = vertices[clockwise, 1]
    return ordered
