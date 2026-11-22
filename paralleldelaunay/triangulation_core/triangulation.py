"""Guibas and Stolfi divide-and-conquer Delaunay triangulation.

The kernel stores the quad-edge as parallel integer arrays instead of edge
objects. Directed edges occupy a contiguous table:

* ``org[e]`` is the origin vertex of directed edge ``e``
* ``onext[e]`` is the next edge counterclockwise around that origin
* ``oprev[e]`` is the previous edge around that origin
* the symmetric edge is ``e ^ 1``

Each undirected edge is a pair. Pair ``k`` occupies indices ``2k`` and
``2k + 1``, which is why the symmetric partner is a bit flip rather than a
stored pointer. ``oprev`` is the third array because splicing an edge into a
vertex ring, and walking the left candidate during a merge, both need the
clockwise neighbour in constant time.

The merge loops are compiled with Numba. One call uses a single edge span.
``triangulate(..., parallel=True)`` hands disjoint spans to Numba ``prange``,
one bottom-up level at a time.
"""

import numpy as np
from numba import njit

# --------------------------------- Predicates --------------------------------


@njit
def _on_right(pts, p1, p2, p3):
    """True when vertex ``p3`` is to the right of directed line ``p1`` -> ``p2``."""
    p1x = pts[p1, 0]
    p1y = pts[p1, 1]
    p2x = pts[p2, 0]
    p2y = pts[p2, 1]
    p3x = pts[p3, 0]
    p3y = pts[p3, 1]
    cross = (p1x - p3x) * (p2y - p3y) - (p1y - p3y) * (p2x - p3x)
    return cross > 0.0


@njit
def _on_left(pts, p1, p2, p3):
    """True when vertex ``p3`` is to the left of directed line ``p1`` -> ``p2``."""
    p1x = pts[p1, 0]
    p1y = pts[p1, 1]
    p2x = pts[p2, 0]
    p2y = pts[p2, 1]
    p3x = pts[p3, 0]
    p3y = pts[p3, 1]
    cross = (p1x - p3x) * (p2y - p3y) - (p1y - p3y) * (p2x - p3x)
    return cross < 0.0


@njit
def _in_circle(pts, a, b, c, d):
    """True when vertex ``d`` is inside the circle through ``a``, ``b``, ``c``."""
    c1 = pts[a, 0] - pts[d, 0]
    c2 = pts[b, 0] - pts[d, 0]
    c3 = pts[c, 0] - pts[d, 0]
    u1 = pts[a, 1] - pts[d, 1]
    u2 = pts[b, 1] - pts[d, 1]
    u3 = pts[c, 1] - pts[d, 1]
    v1 = c1 * c1 + u1 * u1
    v2 = c2 * c2 + u2 * u2
    v3 = c3 * c3 + u3 * u3
    det = (
        c1 * ((u2 * v3) - (v2 * u3))
        - c2 * ((u1 * v3) - (v1 * u3))
        + c3 * ((u1 * v2) - (v1 * u2))
    )
    return det < 0.0


@njit
def _same_point(pts, i, j):
    """True when two vertex indices have identical coordinates."""
    return pts[i, 0] == pts[j, 0] and pts[i, 1] == pts[j, 1]


# ------------------------------- Quad-edge edits -----------------------------


@njit
def _splice(a, b, onext, oprev):
    """Swap the counterclockwise successors of two edges.

    This is the array form of Guibas and Stolfi's ``Splice``. Afterwards,
    ``a`` leads the ring that ``b`` used to lead, and the ``oprev`` entry of
    each successor follows the swap. Predecessors of ``a`` and ``b`` stay put.
    """
    a_next = onext[a]
    b_next = onext[b]
    oprev[a_next] = b
    oprev[b_next] = a
    onext[a] = b_next
    onext[b] = a_next


@njit
def _alloc(
    onext, oprev, free_stack, free_top, span_high, span_limit, span_count
):
    """Return a fresh directed edge, recycling a killed pair when possible.

    ``free_top[0]`` is the number of recycled pairs on ``free_stack``. Fresh
    pairs come from the first span that still has room: ``span_high[s]`` is
    that span's even cursor and ``span_limit[s]`` is its exclusive end. The
    returned index is even; its symmetric partner is ``index ^ 1``.
    """
    top = free_top[0]
    if top > 0:
        top -= 1
        free_top[0] = top
        edge = free_stack[top]
    else:
        edge = -1
        span = 0
        while span < span_count:
            cursor = span_high[span]
            if cursor + 2 <= span_limit[span]:
                edge = cursor
                span_high[span] = cursor + 2
                break
            span += 1
        if edge < 0:
            raise RuntimeError("edge capacity exceeded")
    mirror = edge ^ 1
    onext[edge] = edge
    onext[mirror] = mirror
    oprev[edge] = edge
    oprev[mirror] = mirror
    return edge


@njit
def _kill(
    edge,
    org,
    onext,
    oprev,
    free_stack,
    free_top,
    span_high,
    span_limit,
    span_count,
):
    """Unlink an edge from both endpoints and recycle the pair."""
    _splice(edge, oprev[edge], onext, oprev)
    mirror = edge ^ 1
    _splice(mirror, oprev[mirror], onext, oprev)
    org[edge] = -1
    org[mirror] = -1
    pair = edge ^ 1 if edge & 1 else edge
    free_stack[free_top[0]] = pair
    free_top[0] += 1


@njit
def _connect(
    a,
    b,
    org,
    onext,
    oprev,
    free_stack,
    free_top,
    span_high,
    span_limit,
    span_count,
):
    """Create an edge from the destination of ``a`` to the origin of ``b``."""
    edge = _alloc(
        onext, oprev, free_stack, free_top, span_high, span_limit, span_count
    )
    org[edge] = org[a ^ 1]
    org[edge ^ 1] = org[b]
    _splice(edge, oprev[a ^ 1], onext, oprev)
    _splice(edge ^ 1, b, onext, oprev)
    return edge


# --------------------------------- Primitives --------------------------------


@njit
def _add_segment(
    org,
    onext,
    oprev,
    free_stack,
    free_top,
    span_high,
    span_limit,
    span_count,
    inner,
    outer,
    count,
    a,
    b,
):
    """Append the primitive consisting of one edge and its symmetric."""
    edge = _alloc(
        onext, oprev, free_stack, free_top, span_high, span_limit, span_count
    )
    org[edge] = a
    org[edge ^ 1] = b
    inner[count] = edge
    outer[count] = edge ^ 1
    return count + 1


@njit
def _add_triangle(
    pts,
    org,
    onext,
    oprev,
    free_stack,
    free_top,
    span_high,
    span_limit,
    span_count,
    inner,
    outer,
    count,
    a,
    b,
    c,
):
    """Append a triangle, or an open chain when the three points are collinear."""
    first = _alloc(
        onext, oprev, free_stack, free_top, span_high, span_limit, span_count
    )
    org[first] = a
    org[first ^ 1] = b
    second = _alloc(
        onext, oprev, free_stack, free_top, span_high, span_limit, span_count
    )
    org[second] = b
    org[second ^ 1] = c
    _splice(first ^ 1, second, onext, oprev)

    if _on_right(pts, a, b, c):
        _connect(
            second,
            first,
            org,
            onext,
            oprev,
            free_stack,
            free_top,
            span_high,
            span_limit,
            span_count,
        )
        inner[count] = first
        outer[count] = second ^ 1
    elif _on_left(pts, a, b, c):
        closing = _connect(
            second,
            first,
            org,
            onext,
            oprev,
            free_stack,
            free_top,
            span_high,
            span_limit,
            span_count,
        )
        inner[count] = closing ^ 1
        outer[count] = closing
    else:
        inner[count] = first
        outer[count] = second ^ 1
    return count + 1


# ----------------------------------- Merge -----------------------------------


@njit
def _lowest_common_tangent(pts, org, onext, oprev, left_e, right_e):
    """Walk both hulls to the lower tangent between them.

    ``left_e`` starts as the rightmost edge of the left hull. ``right_e``
    starts as the leftmost edge of the right hull. Each iteration advances
    the endpoint that is not yet on the common lower tangent.
    """
    limit = pts.shape[0] * 4 + 8
    steps = 0
    while True:
        steps += 1
        if steps > limit:
            raise RuntimeError("lowest common tangent did not finish")
        if _on_right(pts, org[left_e], org[left_e ^ 1], org[right_e]):
            left_e = onext[left_e ^ 1]
        elif _on_left(pts, org[right_e], org[right_e ^ 1], org[left_e]):
            right_e = oprev[right_e ^ 1]
        else:
            return left_e, right_e


@njit
def _right_candidate(
    pts,
    org,
    onext,
    oprev,
    free_stack,
    free_top,
    span_high,
    span_limit,
    span_count,
    rcand,
    b1,
    b2,
):
    """Delete right-hull edges that fail the incircle test against the base."""
    limit = pts.shape[0] + 2
    steps = 0
    while True:
        steps += 1
        if steps > limit:
            raise RuntimeError("right candidate did not finish")
        nxt = onext[rcand]
        ccw = _on_right(pts, b1, b2, org[nxt ^ 1])
        inside = _in_circle(pts, b2, b1, org[rcand ^ 1], org[nxt ^ 1])
        if ccw and inside:
            _kill(
                rcand,
                org,
                onext,
                oprev,
                free_stack,
                free_top,
                span_high,
                span_limit,
                span_count,
            )
            rcand = nxt
        else:
            return rcand


@njit
def _left_candidate(
    pts,
    org,
    onext,
    oprev,
    free_stack,
    free_top,
    span_high,
    span_limit,
    span_count,
    lcand,
    b1,
    b2,
):
    """Delete left-hull edges that fail the incircle test against the base."""
    limit = pts.shape[0] + 2
    steps = 0
    while True:
        steps += 1
        if steps > limit:
            raise RuntimeError("left candidate did not finish")
        prev = oprev[lcand]
        ccw = _on_right(pts, b1, b2, org[prev ^ 1])
        inside = _in_circle(pts, b2, b1, org[lcand ^ 1], org[prev ^ 1])
        if ccw and inside:
            _kill(
                lcand,
                org,
                onext,
                oprev,
                free_stack,
                free_top,
                span_high,
                span_limit,
                span_count,
            )
            lcand = prev
        else:
            return lcand


@njit
def _left_candidate_wins(pts, org, rcand, lcand, lcand_valid):
    """True when the left candidate is the next base edge."""
    if not lcand_valid:
        return False
    return _in_circle(
        pts, org[rcand ^ 1], org[rcand], org[lcand], org[lcand ^ 1]
    )


@njit
def _zip(
    pts,
    org,
    onext,
    oprev,
    free_stack,
    free_top,
    span_high,
    span_limit,
    span_count,
    base,
):
    """Fill the channel between two hulls, rising from the lower tangent."""
    limit = pts.shape[0] * 4 + 8
    steps = 0
    while True:
        steps += 1
        if steps > limit:
            raise RuntimeError("hull merge did not finish")
        base_org = org[base]
        base_dest = org[base ^ 1]

        rcand = onext[base ^ 1]
        rcand_valid = _on_right(pts, base_org, base_dest, org[rcand ^ 1])
        lcand = oprev[base]
        lcand_valid = _on_right(pts, base_org, base_dest, org[lcand ^ 1])
        if (not rcand_valid) and (not lcand_valid):
            return

        if rcand_valid:
            rcand = _right_candidate(
                pts,
                org,
                onext,
                oprev,
                free_stack,
                free_top,
                span_high,
                span_limit,
                span_count,
                rcand,
                base_org,
                base_dest,
            )
        if lcand_valid:
            lcand = _left_candidate(
                pts,
                org,
                onext,
                oprev,
                free_stack,
                free_top,
                span_high,
                span_limit,
                span_count,
                lcand,
                base_org,
                base_dest,
            )

        if (not rcand_valid) or _left_candidate_wins(
            pts, org, rcand, lcand, lcand_valid
        ):
            base = _connect(
                lcand,
                base ^ 1,
                org,
                onext,
                oprev,
                free_stack,
                free_top,
                span_high,
                span_limit,
                span_count,
            )
        else:
            base = _connect(
                base ^ 1,
                rcand ^ 1,
                org,
                onext,
                oprev,
                free_stack,
                free_top,
                span_high,
                span_limit,
                span_count,
            )


@njit
def _merge_pair(
    pts,
    org,
    onext,
    oprev,
    free_stack,
    free_top,
    span_high,
    span_limit,
    span_count,
    inner,
    outer,
    left,
    right,
    dest,
):
    """Merge two adjacent partial triangulations into slot ``dest``."""
    ldi, rdi = _lowest_common_tangent(
        pts, org, onext, oprev, outer[left], inner[right]
    )
    ldo = inner[left]
    rdo = outer[right]
    base = _connect(
        ldi ^ 1,
        rdi,
        org,
        onext,
        oprev,
        free_stack,
        free_top,
        span_high,
        span_limit,
        span_count,
    )
    if _same_point(pts, org[ldi], org[ldo]):
        ldo = base
    if _same_point(pts, org[rdi], org[rdo]):
        rdo = base ^ 1
    _zip(
        pts,
        org,
        onext,
        oprev,
        free_stack,
        free_top,
        span_high,
        span_limit,
        span_count,
        base,
    )
    inner[dest] = ldo
    outer[dest] = rdo


@njit
def _fill_range(
    pts,
    lo,
    hi,
    org,
    onext,
    oprev,
    free_stack,
    free_top,
    span_high,
    span_limit,
    span_count,
):
    """Triangulate points ``[lo, hi)`` into the spans already set up.

    Point indices are positions in the full ``pts`` array. New edges are
    taken only from ``free_stack`` and ``span_high`` / ``span_limit``. Returns
    the inner and outer hull edges of the finished subset.
    """
    count_points = hi - lo
    if count_points < 2:
        raise RuntimeError("block has fewer than two points")
    # Same grouping as groups_of_3: never leave a single point on its own.
    if count_points % 3 == 1:
        pair_points = 4
    elif count_points % 3 == 2:
        pair_points = 2
    else:
        pair_points = 0

    inner = np.empty(count_points, np.int64)
    outer = np.empty(count_points, np.int64)
    count = 0
    cursor = lo
    pair_end = lo + pair_points
    while cursor < pair_end:
        count = _add_segment(
            org,
            onext,
            oprev,
            free_stack,
            free_top,
            span_high,
            span_limit,
            span_count,
            inner,
            outer,
            count,
            cursor,
            cursor + 1,
        )
        cursor += 2
    while cursor < hi:
        count = _add_triangle(
            pts,
            org,
            onext,
            oprev,
            free_stack,
            free_top,
            span_high,
            span_limit,
            span_count,
            inner,
            outer,
            count,
            cursor,
            cursor + 1,
            cursor + 2,
        )
        cursor += 3

    # Bottom-up pairing inside this span. A later level can merge the hull
    # this returns with a sibling span, but that sibling is not visible here.
    while count > 1:
        merged = 0
        index = 0
        while index < count:
            if index + 1 < count:
                _merge_pair(
                    pts,
                    org,
                    onext,
                    oprev,
                    free_stack,
                    free_top,
                    span_high,
                    span_limit,
                    span_count,
                    inner,
                    outer,
                    index,
                    index + 1,
                    merged,
                )
                merged += 1
                index += 2
            else:
                inner[merged] = inner[index]
                outer[merged] = outer[index]
                merged += 1
                index += 1
        count = merged
    return inner[0], outer[0]


@njit(cache=True)
def _triangulate(pts):
    """Triangulate lexicographically sorted points.

    Returns ``org``, ``onext``, ``oprev``, and the directed-edge high-water
    mark. Slots with ``org[e] < 0`` are killed edges waiting on the free list.
    """
    n = pts.shape[0]
    # A triangulation has at most 3n - 6 undirected edges. The free list
    # reuses killed pairs, so a small constant times n directed slots is enough.
    cap = 8 * n + 32
    org = np.empty(cap, np.int64)
    onext = np.empty(cap, np.int64)
    oprev = np.empty(cap, np.int64)
    free_stack = np.empty(cap // 2, np.int64)
    free_top = np.zeros(1, np.int64)
    span_high = np.zeros(1, np.int64)
    span_limit = np.empty(1, np.int64)
    span_limit[0] = cap
    _fill_range(
        pts,
        0,
        n,
        org,
        onext,
        oprev,
        free_stack,
        free_top,
        span_high,
        span_limit,
        1,
    )
    return org, onext, oprev, span_high[0]


# -------------------------------- Public type --------------------------------


class Triangulation:
    """Quad-edge arrays for one Delaunay triangulation.

    ``num_edges`` is the directed-edge high-water mark. Indices below that
    are either live or killed. A killed edge has ``org[e] < 0`` and is not
    part of any ring. For a live edge ``e``:

    * ``org[e]`` is the origin vertex
    * ``org[e ^ 1]`` is the destination vertex
    * ``onext[e]`` is the next edge counterclockwise around the origin
    * ``oprev[e]`` is the previous edge around the origin

    Parameters
    ----------
    points : ndarray, shape (n, 2)
        Input coordinates, in the index order used by ``org``.
    org, onext, oprev : ndarray of int64
        Parallel quad-edge arrays.
    num_edges : int
        Number of directed slots in use, including killed holes.
    """

    def __init__(self, points, org, onext, oprev, num_edges):
        self.points = points
        self.org = org
        self.onext = onext
        self.oprev = oprev
        self.num_edges = num_edges


def _as_points(pts_subset):
    """Return a contiguous float64 array of shape ``(n, 2)``."""
    pts = np.asarray(pts_subset, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if pts.shape[0] < 2:
        raise ValueError("need at least two points")
    return np.ascontiguousarray(pts)


def _serial_result(pts):
    """Run the one-core kernel and trim the edge arrays to its high-water mark."""
    org, onext, oprev, num_edges = _triangulate(pts)
    num_edges = int(num_edges)
    return Triangulation(
        pts,
        np.ascontiguousarray(org[:num_edges]),
        np.ascontiguousarray(onext[:num_edges]),
        np.ascontiguousarray(oprev[:num_edges]),
        num_edges,
    )


def triangulate(pts_subset, parallel=True):
    """Triangulate a lexicographically sorted list of 2D points.

    Points must already be sorted by increasing x, then increasing y. The
    divide step consumes them as consecutive pairs and triples, so a point
    out of order is not a different algorithm: it is the wrong set of merges.

    Parameters
    ----------
    pts_subset : array-like, shape (n, 2)
        Sorted coordinates. ``n`` must be at least 2.
    parallel : bool, optional
        When True, the default, Numba ``prange`` runs the independent blocks
        of each merge level together. ``False`` keeps the whole merge on one
        core. A point set too small to split still uses that one-core kernel.

    Returns
    -------
    Triangulation
        Quad-edge arrays for the Delaunay triangulation.
    """
    pts = _as_points(pts_subset)
    if not parallel:
        return _serial_result(pts)
    from paralleldelaunay.triangulation_core.triangulation_parallel import (
        triangulate_prange,
    )

    return triangulate_prange(pts)
