"""Divide-and-conquer triangulation scheduled with Numba ``prange``.

The sorted points are cut into ``2^k`` contiguous blocks. Each block owns a
private span of the quad-edge arrays, sized from the ``3m - 6`` planar bound
with the same slack the serial kernel uses. A merge appends the right block's
free list and spans onto the left block, then allocates only from that private
memory. Sibling merges at one level never share a cursor.

``prange`` runs every leaf at once, then one pass per tree level. It does not
parallelise a recursive call. The tree is a bottom-up loop, and each iteration
of a level is independent.
"""

import os

import numpy as np
from numba import njit, prange

from paralleldelaunay.triangulation_core.triangulation import (
    Triangulation,
    _as_points,
    _fill_range,
    _merge_pair,
    _serial_result,
)

# int64 slots of padding so neighbouring free-list counters do not share a
# cache line. Eight int64s is 64 bytes.
_PAD = 8


def choose_blocks(n_points, requested):
    """Return a power-of-two block count whose blocks each hold two points."""
    if requested < 1:
        requested = os.cpu_count() or 1
    blocks = 1
    while blocks * 2 <= requested:
        blocks *= 2
    while blocks > 1 and n_points // blocks < 2:
        blocks //= 2
    return blocks


def _ranges(n_points, n_blocks):
    """Split ``n_points`` into ``n_blocks`` contiguous, nearly equal ranges."""
    lo = np.empty(n_blocks, np.int64)
    hi = np.empty(n_blocks, np.int64)
    base_size = n_points // n_blocks
    remainder = n_points % n_blocks
    cursor = 0
    for block in range(n_blocks):
        size = base_size + (1 if block < remainder else 0)
        lo[block] = cursor
        cursor += size
        hi[block] = cursor
    return lo, hi


def _arena(pts, n_blocks):
    """Allocate disjoint edge spans, one per block, plus per-block free lists."""
    n_blocks = choose_blocks(pts.shape[0], n_blocks)
    lo, hi = _ranges(pts.shape[0], n_blocks)
    edge_base = np.empty(n_blocks, np.int64)
    edge_limit = np.empty(n_blocks, np.int64)
    cursor = 0
    for block in range(n_blocks):
        cap = 8 * int(hi[block] - lo[block]) + 32
        edge_base[block] = cursor
        cursor += cap
        edge_limit[block] = cursor
    org = np.full(cursor, -1, np.int64)
    onext = np.empty(cursor, np.int64)
    oprev = np.empty(cursor, np.int64)
    free_rows = np.empty((n_blocks, cursor // 2 + 2), np.int64)
    # Column 0 is the counter. The rest of the row keeps it off a shared line.
    free_top_pad = np.zeros((n_blocks, _PAD), np.int64)
    span_high = np.zeros((n_blocks, n_blocks), np.int64)
    span_limit = np.zeros((n_blocks, n_blocks), np.int64)
    span_counts = np.zeros(n_blocks, np.int64)
    hull_in = np.zeros(n_blocks, np.int64)
    hull_out = np.zeros(n_blocks, np.int64)
    for block in range(n_blocks):
        span_high[block, 0] = edge_base[block]
        span_limit[block, 0] = edge_limit[block]
    return (
        n_blocks,
        lo,
        hi,
        org,
        onext,
        oprev,
        free_rows,
        free_top_pad,
        span_high,
        span_limit,
        span_counts,
        hull_in,
        hull_out,
    )


@njit
def _absorb(
    left, right, free_rows, free_top_pad, span_high, span_limit, span_counts
):
    """Give ``left`` the free slots and the unused tails of both sides.

    The result stays in ``left``. ``right`` is not used again at this level,
    and no other merge at this level has ``left`` as an index.
    """
    left_top = free_top_pad[left, 0]
    right_top = free_top_pad[right, 0]
    index = 0
    while index < right_top:
        free_rows[left, left_top + index] = free_rows[right, index]
        index += 1
    free_top_pad[left, 0] = left_top + right_top
    left_spans = span_counts[left]
    right_spans = span_counts[right]
    index = 0
    while index < right_spans:
        span_high[left, left_spans + index] = span_high[right, index]
        span_limit[left, left_spans + index] = span_limit[right, index]
        index += 1
    span_counts[left] = left_spans + right_spans


@njit
def _build_leaf(
    block,
    pts,
    lo,
    hi,
    org,
    onext,
    oprev,
    free_rows,
    free_top_pad,
    span_high,
    span_limit,
    span_counts,
    hull_in,
    hull_out,
):
    """Triangulate one block into its own span."""
    free_top_pad[block, 0] = 0
    span_counts[block] = 1
    inner, outer = _fill_range(
        pts,
        lo[block],
        hi[block],
        org,
        onext,
        oprev,
        free_rows[block],
        free_top_pad[block, :1],
        span_high[block],
        span_limit[block],
        1,
    )
    hull_in[block] = inner
    hull_out[block] = outer


@njit
def _merge_level(
    pair,
    width,
    pts,
    org,
    onext,
    oprev,
    free_rows,
    free_top_pad,
    span_high,
    span_limit,
    span_counts,
    hull_in,
    hull_out,
):
    """Merge one pair of components ``width`` slots apart."""
    left = pair * (width * 2)
    right = left + width
    _absorb(
        left,
        right,
        free_rows,
        free_top_pad,
        span_high,
        span_limit,
        span_counts,
    )
    edges_in = np.empty(2, np.int64)
    edges_out = np.empty(2, np.int64)
    edges_in[0] = hull_in[left]
    edges_out[0] = hull_out[left]
    edges_in[1] = hull_in[right]
    edges_out[1] = hull_out[right]
    _merge_pair(
        pts,
        org,
        onext,
        oprev,
        free_rows[left],
        free_top_pad[left, :1],
        span_high[left],
        span_limit[left],
        span_counts[left],
        edges_in,
        edges_out,
        0,
        1,
        0,
    )
    hull_in[left] = edges_in[0]
    hull_out[left] = edges_out[0]


@njit(parallel=True, cache=True)
def _build_prange(
    pts,
    lo,
    hi,
    org,
    onext,
    oprev,
    free_rows,
    free_top_pad,
    span_high,
    span_limit,
    span_counts,
    hull_in,
    hull_out,
):
    """Build every leaf at once, then one parallel pass per tree level."""
    n_blocks = lo.shape[0]
    for block in prange(n_blocks):
        _build_leaf(
            block,
            pts,
            lo,
            hi,
            org,
            onext,
            oprev,
            free_rows,
            free_top_pad,
            span_high,
            span_limit,
            span_counts,
            hull_in,
            hull_out,
        )
    width = 1
    while width < n_blocks:
        pairs = n_blocks // (width * 2)
        for pair in prange(pairs):
            _merge_level(
                pair,
                width,
                pts,
                org,
                onext,
                oprev,
                free_rows,
                free_top_pad,
                span_high,
                span_limit,
                span_counts,
                hull_in,
                hull_out,
            )
        width *= 2


def _finish(pts, org, onext, oprev):
    """Wrap a finished arena. Unused slots keep ``org[e] < 0``."""
    return Triangulation(pts, org, onext, oprev, int(org.shape[0]))


def triangulate_prange(pts_subset, n_blocks=0):
    """Triangulate with ``prange`` over the leaves and over each merge level.

    A single block, which is what a tiny point set reduces to, uses the serial
    kernel. Every larger split keeps each block on its own edge span.

    Parameters
    ----------
    pts_subset : array-like, shape (n, 2)
        Lexicographically sorted coordinates.
    n_blocks : int, optional
        Requested block count. The value actually used is the greatest power
        of two that still gives every block two points. ``0`` uses the
        machine's CPU count.
    """
    pts = _as_points(pts_subset)
    (
        blocks,
        lo,
        hi,
        org,
        onext,
        oprev,
        free_rows,
        free_top_pad,
        span_high,
        span_limit,
        span_counts,
        hull_in,
        hull_out,
    ) = _arena(pts, n_blocks)
    if blocks == 1:
        return _serial_result(pts)
    _build_prange(
        pts,
        lo,
        hi,
        org,
        onext,
        oprev,
        free_rows,
        free_top_pad,
        span_high,
        span_limit,
        span_counts,
        hull_in,
        hull_out,
    )
    return _finish(pts, org, onext, oprev)
