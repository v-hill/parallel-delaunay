"""Time the default prange triangulation against the one-core kernel.

The first call pays for Numba compilation. Timed samples are taken after
that warmup. ``parallel=True`` is the default on ``triangulate``.
"""

import time

import numpy as np

from paralleldelaunay.triangulation_core.triangulation import triangulate


def _points(n_points, seed):
    """Return ``n_points`` lexicographically sorted random coordinates."""
    raw = np.random.default_rng(seed).random((n_points, 2))
    order = np.lexsort((raw[:, 1], raw[:, 0]))
    return np.ascontiguousarray(raw[order])


def _median(samples):
    ordered = sorted(samples)
    return ordered[len(ordered) // 2]


def _time(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return _median(samples)


def main():
    """Print one-core and prange medians for a few sizes."""
    warmup = _points(256, 0)
    triangulate(warmup, parallel=False)
    triangulate(warmup)

    print(f"{'n':>8}  {'serial':>10}  {'prange':>10}")
    for n_points in (10_000, 50_000, 200_000):
        pts = _points(n_points, 1)
        serial = _time(lambda pts=pts: triangulate(pts, parallel=False), 5)
        parallel = _time(lambda pts=pts: triangulate(pts), 5)
        print(
            f"{n_points:8d}  {serial * 1000:8.1f} ms  {parallel * 1000:8.1f} ms"
        )


if __name__ == "__main__":
    main()
