"""Test of the full Delaunay triangulation algorithm."""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

import paralleldelaunay.triangulation_core.points_tools.generate_values as generate_values
from paralleldelaunay.triangulation_core.linear_algebra import (
    lexicographic_sort,
)
from paralleldelaunay.triangulation_core.triangulation import triangulate
from paralleldelaunay.utilities.settings import World

# -----------------------------------------------------------------------------

# Compile the prange kernel before the timed runs. Three points never split.
warmup = np.random.default_rng(0).random((256, 2))
warmup = warmup[np.lexsort((warmup[:, 1], warmup[:, 0]))]
triangulate(warmup)
Delaunay(warmup)

start = time.time()
world_size = [0, 1000, 0, 1000]
world = World(world_size)

num_points = np.linspace(3, 5, 6)
num_points = [int(10**n) for n in num_points]

y1 = []
y2 = []

for num in num_points:
    samples1 = []
    samples2 = []
    for _ in range(5):
        positions = generate_values.random(num, world)
        start = time.perf_counter()
        ordered = lexicographic_sort(positions)
        triangulate(ordered)
        samples1.append(time.perf_counter() - start)

        start = time.perf_counter()
        Delaunay(ordered)
        samples2.append(time.perf_counter() - start)
    samples1.sort()
    samples2.sort()
    elapsed1 = samples1[len(samples1) // 2]
    elapsed2 = samples2[len(samples2) // 2]
    y1.append(elapsed1 * 1000)
    y2.append(elapsed2 * 1000)
    print(f"vhill: {num:<8} points in {elapsed1 * 1000:0.1f} ms")
    print(f"scipy: {num:<8} points in {elapsed2 * 1000:0.1f} ms")
    print(f"    scale factor {elapsed1 / elapsed2:0.1f}x \n")

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

plt.scatter(num_points, y1)
plt.plot(num_points, y1, "--", label="vhill DT")
plt.scatter(num_points, y2)
plt.plot(num_points, y2, "--", label="SciPy DT")

ax.set_yscale("log")
ax.set_xscale("log")
plt.title("Delaunay triangulation benchmarks")
plt.xlabel("Number of points to triangulate")
plt.ylabel("Time (in ms)")
plt.grid()
plt.legend(loc="lower right")
figure_path = Path("images") / "Figure 2026-11-22 benchmark.png"
fig.savefig(figure_path, bbox_inches="tight")
print(f"wrote {figure_path}")
