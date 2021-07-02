"""
This script is a test of the full Delaunay triangulation algorithm.
"""

# Standard library imports
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.spatial import Delaunay

# Repo module imports
from utilities.settings import World
from triangulation_core.linear_algebra import lexigraphic_sort
import triangulation_core.points_tools.generate_values as generate_values
from triangulation_core.triangulation import triangulate

# -----------------------------------------------------------------------------

start = time.time()
world_size = [0, 1000, 0, 1000]
world = World(world_size)

num_points = np.linspace(3, 5, 6)
num_points = [int(10**n) for n in num_points]

y1 = []
y2 = []

for num in num_points:
    positions = generate_values.random(num, world)

    start = time.time()
    positions = lexigraphic_sort(positions)
    triangulation = triangulate(positions)
    elapsed1 = time.time() - start
    y1.append(elapsed1)
    print(f"vhill: {num:<8} points in {elapsed1*1000:0.1f} ms")

    start = time.time()
    tri = Delaunay(positions)
    elapsed2 = time.time() - start
    y2.append(elapsed2)
    print(f"scipy: {num:<8} points in {elapsed2*1000:0.1f} ms")
    print(f'    scale factor {elapsed1/elapsed2:0.1f}x \n')

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

plt.scatter(num_points, y1)
plt.plot(num_points, y1, '--', label='vhill DT')
plt.scatter(num_points, y2)
plt.plot(num_points, y2, '--', label='SciPy DT')

ax.set_yscale('log')
ax.set_xscale('log')
plt.title("Delaunay triangulation benchmarks")
plt.xlabel("Number of points to triangulate")
plt.ylabel("Time (in ms)")
plt.grid()
plt.legend(loc='lower right')
plt.show()
