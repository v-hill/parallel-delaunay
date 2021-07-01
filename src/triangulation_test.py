"""
This script is a test of the full Delaunay triangulation algorithm.
"""

# Standard library imports
import argparse
import timeit
import time

# Repo module imports
from delaunay_triangulation.utilities.settings import World
from delaunay_triangulation.triangulation_core.linear_algebra import lexigraphic_sort
import delaunay_triangulation.triangulation_core.points_tools.generate_values as generate_values
from delaunay_triangulation.triangulation_core.triangulation import triangulate

# -----------------------------------------------------------------------------

start = time.time()
world_size = [0, 1000, 0, 1000]
world = World(world_size)
num_points = 1000

positions = generate_values.random(num_points, world)
positions = lexigraphic_sort(positions)

start = time.time()
triangulation = triangulate(positions)
elapsed = time.time() - start
print(f"{num_points} {elapsed*1000:0.3f} ms")
