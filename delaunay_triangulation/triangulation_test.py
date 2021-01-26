"""
This script is a test of the full Delaunay triangulation algorithm.
"""

# Standard library imports
import argparse
import timeit
import time

# Repo module imports
from utilities.settings import World
from triangulation_core.linear_algebra.linear_algebra import lexigraphic_sort
import triangulation_core.points_tools.generate_values as generate_values
from triangulation_core.triangulation import triangulate
# from triangulation_core.triangulation import points_splitter_3
# from triangulation_core.triangulation import make_primitives
# from triangulation_core.triangulation import recursive_group_merge

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
