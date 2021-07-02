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

num_points = 1000
positions = generate_values.random(num_points, world)

start = time.time()
positions = lexigraphic_sort(positions)
triangulation = triangulate(positions)
elapsed1 = time.time() - start
print(f"vhill: {num_points} points in {elapsed1*1000:0.1f} ms")

start = time.time()
tri = Delaunay(positions)
elapsed2 = time.time() - start
print(f"scipy: {num_points} points in {elapsed2*1000:0.1f} ms")
print(f'    scale factor {elapsed1/elapsed2:0.1f}x \n')
