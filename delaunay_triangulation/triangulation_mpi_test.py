"""
This script tests the full MPI Delaunay triangulation algorithm.
"""

from mpi4py import MPI

# Code from local files
from utilities.settings import World
import triangulation_core.points_tools.generate_values as generate_values
from triangulation_core.linear_algebra.linear_algebra import lexigraphic_sort
from triangulation_core.triangulation import points_splitter_3
from triangulation_core.triangulation import make_primitives
from triangulation_core.triangulation import recursive_group_merge

num_points = 10000

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
wt_start = MPI.Wtime()

if rank == 0:
    world_size = [0, 1000, 0, 1000]
    world = World(world_size)

    positions = generate_values.random(num_points, world)
    positions = lexigraphic_sort(positions)
    split_pts = points_splitter_3(positions)
    pts_per_core = int(len(split_pts)/size)+1
    data = [split_pts[i:i + pts_per_core] for i in range(0, len(split_pts), pts_per_core)]
else:
    data = None
data = comm.scatter(data, root=0)

primitives = make_primitives(data)
groups = [primitives[i:i+2] for i in range(0, len(primitives), 2)]
triangulation = recursive_group_merge(groups)
print(f"Rank: {rank}, elapsed time: {(MPI.Wtime()-wt_start)*1000:0.3f} ms")

new_groups = comm.gather(triangulation,root=0)

if rank == 0:
    print(f"After recombination, elapsed time: {(MPI.Wtime()-wt_start)*1000:0.3f} ms")
    final_groups = []
    for i in range(0, size, 2):
        group = [new_groups[i][0][0], new_groups[i+1][0][0]]
        final_groups.append(group)

    triangulation = recursive_group_merge(final_groups)
    triangulation = triangulation[0][0]
    wt_end = MPI.Wtime()
    elapsed = wt_end - wt_start
    print(f"Total elapsed time: {elapsed*1000:0.3f} ms")

# Run using the following command format 
# mpiexec -np 4 python .\triangulation_mpi_test.py
