"""
The following code implements the Guibas and Stolfi divide-and-conquer 
algorithm as presented in: 
    L. J. Guibas, J. Stolfi, "Primitives for the manipulation of general 
    subdivisions and the computation of Voronoi diagrams" (1985)
This algorithm computes the Delaunay triangulation of a set of input points.
"""

# ---------------------------------- Imports ----------------------------------

# Repo module imports
import triangulation_core.linear_algebra.linear_algebra as linalg
from triangulation_core.triangulation_primitives import make_primitives

# ------------------------- Functions of points lists -------------------------

def split_points(input_points):
    """
    This function takes in a list of points, splits this list in half and
    return the two new lists containing each subset of the input points.
    
    Parameters
    ----------
    input_points : list
        The set of points to be split

    Returns
    -------
    left : list
        The first half of the input points
    right : list
        The second half of the input points
    """
    mid_val = (len(input_points) + 1) // 2
    
    left = input_points[:mid_val]
    right = input_points[mid_val:]
    return left, right

def points_splitter_2(points):
    num = len(points)
    if num%2==0:
        return [points[i:i+2] for i in range(0, num, 2)]
    elif num%2==1:
        split = [[points[0], points[1], points[3]]]
        split2 = [points[i:i+2] for i in range(3, num, 2)]
    return split + split2

def points_splitter_3(points):
    num = len(points)
    if num%3==0:
        return [points[i:i+3] for i in range(0, num, 3)]
    elif num%3==1:
        split = [points[i:i+2] for i in range(0, 4, 2)]
        split2 = [points[i:i+3] for i in range(4, num, 3)]
    elif num%3==2:
        split = [[points[0], points[1]]]
        split2 = [points[i:i+3] for i in range(2, num, 3)]
    return split + split2
