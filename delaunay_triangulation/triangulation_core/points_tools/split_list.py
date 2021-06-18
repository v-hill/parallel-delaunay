"""
This module implements functions for splitting python lists in various ways.
"""

# ------------------------- Functions of points lists -------------------------

def split_in_half(input_points):
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


def groups_of_2(points):
    num = len(points)
    if num % 2 == 0:
        return [points[i:i + 2] for i in range(0, num, 2)]
    elif num % 2 == 1:
        split = [[points[0], points[1], points[3]]]
        split2 = [points[i:i + 2] for i in range(3, num, 2)]
    return split + split2


def groups_of_3(points):
    num = len(points)
    if num % 3 == 0:
        return [points[i:i + 3] for i in range(0, num, 3)]
    elif num % 3 == 1:
        split = [points[i:i + 2] for i in range(0, 4, 2)]
        split2 = [points[i:i + 3] for i in range(4, num, 3)]
    elif num % 3 == 2:
        split = [[points[0], points[1]]]
        split2 = [points[i:i + 3] for i in range(2, num, 3)]
    return split + split2
