"""Functions for splitting python lists in various ways."""


def split_in_half(input_points):
    """Split list into two halves.

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


def groups_of_3(points):
    """Split a list of points into groups of 3.

    Parameters
    ----------
    points : list
        List of points to split.

    Returns
    -------
    list
        List of sublists, where each sublist contains 3 points. The last
        sublist may contain 1 or 2 points if the length of 'points' is not a
        multiple of 3.
    """
    num = len(points)
    if num % 3 == 0:
        return [points[i : i + 3] for i in range(0, num, 3)]
    elif num % 3 == 1:
        split = [points[i : i + 2] for i in range(0, 4, 2)]
        split2 = [points[i : i + 3] for i in range(4, num, 3)]
    elif num % 3 == 2:
        split = [[points[0], points[1]]]
        split2 = [points[i : i + 3] for i in range(2, num, 3)]
    return split + split2
