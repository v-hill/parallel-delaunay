"""Functions for algebra on python list objects."""


from math import sqrt

# ------------------------------ Vector algebra -------------------------------


def vector_add(vec1, vec2):
    """Return the sum of two 2D vectors.

    Parameters
    ----------
    vec1 : list
        A list or tuple representing the first 2D vector as [x,y].
    vec2 : list
        A list or tuple representing the second 2D vector as [x,y].

    Returns
    -------
    list
        A list representing the sum of the two input vectors as [x,y].
    """
    return [vec1[0] + vec2[0], vec1[1] + vec2[1]]


def vector_sub(vec1, vec2):
    """
    Subtract two vectors.

    Parameters
    ----------
    vec1 : list
        A list or tuple representing the first 2D vector as [x,y].
    vec2 : list
        A list or tuple representing the second 2D vector as [x,y].

    Returns
    -------
    list
        The result of subtracting vec2 from vec1.
    """
    return [vec1[0] - vec2[0], vec1[1] - vec2[1]]


def perpendicular(vec):
    """Compute the vector perpendicular to the input vector.

    Parameters
    ----------
    vec : list
        The input vector.

    Returns
    -------
    list
        The vector perpendicular to vec.
    """
    return [-vec[1], vec[0]]


def list_equal(list1, list2):
    """Check if the x and y components of two points are identical.

    Parameters
    ----------
    list1 : list of 2 elements
        First list to compare.
    list2 : list of 2 elements
        Second list to compare.

    Returns
    -------
    bool
        True if all components are equal, False otherwise.
    """
    val1 = list1[0] == list2[0]
    val2 = list1[1] == list2[1]
    return val1 and val2


def list_divide(vec, val):
    """Divide a two-element list by a scalar value.

    Parameters
    ----------
    vec : list
        A list of two elements representing a vector or point.
    val : float
        The value to divide the vector elements by.

    Returns
    -------
    list
        A new list with the elements of 'vec' divided by 'val'.
    """
    return [vec[0] / val, vec[1] / val]


def normalise(vector, length=1):
    """Normalise an input vector to the specified length.

    Parameters
    ----------
    vector : list
        A list of two or three numerical values representing a 2D or 3D vector.
    length : float, optional
        The desired length of the normalised vector. Default is 1.

    Returns
    -------
    list
        A new list representing the normalised vector. The elements of the
        list are scaled such that the Euclidean norm of the vector is equal
        to the specified length.

    Raises
    ------
    ValueError
        If the input vector has length 0.
    """
    norm = sqrt(vector[0] ** 2 + vector[1] ** 2)
    if norm == 0:
        raise ValueError("Cannot normalise a zero-length vector")
    scale_factor = length / norm
    return [vector[0] * scale_factor, vector[1] * scale_factor]


# ----------------------------- Sorting functions -----------------------------


def lexicographic_sort(
    points: list[list[int | float]],
) -> list[list[int | float]]:
    """Sort a list of 2D points in lexicographic order.

    Parameters
    ----------
    points : list[list[int | float]]
        A list of 2D points, where each point is represented as a tuple of two
        floating-point numbers.

    Returns
    -------
    list[list[int | float]]
        Points in lexicographic order.
    """
    points_sorted = sorted(points, key=lambda k: [k[0], k[1]])
    return points_sorted


# ------------------------------- Linear algebra ------------------------------


def in_circle(a, b, c, d):
    """Test if a point is within a circle defined by three points.

    This function is used to check whether a point 'd' is contained by the
    circle defined by three other points 'a', 'b', 'c'. This is achieved by
    calculating the sign of the following 4x4 matrix determinant.
        │ a.x  a.y  a.x**2+a.y**2  1 │
        │ b.x  b.y  b.x**2+b.y**2  1 │
        │ c.x  c.y  c.x**2+c.y**2  1 │
        │ d.x  d.y  d.x**2+d.y**2  1 │

    Parameters
    ----------
    a, b, c : list
        The three points that define the circle.
    d : list
        Testing if this point is in the circle.

    Returns
    -------
    out : Bool
        True if the point 'd' is within the circle defined by 'a', 'b', 'c'.
    """
    c1 = a[0] - d[0]
    c2 = b[0] - d[0]
    c3 = c[0] - d[0]

    u1 = a[1] - d[1]
    u2 = b[1] - d[1]
    u3 = c[1] - d[1]

    v1 = c1**2 + u1**2
    v2 = c2**2 + u2**2
    v3 = c3**2 + u3**2

    det = (
        (c1 * ((u2 * v3) - (v2 * u3)))
        - (c2 * ((u1 * v3) - (v1 * u3)))
        + (c3 * ((u1 * v2) - (v1 * u2)))
    )
    return det < 0


def ccw_angle(p1, p2, p3):
    """Calculate acute angle with cross product for 3 points.

    Determine the acute angle defined by three points in a right-handed
    coordinate system using the cross product.

    Parameters
    ----------
    p1, p2, p3 : list
        The three points being tested.

    Returns
    -------
    angle : float
        The angle, in radians, between the lines defined by p1-p3 and p2-p3.
        If p3 lies to the right of the line defined by p1-p2, angle is +ve.
        If p3 lies to the left of the line defined by p1-p2, angle is -ve.
        If the points are collinear, the angle is 0.
    """
    angle = (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (
        p2[0] - p3[0]
    )
    return angle


def on_right(p1, p2, p3):
    """
    Determine if p3 is on the right side of the line from p1 to p2.

    The function uses the ccw_angle function to calculate the angle formed by
    the three points and checks if it is greater than 0 to determine if the
    point is on the right.

    Parameters
    ----------
    p1 : List[float]
        Coordinate of the first point.
    p2 : List[float]
        Coordinate of the second point.
    p3 : List[float]
        Coordinate of the point to test.

    Returns
    -------
    bool
        Returns True if the point p3 is on the right side of the line defined
        between the points p1 and p2.
    """
    return ccw_angle(p1, p2, p3) > 0


def on_left(p1, p2, p3):
    """
    Determine if p3 is on the left side of the line from p1 to p2.

    The function uses the ccw_angle function to calculate the angle formed by
    the three points and checks if it is less than 0 to determine if the point
    is on the left.

    Parameters
    ----------
    p1 : List[float]
        Coordinate of the first point.
    p2 : List[float]
        Coordinate of the second point.
    p3 : List[float]
        Coordinate of the point to test.

    Returns
    -------
    bool
        Returns True if the point p3 is on the left side of the line defined
        between the points p1 and p2.
    """
    return ccw_angle(p1, p2, p3) < 0
