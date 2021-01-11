# -*- coding: utf-8 -*-
"""
This module contains functions for algebra on python list objects.
"""

# Standard library imports
from math import sqrt

# ------------------------------ Vector algebra -------------------------------

def vector_add(vec1, vec2):
    return [vec1[0]+vec2[0], vec1[1]+vec2[1]]

def vector_sub(vec1, vec2):
    return [vec1[0]-vec2[0], vec1[1]-vec2[1]]

def perpendicular(vec):
    return [-vec[1], vec[0]]

def list_equal(list1, list2):
    val1 = list1[0] == list2[0]
    val2 = list1[1] == list2[1]
    return val1 and val2

def list_divide(vec, val):
    """
    Returns
    -------
    out : list
        Input list 'vec' divided by the value 'val'.
    """
    return [vec[0]/val, vec[1]/val]

def normalise(vector, length=1):
    """
    Returns
    -------
    out : list
        Input list 'vec' normalsied to the value of 'length'.
    """
    norm = sqrt(vector[0]**2 + vector[1]**2)/length
    return list_divide(vector, norm)

# ----------------------------- Sorting functions -----------------------------

def lexigraphic_sort(points):
    points_sorted = sorted(points, key=lambda k: [k[0], k[1]])
    return points_sorted

# ------------------------------- Linear algebra ------------------------------

def in_circle(a, b, c, d):
    """
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
        The three points that define the circle
    d : list
        Testing if this point is in the circle
        
    Returns
    -------
    out : Bool
        True if the point 'd' is within the circle defined by 'a', 'b', 'c'
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
    
    det = c1*((u2*v3)-(v2*u3)) - c2*((u1*v3)-(v1*u3)) + c3*((u1*v2)-(v1*u2))
    
    return det < 0

def ccw_angle(p1, p2, p3):
    """
    Use the cross product to determine the acute angle defined by three
    points p1, p2, p3. Given this is a "right-handed" coordinate system there
    are three possible outcomes for the angle:
    1.  +ve angle, p3 lies to the right of the line defined by p1 p2
    2.  -ve angle, p3 lies to the left of the line defined by p1 p2
    3.  angle of 0, the points are collinear
    
    Parameters
    ----------
    p1, p2, p3 : list
        The three points being tested

    Returns
    -------
    angle : float
    """
    angle = (p1[0]-p3[0]) * (p2[1]-p3[1]) - (p1[1]-p3[1]) * (p2[0]-p3[0])
    return angle

def on_right(p1, p2, p3):
    """
    Returns
    -------
    out : Bool
        Return true if the point p3 is on the right of the line defined by the 
        points p1 and p3.
    """
    return ccw_angle(p1, p2, p3) > 0
    
def on_left(p1, p2, p3):
    """
    Returns
    -------
    out : Bool
        Return true if the point p3 is on the left of the line defined by the 
        points p1 and p3.
    """
    return ccw_angle(p1, p2, p3) < 0
