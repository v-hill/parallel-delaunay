"""Functions for creating triangulations.

This module contains functions for creating triangulations from 2 and 3 points,
respectively known as the line-primitive and the triangle-primitive.
"""


import triangulation_core.edge_topology as edge_topology
import triangulation_core.linear_algebra as linalg

# -------------------------------- Definitions --------------------------------


def line_primitive(pts_subset):
    """Construct edge and its symmetric, add to TriangulationEdges object.

    This function takes in a list of two points and forms an edge.
    The symmetric edge, where the origin and destination points are reversed,
    is also constructed. These two edge are added into a new TriangulationEdges
    class object which is returned by the function.

    Parameters
    ----------
    pts_index : list
        List of the indices of the two points.
    pts_subset : lists of lists
        A set of two points with the form [ [x1, y1], [x2, y2] ].

    Returns
    -------
    left_most_edge : int
        Index of edge with the left most point.
    right_most_edge : int
       Index of the edge with the right most point.
    line : TriangulationEdges
        The resulting triangulation of two points.
    """
    p1, p2 = 0, 1
    edge, edge_sym = edge_topology.setup_edge(p1, p2, 0)
    line = edge_topology.TriangulationEdges(pts_subset)
    line.push_back(edge)
    line.push_back(edge_sym)

    left_most_edge = edge.index
    right_most_edge = line.edges[edge.index].sym

    line.set_extreme_edges(left_most_edge, right_most_edge)
    return line


def triangle_primitive(pts_subset):
    """Create a triangle from 3 points with CCW orientation.

    This function takes a list of three points and forms three edges to
    create a single triangle. This triangle has the property that the origin
    of one edge is connected to the destination of the next edge in a CCW
    orientation.

    Parameters
    ----------
    pts_index : list
        List of the indices of the three points.
    pts_subset : lists of lists
        A set of three points with the form [ [x1, y1], [x2, y2] , [x3, y3] ].

    Returns
    -------
    out1 : int
        Index of edge with the left most point.
    ou2 : int
       Index of the edge with the right most point.
    edges : TriangulationEdges
        The resulting triangulation of three points.
    """
    p1, p2, p3 = 0, 1, 2
    triang = edge_topology.TriangulationEdges(pts_subset)

    # Create the first two edges of the triangle
    edge1, edge1_sym = edge_topology.setup_edge(p1, p2, 0)
    triang.push_back(edge1)
    triang.push_back(edge1_sym)

    edge2, edge2_sym = edge_topology.setup_edge(p2, p3, 2)
    triang.push_back(edge2)
    triang.push_back(edge2_sym)

    triang.splice(edge1_sym.index, edge2.index)

    # To maintain the counter-clockwise orientation of the edges in the
    # triangle, we determine where p3 is in relation to the two existing edges.
    pt1 = pts_subset[triang.edges[edge1.index].org]
    pt2 = pts_subset[triang.edges[edge1.index].dest]
    pt3 = pts_subset[p3]

    if linalg.on_right(pt1, pt2, pt3):
        # Points are in CCW orientiaton
        c = triang.connect(edge2.index, edge1.index)
        triang.set_extreme_edges(edge1.index, edge2_sym.index)
        return triang

    if linalg.on_left(pt1, pt2, pt3):
        # Points are in CW orientiaton
        c = triang.connect(edge2.index, edge1.index)
        triang.set_extreme_edges(triang.edges[c].sym, c)
        return triang

    # Points are collinear
    triang.set_extreme_edges(edge1.index, edge2_sym.index)
    return triang


def make_primitives(split_pts):
    """Create a list of geometric primitives from a list of sets of points.

    Parameters
    ----------
    split_pts : list of list of tuple
        List of sets of x-y coordinates for points.

    Returns
    -------
    list of Primitive
        List of geometric primitives. Each primitive is either a LinePrimitive
        or a TrianglePrimitive object.

    Raises
    ------
    Exception
        If a set of points contains no points.
    """
    primitives = []
    for pts_subset in split_pts:
        if len(pts_subset) == 2:
            # 2 points define a single edge
            primitives.append(line_primitive(pts_subset))
        elif len(pts_subset) == 3:
            # 3 points define a single triangle
            primitives.append(triangle_primitive(pts_subset))
        elif len(pts_subset) == 0:
            raise Exception("Unexpected number of points in pts_subset")
    return primitives
