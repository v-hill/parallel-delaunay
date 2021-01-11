# -*- coding: utf-8 -*-
"""
This module contains functions for creating triangulations from 2 and 3 points,
respectively known as the line-primitive and the triangle-primitive.
"""

# ---------------------------------- Imports ----------------------------------

# Repo module imports
import triangulation_core.edge_topology as edge_topology
import triangulation_core.linear_algebra.linear_algebra as linalg

# -------------------------------- Definitions --------------------------------

def line_primitive(pts_subset):
    """
    This function takes in a list of two points and forms an edge. 
    The symetric edge, where the origin and destination points are reversed, 
    is also constructed. These two edge are added into a new TriangulationEdges
    class object which is returned by the function. 
    
    Parameters
    ----------
    pts_index : list
        List of the indices of the two points
    pts_subset : lists of lists
        A set of two points with the form [ [x1, y1], [x2, y2] ]

    Returns
    -------
    left_most_edge : int
        Index of edge with the left most point
    right_most_edge : int
       Index of the edge with the right most point
    line : TriangulationEdges
        The resulting triangulation of two points
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
