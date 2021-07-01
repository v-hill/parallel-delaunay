"""
This module implements the quad-edge data structure used by Gubias and Stolfi,
as well as defining classes for storing multiple edge objects.
"""

# -------------------- Allow imports from sibling folders ---------------------

# Standard library imports
from triangulation_core.linear_algebra import list_equal

# --------------------------------- Edge class --------------------------------


class Edge():
    """
    This class represents a single edge as defined in the Guibas and Stolfi
    edge algebra formalism.

    Attributes
    ----------
    index : int
        unique edge index
    org : int
        index of edge origin point
    dest : int
        index of edge destination point
    sym : int
        index of symetric edge
    onext : int
        index of next ccw edge connected to the origin point
    oprev : int
        index of previous ccw edge connected to the origin point
    deactivate : bool
        status of edge in triangulation. False if the edge is still part of
        the triangulation.
    """

    def __init__(self, idx, org, dst, s, onxt, oprv):
        self.index = idx
        self.org = org
        self.dest = dst
        self.sym = s
        self.onext = onxt
        self.oprev = oprv
        self.deactivate = False

    def __repr__(self):
        return (
            f"[{self.index}, {self.org}, "
            f"{self.dest}, {self.sym}, "
            f"{self.onext}, {self.oprev}, "
            f"{self.deactivate}]"
        )

    def return_point(self):
        return [self.org, self.dest]

    def shift_indices(self, shift_edges, shift_points):
        """
        This function is used to shift the indices of an edge. This is used
        when merging sets of edges to ensure each edge has a unique index.

        Parameters
        ----------
        shift_edges : int
            integer to shift edge indices by
        shift_points : int
            integer to shift orgin and destination points by
        """
        self.index += shift_edges
        self.org += shift_points
        self.dest += shift_points
        self.sym += shift_edges
        self.onext += shift_edges
        self.oprev += shift_edges

    def find_connections(self, edges):
        """
        Find all the edges in the triangulation connected to the origin point
        of this edge. This gives a list of the points that the boid is to
        consider as neighbours.

        Parameters
        ----------
        unique_edges : list of Edge class objects
            The list of unique edges in the triangulation

        Returns
        -------
        pts_subset : list
            List of the neighbour points
        """
        pts_subset = [self.return_point()]
        next_edge = edges[self.onext]

        while not list_equal(next_edge.return_point(), self.return_point()):
            pts_subset.append(next_edge.return_point())
            next_edge = edges[next_edge.onext]
        return pts_subset


def setup_edge(origin, dest, edge_idx):
    """
    This function takes in the index of two points and creates an edge array,
    as well as an edge for the symetric edge.

    Parameters
    ----------
    origin : int
        index of origin point
    dest : int
        index of destination point
    edge_idx : int
        The index of the new edge

    Returns
    -------
    edge : Edge class
        Edge connecting org to dest
    edge_sym : Edge class
        Edge connecting dest to org
    """
    e1_idx = edge_idx
    e2_idx = edge_idx + 1

    edge = Edge(e1_idx, origin, dest, e2_idx, e1_idx, e1_idx)
    edge_sym = Edge(e2_idx, dest, origin, e1_idx, e2_idx, e2_idx)

    return edge, edge_sym

# -------------------------------- Edges class --------------------------------


class Edges():
    def __init__(self):
        self.edges = []
        self.num_edges = 0
        self.inner = None
        self.outer = None

    def push_back(self, new_edge):
        self.edges.append(new_edge)
        self.num_edges += 1

    def set_extreme_edges(self, left_most_edge, right_most_edge):
        self.inner = left_most_edge
        self.outer = right_most_edge

    def splice(self, edge1, edge2):
        """
        This function is used when adding another edge to a point in the
        triangulation which has existing edges connected to it. The next and
        previous ccw edges are updated accordingly for each of the input edges.

        Parameters
        ----------
        edge1 : Edge class
        edge2 : Edge class
        """
        self.edges[self.edges[edge1].onext].oprev = edge2
        self.edges[self.edges[edge2].onext].oprev = edge1

        onext_2 = self.edges[edge2].onext
        onext_1 = self.edges[edge1].onext

        self.edges[edge1].onext = onext_2
        self.edges[edge2].onext = onext_1

    def connect(self, edge1, edge2):
        """
        This function takes two seperated edges and creates a new edge
        connecting the two.

        Parameters
        ----------
        edge1 : Edge class
        edge2 : Edge class

        Returns
        -------
        out : int
            index of the created edge
        """

        current_index = self.num_edges
        edge, edge_sym = setup_edge(
            self.edges[edge1].dest, self.edges[edge2].org, current_index)

        self.push_back(edge)
        self.push_back(edge_sym)

        edge1_sym_oprev = self.edges[self.edges[edge1].sym].oprev

        self.splice(edge.index, edge1_sym_oprev)
        self.splice(self.edges[edge.index].sym, edge2)

        return edge.index

    def kill_edge(self, e):
        """
        This function removes an edge from the triangulation by setting the
        status of edge.deactivate to True. The function also fixed the
        connecting edges too.

        Parameters
        ----------
        e : Edge class
            edge to remove from the triangulation
        """
        # Fix the local triangulation
        self.splice(e, self.edges[e].oprev)
        self.splice(self.edges[e].sym, self.edges[self.edges[e].sym].oprev)

        # Set the status of the edge and it's symetric edge to kill
        self.edges[e].deactivate = True
        self.edges[self.edges[e].sym].deactivate = True

    def filter_deactivated(self):
        self.edges = [e for e in self.edges if e.deactivate == False]

    def get_unique(self, num):
        unique = [''] * num
        points_seen = []
        for edge in self.edges:
            if edge.org not in points_seen:
                unique[edge.org] = edge
                points_seen.append(edge.org)

        return unique

# ---------------------------- Triangulation class ----------------------------


class TriangulationEdges(Edges):
    def __init__(self, points_subset):
        super().__init__()
        self.points = points_subset

    def shift_indices(self, shift_edges, shift_points):
        for edge in self.edges:
            edge.shift_indices(shift_edges, shift_points)

    def merge_hulls(self, second_hull):
        # Calculate the capacity of the new edges array
        len1 = self.num_edges
        len2 = second_hull.num_edges

        # Set the correct indices for the second hull
        second_hull.shift_indices(len1, len(self.points))

        # Combine the edges data from the two triangulations
        self.edges += second_hull.edges
        self.num_edges = len1 + len2

    def combine_triangulations(self, triangulation):
        self.merge_hulls(triangulation)
        self.points += triangulation.points
        return self
