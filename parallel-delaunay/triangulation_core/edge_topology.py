"""Implementation of the quad-edge data structure used by Gubias and Stolfi."""


from triangulation_core.linear_algebra import list_equal

# --------------------------------- Edge class --------------------------------


class Edge:
    """Representation of a single quad-edge.

    In the Gubias and Stolfi algorithm, an edge is represented as a directed
    edge (d-edge) which consists of two half-edges. Each half-edge represents
    one side of the edge and has pointers to its origin vertex, its target
    vertex, the next half-edge in the counterclockwise direction around its
    origin, and the opposite half-edge. Each d-edge is represented by four
    half-edges, which are organized into a "quad-edge".

    Attributes
    ----------
    index : int
        Unique edge index.
    org : int
        Index of edge origin point..
    dest : int
        Index of edge destination point.
    sym : int
        Index of symmetric edge.
    onext : int
        Index of next ccw edge connected to the origin point.
    oprev : int
        Index of previous ccw edge connected to the origin point.
    deactivate : bool
        Status of edge in triangulation. False if the edge is still part of the
        triangulation.
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
        """Return a string representation of the quad-edge object.

        Returns
        -------
        str
            A string representation of the quad-edge object.
        """
        return (
            f"[{self.index}, {self.org}, "
            f"{self.dest}, {self.sym}, "
            f"{self.onext}, {self.oprev}, "
            f"{self.deactivate}]"
        )

    def return_point(self):
        """Get the edge origin and destination as a list.

        Returns
        -------
        list
            Origin and destination as a list.
        """
        return [self.org, self.dest]

    def shift_indices(self, shift_edges, shift_points):
        """Shift the indices of an edge.

        This is used when merging sets of edges to ensure each edge has a
        unique index.

        Parameters
        ----------
        shift_edges : int
            Integer to shift edge indices by.
        shift_points : int
            Integer to shift origin and destination points by.
        """
        self.index += shift_edges
        self.org += shift_points
        self.dest += shift_points
        self.sym += shift_edges
        self.onext += shift_edges
        self.oprev += shift_edges

    def find_connections(self, edges):
        """Find edges connected to the origin point.

        Find all the edges in the triangulation connected to the origin point
        of this edge. This gives a list of the points that the boid is to
        consider as neighbours.

        Parameters
        ----------
        unique_edges : list of Edge class objects
            The list of unique edges in the triangulation.

        Returns
        -------
        pts_subset : list
            List of the neighbour points.
        """
        pts_subset = [self.return_point()]
        next_edge = edges[self.onext]

        while not list_equal(next_edge.return_point(), self.return_point()):
            pts_subset.append(next_edge.return_point())
            next_edge = edges[next_edge.onext]
        return pts_subset


def setup_edge(origin, dest, edge_idx):
    """Create symmetric edge for two points.

    This function takes in the index of two points and creates an edge array,
    as well as an edge for the symmetric edge.

    Parameters
    ----------
    origin : int
        Index of origin point.
    dest : int
        Index of destination point.
    edge_idx : int
        The index of the new edge.

    Returns
    -------
    edge : Edge class
        Edge connecting org to dest.
    edge_sym : Edge class
        Edge connecting dest to org.
    """
    e1_idx = edge_idx
    e2_idx = edge_idx + 1

    edge = Edge(e1_idx, origin, dest, e2_idx, e1_idx, e1_idx)
    edge_sym = Edge(e2_idx, dest, origin, e1_idx, e2_idx, e2_idx)

    return edge, edge_sym


# -------------------------------- Edges class --------------------------------


class Edges:
    """Represent a set of edges in a triangulation.

    Stores methods to add edges and keep track of the number of edges, as well
    as methods to set the inner and outer edges used to construct triangle
    primitives.
    """

    def __init__(self):
        self.edges = []
        self.num_edges = 0
        self.inner = None
        self.outer = None

    def push_back(self, new_edge):
        """Append a new edge to the end of the list of edges.

        Parameters
        ----------
        new_edge : Edge
            The new 'Edge' object to be added to the end of the list of edges.
        """
        self.edges.append(new_edge)
        self.num_edges += 1

    def set_extreme_edges(self, left_most_edge, right_most_edge):
        """Set the inner and outer edges for constructing primitives.

        Parameters
        ----------
        left_most_edge : Edge
            The left-most edge in the triangulation, which will be used as the
            inner edge.
        right_most_edge : Edge
            The right-most edge in the triangulation, which will be used as the
            outer edge.
        """
        self.inner = left_most_edge
        self.outer = right_most_edge

    def splice(self, edge1, edge2):
        """Update the next and previous edges in a counterclockwise direction.

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
        """Take two separated edges and creates a new edge connecting the two.

        Parameters
        ----------
        edge1 : Edge class
        edge2 : Edge class

        Returns
        -------
        out : int
            Index of the created edge.
        """
        current_index = self.num_edges
        edge, edge_sym = setup_edge(
            self.edges[edge1].dest, self.edges[edge2].org, current_index
        )
        self.push_back(edge)
        self.push_back(edge_sym)
        edge1_sym_oprev = self.edges[self.edges[edge1].sym].oprev
        self.splice(edge.index, edge1_sym_oprev)
        self.splice(self.edges[edge.index].sym, edge2)
        return edge.index

    def kill_edge(self, e):
        """Remove edge, deactivate status, fix connections.

        This function removes an edge from the triangulation by setting the
        status of edge.deactivate to True. The function also fixed the
        connecting edges too.

        Parameters
        ----------
        e : Edge class
            Edge to remove from the triangulation.
        """
        # Fix the local triangulation
        self.splice(e, self.edges[e].oprev)
        self.splice(self.edges[e].sym, self.edges[self.edges[e].sym].oprev)

        # Set the status of the edge and it's symmetric edge to kill
        self.edges[e].deactivate = True
        self.edges[self.edges[e].sym].deactivate = True

    def filter_deactivated(self):
        """Remove deactivated edges from the edges list."""
        self.edges = [e for e in self.edges if not e.deactivate]

    def get_unique(self, num):
        """Return a list of unique edges corresponding to each point index.

        This method returns a list of unique edges, where each edge corresponds
        to a unique point index in the 'Edges' object. The method iterates over
        the edges in the 'Edges' object and checks if the origin vertex of the
        edge has been seen before. If the origin vertex is unique, the edge is
        added to the corresponding index in the 'unique' list. The method
        returns the 'unique' list.

        Parameters
        ----------
        num : int
            The number of unique points in the 'Edges' object.

        Returns
        -------
        list
            A list of unique edges corresponding to each point index.
        """
        unique = [""] * num
        points_seen = []
        for edge in self.edges:
            if edge.org not in points_seen:
                unique[edge.org] = edge
                points_seen.append(edge.org)
        return unique


# ---------------------------- Triangulation class ----------------------------


class TriangulationEdges(Edges):
    """A class for storing the edges of a Delaunay triangulation.

    This class inherits from the 'Edges' class and adds a reference to the
    points of the triangulation. It also includes methods for shifting the
    indices of the edges and points when merging two triangulations, and for
    merging the hulls of two triangulations.
    """

    def __init__(self, points_subset):
        """
        Initialize the TriangulationEdges object.

        Parameters
        ----------
        points_subset : array-like
            An array of points in the triangulation.
        """
        super().__init__()
        self.points = points_subset

    def shift_indices(self, shift_edges, shift_points):
        """Shift the indices of the edges and points.

        Parameters
        ----------
        shift_edges : int
            The number of edges to shift the indices by.
        shift_points : int
            The number of points to shift the indices by.
        """
        for edge in self.edges:
            edge.shift_indices(shift_edges, shift_points)

    def merge_hulls(self, second_hull):
        """Merge the hulls of two triangulations.

        Parameters
        ----------
        second_hull : TriangulationEdges
            The hull of a second triangulation to merge with this one.
        """
        # Calculate the capacity of the new edges array
        len1 = self.num_edges
        len2 = second_hull.num_edges

        # Set the correct indices for the second hull
        second_hull.shift_indices(len1, len(self.points))

        # Combine the edges data from the two triangulations
        self.edges += second_hull.edges
        self.num_edges = len1 + len2

    def combine_triangulations(self, triangulation):
        """Combine another triangulation with this one.

        This method merges the convex hull of another 'TriangulationEdges'
        object with this object's convex hull. The points of the other
        triangulation are added to the current object's points array.

        Parameters
        ----------
        triangulation : TriangulationEdges
            The triangulation to be combined with this one.

        Returns
        -------
        TriangulationEdges
            The 'TriangulationEdges' object resulting from the combination of
            the two triangulations.
        """
        self.merge_hulls(triangulation)
        self.points += triangulation.points
        return self
