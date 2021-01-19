"""
The following code implements the Guibas and Stolfi divide-and-conquer 
algorithm as presented in: 
    L. J. Guibas, J. Stolfi, "Primitives for the manipulation of general 
    subdivisions and the computation of Voronoi diagrams" (1985)
This algorithm computes the Delaunay triangulation of a set of input points.
"""

# ---------------------------------- Imports ----------------------------------

# Repo module imports
import triangulation_core.points_tools.split_list as split_list
import triangulation_core.linear_algebra.linear_algebra as linalg
from triangulation_core.triangulation_primitives import make_primitives

# --------------------------- Edge finding functions --------------------------

def lowest_common_tangent(h_left, h_right):
    """
    Given two fully triangulated sets of points, this function finds an
    edge connecting the two triangulations. Each triangulation forms a convex 
    hull of edges. The edge to be found by this function is the edge with the 
    lowest y-value point which is still tangential to both hulls. This is
    known as the 'base' edge, as it is the first edge connecting two 
    separately triangulated point sets. 

    Parameters
    ----------
    h_left : TriangulationEdges
    h_right : TriangulationEdges

    Returns
    -------
    left_e : int
        The index of the edge in the right hull which forms one end of the 
        base edge
    right_e : int
        The index of the edge in the left hull which forms the other end of 
        the base edge
    """
    left_e = h_left.outer
    right_e = h_right.inner
    
    pts_left = h_left.points
    pts_right = h_right.points
    
    p1 = pts_left[h_left.edges[left_e].org]
    p2 = pts_left[h_left.edges[left_e].dest]

    p4 = pts_right[h_right.edges[right_e].org]
    p5 = pts_right[h_right.edges[right_e].dest] 

    while True:
        if linalg.on_right(p1, p2, pts_right[h_right.edges[right_e].org]):
            left_e = h_left.edges[h_left.edges[left_e].sym].onext

            p1 = pts_left[h_left.edges[left_e].org]
            p2 = pts_left[h_left.edges[left_e].dest]

        elif linalg.on_left(p4, p5, pts_left[h_left.edges[left_e].org]):
            right_e = h_right.edges[h_right.edges[right_e].sym].oprev
            p4 = pts_right[h_right.edges[right_e].org]
            p5 = pts_right[h_right.edges[right_e].dest]
            
        else:
            return left_e, right_e

def rcand_func(rhull, rcand, b1, b2):
    """
    This function finds the candidate edge from the right hull triangulation.
    An initial candidate 'rcand' is given. This candidate is tested. If the
    candidate fails it is deleted from the triangulation and the next 
    potential candiate is considered. While a valid candidate has not been 
    found this process continues until a valid candidate is found.

    Parameters
    ----------
    rhull : TriangulationEdges
        The triangulation of edges on the right hand side
    rcand : TYPE
        DESCRIPTION.
    b1 : list
        DESCRIPTION.
    b2 : list
        DESCRIPTION.

    Returns
    -------
    rhull : TriangulationEdges
        DESCRIPTION.
    rcand : TYPE
        DESCRIPTION.
    """
    completed = False
    while not completed:
        rcand_onext_dest = rhull.edges[rhull.edges[rcand].onext].dest
        rcand_dest = rhull.edges[rcand].dest
        ccw_test = linalg.on_right(b1, b2, rhull.points[rcand_onext_dest])
        next_cand_invalid = linalg.in_circle(b2, b1, 
                                             rhull.points[rcand_dest], 
                                             rhull.points[rcand_onext_dest])
        if ccw_test and next_cand_invalid:
            t = rhull.edges[rcand].onext
            rhull.kill_edge(rcand)
            rcand = t
        else:
            completed = True
    return rhull, rcand

def lcand_func(lhull, lcand, b1, b2):
    """
    This function performs the same task as the above 'rcand_func' but testing
    for the left candidate edge. 
    """
    completed = False
    while not completed:
        lcand_oprev_dest = lhull.edges[lhull.edges[lcand].oprev].dest
        lcand_dest = lhull.edges[lcand].dest
        ccw_test = linalg.on_right(b1, b2, lhull.points[lcand_oprev_dest])
        next_cand_invalid = linalg.in_circle(b2, b1, 
                                             lhull.points[lcand_dest], 
                                             lhull.points[lcand_oprev_dest])
        if ccw_test and next_cand_invalid:
            t = lhull.edges[lcand].oprev
            lhull.kill_edge(lcand)
            lcand = t
        else:
            completed = True
    return lhull, lcand

def candidate_decider(rcand, lcand, lcand_valid, triangulation):
    """
    Given two potential edges which could be added to the triangulation, 
    decide which of the edges is the correct one to add.

    Parameters
    ----------
    rcand : int
        index of right candidate edge
    lcand : int
        index of left candidate edge
    lcand_valid : TYPE
        DESCRIPTION.
    triangulation : TriangulationEdges
        DESCRIPTION.

    Returns
    -------
    result : bool
        DESCRIPTION.
    """
    pt1 = triangulation.points[triangulation.edges[rcand].dest]
    pt2 = triangulation.points[triangulation.edges[rcand].org]
    pt3 = triangulation.points[triangulation.edges[lcand].org]
    pt4 = triangulation.points[triangulation.edges[lcand].dest]
    result = lcand_valid and linalg.in_circle(pt1, pt2, pt3, pt4)
    return result

# ----------------------------- Merging functions -----------------------------

def combine_triangulations(ldi, rdi, hull_left, hull_right):
    """
    This function takes two TriangulationEdges class objects and combines
    them into a single TriangulationEdges object.

    Parameters
    ----------
    ldi : TYPE
        DESCRIPTION.
    rdi : TYPE
        DESCRIPTION.
    hull_left : TriangulationEdges
        DESCRIPTION.
    hull_right : TriangulationEdges
        DESCRIPTION.

    Returns
    -------
    base : TYPE
        DESCRIPTION.
    edges : TriangulationEdges
        DESCRIPTION.
    """
    ldo = hull_left.inner
    rdo = hull_right.outer
    rdi += hull_left.num_edges
    rdo += hull_left.num_edges
    
    edges = hull_left.combine_triangulations(hull_right)
    base = edges.connect(edges.edges[ldi].sym, rdi)
    
    # Correct the base edge
    ldi_org = edges.points[edges.edges[ldi].org]
    ldo_org = edges.points[edges.edges[ldo].org]
    rdi_org = edges.points[edges.edges[rdi].org]
    rdo_org = edges.points[edges.edges[rdo].org]
    
    if linalg.list_equal(ldi_org, ldo_org):
        ldo = base
    if linalg.list_equal(rdi_org, rdo_org):
        rdo = edges.edges[base].sym
    
    edges.set_extreme_edges(ldo, rdo)
    
    return base, edges

def zip_hulls(base, triang):
    """
    Given a triangulation containing two seperate hulls and the base edge 
    connecting the hulls, triangulate the space between the hulls. This is
    refered to as 'zipping' the hulls together.
    
    Parameters
    ----------
    base : int
        index of base edge
    triang : TriangulationEdges 
        Incomplete triangulation, with known base edge

    Returns
    -------
    d_triang : TriangulationEdges 
        Instance of TriangulationEdges class object containing the finished
        Delaunay triangulation of the input triangulation. 
    """
    while True:
        # Make variables for commonly used base edge points
        base1 = triang.points[triang.edges[base].org]
        base2 = triang.points[triang.edges[base].dest]
        
        # Find the first candidate edges for triangulation from each subset
        rcand = triang.edges[triang.edges[base].sym].onext
        pt1 = triang.points[triang.edges[rcand].dest]
        rcand_valid = linalg.on_right(base1, base2, pt1)
        
        lcand = triang.edges[base].oprev
        pt2 = triang.points[triang.edges[lcand].dest]
        lcand_valid = linalg.on_right(base1, base2, pt2)
        
        # If neither candidate is valid, hull merge is complete
        if not rcand_valid and not lcand_valid:
            break
        
        if rcand_valid:
            triang, rcand = rcand_func(triang, rcand, base1, base2)

        if lcand_valid:
            triang, lcand = lcand_func(triang, lcand, base1, base2)
        
        lcand_strong_valid = candidate_decider(rcand, lcand, lcand_valid, triang)
        
        if not rcand_valid or lcand_strong_valid:
            base = triang.connect(lcand, triang.edges[base].sym)
        else:
            base = triang.connect(triang.edges[base].sym, triang.edges[rcand].sym)
    return triang

def merge_triangulations(groups):
    """
    Each entry of the groups list is a list of two (or one) triangulations. 
    This function takes each pair of triangulations and combines them. 

    Parameters
    ----------
    groups : list
        List of pairs of triangulations

    Returns
    -------
    list
        List of merged triangulations
    """
    triangulations = []
    for group in groups:
        if len(group)==2:
            # Find the first edges to connect the seperate triangulations
            ldi, rdi = lowest_common_tangent(group[0], group[1])
            
            # Combine the two hulls into a single set of edges
            base, d_triang = combine_triangulations(ldi, rdi, group[0], group[1])
            
            # Given the starting base edge, fill in the edges between the hulls
            d_triang = zip_hulls(base, d_triang)
            triangulations.append(d_triang)
        else:
            triangulations.append(group[0])
    return [triangulations[i:i+2] for i in range(0, len(triangulations), 2)]
    
