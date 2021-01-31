# delaunay-triangulation
This library is used for computing the [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) of 2D point sets. 

**Why is this library special?**

Speed! This is a pure python implemenation of the Guibas &amp; Stolfi's divide and conquer algorithm. The divide and conquer algorithm has been shown to be the [fastest](https://people.eecs.berkeley.edu/~jrs/meshpapers/SuDrysdale.pdf) DT generation technique, with O(*n* log *n*) running time. Furthermore, this code has been parallelised using the MPI for Python ([mpi4py](https://github.com/mpi4py/mpi4py)) library to utilise multiple CPU cores. This allows the algorithm to be efficiently scaled for distributed computing across supercomputer nodes.

## PLEASE NOTE

**This library is currently a work in progress. As such it is currently about 1000 lines of function definitions with no running script.**

## Structure
This repository is currently structured as follows.

    ├── delaunay_triangulation       
        ├── triangulation_core
            ├── linear_algebra
                └── linear_algebra.py
            ├── points_tools   
                ├── generate_values.py
                └── split_list.py
            ├── edge_topology.py
            ├── triangulation.py
            └── triangulation_primitives.py  
        ├── utilities    
            └── settings.py
        ├── triangulation_mpi_test.py
        └── triangulation_test.py
        
## References
<a id="1">[1]</a> 
Guibas, Leonidas and Stolfi, Jorge
'Primitives for the Manipulation of General Subdivisions and the Computation of Voronoi'
In: ACM Trans. Graph.4.2  (Apr.1985),  pp.  74–123.
issn:  0730-0301
doi:10.1145/282918.282923
