# parallel-delaunay
This library is used for computing the [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) of 2D point sets.

An example Delaunay triangulation output for 32 points is shown below.
<img src="./images/Figure 2021-02-24 110617.png" alt="drawing" width="500"/>


**Why is this library special?**

It is a Numba port of Guibas &amp; Stolfi's divide-and-conquer algorithm, which runs in O(*n* log *n*) and is among the [fastest](https://people.eecs.berkeley.edu/~jrs/meshpapers/SuDrysdale.pdf) Delaunay techniques. The kernel does not use edge objects. Each directed edge is an index into parallel arrays: `org[e]` is the origin vertex and `onext[e]` is the next edge counterclockwise around that origin. The symmetric edge is `e ^ 1`, because each undirected edge occupies the pair of slots `2k` and `2k + 1`. The merge loops are marked `@njit`, so Numba compiles them to machine code.

`triangulate` runs that kernel across cores. The divide tree is a bottom-up loop, and `prange` takes each level: every leaf first, then each round of independent merges. `parallel=False` keeps the same kernel on one core. The array layout is a rewrite of the old edge objects, and the MPI driver is gone.

# Installation

This repository uses [uv](https://docs.astral.sh/uv/) for Python and dependency management. Python 3.11 or newer is required.

Install the library and development tools from the repository root:

```bash
uv sync
```

Add the published package to another uv project with `uv add parallel-delaunay`.

`scipi_comparison.py` generates a random set of 2D points and runs the triangulation:

```bash
uv run python scipi_comparison.py
```

# Tests

Unit tests cover the geometry helpers. One test runs the full triangulation and compares its triangles with SciPy.

```bash
make test
```

# Formatting

Format the Python sources with Ruff:

```bash
make format
```

# Benchmarking

`triangulation_benchmarks.py` times this kernel against `scipy.spatial.Delaunay`. Numba compiles on the first call, so the script triangulates a tiny point set before the timed loop.

```bash
uv run python triangulation_benchmarks.py
```

`triangulate` splits the points into blocks and runs each merge level with Numba `prange`. That is the default. Pass `parallel=False` for the one-core kernel. `parallel_benchmark.py` times the two:

```bash
uv run python parallel_benchmark.py
```

`scipi_comparison.py` prints one comparison for 1000 points:

```bash
uv run python scipi_comparison.py
```

The figure below is the earlier pure-Python kernel, which ran about 40× slower than SciPy. It is not a timing of the Numba kernel.

<img src="./images/Figure 2021-07-02 210145.png" alt="drawing" width="700"/>

## Structure
This repository is currently structured as follows.

    ├── Makefile
    ├── paralleldelaunay
    │   ├── triangulation_core
    │   │   ├── points_tools
    │   │   │   ├── generate_values.py
    │   │   │   └── split_list.py
    │   │   ├── linear_algebra.py
    │   │   ├── triangulation.py
    │   │   └── triangulation_parallel.py
    │   └── utilities
    │       └── settings.py
    ├── tests
    │   ├── test_linear_algebra.py
    │   ├── test_points_tools.py
    │   ├── test_primitives.py
    │   └── test_triangulate.py
    ├── parallel_benchmark.py
    ├── scipi_comparison.py
    ├── triangulation_benchmarks.py
    └── pyproject.toml

## References
<a id="1">[1]</a>
Guibas, Leonidas and Stolfi, Jorge
'Primitives for the Manipulation of General Subdivisions and the Computation of Voronoi'
In: ACM Trans. Graph.4.2  (Apr.1985),  pp.  74–123.
issn:  0730-0301
doi:10.1145/282918.282923
