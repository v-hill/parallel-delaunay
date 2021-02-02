"""
Command line interface for Delaunay triangulation program.
"""

# ---------------------------------- Imports ----------------------------------

# Standard library imports
import argparse
import time

# Repo module imports
from linear_algebra.linear_algebra import lexigraphic_sort
import points_tools.generate_values as generate_values
from triangulation_core.triangulation import triangulate
from utilities.settings import World
from utilities.settings import world_options
