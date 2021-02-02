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

# -----------------------------------------------------------------------------  

def set_options(world_options):
    points_options = {}
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Triangulation run script')
    
    # Points options
    points = parser.add_argument_group('Points options')
    points.add_argument("number_of_points", 
                        type=int,
                        help=("Number of points to generate \n"
                              "(type: %(type)s)"))
    points.add_argument("--points_distribution",
                        default='random', const='random', nargs="?",
                        choices=['random', 'lattice'],
                        help=("Define how the points are initally arranged "
                              "within the world \n"
                              "(choices: %(choices)s) (default: %(default)s)"))
    
    # Edit world options
    world = parser.add_argument_group('Points world options')
    world.add_argument("-xm", "--max_x_val", 
                        type=float, metavar='',
                        help=("Max x-value of generated points \n"
                              f"(default: {world_options['max_x_val']}) "
                              "(type: %(type)s)"))
    world.add_argument("-ym", "--max_y_val", 
                        type=float, metavar='',
                        help=("Max y-value of generated points \n"
                              f"(default: {world_options['max_y_val']}) "
                              "(type: %(type)s)"))
    
    args = parser.parse_args()
    
    # Points options
    points_options['number_of_points'] = args.number_of_points
    points_options['points_distribution'] = args.points_distribution
    
    # Edit world options
    if args.max_x_val: 
        world_options['max_x_val'] = args.max_x_val
    if args.max_y_val: 
        world_options['max_y_val'] = args.max_y_val

    print("Points options:")
    for key, val in points_options.items():
        print(f"    {key:22} {val}")
        
    print("Points world options:")
    for key, val in world_options.items():
        print(f"    {key:22} {val}")

    options = {**world_options, 
               **points_options}
    
    return options
