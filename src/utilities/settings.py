# -*- coding: utf-8 -*-

"""
This module containings the default settings for the triangulation 
run program.
"""

class World():
    """
    Define the range of possible values to generate points for.
    """
    def __init__(self, world_size):
        self.x_min = world_size[0]
        self.x_max = world_size[1]
        self.y_min = world_size[2]
        self.y_max = world_size[3]

world_options = {'min_x_val' : 0,
                 'max_x_val' : 100,
                 'min_y_val' : 0,
                 'max_y_val' : 100}
