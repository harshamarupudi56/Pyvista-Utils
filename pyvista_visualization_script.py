#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:39:59 2022

@author: sriharsha.marupudi
"""

import pyvista as pv
import trimesh

mesh = pv.read("/gpfs_projects/sriharsha.marupudi/cube_50x50.stl")
pv.set_plot_theme('document')
p = pv.Plotter()
p.add_mesh(mesh,color='white')
# p.add_axes_at_origin()
# p.add_mesh(bar)
p.show()
