#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 20:52:45 2023

@author: sriharsha.marupudi
"""

import numpy as np
import pyvista as pv
import nrrd 
import matplotlib.pyplot as plt 
from skimage.filters import threshold_otsu
from sklearn.neighbors import KDTree

def create_uniform_grid(img, img_name, spacing=0.20064794):
    grid = pv.ImageData()
    grid.dimensions = img.shape
    grid.spacing = (spacing, spacing, spacing)
    grid.point_data[img_name] = img.flatten()
    values = img.flatten()
    thresh = threshold_otsu(grid.point_data[img_name])
    # hist, bins = np.histogram(grid.point_data[img_name], bins=50)
    # max_bin = np.argmax(hist)
    # thresh = bins[max_bin]

    return grid, values, thresh

def create_contour_from_grid(grid, threshold, values, method='marching_cubes'):
    mesh = grid.contour([threshold], values, method=method)
    return mesh

def create_kd_tree(points, leaf_size=2):
    tree = KDTree(points, leaf_size=leaf_size)
    return tree

def plot_mesh(mesh, tree):
    dist,ind = tree.query(mesh.points,k=3)
    pv.set_plot_theme('document')
    p = pv.Plotter()
    p.add_mesh(mesh,scalars=dist, smooth_shading=True,specular=0.5,cmap='fire',clim=[0,2.5])
    p.set_focus(mesh.center)
    p.camera_position = 'xy'
    p.camera.roll += 90
    p.show()

NAME1 = "Bone3"
NAME2 = "03947"

img1,img1_header = nrrd.read(f"/gpfs_projects/sriharsha.marupudi/EOS_Prints_STL_Rescale/{NAME1}_Rescaled.nrrd")
img1 = np.pad(img1,((1,1), (1,1), (1,1)), 'constant')
img2,img2_header = nrrd.read(f"/gpfs_projects/sriharsha.marupudi/Registered_Images/{NAME2}_Registration_uCT_20230302.nrrd")
img2 = np.pad(img2,((1,1), (1,1), (1,1)), 'constant')

grid, values, thresh = create_uniform_grid(img2, "img2")
mesh2 = create_contour_from_grid(grid, thresh, values)

grid1, values1, _ = create_uniform_grid(img1, "img1")
mesh1 = create_contour_from_grid(grid1, 1, values1)

tree = create_kd_tree(mesh2.points)

plot_mesh(mesh1, tree)