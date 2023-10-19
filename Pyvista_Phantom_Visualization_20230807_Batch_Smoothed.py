
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 00:42:53 2023
@author: sriharsha.marupudi
"""

import pyvista as pv 
from glob import glob 
import numpy as np  
import pickle
import os 
import trimesh 
import sys

def isWatertight(vertices, faces):
    # Check if mesh is watertight
    mesh = trimesh.Trimesh(vertices = vertices, faces = faces)
    return mesh.is_watertight

sys.path.append("/gpfs_projects/qian.cao/BoneBox/bonebox/utils")
import mesh # meshing utilities

out_dir = "/gpfs_projects/sriharsha.marupudi/phantom_generation_figs_20230804_simplified/"
os.makedirs(out_dir, exist_ok=True)

filepath = "/gpfs_projects/andrew.wang/20230712_poisson_model_variance3/"
Dir = glob(filepath + "*.pkl")

for txt in Dir:
    NAME = os.path.basename(txt).replace(".pkl", "")
    
    print(f"Processing {NAME}.")

    with open(filepath + NAME + ".pkl", 'rb') as file:
        data = pickle.load(file)
    
    volume = data['volume']
    params = data['params']
    
    # Convert the data to a NumPy array
    volume = np.array(volume)
    filenamePNG = os.path.join(out_dir, f"{NAME}.png")
    
    voxelSize = (0.025,) * 3
    STEPSIZE = 1
    smooth_iterations = 10
    volume[:,:,0] = 0
    volume[:,:,-1] = 0
    volume[0,:,:] = 0
    volume[-1,:,:] = 0
    volume[:,0,:] = 0
    volume[:,-1,:] = 0
    
    try:
        vertices, faces, normals, values = mesh.Voxel2SurfMesh(volume, voxelSize=voxelSize, step_size=STEPSIZE)
        vertices, faces = mesh.smoothSurfMesh(vertices, faces, iterations=smooth_iterations)
        # vertices, faces = mesh.simplifySurfMeshACVD(vertices, faces, target_fraction=0.10)
        
        surface_mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy())
        
        pv.set_plot_theme('document')
        p = pv.Plotter(off_screen=True)
        p.add_mesh(surface_mesh, color='white')
        # # p.add_axes_at_origin()
        p.show(screenshot=os.path.abspath(filenamePNG))  # Use absolute path
        
    except ValueError as e:
        print(f"Error processing {NAME}: {e}")
        continue  # Move to the next volume
