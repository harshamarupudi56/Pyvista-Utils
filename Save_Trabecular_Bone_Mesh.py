#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys 
sys.path.append("/home/qian.cao/projectchrono/chrono_build/build/bin")
import numpy as np
import nrrd
import os 

# meshing
from skimage import measure
import tetgen # tetrahedralization
import trimesh # general mesh ops
import pyvista as pv

from glob import glob 

def cropCubeFromCenter(img,length):
    
    x0,y0,z0 = np.array(img.shape)//2
    R = length//2
    
    return img[slice(x0-R,x0+R+1),
               slice(y0-R,y0+R+1),
               slice(z0-R,z0+R+1)]


def set_volume_bounds(volume, airValue=None, bounds = 1):
    # set boundaries of volume to airValue
    
    if airValue is None:
        airValue = np.min(volume)
        
    volume[:(bounds+1),:,:] = airValue
    volume[-(bounds+1):,:,:] = airValue
    volume[:,:(bounds+1),:] = airValue
    volume[:,-(bounds+1):,:] = airValue
    volume[:,:,:(bounds+1)] = airValue
    volume[:,:,-(bounds+1):] = airValue
    
    return volume

def filter_connected_volume(volume):
    # filter out components unconnected to the main bone structure
    # performs connected component analysis and preserves only the largest connected component
    
    labels = measure.label(volume,connectivity=1)
    values = np.unique(labels) # get all labels
    values.sort()
    values = values[1:] # discard zeros (background)
    num_voxels = [np.sum(labels==x) for x in values]
    largest_component_label = values[np.argmax(num_voxels)]
    
    vmin = np.min(volume)
    vmax = np.max(volume)
    
    volume_out = np.ones(volume.shape,dtype=volume.dtype) * vmin
    volume_out[labels==largest_component_label] = vmax
    
    return volume_out

def filter_connected_mesh(faces):
    # filter out components unconnected to the main bone structure
    pass

def Voxel2SurfMesh(volume, voxelSize=(1,1,1), origin=None, level=None, step_size=1, allow_degenerate=False):
    # Convert voxel image to surface
    
    if level == None:
        level = (np.max(volume))/2
    
    # vertices, faces, normals, values = \
    #     measure.marching_cubes_lewiner(volume = volume, level = level, spacing = voxelSize, \
    #                                    step_size = step_size, allow_degenerate = allow_degenerate)
    vertices, faces, normals, values = \
        measure.marching_cubes(volume = volume, level = level, spacing = voxelSize, \
                               step_size = step_size, allow_degenerate = allow_degenerate)
            
    return vertices, faces, normals, values


def Surf2TetMesh(vertices, faces, order=1, verbose=1, **tetkwargs):
    # Convert surface mesh to tetrahedra
    # https://github.com/pyvista/tetgen/blob/master/tetgen/pytetgen.py
    
    tet = tetgen.TetGen(vertices,faces)
    tet.tetrahedralize(order=order, verbose=verbose, **tetkwargs)
    

    grid = tet.grid
    # grid.plot(show_edges = True)
    
    nodes = tet.node
    elements = tet.elem
    
   
    mesh = pv.read(filenameSTL)  
    mesh1 = mesh.slice_along_axis(n=1, axis="y")
    # mesh.plot(show_edges = True)
    cells = grid.cells.reshape(-1, 5)[:, 1:]
    cell_center = grid.points[cells].mean(1)
    
    # extract cells below the 0 xy plane
    mask = cell_center[:, 0] < 1000
    cell_ind = mask.nonzero()[0]
    subgrid = grid.extract_cells(cell_ind)
    subgrid1 = mesh.slice_along_axis(n=1, axis="y")
    
 
    # from ansys.mapdl.reader import quality
    # cell_qual = quality(subgrid)
    # subgrid.plot(scalars=cell_qual, stitle='Quality', cmap='bwr', clim=[0, 1],
    #               flip_scalars=True, show_edges=True)
    
    return nodes, elements, tet
        
   

def smoothSurfMesh(vertices, faces, **trimeshkwargs):
    # smooths surface mesh using Mutable Diffusion Laplacian method
    
    mesh = trimesh.Trimesh(vertices = vertices, faces = faces)
    trimesh.smoothing.filter_mut_dif_laplacian(mesh, **trimeshkwargs)
    
    return mesh.vertices, mesh.faces

def simplifySurfMeshACVD(vertices, faces, target_fraction):
    # simplify surface mesh, use pyacvd
    
    import pyacvd
    
    Nfaces = faces.shape[0]
    target_count = round(Nfaces*target_fraction)
    
    mesh = trimesh.Trimesh(vertices = vertices, faces = faces)
    mesh = pv.wrap(mesh)
    
    clus = pyacvd.Clustering(mesh)
    clus.cluster(target_count)
    
    mesh = clus.create_mesh()
    
    # https://github.com/pyvista/pyvista/discussions/2268
    faces_as_array = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:]
    tmesh = trimesh.Trimesh(vertices = mesh.points, faces = faces_as_array)
    
    return tmesh.vertices, tmesh.faces

def repairSurfMesh(vertices, faces):
    import pymeshfix
    vclean, fclean = pymeshfix.clean_from_arrays(vertices, faces)
    return vclean, fclean

def isWatertight(vertices, faces):
    # Check if mesh is watertight
    mesh = trimesh.Trimesh(vertices = vertices, faces = faces)
    return mesh.is_watertight

 

if __name__ == "__main__":
    
  
    out_dir1 = "/gpfs_projects/sriharsha.marupudi/Segmentations_Otsu_L1_Mesh_Figs_20230804_Smoothed/"
    os.makedirs(out_dir1, exist_ok=True)
    
    
    ROIDir = "/gpfs_projects/sriharsha.marupudi/Segmentations_Otsu_L1/"

    ROINRRD = glob(ROIDir+"*.nrrd")


    for txt in ROINRRD:
        NAME = os.path.basename(txt).replace("Segmentation-grayscale-","").replace(".nrrd","")


        print(f"output {NAME}.")
    
          
        voxelSize = (0.025, 0.025, 0.025) # mm
        cubeShape = (120,120,120)
        plattenThicknessVoxels = 10 # voxels
        plattenThicknessMM = plattenThicknessVoxels * voxelSize[0] # mm
        STEPSIZE = 1 
        # cubeShape = (202, 202, 202)
        
        camera_position = [(33.77241683272833, 20.37339381595352, 4.05313061246571),
         (4.9999999813735485, 4.9999999813735485, 4.9999999813735485),
         (0.03299032706089477, -0.000185872956304527, 0.9994556537293985)]
        
        
        filenameNRRD = ROIDir+f"Segmentation-grayscale-{NAME}.nrrd"
        filenamePNG = out_dir1+f"{NAME}.png"
    
        # Elastic Modulus of a real bone ROI
        volume,header = nrrd.read(filenameNRRD)
        volume = cropCubeFromCenter(volume,cubeShape[0]) # crop the ROI to a 1.8 cm^3 volume
        # volume = addPlatten(volume, plattenThicknessVoxels)
        volume = set_volume_bounds(volume, airValue=None,bounds=2) # set edge voxels to zero
        volume = filter_connected_volume(volume) # connected components analysis
        
            
        # Finite Element
        vertices, faces, normals, values = Voxel2SurfMesh(volume, voxelSize=voxelSize, step_size=STEPSIZE)

        vertices, faces = smoothSurfMesh(vertices, faces, iterations=10)
        # vertices, faces = simplifySurfMeshACVD(vertices, faces, 0.125)
          
        # if not isWatertight(vertices, faces):
        #     vertices, faces = repairSurfMesh(vertices, faces)    
        #     print("Surface Mesh Repaired")
        #     print("Is watertight? " + str(isWatertight(vertices, faces)))
        # assert isWatertight(vertices, faces), "surface not watertight after repair"
        surface_mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy())
          
          
        pv.set_plot_theme('document')
        p = pv.Plotter(off_screen=True)
        p.add_mesh(surface_mesh,color='white')
        # # p.add_axes_at_origin()
        p.show(screenshot=filenamePNG)

           