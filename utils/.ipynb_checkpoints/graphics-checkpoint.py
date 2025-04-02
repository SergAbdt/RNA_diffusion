import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import h5py
import os
import re
import moviepy.editor as mp
import sys


sys.path.append(os.path.dirname(os.path.abspath('graphics.py')))

from utils.graphics_3d import *


mpl.rcParams.update(mpl.rcParamsDefault)
rc('text', usetex=True)


def plot_spvars(dim, mesh, sparse, spvar, variable_name, colormap, output_dir, steps=None, val_lim=None, bin_r=None, sigma_chr=None, plane_pos=0.5, thres=None, d_chr=None, num_segments=1, global_max=0, visibility=1):
    """
    Function to generate images of the spatial profiles of different cellVariables()

    **Input**

    -   dim   =   Dimension number
    -   mesh   =   FiPy mesh object
    -   sparse   =   Whether the chromatin is sparse
    -   spvar   =   Spatial variable to plot
    -   variable_name   =   Name of the output file
    -   output_dir   =   Path to the output directory
    -   steps   =   Current simulation step
    -   val_lim   =   Spatial variable value limits (values exceeding these will be clipped)
    -   bin_r   =   Chromatin bins coordinates
    -   sigma_chr   =   Chromatin thickness (for dense chromatin model)
    -   plane_pos   =   Projection plane position (fraction of the total z thickness)
    -   thres   =   Max distance from plotted control volume center to the projection plane (fraction of the total z thickness)
    -   d_chr   =   Distance to the chromatin array / FiPy CellVariable
    -   num_segments    =   Number of nodes (current function returns mesh with ~2-fold number of nodes)
    -   global_max    =   Whether to norm transparency on max distance to the projection plane (if 0)
                          or to norm it on max z difference in the given space (if 1)
    -   visibility   =   One-sided distance to the projection plane
                         relative to the distance norm (see global_max parameter description)
                         within which the curve is visible
    """

    name = spvar.name
    
    # create masked triangles
    if dim == 2:
        triang = tri.Triangulation(mesh.x.value, mesh.y.value)
        mask=np.any(mesh.exteriorFaces.value[mesh.cellFaceIDs.data], axis=0)
        triang.set_mask(np.all(mask[triang.triangles], axis=1))
    elif dim == 3:
        plane = mesh.z.value.min() + plane_pos * (mesh.z.value.max() - mesh.z.value.min())
        mask_z = (abs(mesh.z.value - plane) < thres * (1 + d_chr/d_chr.max()))
        triang = mesh_triang(mesh, sparse, mask_z, bin_r, sigma_chr, mode_cells=1)
        spvar = spvar[mask_z]

    if not sparse:
        triang.set_mask(np.any(np.isnan(spvar.value)[triang.triangles], axis=1) | triang.mask)
    else:
        triang.set_mask(np.any(np.isnan(spvar.value)[triang.triangles], axis=1))
    
    if val_lim is not None:
        max_val = val_lim[1]
        min_val = val_lim[0]
    elif spvar.value[np.isfinite(spvar.value)].size:
        max_val = spvar.value[np.isfinite(spvar.value)].max()
        min_val = spvar.value[np.isfinite(spvar.value)].min()
    else:
        max_val = None
        min_val = None

    if min_val != max_val:
        levels = np.linspace(min_val, max_val, 256)
    
    spvar.value[spvar.value==np.inf] = max_val
    spvar.value[spvar.value==-np.inf] = min_val

#    triang = tri.Triangulation(mesh.faceCenters.value[0], mesh.faceCenters.value[1])
#    triang.set_mask(np.all(mesh.exteriorFaces.value[triang.triangles], axis=1))
    
    fig, ax = plt.subplots()
    if spvar.value[np.isfinite(spvar.value)].size and min_val != max_val:
        cs = ax.tricontourf(
            triang,
            spvar.value,
            cmap = plt.cm.get_cmap(colormap),
            levels = levels,
            extend='both')
        fig.colorbar(cs)

        if dim == 3:
            plot_curve_with_opacity(ax, bin_r, z_plane=plane, num_segments=num_segments,
                                    global_max=global_max, visibility=visibility)
            
            ax.set_xlim(np.min(mesh.x.value[mask_z])*1.05, np.max(mesh.x.value[mask_z])*1.05)
            ax.set_ylim(np.min(mesh.y.value[mask_z])*1.05, np.max(mesh.y.value[mask_z])*1.05)
                            
#    if bin_r is not None:
#        plt.scatter(bin_r[:, 0], bin_r[:, 1], c='black', edgecolor='k', s=5)
#        ax.plot(bin_r[:, 0], bin_r[:, 1], c='black')
        
    try:
        ax.set_title(name)
    except:
        print("No name given for the spatial variable while plotting")
        raise
    
    ax.set_aspect('equal', adjustable='box')
    
    ext = '_{step}.png'.format(step=steps) if steps is not None else '.png'
    fig.savefig(fname=output_dir + '/' + variable_name + ext, dpi=300, format='png') # + 'Images/'
    # fig.savefig(fname=output_dir + '/Images/' + variable_name + '_{step}.svg'.format(step=steps),dpi=600,format='svg')
    # pkl.dump((fig,ax),file(output_dir + '/Images/' + variable_name +'_{step}.pickle'.format(step=steps),'w'))
    plt.close()

    
def write_movie_from_images(PATH, names, fps=3):
    """
    Function to write movies from png files

    **Input**

    -   PATH   =   Path to the directory containing the input files
    -   names   =   Input files prefixes
    -   fps    =   Frames per second
    """

    def key_funct(x):
        return int(x.split('_')[-1].rstrip('.png'))

    # # make directory
    # try:
    #     os.mkdir(os.path.join(PATH, 'movies'))
    # except:
    #     print("/movies directory already exists")
    
    for name in names:
        file_names = sorted(list((fn for fn in os.listdir(os.path.join(PATH, 'Images')) if re.match(fr'{name}.*\.png', fn))), key=key_funct)
        file_paths = [os.path.join(PATH, 'Images', f) for f in file_names]
        clip = mp.ImageSequenceClip(file_paths, fps=fps)
        clip.write_videofile(os.path.join(PATH, 'movies', f'{name}_scaled.mp4'), fps=fps)
        clip.close()

        # delete individual images
        for f in file_paths:
            os.remove(f)


def write_movie_from_hdf5(PATH, names, dim, mesh, sparse, fps=3, val_lim=None, bin_r=None, plane_pos=0.5, thres=None, d_chr=None, sigma_chr=None, num_segments=1, global_max=0, visibility=1):
    """
    Function to write movies from hdf5 files

    **Input**

    -   PATH   =   Path to the directory containing the input files
    -   names   =   Input files prefixes
    -   dim   =   Dimension number
    -   mesh   =   FiPy mesh object
    -   sparse   =   Whether the chromatin is sparse
    -   fps    =   Frames per second
    -   val_lim   =   Spatial variable value limits (values exceeding these will be clipped)
    -   bin_r   =   Chromatin bins coordinates
    -   plane_pos   =   Projection plane position (fraction of the total z thickness)
    -   thres   =   Max distance from plotted control volume center to the projection plane (fraction of the total z thickness)
    -   d_chr   =   Distance to the chromatin array / FiPy CellVariable
    -   sigma_chr   =   Chromatin thickness (for dense chromatin model)
    -   num_segments    =   Number of nodes (current function returns mesh with ~2-fold number of nodes)
    -   global_max    =   Whether to norm transparency on max distance to the projection plane (if 0)
                          or to norm it on max z difference in the given space (if 1)
    -   visibility   =   One-sided distance to the projection plane
                         relative to the distance norm (see global_max parameter description)
                         within which the curve is visible
    """

    def key_funct(x):
        return int(x.split('_')[-2])

    # make directory
    if not os.path.exists(os.path.join(PATH, 'movies')):
        os.mkdir(os.path.join(PATH, 'movies'))
    
    # create triangles
    if dim == 2:
        triang = tri.Triangulation(mesh.x.value, mesh.y.value)
        mask_z = np.full((mesh.x.value.shape[0]), True)
    elif dim == 3:
        plane = mesh.z.value.min() + plane_pos * (mesh.z.value.max() - mesh.z.value.min())
        mask_z = (abs(mesh.z.value - plane) < thres * (1 + d_chr/d_chr.max()))
        triang = mesh_triang(mesh, sparse, mask_z, bin_r, sigma_chr, mode_cells=1)
    
    with h5py.File(os.path.join(PATH, "spatial_variables_2.hdf5"), mode="r") as df_total:
    
        for name in names:
            
            df_ = df_total[name][:,mask_z]
            max_val = val_lim[name][1] if val_lim and val_lim[name][1] else df_[np.isfinite(df_)].max()
            min_val = val_lim[name][0] if val_lim and val_lim[name][0] else df_[np.isfinite(df_)].min()

            if len(df_total[name][:].shape) == 2:
                df_ = [df_]
            elif len(df_total[name][:].shape) == 3:
                df_ = [df_total[name][:,0,mask_z], df_total[name][:,1,mask_z]]
            
            for idx, df in enumerate(df_):
                
                for i in range(df.shape[0]):
                    
                    df_fin = df[i][np.isfinite(df[i])]
                    max_val_local = df_fin.max() if df_fin.size else max_val
                    min_val_local = df_fin.min() if df_fin.size else min_val
                    val_lims = [[min_val, max_val], [min_val_local, max_val_local]]
                    
                    # mask triangles
                    if not sparse:
                        triang.set_mask(np.any(np.isnan(df[i])[triang.triangles], axis=1) | triang.mask)
                    else:
                        triang.set_mask(np.any(np.isnan(df[i])[triang.triangles], axis=1))
                    
                    for m, mode in enumerate(['non-scaled', 'scaled']):
                        
                        df[i][df[i]==np.inf] = val_lims[m][1]
                        df[i][df[i]==-np.inf] = val_lims[m][0]
                        
                        if val_lims[m][0] < 0.0 and val_lims[m][1] and np.abs(val_lims[m][0]/val_lims[m][1]) > 0.3:
                            color_map = 'coolwarm'
                        else:
                            color_map = 'Reds'
                    
                        fig, ax = plt.subplots()
                        if df[i][np.isfinite(df[i])].size:
                            cs = ax.tricontourf(triang,
                                                df[i],
                                                levels=np.linspace(val_lims[m][0]*(1-1e-3*np.sign(val_lims[m][0])),
                                                                   val_lims[m][1]*(1+1e-3*np.sign(val_lims[m][1])),
                                                                   256),
                                                cmap=color_map,
                                                extend='both')
                            cbar=fig.colorbar(cs)
                            cbar=cbar.formatter.set_powerlimits((0, 0))

                        if dim == 3:
                            plot_curve_with_opacity(ax, bin_r, z_plane=plane, num_segments=num_segments,
                                                    global_max=global_max, visibility=visibility)

                            ax.set_xlim(np.min(mesh.x.value[mask_z])*1.05, np.max(mesh.x.value[mask_z])*1.05)
                            ax.set_ylim(np.min(mesh.y.value[mask_z])*1.05, np.max(mesh.y.value[mask_z])*1.05)
                            
                        ax.set_aspect('equal', adjustable='box')
                        ax.set_title(name)
    
                        fig.savefig(fname=PATH +'/movies/{n}_step_{step}_{mode}.png'.format(n=name, step=i, mode=mode),dpi=300,format='png')
                        plt.close(fig)

                for mode in ['non-scaled', 'scaled']:
                    file_names = sorted(list((fn for fn in os.listdir(os.path.join(PATH, 'movies')) if fn.endswith(f'{mode}.png'))), key=key_funct)
                    file_paths = [os.path.join(PATH, 'movies', f) for f in file_names]
                    clip = mp.ImageSequenceClip(file_paths, fps=fps)
                    clip.write_videofile(os.path.join(PATH, 'movies','{n}_{idx}_{mode}.mp4'.format(n=name, idx=idx, mode=mode)), fps=fps)
                    clip.close()
    
                    # delete individual images
                    for f in file_paths:
                        os.remove(f)