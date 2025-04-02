import h5py
import os
import numpy as np
from numpy.linalg import norm
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
import fipy as fp
import moviepy.editor as mp


mpl.rcParams.update(mpl.rcParamsDefault)
rc('text', usetex=True)


def plot_curve_with_opacity(ax, points, z_plane, num_segments=1, global_max=0, visibility=1):
    """
    Function plots a 3D curve from the given points projected onto the given 2D plane
    with transparency proportional to the distance to the plane 
    and line style corresponding to the position relative to the projection plane
    (solid in front of the plane and dotted behind it)

    **Input**

    -   ax   =   ax to plot on
    -   points    =   Points coordinates
    -   z_plane   =   z coordinate of the projection plane
    -   num_segments    =   Number of nodes (current function returns mesh with ~2-fold number of nodes)
    -   global_max    =   Whether to norm transparency on max distance to the projection plane (if 0)
                          or to norm it on max z difference in the given space (if 1)
    -   visibility   =   One-sided distance to the projection plane
                         relative to the distance norm (see global_max parameter description)
                         within which the curve is visible
    """

    points = np.array(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Normalize z-values to determine alpha
    if global_max:
        zmax = np.max(z) - np.min(z)
    else:
        zmax = np.max(np.abs(z - z_plane))
    
    if not zmax:
        alphas = np.ones_like(z)  # Handle case where all z are equal
    else:
        alphas = (1 - 0.95 * np.abs(z - z_plane) / (zmax * visibility)).clip(0, 1)
    
    # Plot each line segment with interpolated opacity
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i+1]
        alpha0, alpha1 = alphas[i], alphas[i+1]
        
        # Generate interpolated points along the line segment
        t = np.linspace(0, 1, num_segments + 1)
        x_interp = p0[0] + t * (p1[0] - p0[0])
        y_interp = p0[1] + t * (p1[1] - p0[1])
        z_interp = p0[2] + t * (p1[2] - p0[2]) - z_plane
        
        # Calculate interpolated alpha values
        alpha_interp = alpha0 + t * (alpha1 - alpha0)
        
        # Create segments between consecutive points
        points_interp = np.column_stack([x_interp, y_interp])
        segments = np.array([points_interp[:-1], points_interp[1:]]).transpose(1, 0, 2)
        
        # Calculate average alpha for each small segment
        alpha_avg = (alpha_interp[:-1] + alpha_interp[1:]) / 2
                                                 
        # Determine the position of the segments relative to the image plane
        sign = np.heaviside(z_interp[:-1] + z_interp[1:], 1)
        
        # Create RGBA colors (black with varying alpha)
        rgba = np.zeros((len(segments), 4))
        rgba[:, 3] = alpha_avg  # Set alpha channel
        
        # Add line collection to plot
        for j in range(num_segments):
            lc = LineCollection([segments[j]], colors=[rgba[j]], linewidths=1.0)
            if not sign[j]:
                lc.set(ls=':')
            ax.add_collection(lc)


def mesh_triang(mesh, sparse, mask_z, bin_r, sigma_chr, mode_cells=1):
    """
    Function triangulates the given mesh elements (specified with mask_z)
    and masks triangles corresponding to the chromatin

    **Input**

    -   mesh   =   FiPy mesh object
    -   sparse   =   Whether the chromatin is sparse
    -   mask_z   =   Mask of mesh elements to be used
    -   bin_r   =   Chromatin bins coordinates
    -   sigma_chr   =   Chromatin thickness (for dense chromatin model)
    -   model_cells   =   Whether to triangulate cell centers (if 1) or face centers (if 0)
    """
    
    if mode_cells:
        triang = tri.Triangulation(mesh.x.value[mask_z], mesh.y.value[mask_z])
        if not sparse:
            mask_b_1 = np.all(np.any(mesh.exteriorFaces.value[mesh.cellFaceIDs.data[:, mask_z]], axis=0)[triang.triangles], axis=1)
            mask_b_2 = (KDTree(bin_r).query(np.mean(mesh.cellCenters.value[:,mask_z].T[triang.triangles], axis=1))[0]<sigma_chr)
            triang.set_mask(mask_b_1 | mask_b_2)
    else:
        triang = tri.Triangulation(mesh.faceCenters.value[0], mesh.faceCenters.value[1])
        triang.set_mask(np.all(mesh.exteriorFaces.value[triang.triangles], axis=1))

    return triang


def z_scan(mesh, sparse, bin_r, out_f, spvar, thres, sigma_chr, num_segments=1, global_max=1, visibility=1, nframes=20, fps=4, out_format='mp4'):
    """
    Function scans a 3D space with a 2D plane
    and plots the given spatial variable and a 3D curve from the given points
    projected onto the plane at each plane position
    with transparency proportional to the distance to the plane 
    and line style corresponding to the position relative to the projection plane
    (solid in front of the plane and dotted behind it)

    **Input**

    -   mesh   =   FiPy mesh object
    -   sparse   =   Whether the chromatin is sparse
    -   bin_r   =   Chromatin bins coordinates
    -   out_f   =   Path to the output file without extension
    -   spvar   =   Spatial variable to plot
    -   thres   =   Max distance from plotted control volume center to the projection plane (fraction of the total z thickness)
    -   sigma_chr   =   Chromatin thickness (for dense chromatin model)
    -   num_segments    =   Number of nodes (current function returns mesh with ~2-fold number of nodes)
    -   global_max    =   Whether to norm transparency on max distance to the projection plane (if 0)
                          or to norm it on max z difference in the given space (if 1)
    -   visibility   =   One-sided distance to the projection plane
                         relative to the distance norm (see global_max parameter description)
                         within which the curve is visible
    -   nframes    =   Total frame number
    -   fps    =   Frames per second
    """
    
    file_paths = []
    for i, z_plane in tqdm(enumerate(np.linspace(mesh.z.value.min(),  mesh.z.value.max(), nframes))):
        file_paths.append(f'{out_f}_z_plane_{i}.jpg')
        mask_z = (abs(mesh.z.value - z_plane) / (mesh.z.value.max() - mesh.z.value.min()) < thres * (1 + d_chr/d_chr.max()))
        triang = mesh_triang(mesh, sparse, mask_z, bin_r, sigma_chr, mode_cells=1)
        
        fig, ax = plt.subplots(figsize=(8,5))
        cs = ax.tricontourf(triang,
                    spvar[mask_z],
                    levels=np.linspace(0, spvar.max(), 256),
                    cmap="Reds",
                    extend="both")
        cbar=fig.colorbar(cs)
        cbar.formatter.set_powerlimits((0, 0))
        plot_curve_with_opacity(ax, bin_r, z_plane=z_plane, num_segments=num_segments,
                                global_max=global_max, visibility=visibility)

        ax.set_xlim(np.min(mesh.x.value), np.max(mesh.x.value))
        ax.set_ylim(np.min(mesh.y.value), np.max(mesh.y.value))
        ax.set_aspect('equal', adjustable='box')
        fig.savefig(file_paths[-1], bbox_inches='tight', dpi=300, format='jpg')
        plt.close()
        
    clip = mp.ImageSequenceClip(file_paths, fps=fps)
    
    clip.write_videofile(f'{out_f}.{out_format}', fps=fps)
    clip.close()
    
    # delete individual images
    for f in file_paths:
        os.remove(f)