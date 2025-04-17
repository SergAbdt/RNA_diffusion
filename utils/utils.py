import numpy as np
from numpy.linalg import norm
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.spatial import KDTree
import fipy as fp


def parse_chrom(chrom_f, chrom_size, space_scale, n_interp=10):
    """
    Function transforms 3D-chromatin coordinates from .pdb (ParticleChromo3D format)
    into a form suitable for making 2D plots.

    **Input**

    -   chrom_f   =   .pdb file with chromatin coordinates (ParticleChromo3D .pdb format)
    -   space_size    =   Size of the longest axis (x axis)
    -   n_interp   =   Number of spline points per chromatin segment
    """
    
    names = ['ATOM', 'num', 'type', 'alt_loc', 'aa', 'chain', 'seq_num', 'res_insert',
         'x', 'y', 'z', 'occupancy', 'tempfactor', 'seg_id', 'elt_symb', 'charge']
    colspecs = [(0, 4), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26),
                (26, 27), (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (72, 76),
                (76, 78), (78, 80)]
    bin_r = pd.read_fwf(chrom_f, names=names, colspecs=colspecs)\
                  .query('ATOM=="ATOM"')[['x', 'y', 'z']].to_numpy()
    
    bin_r = bin_r - np.mean(bin_r, axis=0)
    U, S, Vt = np.linalg.svd(bin_r)
    bin_r = bin_r @ Vt.T
    bin_r *= chrom_size / space_scale / np.max(np.abs(bin_r))

    dist = np.linalg.norm(bin_r[1:]-bin_r[:-1], axis=1)
    u_c = np.r_[0, np.cumsum(dist**0.5)]
    spl = make_interp_spline(u_c, bin_r, axis=0)
    uu = np.linspace(u_c[0], u_c[-1], len(dist)*n_interp + 1)
    bin_r_spline = spl(uu)

    return bin_r, bin_r_spline


def lin_interp(points, n):
    """
    Function linearly interpolates a curve defined by given points

    **Input**

    -   points   =   Points to be interpolated
    -   n    =   Distance to the vertices of the polygonal chain for each point
    """
    
    out = []
    for i in range(points.shape[0] - 1):
        for j in range(n):
            out.append(points[i] + j/(n+1) * (points[i+1] - points[i]))
    out.append(points[-1])

    return np.array(out)


def mindist(r, dist_to_bin, bin_r, bin_=None):
    """
    Function calculates min distance to the segments
    following the given vertices of a polygonal chain

    **Input**

    -   r   =   Points to which the distance is calculated (np.array)
    -   dist_to_bin    =   Distance to the vertices of the polygonal chain for each point
    -   bin_r    =   Vertices of the polygonal chain
    -   bin_    =   Indices of the polygonal chain vertices to be used
    """
    if isinstance(bin_, np.ndarray):
        bin_ = bin_[bin_>=0]
    else:
        bin_ = range(len(bin_r) - 1)
        
    d_chr = []
    for i in bin_:
        dot_f = np.dot(r - bin_r[i], bin_r[i + 1] - bin_r[i])
        dot_r = np.dot(r - bin_r[i + 1], bin_r[i] - bin_r[i + 1])
        if dot_f > 0 and dot_r > 0:
            d_chr.append(norm(np.cross(bin_r[i + 1] - bin_r[i], bin_r[i] - r)) /
                         norm(bin_r[i + 1] - bin_r[i]))
    
    d_chr = min(d_chr + [dist_to_bin])
    return d_chr


def dist_to_polygonal_chain_naive(r, bin_r):
    """
    Function calculates min distance to a polygonal chain

    **Input**

    -   r   =   Points to which the distance is calculated (np.array)
    -   bin_r    =   Vertices of the polygonal chain
    """
    tree = KDTree(bin_r)
    dist_to_bin = tree.query(r)[0]
    d_chr = np.vectorize(mindist, excluded=['bin_r'],
                         signature=f'({r.shape[1]}),()->()')(r, dist_to_bin, bin_r=bin_r)
    return d_chr


def process_closest_segments(b, l, k):
    """
    Function returns the vertices preceding the segments
    adjacent to the given vertices of a polygonal chain

    **Input**

    -   b   =   Indices of the given vertices of the polygonal chain
    -   l    =   Number of vertices of the polygonal chain
    -   k    =   Number of closest vertices to use
    """
    
    arr = np.setdiff1d(np.union1d(b, b - 1), [-1, l - 1])
    arr = np.pad(arr, (0, 2*k - len(arr)), constant_values=(-1))
    return arr


def dist_to_polygonal_chain_knn(r, bin_r, k):
    """
    Function calculates min distance to a polygonal chain
    as min distance to the segments adjacent to k closest vertices

    **Input**

    -   r   =   Points to which the distance is calculated (np.array)
    -   bin_r    =   Vertices of the polygonal chain
    -   k    =   Number of closest vertices to use
    """
    
    tree = KDTree(bin_r)
    bin_ = tree.query(r, k)
    if k == 1:
        dist_to_bin = bin_[0]
        bin_ = bin_[1].reshape(-1, 1)
    else:
        dist_to_bin = bin_[0].T[0]
        bin_ = bin_[1]
    bin_ = np.vectorize(process_closest_segments, excluded=['l', 'k'],
                        signature=f'({bin_.shape[1]})->({2*k})')\
           (bin_, l=len(bin_r), k=k)
    
    d_chr = np.vectorize(mindist, excluded=['bin_r'],
                         signature=f'({r.shape[1]}),(),({2*k})->()')\
            (r, dist_to_bin, bin_=bin_, bin_r=bin_r)
    
    return d_chr


def dist_to_chrom(mesh, sparse, bin_r_spline, sigma_chr, space_scale):
    """
    The function calculates distance to the chromatin
    for the mesh cellCenters

    **Input**

    -   mesh   =   FiPy mesh object
    -   sparse   =   Whether the chromatin is sparse
    -   bin_r    =   Interpolated chromatin bins
    -   sigma_chr   =   Chromatin thickness (for dense chromatin model)
    -   space_scale    =   Scaling factor of the convex hull of the chromatin bins
    """
    
    if not sparse:
        dist = np.linalg.norm(bin_r_spline[1:]-bin_r_spline[:-1], axis=1)
        ext_faces = mesh.faceCenters.value.T[mesh.exteriorFaces]
        chr_faces = KDTree(ext_faces[KDTree(bin_r_spline).query(ext_faces)[0] < space_scale*(sigma_chr**2 + max(dist)**2)**0.5])
        d_chr = fp.CellVariable(mesh=mesh, name=r'$d_{chr}$', value = chr_faces.query(mesh.cellCenters.value.T)[0] + sigma_chr)
    else:
        d_chr = fp.CellVariable(mesh=mesh, name=r'$d_{chr}$', value = dist_to_polygonal_chain_knn(mesh.cellCenters.value.T, bin_r_spline, 20))

    return d_chr


def sum_in_domain(mesh, var, mask):
    """
    The function calculates the sum of the values
    of a given spatial variable in a given domain

    **Input**

    -   mesh   =   FiPy mesh object
    -   var    =   Spatial variable
    -   mask    =   Mask of nodes from a given domain
    """
    
    sum_ = np.sum(mesh.cellVolumes[mask]*var.value[mask])
    return sum_


def av_in_domain(mesh, var, mask):
    """
    The function calculates the average of the values
    of a given spatial variable in a given domain

    **Input**

    -   mesh   =   FiPy mesh object
    -   var    =   Spatial variable
    -   mask    =   Mask of nodes from a given domain
    """
    
    av = sum_in_domain(mesh, var, mask)/np.sum(mesh.cellVolumes[mask])
    return av