import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull
import gmsh


def chrom_2d(points, radius, n_nodes, scale, sigma, mesh_f):
    """
    Function creates circular 2D mesh with exclusion zone with no nodes and edges corresponding to chromatin.
    Mesh density is uniform in the region beyond the sigma*scale from the exclusion zone and
    follows a power-law dependence with an exponent of -2 within the sigma*scale from the exclusion zone

    **Input**

    -   points   =   Chromatin bins' coordinates
    -   radius    =   Radius of mesh
    -   n_nodes    =   Number of nodes (current function returns mesh with ~2-fold number of nodes)
    -   scale   =   Distance from the exclusion zone in sigma units within which mesh density
                    follows a power-law dependence with an exponent of -2
    -   sigma   =   Chromatin thickness (half-width of the exclusion zone with no nodes and edges)
    -   mesh_f    =   Path to the output .msh2 file
    """
    
    gmsh.initialize()
    gmsh.model.add("mesh")
    
    # Create main domain (large circle)
    main_circle = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
    
    # Create original spline
    p_tags = []
    for i, p in enumerate(points):
        tag = 1000 + i
        gmsh.model.occ.addPoint(*p, 0, 1, tag)
        p_tags.append(tag)
    spline_tag = 1000
    gmsh.model.occ.addSpline(p_tags, spline_tag)
    
    #beg_circle = gmsh.model.occ.addDisk(*points[0], 0, 0.4*sigma*2, 0.4*sigma*2)
    #end_circle = gmsh.model.occ.addDisk(*points[-1], 0, 0.4*sigma*2, 0.4*sigma*2)
    
    # Calculate tangents for offset points
    tangents = []
    for i in range(len(points)):
        if i == 0:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
        elif i == len(points) - 1:
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
        else:
            dx_prev = points[i][0] - points[i-1][0]
            dy_prev = points[i][1] - points[i-1][1]
            dx_next = points[i+1][0] - points[i][0]
            dy_next = points[i+1][1] - points[i][1]
            dx = (dx_prev + dx_next) / 2
            dy = (dy_prev + dy_next) / 2
        length = np.hypot(dx, dy)
        if length == 0:
            tx, ty = 0, 0
        else:
            tx = dx / length
            ty = dy / length
        tangents.append((tx, ty))
    
    # Create offset points for left and right curves
    left_tags, right_tags = [], []
    for i in range(len(points)):
        x, y = points[i]
        tx, ty = tangents[i]
        nx, ny = -ty, tx  # Perpendicular to tangent
        
        # Left offset
        x_left = x + nx * sigma
        y_left = y + ny * sigma
        left_tag = 2000 + i
        gmsh.model.occ.addPoint(x_left, y_left, 0, 1, left_tag)
        left_tags.append(left_tag)
        
        # Right offset
        x_right = x - nx * sigma
        y_right = y - ny * sigma
        right_tag = 3000 + i
        gmsh.model.occ.addPoint(x_right, y_right, 0, 1, right_tag)
        right_tags.append(right_tag)
    
    # Create splines for left and right curves
    left_spline = gmsh.model.occ.addSpline(left_tags, 2000)
    right_spline = gmsh.model.occ.addSpline(right_tags, 3000)
    
    # Connect start and end points
    line_start = gmsh.model.occ.addLine(right_tags[0], left_tags[0], 4000)
    line_end = gmsh.model.occ.addLine(left_tags[-1], right_tags[-1], 4001)
    
    # Form curve loop (left, end_line, reversed right, start_line) and subtract resulting surface from main domain
    curve_loop = gmsh.model.occ.addCurveLoop([left_spline, line_end, -right_spline, line_start], 5000)
    surface = gmsh.model.occ.addPlaneSurface([curve_loop], 6000)
    cut_result = gmsh.model.occ.cut([(2, main_circle)], [(2, surface)])#, (2, beg_circle), (2, end_circle)])
    gmsh.model.occ.synchronize()

    # Evaluate min mesh size for generating required number of nodes
    length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    dense_area = 2*(scale+1)*sigma*(length+2*(scale+1)*sigma) - 2*sigma*(length+2*sigma)
    light_area = np.pi*radius**2-dense_area
    min_size=(dense_area/(n_nodes*scale**2)+light_area/(n_nodes*scale**4))**0.5
    
    # Configure mesh size field (graded from exclusion_radius)
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", [spline_tag])
    gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 100)
    
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", min_size)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", min_size*scale**2)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", sigma)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", sigma*(scale+1))
    
    # Remove original spline for cleaner mesh
    gmsh.model.occ.remove([(1, spline_tag)])

    # Generate and save the mesh
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.model.mesh.generate(2)
    gmsh.write(mesh_f)
    
    gmsh.finalize()


def gmsh_add_line_if_not_exists(pa, pb, line_check):
    """
    Function creates a GMSH line if it does not already exist

    **Input**

    -   pa   =   GMSH tag of the first point
    -   pb    =   GMSH tag of the second point
    -   line_check    =   List of lines already created
    """
    
    line_f = line_check.get((pa, pb), 0)
    line_r = line_check.get((pb, pa), 0)
    if line_f:
        line = line_f
    elif line_r:
        line = -line_r
    else:
        line = gmsh.model.occ.addLine(pa, pb)
        line_check[(pa, pb)] = line
    return line


def gmsh_convex_hull_3d(points, scale=1.2):
    """
    Function creates GMSH entities for the enlarged convex hull of the given points

    **Input**

    -   points   =   Points coordinates
    -   scale   =   Convex hull scale factor
    """
    
    # Compute the convex hull
    hull = ConvexHull(points)
    
    # Extract hull vertices and original points
    hull_vertex_indices = hull.vertices
    original_hull_pts = points[hull_vertex_indices]
    
    # Calculate centroid of the original hull
    centroid = np.mean(original_hull_pts, axis=0)
    
    # Scale points
    scaled_hull_pts = centroid + (original_hull_pts - centroid) * scale
    
    # Create a mapping from original indices to scaled points
    vertex_index_map = {vid: i for i, vid in enumerate(hull_vertex_indices)}
    
    # Add scaled points to the geometry
    scaled_geom_pts = [gmsh.model.occ.add_point(*pt) for pt in scaled_hull_pts]

    surfaces = []
    line_check={}
    for simplex in hull.simplices:
        a, b, c = simplex

        # Correct normal orientation
        pa, pb, pc = points[a], points[b], points[c]
        tri_centroid = (pa + pb + pc) / 3
        v1 = pb - pa
        v2 = pc - pa
        normal = np.cross(v1, v2)
        direction = centroid - tri_centroid

        if np.dot(normal, direction) > 0:
            a, b, c = a, c, b  # Reverse order

        # Get indices in scaled points
        ia = vertex_index_map[a]
        ib = vertex_index_map[b]
        ic = vertex_index_map[c]

        # Create triangle surface
        lab = gmsh_add_line_if_not_exists(scaled_geom_pts[ia], scaled_geom_pts[ib], line_check)
        lbc = gmsh_add_line_if_not_exists(scaled_geom_pts[ib], scaled_geom_pts[ic], line_check)
        lca = gmsh_add_line_if_not_exists(scaled_geom_pts[ic], scaled_geom_pts[ia], line_check)
        
        triangle = gmsh.model.occ.addCurveLoop([lab, lbc, lca])
        surfaces.append(gmsh.model.occ.addPlaneSurface([triangle]))

    # Create volume from surfaces
    surface_loop = gmsh.model.occ.add_surface_loop(surfaces)
    space = gmsh.model.occ.add_volume([surface_loop])
    gmsh.model.occ.synchronize()

    return space, hull.volume * scale


def mesh_min_size_chrom_3d(points, n_nodes, sigma, scale, sp_vol):
    '''
    (!!! Does not work properly !!!)
    (Because the last expression comes from the mesh density integral for 2d case)
    Function evaluates min mesh size for generating required number of nodes
    The volume of the region with variable mesh density is approximated
    as if the chromatin were a straight line
    (there is a positive bias up to (pi-1)*100%)

    **Input**

    -   points   =   Points coordinates
    -   n_nodes   =   Conves hull scale factor
    -   sigma   =   Chromatin thickness (half-width of the exclusion zone with no nodes and edges)
    -   scale   =   Distance from the exclusion zone in sigma units within which mesh density
                    follows a power-law dependence with an exponent of -2
    -   sp_vol   =   Points coordinates
    '''
    
    length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    dense_vol = np.pi*(sigma*(scale+1))**2 * (length+2*(scale+1)*sigma) - np.pi*sigma**2 * (length+2*sigma)
    dense_vol = np.pi*sigma**2
    light_vol = sp_vol-dense_vol
    min_size=(dense_vol/(n_nodes*scale**2)+light_vol/(n_nodes*scale**4))**0.5
    
    return min_size


def split_points_into_sets(points, sigma):
    """
    Function splits the given points into sets
    each of which can be interpolated using a GMSH spline
    which creates a non-self-intersecting GMSH pipe.
    This is a sloppy DeepSeek-R1 O(N^2) implementation (still impressive that it works)
    of the original O(N) algorithm, but it's still not a bottleneck so it's bearable

    **Input**

    -   points   =   Points coordinates
    -   sigma   =   Pipe thickness
    """
    
    n = len(points)
    if n < 3:
        return {'spline_sets': [], 'sphere_cylinder': list(range(n))}
    
    # Compute angles between consecutive segments for points 1 to n-2 (0-based)
    angles = []
    for i in range(1, n-1):
        vec_prev = np.array(points[i]) - np.array(points[i-1])
        vec_next = np.array(points[i+1]) - np.array(points[i])
        norm_prev = norm(vec_prev)
        norm_next = norm(vec_next)
        if norm_prev == 0 or norm_next == 0:
            # Handle zero-length segments (angle is 0)
            angles.append(0.0)
            continue
        cos_theta = np.dot(vec_prev, vec_next) / (norm_prev * norm_next)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        angles.append(theta)
    
    # Compute prefix sum of angles
    prefix_sum = [0.0]
    for angle in angles:
        prefix_sum.append(prefix_sum[-1] + angle)
    
    spline_sets = []
    sphere_cylinder = set()
    i = 0
    
    while i < n:
        j = i + 2
        while j < n:
            valid = True
            if j != i + 1:
                vec_prev = np.array(points[j-1]) - np.array(points[j-2])
                vec_next = np.array(points[j]) - np.array(points[j-1])
                norm_prev = norm(vec_prev)
                vec_next = norm(vec_next)
                thres = 2*sigma*np.sin(angles[j-2]) # 1.3
            if j != i + 1 and (norm_prev < thres or norm_next < thres):
                # print(i, j, norm_prev, vec_next, thres)
                break
            for k in range(i, j-1):
                cum_angle = prefix_sum[j-1] - prefix_sum[k]
                if cum_angle >= np.pi / 2:
                    point_k = np.array(points[k])
                    point_j = np.array(points[j])
                    dist = np.linalg.norm(point_k - point_j)
                    if dist <= 2.5 * sigma: # 2
                        # print(i, j, k, dist)
                        valid = False
                        break
            if valid:
                j += 1
            else:
                break
        # Determine if the current segment is a valid spline set
        if j - i >= 3:
            spline_sets.append(list(range(i, j)))
            i = j - 1
        else:
            sphere_cylinder.add(i)
            i = j - 1
    
    # Convert the set to a sorted list
    sphere_cylinder = sorted(sphere_cylinder)
    
    return {
        'spline_sets': spline_sets,
        'sphere_cylinder': sphere_cylinder
    }


def gmsh_spline_with_volume_3d(points, sigma):
    """
    Function interpolates the given points
    with a GMSH spline and generates a pipe from it

    **Input**

    -   points   =   Points coordinates
    -   sigma   =   Pipe thickness
    """
    
    # Create original spline
    p_tags = []
    # l_tags = []
    for i in range(len(points)):
        p_tags.append(gmsh.model.occ.addPoint(*points[i], 0))
        # if i:
        #     l_tags.append(gmsh.model.occ.addLine(p_tags[-2], p_tags[-1]))
    spline = gmsh.model.occ.addSpline(p_tags)
    
    #beg_circle = gmsh.model.occ.addDisk(*points[0], 0, 0.4*sigma*2, 0.4*sigma*2)
    #end_circle = gmsh.model.occ.addDisk(*points[-1], 0, 0.4*sigma*2, 0.4*sigma*2)

    # A wire is like a curve loop, but open:
    #wire = gmsh.model.occ.addWire(l_tags)
    wire = gmsh.model.occ.addWire([spline])
    
    # We define the shape we would like to extrude along the spline (a circle):
    disk = gmsh.model.occ.addDisk(*points[0], sigma, sigma)
    
    # Compute the axis of rotation (cross product of spline tangent and circle normal)
    spline_tan = points[1] - points[0]
    rotation_axis = np.cross(spline_tan, [0, 0, 1])
    rotation_axis /= np.linalg.norm(rotation_axis)
    
    # We use the dot product between the line vector and the z-axis to get the rotation angle
    angle = np.arcsin(np.abs(spline_tan[2]) / np.linalg.norm(spline_tan)) - np.pi / 2
    gmsh.model.occ.rotate([(2, disk)], *points[0], *rotation_axis, angle)
    
    # We extrude the disk along the spline to create a pipe (other sweeping types
    # can be specified; try e.g. 'Frenet' instead of 'DiscreteTrihedron'):
    chrom = gmsh.model.occ.addPipe([(2, disk)], wire, 'DiscreteTrihedron')
    gmsh.model.occ.remove([(1, spline)])
    gmsh.model.occ.remove([(2, disk)])

    return chrom


def chrom_3d(points, min_size, scale, sigma, mesh_f, space_scale=1.2, spheres=1, cylinders=1, splines=1):
    """
    Function creates circular 3D mesh with exclusion zone with no nodes and edges corresponding to chromatin.
    Mesh density is uniform in the region beyond the sigma*scale from the exclusion zone and
    follows a power-law dependence with an exponent of -2 within the sigma*scale from the exclusion zone

    **Input**

    -   points   =   Chromatin bins' coordinates
    -   sigma   =   Chromatin thickness (half-width of the exclusion zone with no nodes and edges)
    -   min_size    =   Mesh minimum density
    -   n_nodes    =   (Temporarily non-functional) Number of nodes (current function returns mesh with ~2-fold number of nodes)
    -   scale   =   Distance from the exclusion zone in sigma units within which mesh density
                    follows a power-law dependence with an exponent of -2
    -   output_dir    =   Output dir for mesh .msh2 file
    -   space_scale    =   Scaling factor of the convex hull of the chromatin bins
    -   spheres   =   Whether to generate spheres
    -   cylinders   =   Whether to generate cylinders
    -   splines   =   Whether to generate splines
                      (GMSH does not typically handle cases where there are more than 100 bins in this mode)
    """
    
    gmsh.initialize()
    gmsh.model.add("mesh")
    # gmsh.option.setNumber("General.Verbosity", 3)
    gmsh.option.setNumber("General.NumThreads", 16)
    gmsh.option.setNumber("Geometry.NumSubEdges", 1000)
    
    # Create main domain (scaled convex hull)
    space, sp_vol = gmsh_convex_hull_3d(points, scale=space_scale)
    space = [(3, space)]

    # Create chromatin
    spline_sets = []
    if splines:
        sets = split_points_into_sets(points, sigma)
        spline_sets = sets['spline_sets']
        sphere_cylinder = sets['sphere_cylinder']
    else:
        sphere_cylinder = range(len(points))

    # splines_ = []
    chrom = []

    for spline in spline_sets:
        # space = gmsh.model.occ.cut(space, [(3, gmsh.model.occ.addSphere(*points[spline[0]], 1.005*sigma))])[0]
        chrom.extend(gmsh_spline_with_volume_3d(points[spline[0]:(spline[-1]+1)], sigma))
        # space = gmsh.model.occ.cut(space, gmsh_spline_with_volume_3d(points[spline[0]:(spline[-1]+1)], sigma))[0]

    # for spline in spline_sets:
    #     chrom.append((3, gmsh.model.occ.addSphere(*points[spline[0]], 1.005*sigma)))
    
    for i in sphere_cylinder:
        if spheres:
            chrom.append((3, gmsh.model.occ.addSphere(*points[i], 1.005*sigma)))
            # space = gmsh.model.occ.cut(space, [(3, gmsh.model.occ.addSphere(*points[i], 1.005*sigma))])[0]
    
    dist = norm(points[1:]-points[:-1], axis=1)
    for i in sphere_cylinder:
        if i != len(points) - 1 and cylinders: # and dist[i] > (4*min_size*(2*sigma - min_size))**0.5/2:
            chrom.append((3, gmsh.model.occ.addCylinder(*points[i], *(points[i+1]-points[i]), sigma)))
            # space = gmsh.model.occ.cut(space, [(3, gmsh.model.occ.addCylinder(*points[i], *(points[i+1]-points[i]), sigma))])[0]

    chrom = gmsh.model.occ.fuse(chrom, chrom)[0]
    # print(gmsh.model.occ.getEntities(dim=3))
    space = gmsh.model.occ.cut(space, chrom)[0]
    # if splines_:
    #     space = gmsh.model.occ.cut(space, splines_)[0]
    gmsh.model.occ.synchronize()
    
    # (Temporarily non-functional) Evaluate min mesh size for generating required number of nodes
    # min_size=mesh_min_size_chrom_3d(points, n_nodes, sigma, scale, sp_vol)

    # Get chromatin surface
    s_tags=[s[1] for s in gmsh.model.occ.getEntities(dim=2) if gmsh.model.getType(*s) != 'Plane']
    
    # Configure mesh size field (graded from exclusion_radius)
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "FacesList", s_tags)
    
    # Use for large distances between adjacent bins
    # gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 100)
    
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", min_size)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", min_size*scale**2)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", sigma*scale)
    
    # Remove entities for cleaner mesh
    # gmsh.model.occ.remove(gmsh.model.occ.getEntities(dim=1))
    # gmsh.model.occ.remove(gmsh.model.occ.getEntities(dim=2))
    # gmsh.model.occ.synchronize()
    
    # Generate and save the mesh
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    # This may resolve some BOPAlgo alerts and related errors
    # gmsh.model.mesh.generate(2)
    # gmsh.model.mesh.refine()
    
    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_f)
    
    gmsh.finalize()


def chrom_3d_sparse(points, min_size, scale, sigma, mesh_f, space_scale=1.2):
    """
    Function creates circular 3D mesh with exclusion zone with no nodes and edges corresponding to chromatin.
    Mesh density is uniform in the region beyond the sigma*scale from the exclusion zone and
    follows a power-law dependence with an exponent of -2 within the sigma*scale from the exclusion zone

    **Input**

    -   points   =   Chromatin bins' coordinates
    -   sigma   =   Chromatin thickness (half-width of the exclusion zone with no nodes and edges)
    -   min_size    =   Mesh minimum density
    -   n_nodes    =   (Temporarily non-functional) Number of nodes (current function returns mesh with ~2-fold number of nodes)
    -   scale   =   Distance from the exclusion zone in sigma units within which mesh density
                    follows a power-law dependence with an exponent of -2
    -   output_dir    =   Output dir for mesh .msh2 file
    -   space_scale    =   Scaling factor of the convex hull of the chromatin bins
    """
    
    gmsh.initialize()
    gmsh.model.add("mesh")
    # gmsh.option.setNumber("General.Verbosity", 3)
    gmsh.option.setNumber("General.NumThreads", 16)
    gmsh.option.setNumber("Geometry.NumSubEdges", 1000)
    
    # Create main domain (scaled convex hull)
    space, sp_vol = gmsh_convex_hull_3d(points, scale=space_scale)
    space = [(3, space)]

    # Create chromatin
    p_tags = []
    l_tags = []
    for i in range(len(points)):
        p_tags.append(gmsh.model.occ.addPoint(*points[i], 0))
        if i:
            l_tags.append(gmsh.model.occ.addLine(p_tags[-2], p_tags[-1]))

    gmsh.model.occ.synchronize()
    
    # Configure mesh size field (graded from exclusion_radius)
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", l_tags)
    
    # Use for large distances between adjacent bins
    # gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 100)
    
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", min_size)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", min_size*scale**2)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", sigma*scale)
    
    # Remove entities for cleaner mesh
    # gmsh.model.occ.remove(gmsh.model.occ.getEntities(dim=1))
    # gmsh.model.occ.remove(gmsh.model.occ.getEntities(dim=2))
    # gmsh.model.occ.synchronize()
    
    # Generate and save the mesh
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 1)

    # This may resolve some BOPAlgo alerts and related errors
    # gmsh.model.mesh.generate(2)
    # gmsh.model.mesh.refine()
    
    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_f)
    
    gmsh.finalize()