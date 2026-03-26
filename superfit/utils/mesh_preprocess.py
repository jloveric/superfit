import os
import numpy as np
import trimesh
import torch as th
import sys
from .mesh_sdf import renorm_target_sdf, get_target_cubvh, sdf_to_mesh
from .constants import MIN_VOLUME_LIMIT
from .logger import logger
import cc3d
import igl
import cubvh

P_INSIDE_THRESHOLD = 0.25
CD_THRESHOLD = 0.001
INFATE_AMOUNT = 0.02

def quick_sample_points(mesh, sketcher, n_points=10000):
    # Ensure normals exist (needed only if you want smooth normals)
    # Sample uniformly on the surface
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    points = th.from_numpy(points).float().to(sketcher.device)
    return points
    
def extract_mesh(cur_file, *, concatenate=True, process=False, fix_mirrors=True):
    """
    Return the scene as either a single mesh (concatenate=True)
    or a list of per-instance meshes with transforms baked in.

    - Uses `force='mesh'` for the single-mesh case (lets Trimesh bake transforms).
    - Uses `graph.to_flattened()` for the per-part case (bakes each node exactly once).
    """
    # Fast path: ask Trimesh to give you one world-baked mesh
    if os.path.isdir(cur_file):
        return extract_from_folder(cur_file, fix_mirrors=fix_mirrors, process=process)
    else:
        return extract_from_file(cur_file, concatenate=concatenate, fix_mirrors=fix_mirrors, process=process)

def extract_from_file(cur_file, concatenate=True, fix_mirrors=True, process=False):
    if concatenate:
        mesh = trimesh.load(cur_file, force='mesh', process=process)
        if isinstance(mesh, trimesh.Trimesh):
            if fix_mirrors:
                _fix_if_mirrored(mesh)
            return mesh
        # Fall back to manual flattening if needed

    # Manual per-node flattening (one mesh per instance)
    scene_or_mesh = trimesh.load(cur_file, force='scene', process=process)
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        # File already a single mesh
        if fix_mirrors:
            _fix_if_mirrored(scene_or_mesh)
        return scene_or_mesh if not concatenate else scene_or_mesh.copy()

    scene = scene_or_mesh
    if not getattr(scene, "geometry", None):
        # Nothing in scene; try direct mesh load
        mesh = trimesh.load(cur_file, process=process)
        if isinstance(mesh, trimesh.Trimesh) and fix_mirrors:
            _fix_if_mirrored(mesh)
        return mesh

    parts = []
    flat = scene.graph.to_flattened()  # {node_name: {'geometry': <name>, 'transform': 4x4}}
    for _, info in flat.items():
        geom_name = info.get("geometry")
        if geom_name is None:
            continue
        base = scene.geometry.get(geom_name)
        if not isinstance(base, trimesh.Trimesh):
            continue
        T = np.asarray(info["transform"], dtype=float)
        m = base.copy()
        m.apply_transform(T)
        if fix_mirrors:
            _fix_if_mirrored(m, T)  # use T to decide winding
        parts.append(m)

    if not parts:
        return trimesh.load(cur_file, process=process)

    if concatenate:
        # Merge geometry only; note: this cannot preserve multiple distinct materials.
        return trimesh.util.concatenate(parts)

    return parts


def _fix_if_mirrored(mesh: trimesh.Trimesh, T: np.ndarray | None = None):
    """
    If a negative-determinant transform was applied, triangle winding flips.
    Flip faces back to restore outward normals.
    If T is None, we skip determinant test (can’t know), but we do NOT touch faces.
    """
    if T is None:
        return
    T = np.asarray(T, dtype=float)
    if np.linalg.det(T[:3, :3]) < 0:
        mesh.faces = mesh.faces[:, [0, 2, 1]]
        # (Normals will be re-derived on export/render; no need to do heavy repairs here.)


def normalize_mesh(input_mesh):    
    verts = np.asarray(input_mesh.vertices)
    mesh_range = verts.max(axis=0) - verts.min(axis=0)
    mesh_range = mesh_range.max()
    input_mesh.apply_translation(-verts.min(axis=0))
    input_mesh.apply_scale(2.0/mesh_range)
    input_mesh.apply_translation((-1.0, -1.0, -1.0))
    input_mesh.apply_scale(0.9)
    return input_mesh

def normalize_to_unit_cube(mesh: trimesh.Trimesh, margin=0.9):
    """
    Normalize a mesh to fit within [-margin, margin]^3 (e.g., 90% of unit cube),
    centered at the origin. Done with a single transform.
    """
    # Compute bounding box
    bmin, bmax = mesh.bounds
    center = 0.5 * (bmin + bmax)
    extent = bmax - bmin
    max_extent = extent.max()

    # Build transform: translate to center, then scale
    scale = (2 * margin) / max_extent  # so it fits inside [-margin, margin]
    
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = -center                  # move center to origin
    T = np.dot(np.diag([scale, scale, scale, 1.0]), T)  # scale * translate

    mesh.apply_transform(T)
    return mesh


def normalize_to_unit_cube_with_transform(mesh: trimesh.Trimesh, margin=0.9):
    """
    Normalize mesh to fit within [-margin, margin]^3. Apply translation first, then scale.
    Returns (mesh, translation, scale) where translation is the 3D offset applied first
    (v' = scale * (v + translation)), and scale is the uniform scale factor.
    Useful for partwise fitting so the transform can be saved and applied inversely later.
    """
    bmin, bmax = mesh.bounds
    center = 0.5 * (bmin + bmax)
    extent = bmax - bmin
    max_extent = extent.max()
    scale = (2 * margin) / max_extent
    translation = -center  # applied first: v -> v + translation, then v -> scale * v
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = -center                  # move center to origin
    T = np.dot(np.diag([scale, scale, scale, 1.0]), T)  # scale * translate
    mesh = mesh.copy()
    mesh.apply_transform(T)
    return mesh, np.asarray(translation, dtype=np.float64), float(scale)


def get_mask_scaled_aabb(points: th.Tensor,
                                  mesh: trimesh.Trimesh,
                                  padding: float = 0.15) -> th.Tensor:
    """
    Return subset of points inside the scaled bounding box of a mesh.

    Args:
        points: (N, 3) th.Tensor of 3D coordinates
        mesh: trimesh.Trimesh object with .bounds
        scale: float, scaling factor applied about the bbox center

    Returns:    
        th.Tensor of shape (M,) where M <= N
    """

    bmin, bmax = mesh.bounds  # numpy arrays
    padded_bmin = bmin - padding
    padded_bmax = bmax + padding

    padded_bmin = th.tensor(padded_bmin, dtype=points.dtype, device=points.device)
    padded_bmax = th.tensor(padded_bmax, dtype=points.dtype, device=points.device)

    mask = (points >= padded_bmin) & (points <= padded_bmax)
    inside_mask = mask.all(dim=1)
    return inside_mask


def target_cleanup_v2(target_sdf, sketcher_3d, min_volume_limit=MIN_VOLUME_LIMIT):
    pos_target = target_sdf.clone()
    mesh_shape = (sketcher_3d.resolution, sketcher_3d.resolution, sketcher_3d.resolution)
    reshaped_mask = (pos_target<=0).reshape(*mesh_shape).cpu().numpy()
    labels_out, N = cc3d.connected_components(reshaped_mask, return_N=True, connectivity=6) # free
    # Remove dust. 
    vox_grid_size = np.prod(mesh_shape)
    logger.info(f"Found {N} parts in target")
    # Image statistics like voxel counts, bounding boxes, and centroids.
    stats = cc3d.statistics(labels_out)
    volume_measure = [x/vox_grid_size for x in stats['voxel_counts']]
    reject_index = [ind for ind, x in enumerate(volume_measure) if x < min_volume_limit]


    reshaped_labels_out = th.from_numpy(labels_out.reshape(-1)).to(sketcher_3d.device).long()
    reject_index = th.tensor(reject_index).to(sketcher_3d.device).long()
    num_labels = int(reshaped_labels_out.max().item()) + 1   # max label ID + 1
    label_mask = th.zeros(num_labels, dtype=th.bool, device=reshaped_labels_out.device)


    # Mark rejected labels in the lookup table
    label_mask[reject_index] = True

    # Now use gather to map labels_out → mask
    reject_mask = label_mask[reshaped_labels_out]   # (N,) bool mask (no broadcast)

    pos_target[reject_mask] = 1.0
    logger.info(f"Rejecting {len(reject_index)} parts by volume fraction")
    # for ind in reject_index:
    #     # we need to find nhbd sign. 
    #     pos_target[reshaped_labels_out==ind] = 1.0

    pos_target = renorm_target_sdf(pos_target, sketcher_3d)

    flipped_sdf = -pos_target.clone()

    reshaped_mask = (flipped_sdf<=0).reshape(*mesh_shape).cpu().numpy()
    labels_out, N = cc3d.connected_components(reshaped_mask, return_N=True, connectivity=6) # free
    # Remove dust. 
    vox_grid_size = np.prod(mesh_shape)
    logger.info(f"Found {N} parts in target")
    # Image statistics like voxel counts, bounding boxes, and centroids.
    stats = cc3d.statistics(labels_out)
    volume_measure = [x/vox_grid_size for x in stats['voxel_counts']]
    reject_index = [ind for ind, x in enumerate(volume_measure) if x < min_volume_limit]

    reshaped_labels_out = th.from_numpy(labels_out.reshape(-1)).to(sketcher_3d.device).long()
    reject_index = th.tensor(reject_index).to(sketcher_3d.device).long()
    num_labels = int(reshaped_labels_out.max().item()) + 1   # max label ID + 1
    label_mask = th.zeros(num_labels, dtype=th.bool, device=reshaped_labels_out.device)


    # Mark rejected labels in the lookup table
    label_mask[reject_index] = True

    # Now use gather to map labels_out → mask
    reject_mask = label_mask[reshaped_labels_out]   # (N,) bool mask (no broadcast)

    flipped_sdf[reject_mask] = 1.0
    logger.info(f"Rejecting {len(reject_index)} parts by volume fraction")
    # reshaped_labels_out = labels_out.reshape(-1)
    # print("rejecting", reject_index)
    # for ind in reject_index:
    #     # we need to find nhbd sign. 
    #     flipped_sdf[reshaped_labels_out==ind] = 1.0

    final_sdf = -flipped_sdf.clone()

    final_sdf = renorm_target_sdf(final_sdf, sketcher_3d)
    return final_sdf


def process_mesh_to_sdf(input_mesh_file, sketcher_3d, inflate=False, inflate_amount=INFATE_AMOUNT):
    """
    Process a mesh file through the full pipeline to generate SDF and mesh.
    
    Args:
        input_mesh_file: Path to the input mesh file
        sketcher_3d: Sketcher instance for SDF computation
        
    Returns:
        mesh: Processed trimesh.Trimesh object
        target_sdf: Processed SDF tensor
    """
    input_mesh = extract_mesh(input_mesh_file)
    input_mesh = normalize_to_unit_cube(input_mesh)
    target_sdf = get_target_cubvh(input_mesh, sketcher_3d, mode="raystab")
    # Inflate?
    if inflate:
        target_sdf = target_sdf - inflate_amount
    target_sdf = renorm_target_sdf(target_sdf, sketcher_3d)
    target_sdf = target_cleanup_v2(target_sdf, sketcher_3d)
    target_sdf = renorm_target_sdf(target_sdf, sketcher_3d)
    
    mesh = sdf_to_mesh(target_sdf, sketcher_3d)
    return mesh, target_sdf

def process_v2_inflate_mesh(input_mesh_file, sketcher_3d, inflate_amount=INFATE_AMOUNT):

    mesh1, target_sdf_1 = process_mesh_to_sdf(input_mesh_file, sketcher_3d, inflate=False, inflate_amount=inflate_amount)
    input_mesh = extract_mesh(input_mesh_file)
    input_mesh = normalize_to_unit_cube(input_mesh)
    target_sdf = get_target_cubvh(input_mesh, sketcher_3d, mode="raystab")
    target_sdf = target_sdf - inflate_amount
    target_sdf = renorm_target_sdf(target_sdf, sketcher_3d)
    out_sdf = target_cleanup_v2(target_sdf, sketcher_3d)
    out_sdf = renorm_target_sdf(out_sdf, sketcher_3d)

    # Make the orig inflated
    inflated_orig = target_sdf_1 - inflate_amount
    inflated_orig = renorm_target_sdf(inflated_orig, sketcher_3d)

    rem = th.maximum(out_sdf, -inflated_orig)
    rem = renorm_target_sdf(rem, sketcher_3d)
    final_sdf = th.minimum(target_sdf_1, rem)
    final_sdf = renorm_target_sdf(final_sdf, sketcher_3d)
    final_sdf = target_cleanup_v2(final_sdf, sketcher_3d)
    final_sdf = renorm_target_sdf(final_sdf, sketcher_3d)

    final_mesh = sdf_to_mesh(final_sdf, sketcher_3d)
    return final_mesh, final_sdf

def extract_from_folder(mesh_folder, fix_mirrors=True, process=False):
    # Assume all meshes are in the folder.
    mesh_list = [os.path.join(mesh_folder, x) for x in os.listdir(mesh_folder) if x.endswith(".obj")]
    all_meshes = []
    for mesh_file in mesh_list:
        mesh = trimesh.load(mesh_file, force='scene', process=process)
        if isinstance(mesh, trimesh.Trimesh):
            if fix_mirrors:
                _fix_if_mirrored(mesh)
        all_meshes.append(mesh)
    
    parts = []
    for scene in all_meshes:
        flat = scene.graph.to_flattened()  # {node_name: {'geometry': <name>, 'transform': 4x4}}
        for _, info in flat.items():
            geom_name = info.get("geometry")
            if geom_name is None:
                continue
            base = scene.geometry.get(geom_name)
            if not isinstance(base, trimesh.Trimesh):
                continue
            T = np.asarray(info["transform"], dtype=float)
            m = base.copy()
            m.apply_transform(T)
            if fix_mirrors:
                _fix_if_mirrored(m, T)  # use T to decide winding
            parts.append(m)

    return trimesh.util.concatenate(parts)

def cd_based_process_mesh_to_sdf(input_mesh_file, sketcher_3d, inflate=False, inflate_amount=INFATE_AMOUNT):
    # if its a folder do something else.
    input_mesh = extract_mesh(input_mesh_file)
    input_mesh = normalize_to_unit_cube(input_mesh)
    mesh_v1, target_sdf_v1 = process_mesh_to_sdf(input_mesh_file, sketcher_3d, inflate=inflate, inflate_amount=inflate_amount)
    if inflate:
        return mesh_v1, target_sdf_v1, 0.0
    do_inflate = False
    if mesh_v1.faces.shape[0] == 0:
        cd_avg = 100.0
        do_inflate = True
    else:
        pc_1 = quick_sample_points(input_mesh, sketcher_3d)
        pc_2 = quick_sample_points(mesh_v1, sketcher_3d)
        cd = th.cdist(pc_1, pc_2, p=2) ** 2
        cd_1 = th.min(cd, dim=1)[0]
        cd_2 = th.min(cd, dim=0)[0] 
        cd_avg = (th.mean(cd_1) + th.mean(cd_2)) / 2.0
        if cd_avg >= CD_THRESHOLD:
            do_inflate = True
    if do_inflate:
        mesh_v2, target_sdf_v2 = process_v2_inflate_mesh(input_mesh_file, sketcher_3d, inflate_amount=inflate_amount)
        return mesh_v2, target_sdf_v2, cd_avg
    else:
        return mesh_v1, target_sdf_v1, cd_avg


def winding_to_p_inside_torch(w: th.Tensor, t: float = 0.15, sigma: float = 0.20, eps: float = 1e-6):
    """
    w: Tensor of winding numbers (any shape), float32/float64, CPU or CUDA
    Returns:
      p_inside in [0,1], same shape as w
      conf in [0,1], same shape as w
    """
    a = w.abs()

    # nearest integer without torch.rint:
    # round(x) = floor(x + 0.5) for x>=0 (we use abs so it's >=0)
    k = (a + 0.5).floor()

    delta = (a - k).abs()

    # membership: outside ~0, inside-ish ~1 once a>0.5
    member = th.sigmoid((a - 0.5) / t)

    # confidence: 1 near integers, ~0 for fractional ambiguous values
    conf = th.exp(-0.5 * (delta / (sigma + eps)) ** 2)

    p = (member * conf).clamp(0.0, 1.0)
    return p, conf

def mesh_with_segments_to_part_targets(mesh, instance_ids):
    n_index = len(np.unique(instance_ids))
    part_targets = []
    for selected_index in range(n_index):

        face_mask = (instance_ids == selected_index)
        face_idx = np.nonzero(face_mask)[0]

        # returns a single Trimesh when append=Tru4
        new_mesh = mesh.submesh([face_idx], append=True, repair=False, only_watertight=False)

        # optionally avoid trimesh "processing" changing things:
        new_mesh.process(validate=True)
        # make mesh both sided
        part_targets.append(new_mesh)
    return part_targets

def open_mesh_to_closed_mesh(mesh, sketcher_3d):
    coords = sketcher_3d.get_base_coords()
    coords_np = coords.cpu().numpy()
    # Winding Numbers:
    V = np.asarray(mesh.vertices, dtype=np.float64, order="C")
    F = np.asarray(mesh.faces, dtype=np.int32, order="C")
    Q = np.asarray(coords_np, dtype=np.float64, order="C")
    winding_number = igl.fast_winding_number(V, F, coords_np)
    winding_th = th.from_numpy(winding_number).cuda()
    p_inside, _ = winding_to_p_inside_torch(winding_th)

    BVH = cubvh.cuBVH(mesh.vertices, mesh.faces) # build with numpy.ndarray/torch.Tensor
    # BVH = cubvh.cuBVH(mesh_data.vertices, mesh_data.faces) # build with numpy.ndarray/torch.Tensor
    distances, face_id, uvw = BVH.unsigned_distance(coords, return_uvw=False)

    signed_distances = distances * -th.sign(p_inside-P_INSIDE_THRESHOLD).float()
    new_mesh = sdf_to_mesh(signed_distances, sketcher_3d)
    return new_mesh

