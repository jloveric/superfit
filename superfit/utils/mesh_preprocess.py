import os
import numpy as np
import trimesh
import torch as th
import sys
from .mesh_sdf import renorm_target_sdf
from .constants import MIN_VOLUME_LIMIT
import cc3d

    
def extract_mesh(cur_file, *, concatenate=True, process=False, fix_mirrors=True):
    """
    Return the scene as either a single mesh (concatenate=True)
    or a list of per-instance meshes with transforms baked in.

    - Uses `force='mesh'` for the single-mesh case (lets Trimesh bake transforms).
    - Uses `graph.to_flattened()` for the per-part case (bakes each node exactly once).
    """
    # Fast path: ask Trimesh to give you one world-baked mesh
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

# def normalize_to_unit_cube(mesh: trimesh.Trimesh, margin=0.9):
#     """
#     Move mesh so its bbox min goes to (0,0,0), scale to fit in [-1,1]^3,
#     translate to (-1,-1,-1), then apply a margin. Done with ONE transform.
#     """
#     # Use mesh.bounds for fast bbox
#     bmin, bmax = mesh.bounds  # shape (3,)
#     extent = (bmax - bmin)
#     mesh_range = extent.max()
#     print("max range", mesh_range)

#     # Build transforms (right-multiply order)
#     def T_translate(t):
#         T = np.eye(4, dtype=np.float64)
#         T[:3, 3] = t
#         return T

#     def T_scale(s):
#         T = np.eye(4, dtype=np.float64)
#         T[0, 0] = T[1, 1] = T[2, 2] = s
#         return T

#     T = np.eye(4, dtype=np.float64)
#     T = T @ T_translate(-bmin)                 # shift bbox min to origin
#     T =  T_scale(2/mesh_range)  @ T       # scale to ~[-1,1] after next translation
#     T = T_translate((-1.0, -1.0, -1.0)) @T   # move to corner of [-1,1]^3
#     T =  T_scale(margin) @ T                   # shrink a bit (0.9)

#     mesh.apply_transform(T)  # single cache invalidation
#     return mesh
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
    # bmin, bmax = mesh.bounds  # (2, 3)
    # center = (bmin + bmax) / 2.0
    # extent = (bmax - bmin) * scale / 2.0

    # scaled_bmin = center - extent
    # scaled_bmax = center + extent

    # scaled_bmin = th.tensor(scaled_bmin, dtype=points.dtype, device=points.device)
    # scaled_bmax = th.tensor(scaled_bmax, dtype=points.dtype, device=points.device)

    # mask = (points >= scaled_bmin) & (points <= scaled_bmax)
    # inside_mask = mask.all(dim=1)

    bmin, bmax = mesh.bounds  # numpy arrays
    padded_bmin = bmin - padding
    padded_bmax = bmax + padding

    padded_bmin = th.tensor(padded_bmin, dtype=points.dtype, device=points.device)
    padded_bmax = th.tensor(padded_bmax, dtype=points.dtype, device=points.device)

    mask = (points >= padded_bmin) & (points <= padded_bmax)
    inside_mask = mask.all(dim=1)
    return inside_mask


def toy_4kload_and_process_mesh(mesh_file):
    # Load raw mesh (don't let trimesh auto-process)
    mesh = trimesh.load(mesh_file, process=False)

    # If the file contains multiple geometries (Scene), merge them
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(g for g in mesh.geometry.values())
        )

    # Ensure the mesh is watertight and triangular
    if not mesh.is_watertight:
        print("Warning: mesh is not watertight")

    # If mesh is not triangulated, force triangulation
    if not mesh.is_winding_consistent:
        mesh = mesh.copy()
        mesh.fix_normals()  # make consistent before triangulation
    # mesh = mesh.as_triangles()

    # --- Normal estimation ---

    # Compute vertex normals if missing
    # if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
    #     mesh.vertex_normals = mesh.compute_vertex_normals()

    # Compute face normals (trimesh lazily computes these)
    face_normals = mesh.face_normals  # triggers computation if needed

    print(f"Loaded mesh: {mesh_file}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces:    {len(mesh.faces)}")

    return mesh

def target_cleanup_v2(target_sdf, sketcher_3d, min_volume_limit=MIN_VOLUME_LIMIT):
    pos_target = target_sdf.clone()
    mesh_shape = (sketcher_3d.resolution, sketcher_3d.resolution, sketcher_3d.resolution)
    reshaped_mask = (pos_target<=0).reshape(*mesh_shape).cpu().numpy()
    labels_out, N = cc3d.connected_components(reshaped_mask, return_N=True, connectivity=6) # free
    # Remove dust. 
    vox_grid_size = np.prod(mesh_shape)
    print("Found", N, "parts in target")
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
    print(f"rejecting {len(reject_index)} parts by volume fraction")
    # for ind in reject_index:
    #     # we need to find nhbd sign. 
    #     pos_target[reshaped_labels_out==ind] = 1.0

    pos_target = renorm_target_sdf(pos_target, sketcher_3d)

    flipped_sdf = -pos_target.clone()

    reshaped_mask = (flipped_sdf<=0).reshape(*mesh_shape).cpu().numpy()
    labels_out, N = cc3d.connected_components(reshaped_mask, return_N=True, connectivity=6) # free
    # Remove dust. 
    vox_grid_size = np.prod(mesh_shape)
    print("Found", N, "parts in target")
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
    print(f"rejecting {len(reject_index)} parts by volume fraction")
    # reshaped_labels_out = labels_out.reshape(-1)
    # print("rejecting", reject_index)
    # for ind in reject_index:
    #     # we need to find nhbd sign. 
    #     flipped_sdf[reshaped_labels_out==ind] = 1.0

    final_sdf = -flipped_sdf.clone()

    final_sdf = renorm_target_sdf(final_sdf, sketcher_3d)
    return final_sdf
