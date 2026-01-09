import numpy as np
import torch as th
import fastsweep
from .constants import USE_CUDA, CLEAN_UP_DELTA, MIN_VOLUME_LIMIT
import trimesh
import cubvh
import cc3d
import time
from geolipi.torch_compute import recursive_evaluate
from kaolin.non_commercial.flexicubes import FlexiCubes

if not USE_CUDA:
    from drjit.llvm import TensorXf
else:
    from drjit.cuda import TensorXf


def renorm_target_sdf(target, sketcher):
    sdf_res = sketcher.resolution
    occ = (target >=0.0).float() - 0.5
    occ_reshaped = occ.reshape(sdf_res, sdf_res, sdf_res)
    init = TensorXf(occ_reshaped)
    sdf = fastsweep.redistance(init)
    cur_pt_distance = sdf.torch().reshape(-1)
    return cur_pt_distance


def clean_up_msd_with_opening(target_sdf, sketcher_3d, amount=CLEAN_UP_DELTA):
    new_sdf = target_sdf + amount
    new_reformed_sdf = renorm_target_sdf(new_sdf, sketcher_3d)
    new_sdf = new_reformed_sdf - amount
    new_reformed_sdf = renorm_target_sdf(new_sdf, sketcher_3d)
    return new_reformed_sdf
    

def sdf_to_mesh(sdf, sketcher):
    sdf_res = sketcher.resolution
    th.cuda.empty_cache()
    flexer = FlexiCubes()
    new_points, cube_dx = flexer.construct_voxel_grid(sdf_res-1)
    sampling_center, sampling_scale = 0.0, 1.0
    new_points = (new_points * 2 - sampling_center) / sampling_scale  # Normalize to [-1, 1] range
    out = flexer(new_points, sdf, cube_idx=cube_dx, resolution=sdf_res-1)

    vertices = out[0]# .cpu().numpy()
    faces = out[1]# .cpu().numpy()
    vertices_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()
    out_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
    return out_mesh


def get_target_cubvh(mesh, sketcher, ensure_min_sdf=False, mode="raystab"):
    points = sketcher.get_base_coords()
    # Ideally adjust this to the shape of the target.
    faces = mesh.faces
    BVH = cubvh.cuBVH(mesh.vertices, mesh.faces) # build with numpy.ndarray/torch.Tensor
    distances, face_id, uvw = BVH.signed_distance(points, return_uvw=False, mode=mode) # [N], [N], [N, 3]
    # Option 2: Use mesh2sdf
    target = distances
    # Ensure min value is atleast 
    if ensure_min_sdf:
        min_sdf_req = - 2.0 / sketcher.resolution
        if target.min() > min_sdf_req:
            min_target_val = target.min()
            target = target - (min_target_val - min_sdf_req)
    return target


def target_cleanup(target_sdf, sketcher_3d, min_volume_limit=MIN_VOLUME_LIMIT):
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
    reshaped_labels_out = labels_out.reshape(-1)
    print(f"rejecting {len(reject_index)} parts by volume fraction")
    for ind in reject_index:
        # we need to find nhbd sign. 
        pos_target[reshaped_labels_out==ind] = 1.0

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
    reshaped_labels_out = labels_out.reshape(-1)
    print("rejecting", reject_index)
    for ind in reject_index:
        # we need to find nhbd sign. 
        flipped_sdf[reshaped_labels_out==ind] = 1.0

    final_sdf = -flipped_sdf.clone()

    final_sdf = renorm_target_sdf(final_sdf, sketcher_3d)
    return final_sdf


def get_masked(program, target, sketcher):
    original_exec = recursive_evaluate(program, sketcher, relaxed_eval=False)
    original_exec = renorm_target_sdf(original_exec, sketcher)
    original_exec -= CLEAN_UP_DELTA
    original_exec = renorm_target_sdf(original_exec, sketcher)
    updated_target = th.maximum(target, - original_exec)
    # original_hard_output = (original_exec.detach() <= 0.0)
    # updated_target = th.where(original_hard_output, th.ones_like(target), target)
    updated_target = renorm_target_sdf(updated_target, sketcher)
    # updated_target = clean_up_msd_with_opening(updated_target, sketcher, CLEAN_UP_DELTA)
    # Instead get new OCC and regen sdf

    return updated_target


