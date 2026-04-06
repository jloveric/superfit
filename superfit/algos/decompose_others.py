"""
ADOBE

Copyright 2026 Adobe

All Rights Reserved.

NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.

This file contains functions for convex decomposition that can be used for initializing primitives. 
"""
import trimesh
import torch as th
from superfit.utils.mesh_sdf import sdf_to_mesh, get_target_cubvh, renorm_target_sdf

MAX_SPREAD_ITER = 1000

def spread_segmentation(pruned_parts, hard_target, sketcher_3d):

    res = sketcher_3d.resolution
    conv_kernel = th.ones(1, 1, 3, 3, 3).to(hard_target.device)
    n_parts = len(pruned_parts)
    cur_parts = [x <=0 for x in pruned_parts]
    for i in range(MAX_SPREAD_ITER):
        prunned_occs = cur_parts# [x <=0 for x in cur_parts]
        # prunned_vols = [x.float().sum() for x in prunned_occs]
        # print(prunned_vols)

        prunned_occs_stacked = th.stack(prunned_occs, dim=0)
        program_exec = prunned_occs_stacked.any(dim=0)
        remenent = th.logical_and(hard_target, ~program_exec)#.reshape(1, 1, res, res, res)
        print(remenent.shape, remenent.sum())
        # now use 3x3x3 convolution to grow the parts
        prunned_occs_stacked = prunned_occs_stacked.reshape(-1, 1, res, res, res)
        conv_out = th.nn.functional.conv3d(prunned_occs_stacked.float(), weight=conv_kernel, padding=1)
        conv_out = conv_out.reshape(n_parts, -1)

        conv_out = th.logical_and(conv_out > 0, remenent[None, ...])
        has_updates = conv_out.any()
        if not has_updates:
            print("no updates")
            break
        new_labels = th.zeros_like(program_exec).long()
        for i in range(n_parts):
            new_labels[conv_out[i]] = (i+1)
        new_parts = []
        for i in range(n_parts):
            cur_xtra_part = new_labels == (i+1)
            cur_part = th.logical_or(prunned_occs[i], cur_xtra_part)
            new_parts.append(cur_part)
        # new_vols = [x.float().sum() for x in new_parts]
        # print(new_vols)
        cur_parts = new_parts
    
    cur_parts = [renorm_target_sdf(0.5 - x.float(), sketcher_3d) for x in cur_parts]
    return cur_parts
    
def adjust_partitioning(input_parts, target, sketcher_3d, part_limit=50, respread=False):
    hard_target = (target <= 0)
    volumes = [(part<=0).float().sum() for part in input_parts]
    if len(input_parts) < part_limit:
        return input_parts, volumes
    pruned_parts = spread_segmentation(input_parts, hard_target, sketcher_3d)
    # _pruned_parts = [renorm_target_sdf(0.5 - x.float(), sketcher_3d) for x in _pruned_parts]
    volumes = [(part<=0).float().sum() for part in pruned_parts]
    if len(pruned_parts) > part_limit:
        volumes, pruned_parts = zip(*sorted(zip(volumes, pruned_parts), key=lambda x: -x[0]))
        # _, _all_indices = zip(*sorted(zip(volumes, input_indices), key=lambda x: -x[0]))

        pruned_parts = input_parts[:part_limit]
        if respread:
            pruned_parts = spread_segmentation(pruned_parts, hard_target, sketcher_3d)
        volumes = [(part<=0).float().sum() for part in pruned_parts]
    else:
        pruned_parts = input_parts
    return pruned_parts, volumes


def coacd_decompose(target_sdf, sketcher_3d, **kwargs):
    import coacd
    if "max_convex_hull" in kwargs:
        size_limit = kwargs.pop("max_convex_hull")
    else:
        size_limit = 20
    mesh = sdf_to_mesh(target_sdf, sketcher_3d)
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(coacd_mesh, **kwargs) # a list of convex hulls.
    vertices = [x[0] for x in parts]
    faces = [x[1] for x in parts]
    new_meshes = [trimesh.Trimesh(vertices=vertices[i], faces=faces[i]) for i in range(len(vertices))]
    new_meshes = [x for x in new_meshes if x.volume > 1e-7]
    part_sdfs = []
    for mesh in new_meshes:
        part_sdf = get_target_cubvh(mesh, sketcher_3d, mode="raystab")
        # part_sdf = target_cleanup(part_sdf, sketcher_3d)
        part_sdf = renorm_target_sdf(part_sdf, sketcher_3d)
        part_sdfs.append(part_sdf)
    pruned_parts, volumes = adjust_partitioning(part_sdfs, target_sdf, sketcher_3d, part_limit=size_limit)
    return pruned_parts

def vhacd_decompose(target_sdf, sketcher_3d, size_limit=30, **kwargs):
    mesh = sdf_to_mesh(target_sdf, sketcher_3d)
    cvx_decomp = trimesh.decomposition.convex_decomposition(mesh, maxConvexHulls=size_limit, findBestPlane=True, **kwargs  )
    cvx_decomp = [trimesh.Trimesh(vertices=x['vertices'], faces=x['faces']) for x in cvx_decomp]
    cvx_decomp = [x for x in cvx_decomp if x.volume > 1e-7]
    part_sdfs = []
    for mesh in cvx_decomp:
        part_sdf = get_target_cubvh(mesh, sketcher_3d, mode="raystab")
        # part_sdf = target_cleanup(part_sdf, sketcher_3d)
        part_sdf = renorm_target_sdf(part_sdf, sketcher_3d)
        part_sdfs.append(part_sdf)
    return part_sdfs