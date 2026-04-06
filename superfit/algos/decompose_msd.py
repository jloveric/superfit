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
"""
# Code to decompose a shape into a set of primitive regions. 
import torch as th
import numpy as np
import cc3d
import time
from superfit.utils.constants import CLEAN_UP_DELTA
from superfit.utils.mesh_sdf import renorm_target_sdf, clean_up_msd_with_opening
from superfit.utils.logger import logger


def msd(target_sdf, sketcher_3d, 
            max_msd_iter=100,
            min_eroded_part_size_ratio=0.005, 
            min_part_size_ratio=0.001,
            *args, **kwargs):
    output = decompose_msd(target_sdf, sketcher_3d, 
                    max_msd_iter=max_msd_iter,
                    min_eroded_part_size_ratio=min_eroded_part_size_ratio,
                    min_part_size_ratio=min_part_size_ratio, 
                    *args, **kwargs)
    return output

def update_labels_by_min_eroded_parts(labels_out, stats, min_eroded_part_size):
    N = labels_out.max()
    label_map = {0: 0}
    count = 0
    for x in range(1, N + 1):
        if stats['voxel_counts'][x] > min_eroded_part_size:
            count += 1
            label_map[x] = count
        else:
            label_map[x] = 0
    new_labels_out = th.zeros_like(labels_out)
    for x in range(1, N+1):
        new_labels_out[labels_out == x] = label_map[x]
    
    labels_out = new_labels_out.reshape(-1)
    return labels_out

def filter_by_part_size(cur_target_sdf, 
                     labels_out, 
                     sketcher_3d,
                     cur_threshold, 
                     min_part_size):
    
    abs_threshold = th.abs(cur_threshold)
    selected_parts = []
    # Erosion based partitioning
    N = labels_out.max()
    for i in range(1, N+1):
        cur_mask = (labels_out == i)
        # proxy_sdf = (0.5 - cur_mask.float())
        proxy_sdf = cur_target_sdf.clone() + abs_threshold - 1e-6
        proxy_sdf[~cur_mask] = 1.0
        renormed_proxy_sdf = renorm_target_sdf(proxy_sdf, sketcher_3d)
        renormed_proxy_sdf = renormed_proxy_sdf - (abs_threshold)
        # dilated_mask = (renormed_proxy_sdf <= 0)
        part_size = (renormed_proxy_sdf <= 0).float().sum()
        if part_size > min_part_size:
            selected_parts.append(renormed_proxy_sdf)
    return selected_parts


def find_mo_parts(cur_target_sdf, 
                  sketcher_3d, 
                  basic_jump_size, 
                  min_eroded_part_size_ratio,
                  min_part_size):
    min_val = cur_target_sdf.min()
    # Instead only do the jump size given and then if fewer parts do fewer jumps. 
    n_steps_jump = int(th.abs(min_val) / basic_jump_size)
    res = sketcher_3d.resolution
    selected_parts = []
    mask_shape = (res, res, res)
    # content_size 
    shape_size = (cur_target_sdf <= 0).float().sum()
    min_eroded_part_size = shape_size * min_eroded_part_size_ratio
    min_eroded_part_size = int(min_eroded_part_size)

    all_vals = th.arange(min_val, min(0.0, min_val + basic_jump_size * n_steps_jump/3.0), basic_jump_size).to(sketcher_3d.device).to(sketcher_3d.dtype)
    neg_sdf_vals = cur_target_sdf[cur_target_sdf<=0]
    deltas = neg_sdf_vals[None, :] < all_vals[:, None]
    deltas = deltas.float().sum(axis=-1)
    cond = deltas < min_eroded_part_size
    first_false = (~cond).int().argmax().item()
    # first_false = 0
    logger.info(f"first_false: {first_false}")
    for i in range(first_false, n_steps_jump + 1):
        cur_threshold = min_val + basic_jump_size * i
        if cur_threshold > 0.0:
            cur_threshold = th.clip(cur_threshold, min=0.0)
        cur_mask = cur_target_sdf <= cur_threshold
        total_voxels = cur_mask.float().sum()
        if total_voxels < min_eroded_part_size:
            continue
        reshaped_mask = cur_mask.reshape(*mask_shape).cpu().numpy()
        labels_out, N = cc3d.connected_components(reshaped_mask, return_N=True, binary_image=True) # free

        stats = cc3d.statistics(labels_out)
        voxel_counts = stats['voxel_counts'][1:]
        valid_primitives = voxel_counts > min_eroded_part_size
        # here for each do a reorg and check: 
        if valid_primitives.sum() > 0:
            logger.info(f"Valid primitives found at threshold {cur_threshold:.6f}, iteration {i}")
            labels_out = th.from_numpy(labels_out).to(sketcher_3d.device).long()
            updated_labels = update_labels_by_min_eroded_parts(labels_out, stats, min_eroded_part_size)
            selected_parts = filter_by_part_size(cur_target_sdf, 
                     updated_labels, 
                     sketcher_3d,
                     cur_threshold, 
                     min_part_size)
            n_parts = len(selected_parts)
            logger.info(f"Found {n_parts} parts after min part size filter")
            if len(selected_parts) > 0:
                break

    return selected_parts


def decompose_msd(target_sdf, sketcher_3d, 
                    max_msd_iter=100, n_steps_jump=100, 
                    min_eroded_part_size_ratio=0.0005,
                    min_part_size_ratio=0.0001,
                    clean_up_delta=CLEAN_UP_DELTA,
                    *args, **kwargs):
    # New stuff -> Subtract with some dilation. 
    # General Principle: We would like to consider "part" size - but we cant as it takes too long to do that. 
    # So instead we say atleast k-voxels. Now if k=1, we might get parts which are too small. 
    # Avoid point mass -> after erosion to must have some min size. 
    cur_target_sdf = target_sdf.clone()
    min_sdf_value = cur_target_sdf.min()
    basic_jump_size = th.abs(min_sdf_value) / n_steps_jump
    init_time = time.time()
    all_parts = []
    all_indices = []
    cur_iter = 0
    prog_execution  = None
    shape_size = (cur_target_sdf <= 0).float().sum()
    min_part_size = shape_size * min_part_size_ratio
    min_part_size = int(min_part_size)

    while cur_iter < max_msd_iter:
        # Find minimal level set that is bigger than min_size_ratio. 
        min_sdf = cur_target_sdf.min()
        if min_sdf > 0.0:
            break
        selected_parts = find_mo_parts(cur_target_sdf, 
                                              sketcher_3d, 
                                              basic_jump_size, 
                                              min_eroded_part_size_ratio,
                                              min_part_size,)
        # Valid parts ->
        n_parts = len(selected_parts)
        logger.info(f"Found {n_parts} parts")
        if n_parts == 0:
            break
        
        all_parts.extend(selected_parts)
        logger.debug(f"all_parts: {len(all_parts)}")
        # CLEAN UP AND NEXT TARGET CODE.
        indices = [cur_iter for _ in range(n_parts)]
        all_indices.extend(indices)
        united_mask = th.stack(all_parts, axis=-1)
        prog_execution = (united_mask <= clean_up_delta).any(axis=-1)
        cur_target_sdf = target_sdf.clone()
        cur_target_sdf[prog_execution] = 1.0
        cur_target_sdf = renorm_target_sdf(cur_target_sdf, sketcher_3d)
        # opening based cleanup?
        cur_target_sdf = clean_up_msd_with_opening(cur_target_sdf, sketcher_3d, amount=clean_up_delta)
        logger.debug(f"cur_iter: {cur_iter}, n_parts: {len(all_parts)}")
        cur_iter += 1
    total_time = time.time() - init_time
    logger.info(f"=========== Decomposition MSD: Time taken: {total_time:.3f}s ===========")
    # Now from these "parts" we must generate initializations for optimization.
    # extend the parts:
    return all_parts, all_indices

