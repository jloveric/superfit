import torch as th
import torch.nn.functional as F
from ..utils.config import AlgorithmConfig as AlgConf
from ..torch_compute.compile_friendly import sample_gumbel
from ..utils.logger import logger
PARAM_MUL = 1e6

def get_batched_overlap_loss(prim_execs, full_execution, scale_factor):
    # Fuse operations where possible
    prim_scaled = prim_execs * scale_factor
    output_tanh = th.tanh(prim_scaled)
    output_shape = th.sigmoid(-output_tanh * scale_factor)

    occ_sum = output_shape.sum(dim=0)
    loss_per_cell = th.clamp_min(occ_sum - 1.0, 0.0)
    
    full_scaled = full_execution * scale_factor
    full_output_tanh = th.tanh(full_scaled)
    full_output_shape = th.sigmoid(-full_output_tanh * scale_factor)

    loss = loss_per_cell.sum() / (full_output_shape.sum() + 1e-6)
    return loss


def get_batched_shape_unoverlap_loss(prim_execs, full_execution, scale_factor):
    prim_scaled = prim_execs * scale_factor
    output_tanh = th.tanh(prim_scaled)
    output_shape = th.sigmoid(-output_tanh * scale_factor)

    occ_sum = output_shape.sum(dim=0)
    occupancy_map = th.clamp_max(occ_sum, 1.0)
    #  POST REBUTTAL - What if we just detach gradients here?
    # occupancy_map = occupancy_map.detach()

    full_scaled = full_execution * scale_factor
    full_output_tanh = th.tanh(full_scaled)
    full_output_shape = th.sigmoid(-full_output_tanh * scale_factor)

    loss_per_cell = th.clamp_min(full_output_shape - occupancy_map, 0.0)
    loss = loss_per_cell.sum() / (full_output_shape.sum() + 1e-6)
    return loss


def get_primitive_count_loss(transformed_params):
    temperature = transformed_params[-1]
    logits = transformed_params[-2]
    # SHould this also have GMBL noise? 
    soft = th.softmax(logits / temperature, dim=-1)
    loss = soft[:, 0].sum()
    return loss


def get_param_loss_sf(transformed_params):
    prim_params, su_ops, logits, temperature = transformed_params
    su_loss = su_ops.sum()
    
    prim_size_loss = prim_params[:, 3:6].sum()
    # lower_bound_scale = prim_params[:, 3:6] * prim_params[:, 8:9]
    # lower_bound_loss = th.where(lower_bound_scale < 0.01, -lower_bound_scale * PARAM_MUL, th.zeros_like(lower_bound_scale)).sum()
    # prim_loss = prim_size_loss + lower_bound_loss
    # dilation_loss = prim_params[:, 11:12].sum()
    loss = su_loss + prim_size_loss
    return loss
    

def compute_total_loss(output_shape_occ, hard_target_fl, 
                 output_surface_adj_occ, hard_target_surface_adj_fl, 
                 output_surface_sdf,
                 primitive_sdfs, output_sdf, 
                 mask_shape, mask_surface, mask_surface_adj,
                 transformed_params, 
                 scale_factor, curvature_weights):

    # Shape occupancy loss
    delta_shape = (output_shape_occ - hard_target_fl) ** 2
    mask_shape_sum = mask_shape.sum()
    loss_shape_occ = 0.5 * (mask_shape * delta_shape).sum() / (mask_shape_sum + 1e-8)

    # Surface adj occupancy loss
    delta_surface_adj = (output_surface_adj_occ - hard_target_surface_adj_fl) ** 2
    delta_surface_adj = delta_surface_adj * (1 + curvature_weights)
    mask_surface_adj_sum = mask_surface_adj.sum()
    loss_surface_adj_occ = 0.5 * (mask_surface_adj * delta_surface_adj).sum() / (mask_surface_adj_sum + 1e-8)
    
    # Surface SDF loss
    delta_surface_sdf = th.abs(output_surface_sdf)# ** 2
    # delta_surface_sdf = th.nn.functional.leaky_relu(output_surface_sdf)# ** 2
    delta_surface_sdf = delta_surface_sdf * (1 + curvature_weights)
    # Could make it one sided?
    mask_surface_sum = mask_surface.sum()
    loss_surface_sdf = 0.5 * (mask_surface * delta_surface_sdf).sum() / (mask_surface_sum + 1e-8)

    # param_loss = get_param_loss(transformed_params)
    primitive_count_loss = get_primitive_count_loss(transformed_params)
    overlap_loss = get_batched_overlap_loss(primitive_sdfs, output_sdf, scale_factor)
    shape_unoverlap_loss = get_batched_shape_unoverlap_loss(primitive_sdfs, output_sdf, scale_factor)

    # Build total loss more efficiently
    total_loss = (AlgConf.LOSS_OCC_ALPHA * loss_shape_occ + 
                    AlgConf.LOSS_PRIMITIVE_COUNT_ALPHA * primitive_count_loss + 
                    AlgConf.LOSS_OVERLAP_ALPHA * overlap_loss + 
                    AlgConf.LOSS_SHAPE_UNOVERLAP_ALPHA * shape_unoverlap_loss +
                    # AlgConf.LOSS_PARAM_REGULARIZATION_ALPHA * param_loss + \
                    AlgConf.LOSS_SURFACE_ADJ_OCC_ALPHA * loss_surface_adj_occ + 
                    AlgConf.LOSS_SURFACE_SDF_ALPHA * loss_surface_sdf)
    return total_loss


def compute_reflection_loss(ref_output, transformed_params):
    ref_sdf, ref_prim_index = ref_output[..., 0], ref_output[..., 1].long()
    ref_mask = (ref_sdf <= AlgConf.LOSS_BAND).float()
    prim_params = transformed_params[0][:, 6:]
    ref_params = prim_params[ref_prim_index]
    ref_mask_1, ref_mask_2 = ref_mask.chunk(2, dim=0)
    ref_params_1, ref_params_2 = ref_params.chunk(2, dim=0)
    ref_params_1_detached = ref_params_1.detach()
    ref_params_2_detached = ref_params_2.detach()
    ref_sdf_1, ref_sdf_2 = ref_sdf.chunk(2, dim=0)
    cond_for_1 = th.abs(ref_sdf_1) < th.abs(ref_sdf_2)
    # IF SDF 1 is better, then move sdf2 to it. 
    ref_params_1 = th.where(cond_for_1[..., None], ref_params_1_detached, ref_params_1)
    ref_params_2 = th.where(cond_for_1[..., None], ref_params_2, ref_params_2_detached)
    # delta_ref_params = th.abs(ref_params_1 - ref_params_2).sum(dim=-1)
    delta_ref_params = th.norm(ref_params_1- ref_params_2, dim=-1)
    real_ref_mask = th.logical_or(ref_mask_1, ref_mask_2)
    reflection_loss =(real_ref_mask * delta_ref_params).sum() / (real_ref_mask.sum() + 1e-8)
    return reflection_loss

def compute_semantic_loss(point_soft_assoc, mask, sem_points_labels, n_classes, transformed_params):
    # Hack for now

    sharpness = 0.1
    temperature_2 = transformed_params[-1]

    point_gt_one_hot = sem_points_labels[mask].long()
    point_one_hot_spread = F.one_hot(point_gt_one_hot, num_classes=n_classes + 1).float()

    # Point2Prim Distribution
    point2prim_distr = point_soft_assoc[mask]
    g = sample_gumbel(point2prim_distr.shape, 
        device=point2prim_distr.device, 
        dtype=point2prim_distr.dtype)
    point2prim_distr  = th.softmax((point2prim_distr) \
        / (0.1 * temperature_2), dim=-1)  # (..., 2)

    # Prim2Label Distribution
    hard_alloc = point2prim_distr.argmax(dim=-1)
    hard_alloc = F.one_hot(hard_alloc.long(), 
        num_classes=point2prim_distr.shape[1]).float()
    prim2label_distr = hard_alloc.T @ point_one_hot_spread
    prim2label_distr = prim2label_distr \
        / (prim2label_distr.sum(dim=-1, keepdim=True) + 1e-6)
    prim2label_distr = th.softmax(prim2label_distr/(0.00001 * temperature_2), dim=-1)

    # Point2Label Distribution
    point2label_distr = point2prim_distr.float() @ (prim2label_distr.float())

    loss = F.cross_entropy(point2label_distr, point_gt_one_hot, reduction="mean")
    logger.info(f"Semantic loss: {loss}")
    return loss