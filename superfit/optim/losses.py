import torch as th
from ..utils.config import AlgorithmConfig as AlgConf

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

    full_scaled = full_execution * scale_factor
    full_output_tanh = th.tanh(full_scaled)
    full_output_shape = th.sigmoid(-full_output_tanh * scale_factor)

    loss_per_cell = th.clamp_min(full_output_shape - occupancy_map, 0.0)
    loss = loss_per_cell.sum() / (full_output_shape.sum() + 1e-6)
    return loss


def get_primitive_count_loss(transformed_params, temperature):
    logits = transformed_params[-2]
    soft = th.softmax(logits / temperature, dim=-1)
    loss = soft[:, 0].sum()
    return loss


def get_param_loss(transformed_params):
    prim_params, su_ops, logits, temperature = transformed_params
    su_loss = su_ops.sum()
    
    prim_size_loss = prim_params[:, 6:9].sum()
    lower_bound_scale = prim_params[:, 6:9] * prim_params[:, 11:12]
    lower_bound_loss = th.where(lower_bound_scale < 0.01, -lower_bound_scale * PARAM_MUL, th.zeros_like(lower_bound_scale)).sum()
    prim_loss = prim_size_loss + lower_bound_loss
    
    loss = prim_loss + su_loss
    return loss

def compute_total_loss(output_shape_occ, hard_target_fl, 
                 output_surface_adj_occ, hard_target_surface_adj_fl, 
                 output_surface_sdf,
                 primitive_sdfs, output_sdf, 
                 mask_shape, mask_surface, mask_surface_adj,
                 transformed_params, temperature, 
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
    delta_surface_sdf = output_surface_sdf ** 2
    delta_surface_sdf = delta_surface_sdf * (1 + curvature_weights)
    mask_surface_sum = mask_surface.sum()
    loss_surface_sdf = 0.5 * (mask_surface * delta_surface_sdf).sum() / (mask_surface_sum + 1e-8)

    param_loss = get_param_loss(transformed_params)
    primitive_count_loss = get_primitive_count_loss(transformed_params, temperature)
    overlap_loss = get_batched_overlap_loss(primitive_sdfs, output_sdf, scale_factor)
    shape_unoverlap_loss = get_batched_shape_unoverlap_loss(primitive_sdfs, output_sdf, scale_factor)

    # Build total loss more efficiently
    total_loss = (AlgConf.LOSS_OCC_ALPHA * loss_shape_occ + 
                    AlgConf.LOSS_PRIMITIVE_COUNT_ALPHA * primitive_count_loss + 
                    AlgConf.LOSS_OVERLAP_ALPHA * overlap_loss + 
                    AlgConf.LOSS_SHAPE_UNOVERLAP_ALPHA * shape_unoverlap_loss +
                    AlgConf.LOSS_PARAM_REGULARIZATION_ALPHA * param_loss + \
                    AlgConf.LOSS_SURFACE_ADJ_OCC_ALPHA * loss_surface_adj_occ + 
                    AlgConf.LOSS_SURFACE_SDF_ALPHA * loss_surface_sdf)
    return total_loss