
import time
import torch as th
import numpy as np
import geolipi.symbolic as gls
import superfit.symbolic as sps
import cubvh
from .param_conversion import params_from_variables
from ..symbolic.utils import gather_primitives
from ..utils.config import AlgorithmConfig as AlgConf
from ..utils.logger import logger
from ..utils.stats import Stats
from .utils import (quick_sample_points_and_normals, exponential_temperature_schedule, 
                    recompute_sdf_from_BVH, get_mask_scaled_aabb)
from .curvature import get_points_and_weights
from .measures import get_iou

def get_scale_factor(i, scale_factors):
    if i >= len(scale_factors):
        scale_factor = scale_factors[-1]
    else:
        scale_factor = scale_factors[i]
    return scale_factor


def run_optimization_loop(init_opt_program, target_mesh, target, sketcher, 
                          variable_list, tensor_list, type_annotation, param_groups,
                          compiled_func_relaxed):
    ## Prelims
    opt_program = init_opt_program
    has_temp = isinstance(opt_program, (sps.SuperFrustumPackedBatchedStochasticSU, 
                                        sps.SolidSFPackedBatchedStochasticSU,
                                        )
                        )

    device = sketcher.device
    prim_params = opt_program.get_arg(0)
    n_prims = prim_params.shape[0]

    min_temp = th.tensor([AlgConf.MIN_TEMP_VAL], device=device, dtype=AlgConf.OPT_DTYPE)
    max_temp = th.tensor([AlgConf.MAX_TEMP_VAL], device=device, dtype=AlgConf.OPT_DTYPE)

    # Scale factor calculation
    start = np.log(AlgConf.SCALE_FACTOR_START)
    end = np.log(AlgConf.SCALE_FACTOR_END)
    scale_factors = np.exp(
        np.arange(start, end, (end-start)/float(AlgConf.N_ITERS)))

    sigmoid_func = th.nn.Sigmoid()

    init_temp = max_temp
    
    
    best_params = None
    best_obj = th.tensor([-1.0], device=device, dtype=AlgConf.OPT_DTYPE)
    best_shape_iou = th.tensor([-1.0], device=device, dtype=AlgConf.OPT_DTYPE)
    best_surface_iou = th.tensor([-1.0], device=device, dtype=AlgConf.OPT_DTYPE)
    iterations_without_improvement = 0

    orig_program = opt_program.sympy()
    start_temp_decay = False
    decay_start_iter = 0
    stochastic_precondition_n_iters = AlgConf.N_ITERS//3
    iter_limit = AlgConf.N_ITERS
    base_iters = AlgConf.N_ITERS
    max_iter = AlgConf.MAX_ITER

    ## Process targets:
    st = time.time()
    # target = get_target_cubvh(target_mesh, sketcher, mode="watertight")
    # target = renorm_target_sdf(target, sketcher)
    logger.debug(f"Processing targets: {time.time() - st:.3f}s")
    

    hard_target = (target <= 0)
    hard_target_fl = hard_target.float()

    if AlgConf.TARGET_MODE == "bboxed":
        target_mask = get_mask_scaled_aabb(sketcher.get_base_coords(), target_mesh)
        hard_target_fl = hard_target_fl[target_mask]
        hard_target = hard_target[target_mask]
        target = target[target_mask] 
    elif AlgConf.TARGET_MODE == "dilated":
        target_mask = (target <= AlgConf.TARGET_MODE_DILATION)
        hard_target_fl = hard_target_fl[target_mask]
        hard_target = hard_target[target_mask]
        target = target[target_mask] 
    else:
        target_mask = None

    if AlgConf.TVERSKY_MODE:
        pos_weight = AlgConf.TVERSKY_BETA * hard_target_fl           # for t(x) == 1
        neg_weight = AlgConf.TVERSKY_ALPHA * (1 - hard_target_fl)    # for t(x) == 0
        tversky_weights = pos_weight + neg_weight            # [N]
    else:
        tversky_weights = None
        tversky_weights_surface_adj = None
    
    ## Process Input:
    st = time.time()
    if AlgConf.USE_CURVATURE_WEIGHTS:
        surface_sampled_points, curvature_weights = get_points_and_weights(target_mesh, sketcher, n_points=AlgConf.N_SURFACE_POINTS)
        curvature_weights = AlgConf.CURVATURE_WEIGHTS_SCALE * curvature_weights
    else:
        surface_sampled_points, sampled_normals = quick_sample_points_and_normals(target_mesh, sketcher, n_points=AlgConf.N_SURFACE_POINTS)
        curvature_weights = None

    BVH = cubvh.cuBVH(target_mesh.vertices, target_mesh.faces)
    # make 
    base_coords = sketcher.get_base_coords()
    base_coords = base_coords.unsqueeze(0).expand(1, base_coords.shape[0], base_coords.shape[1])

    logger.debug(f"Creating BVH: {time.time() - st:.3f}s")
    if target_mask is not None:
        base_coords = base_coords[:, target_mask, :]


    ##  ----- OPTIM -- 
    logger.info("Starting optimization loop")
    optim = make_optimizer(param_groups)
    start_time = time.time()
    
    i = 0
    while (i < 10_000):

        ### ITERATION CONFIG
        optim.zero_grad()
        scale_factor = get_scale_factor(i, scale_factors)
        
        if start_temp_decay:
            temperature = exponential_temperature_schedule(i-decay_start_iter, base_iters, max_temp, min_temp, device=device)
        else:
            temperature = max_temp

        # Get transformed parameters
        transformed_params = params_from_variables(variable_list, tensor_list)
        # transformed_params, new_type_annotation = inject_temp_param_compiled(transformed_params, temperature, type_annotation)
        if has_temp:
            transformed_params.append(temperature)
        # For input
        all_coords = []
        cur_coords = base_coords# .clone()
        all_coords.append(cur_coords)
        # Add points from surface
        if i % AlgConf.RENEW_PTS_ITER == 0:
            perturbations = (th.rand_like(surface_sampled_points)  - 0.5) * AlgConf.SURFACE_ADJ_PERTURBATION_SCALE
            surface_adj_points = surface_sampled_points.clone() + perturbations#[..., None] * sampled_normals
            surface_sampled_sdf = recompute_sdf_from_BVH(surface_adj_points, BVH, mode="watertight")
            hard_target_surface_adj = (surface_sampled_sdf <= 0.0)
            hard_target_surface_adj_fl = hard_target_surface_adj.float()

            if AlgConf.TVERSKY_MODE:
                surface_sampled_occ_fl = (surface_sampled_sdf <= 0.0).float()
                pos_weight = AlgConf.TVERSKY_BETA * surface_sampled_occ_fl
                neg_weight = AlgConf.TVERSKY_ALPHA * (1 - surface_sampled_occ_fl)
                tversky_weights_surface_adj = pos_weight + neg_weight 
            batched_surface_adj_points = surface_adj_points.unsqueeze(0).expand(1, surface_adj_points.shape[0], surface_adj_points.shape[1])
            batched_surface_sampled_points = surface_sampled_points.unsqueeze(0).expand(1, surface_sampled_points.shape[0], surface_sampled_points.shape[1])
        
        all_coords.append(batched_surface_adj_points)
        all_coords.append(batched_surface_sampled_points)

        all_sizes = [x.shape[1] for x in all_coords]
        all_coords = th.cat(all_coords, dim=1)
        # if stochastic_precondition:
        if i < stochastic_precondition_n_iters:
            all_coords = perform_batched_stochastic_precondition(all_coords, i-decay_start_iter, stochastic_precondition_n_iters)
            
        all_coords = sketcher.make_homogenous_coords(all_coords)

        primitive_sdfs, output_sdf = compiled_func_relaxed(all_coords, transformed_params)
        output_sdf = output_sdf[0]
        mask = (output_sdf<= AlgConf.LOSS_BAND).float()
        if not mask.sum() > 0:
            logger.warning("No valid points")
            continue
        
        output_shape_sdf = output_sdf[:all_sizes[0]].detach().clone()
        output_surface_adj_sdf = output_sdf[all_sizes[0]:all_sizes[0]+all_sizes[1]].detach().clone()

        output_for_occ = output_sdf[:all_sizes[0]+ all_sizes[1]]
        output_surface_sdf = output_sdf[all_sizes[0]+all_sizes[1]:]
        output_tanh = th.tanh(output_for_occ * scale_factor)
        output_for_occ = sigmoid_func(-output_tanh * scale_factor)
        output_shape_occ = output_for_occ[:all_sizes[0]]
        output_surface_adj_occ = output_for_occ[all_sizes[0]:all_sizes[0]+all_sizes[1]]
        mask_shape = mask[:all_sizes[0]]
        mask_surface_adj = mask[all_sizes[0]:all_sizes[0]+all_sizes[1]]
        mask_surface = mask[all_sizes[0]+all_sizes[1]:]

        ##  LOSSES
        delta_shape = (output_shape_occ - hard_target_fl) ** 2
        if AlgConf.TVERSKY_MODE:
            delta_shape = delta_shape * tversky_weights
        loss_shape_occ = 0.5 * th.sum(mask_shape * delta_shape) / th.sum(mask_shape)

        # delta_shape_sdf = (output_shape_sdf - target) ** 2
        # loss_shape_sdf = 0.5 * th.sum(mask_shape * delta_shape_sdf) / th.sum(mask_shape)

        delta_surface_adj = (output_surface_adj_occ - hard_target_surface_adj_fl) ** 2
        if AlgConf.TVERSKY_MODE:
            delta_surface_adj = delta_surface_adj * tversky_weights_surface_adj
        if AlgConf.USE_CURVATURE_WEIGHTS:
            delta_surface_adj = delta_surface_adj * (1 + curvature_weights)

        loss_surface_adj_occ = 0.5 * th.sum(mask_surface_adj * delta_surface_adj) / th.sum(mask_surface_adj)
        
        delta_surface_sdf = (output_surface_sdf) ** 2
        if AlgConf.USE_CURVATURE_WEIGHTS:
            delta_surface_sdf = delta_surface_sdf * (1 + curvature_weights)
        loss_surface_sdf = 0.5 * th.sum(mask_surface * delta_surface_sdf) / th.sum(mask_surface)

        param_loss = get_param_loss(transformed_params, type_annotation)
        primitive_count_loss = get_primitive_count_loss(transformed_params, temperature)
        overlap_loss = get_batched_overlap_loss(primitive_sdfs, output_sdf, sigmoid_func, scale_factor)
        shape_unoverlap_loss = get_batched_shape_unoverlap_loss(primitive_sdfs, output_sdf, sigmoid_func, scale_factor)

        total_loss = AlgConf.LOSS_OCC_ALPHA * loss_shape_occ + \
                + AlgConf.LOSS_PRIMITIVE_COUNT_ALPHA * primitive_count_loss + \
                + AlgConf.LOSS_OVERLAP_ALPHA * overlap_loss + \
                + AlgConf.LOSS_SHAPE_UNOVERLAP_ALPHA * shape_unoverlap_loss \
                + AlgConf.LOSS_PARAM_REGULARIZATION_ALPHA * param_loss 
                # + AlgConf.LOSS_sdf_ALPHA * loss_shape_sdf + \
        if not AlgConf.SKIP_SURFACE:
            total_loss = total_loss + AlgConf.LOSS_SURFACE_ADJ_OCC_ALPHA * loss_surface_adj_occ + \
                + AlgConf.LOSS_SURFACE_SDF_ALPHA * loss_surface_sdf 
                
        total_loss.backward()
        nan_detected = False
        opt_var_list = [x for ind, x in enumerate(variable_list)]
        for opt_ind, opt_var in enumerate(opt_var_list):
            if not (opt_var.grad is None):
                if th.isnan(opt_var.grad).any():
                    nan_detected = True
                    opt_var.grad[th.isnan(opt_var.grad)] = 0.0
            if nan_detected:
                logger.error("NAN detected")
        # Statistics:
        # if not nan_detected:
        optim.step()

        with th.no_grad():
            hard_output_shape = (output_shape_sdf <= 0.0)
            hard_output_surface_adj = (output_surface_adj_sdf <= 0.0)
            shape_iou = get_iou(hard_output_shape, hard_target)
            surface_adj_iou = get_iou(hard_output_surface_adj, hard_target_surface_adj)
            sdf_error = th.abs(output_surface_sdf.detach()).mean()
             
        # STOPPING DESIGN:

        shape_improve =  (shape_iou > best_shape_iou + AlgConf.MIN_IMPROVEMENT)
        surface_improve = (surface_adj_iou > best_surface_iou + AlgConf.MIN_IMPROVEMENT)

        act_shape_improve = (shape_iou > best_shape_iou)
        act_surface_improve = (surface_adj_iou > best_surface_iou)
        if surface_improve:
            best_surface_iou = surface_adj_iou
        if shape_improve:
            best_shape_iou = shape_iou
            
        if shape_improve or surface_improve:

            Stats.record("best_shape_iou", shape_iou.item(), log=False)
            Stats.record("best_surface_iou", surface_adj_iou.item(), log=False)
            Stats.record("best_obj", shape_iou.item(), log=False)
            Stats.record("best_obj", shape_iou.item(), log=False)
            best_obj = shape_iou
            iterations_without_improvement = 0
            
        else:
            iterations_without_improvement += 1

        if act_shape_improve or act_surface_improve:
            if has_temp:
                t_param = transformed_params[:-1]
            else:
                t_param = transformed_params
            best_params = [x.detach() for x in t_param]
        
        stopping_criteria_1 = i >= (iter_limit-1)
        stopping_criteria_2 = iterations_without_improvement >= AlgConf.SAT_PATIENCE
        stopping_criteria_3 = i >= max_iter

        if stopping_criteria_1:
            # can stop ONLY after this. 
            any_stop = stopping_criteria_2 or stopping_criteria_3
            if any_stop and start_temp_decay:
                # Time to stop:
                logger.info("===========Stopping due to stopping criteria===========")
                logger.info(f"cur_iter: {i}, iter_limit: {iter_limit}, max_iter: {max_iter}")
                logger.info(f"iterations_without_improvement: {iterations_without_improvement}, saturation patience: {AlgConf.SAT_PATIENCE}")
                break
            if any_stop and not start_temp_decay:
                logger.info("===========Starting Temp Decay===========")
                start_temp_decay = True
                decay_start_iter = i
                iterations_without_improvement = 0
                stochastic_precondition_n_iters = i + stochastic_precondition_n_iters
                iter_limit = i + base_iters
                max_iter = max(iter_limit, AlgConf.MAX_ITER)
                
                logger.info(f"---- new max_iter: {max_iter}, new iter_limit: {iter_limit}  ----")

        
        i += 1
        if has_temp:
            transformed_params = transformed_params[:-1]

        if i % AlgConf.LOG_FREQUENCY == 0:
            cur_time = time.time()
            iteration_rate = (cur_time - start_time) / (i + 1)
            logger.info(f"Iteration rate: {iteration_rate:.3f} seconds per iteration")
            logger.info(f"Iteration {i}, Shape IOU: {shape_iou.item():.3f} | Surface Adj IOU: {surface_adj_iou.item():.3f} | Surface SDF Error: {sdf_error.item():.5f}")
            logger.info(f"Total Loss: {total_loss.item():.5f} | Temperature: {temperature.item():.3f} | Scale Factor: {scale_factor:.3f}")
            logger.info(f"Loss Shape Occ: {loss_shape_occ.item():.5f} |  Loss SurfaceAdj OCC: {loss_surface_adj_occ.item():.5f} | Loss Surface Adj SDF: {loss_surface_sdf.item():.5f}")
            logger.info(f"Loss Overlap: {overlap_loss.item():.5f} | Loss Shape Unoverlap: {shape_unoverlap_loss.item():.5f}")
            logger.info(f"Loss Param: {param_loss.item():.5f} | Loss Primitive Count: {primitive_count_loss.item():.5f} | N Prims: {n_prims}")
            logger.debug(f"Iterations without improvement: {iterations_without_improvement}, new_iou: {shape_iou.item():.3f}, best_iou: {best_shape_iou.item():.3f} | new_surface_iou: {surface_adj_iou.item():.3f}, best_surface_iou: {best_surface_iou.item():.3f}")
    logger.info(f"Optimization stopped after {i} iterations - iterations_without_improvement {iterations_without_improvement}")

    # Final results
    if best_params is None:
        best_program = orig_program.tensor(dtype=AlgConf.OPT_DTYPE)
    else:
        best_program = opt_program.inject_tensor_list(best_params)
    
    Stats.record("n_iters", i)
    Stats.record("iterations_without_improvement", iterations_without_improvement)
    
    return best_program


def perform_batched_stochastic_precondition(base_coords, i, base_iters):

    init_val = AlgConf.STOCHASTIC_PRECONDITION_INIT_VAL
    final_val = 0
    # linear interpolate from i to base iters
    frac = min(i / (base_iters), 1.0)
    alpha = init_val * (1 - frac) + final_val * frac
    noise = th.randn_like(base_coords[0]).unsqueeze(0).expand(base_coords.shape[0], base_coords.shape[1], base_coords.shape[2]) * alpha
    base_coords = base_coords + noise
    return base_coords


def get_batched_overlap_loss(prim_execs, full_execution, sigmoid_func, scale_factor):
    output_tanh = th.tanh(prim_execs * scale_factor)
    output_shape = sigmoid_func(-output_tanh * scale_factor)

    occ_sum = th.sum(output_shape, dim=0)
    loss_per_cell = th.maximum(occ_sum - 1.0, th.zeros_like(occ_sum))
    
    full_output_tanh = th.tanh(full_execution * scale_factor)
    full_output_shape = sigmoid_func(-full_output_tanh * scale_factor)

    loss = th.sum(loss_per_cell) / (th.sum(full_output_shape) + 1e-6)
    return loss


def get_batched_shape_unoverlap_loss(prim_execs, full_execution, sigmoid_func, scale_factor):
    output_tanh = th.tanh(prim_execs * scale_factor)
    output_shape = sigmoid_func(-output_tanh * scale_factor)

    occ_sum = th.sum(output_shape, dim=0)
    occupancy_map = th.minimum(occ_sum, th.ones_like(occ_sum))

    full_output_tanh = th.tanh(full_execution * scale_factor)
    full_output_shape = sigmoid_func(-full_output_tanh * scale_factor)

    # Loss where Shape is present, but occupancy map is not
    loss_per_cell = th.maximum(full_output_shape - occupancy_map, th.zeros_like(full_output_shape))
    loss = th.sum(loss_per_cell) / (th.sum(full_output_shape) + 1e-6)
    return loss


def get_primitive_count_loss(transformed_params, temperature):
    # Likelihood should be low. 
    loss = th.tensor([0.0], device=transformed_params[0].device, dtype=AlgConf.OPT_DTYPE)
    if len(transformed_params) == 4:
            logits = transformed_params[-2]
            # g = sample_gumbel((2,), device=transformed_params[0].device)
            # The gradients are too saturated. 
            # likelihood_logits = logits[0] - logits[1]
            soft = th.softmax((logits) / (temperature), dim=-1)  # (2,)
            # term = -th.log((1 - soft[0]).clamp_min(1e-3))
            loss += soft[..., 0].sum()
    return loss

def get_param_loss(transformed_params, type_annotation):
    device = transformed_params[0].device
    loss = th.tensor([0.0], device=device, dtype=AlgConf.OPT_DTYPE)
    if len(transformed_params) == 4:
        prim_params, su_ops, logits, temperature = transformed_params
        # lower su first
    elif len(transformed_params) == 2:
        prim_params, su_ops = transformed_params
    else:
        raise ValueError(f"Invalid number of parameters: {len(transformed_params)}")
    MUL = 1e6
    su_loss = th.sum(su_ops, dim=0)
    if issubclass(type_annotation[0][0], (sps.SuperFrustumPackedBatchedStochasticSU, 
                                            sps.SolidSFPackedBatchedStochasticSU,)):
        # prim_dilation_loss = th.sum(prim_params[..., -3:-2])
        prim_size_loss = th.sum(prim_params[..., 6:9])
        # prim_vol_loss = prim_params[..., 6:9].prod(dim=-1).sum()
        # if scale * size < 0.01 -> penalize heavily.
        lower_bound_scale = prim_params[..., 6:9] * prim_params[..., 11:12]
        lower_bound_loss = th.where(lower_bound_scale< 0.01, -lower_bound_scale * MUL, th.zeros_like(lower_bound_scale))
        lower_bound_loss = lower_bound_loss.sum()
        prim_loss = prim_size_loss + lower_bound_loss
    loss += prim_loss + su_loss
    return loss

import torch as th

def make_optimizer(param_groups):
    """
    Returns a PyTorch optimizer based on AlgConf.OPTIMIZER.
    Supported: ADAM, ADAMW, MUON, SHAMPOO, KFAC, LBFGS
    """
    name = AlgConf.OPTIMIZER.strip().upper()
    lr = AlgConf.OPT_LR_RATE
    wd = AlgConf.WEIGHT_DECAY

    if name == "ADAM":
        logger.info(f"[Optimizer] Using {name} (lr={lr}, weight_decay={0.0})")
        optim = th.optim.Adam(param_groups, lr=lr)
    elif name == "ADAMW":
        logger.info(f"[Optimizer] Using {name} (lr={lr}, weight_decay={wd})")
        optim = th.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer type: {name}")

    return optim


def get_full_loss(compiled_func_relaxed, all_coords, transformed_params, 
                 all_sizes, scale_factor, sigmoid_func, 
                 tversky_weights, tversky_weights_surface_adj, 
                 curvature_weights, 
                 hard_target_fl, hard_target_surface_adj_fl, 
                 mask_shape, mask_surface_adj, mask_surface, 
                 output_shape_occ, output_surface_adj_occ, 
                 output_surface_sdf, type_annotation, temperature):
    
    loss = th.tensor([0.0], device=all_coords.device, dtype=AlgConf.OPT_DTYPE)
    primitive_sdfs, output_sdf = compiled_func_relaxed(all_coords, transformed_params)
    output_sdf = output_sdf[0]
    # mask = (th.abs(output_sdf)<= AlgConf.LOSS_BAND).float()
    mask = (output_sdf<= AlgConf.LOSS_BAND).float()
    if not mask.sum() > 0:
        logger.warning("No valid points")
        return loss
    
    output_shape_sdf = output_sdf[:all_sizes[0]].detach().clone()
    output_surface_adj_sdf = output_sdf[all_sizes[0]:all_sizes[0]+all_sizes[1]].detach().clone()

    output_for_occ = output_sdf[:all_sizes[0]+ all_sizes[1]]
    output_surface_sdf = output_sdf[all_sizes[0]+all_sizes[1]:]
    output_tanh = th.tanh(output_for_occ * scale_factor)
    output_for_occ = sigmoid_func(-output_tanh * scale_factor)
    output_shape_occ = output_for_occ[:all_sizes[0]]
    output_surface_adj_occ = output_for_occ[all_sizes[0]:all_sizes[0]+all_sizes[1]]
    mask_shape = mask[:all_sizes[0]]
    mask_surface_adj = mask[all_sizes[0]:all_sizes[0]+all_sizes[1]]
    mask_surface = mask[all_sizes[0]+all_sizes[1]:]

    ##  LOSSES
    delta_shape = (output_shape_occ - hard_target_fl) ** 2
    if AlgConf.TVERSKY_MODE:
        delta_shape = delta_shape * tversky_weights
    loss_shape_occ = 0.5 * th.sum(mask_shape * delta_shape) / th.sum(mask_shape)

    # delta_shape_sdf = (output_shape_sdf - target) ** 2
    # loss_shape_sdf = 0.5 * th.sum(mask_shape * delta_shape_sdf) / th.sum(mask_shape)

    delta_surface_adj = (output_surface_adj_occ - hard_target_surface_adj_fl) ** 2
    if AlgConf.TVERSKY_MODE:
        delta_surface_adj = delta_surface_adj * tversky_weights_surface_adj
    if AlgConf.USE_CURVATURE_WEIGHTS:
        delta_surface_adj = delta_surface_adj * (1 + curvature_weights)

    loss_surface_adj_occ = 0.5 * th.sum(mask_surface_adj * delta_surface_adj) / th.sum(mask_surface_adj)
    
    delta_surface_sdf = (output_surface_sdf) ** 2
    if AlgConf.USE_CURVATURE_WEIGHTS:
        delta_surface_sdf = delta_surface_sdf * (1 + curvature_weights)
    loss_surface_sdf = 0.5 * th.sum(mask_surface * delta_surface_sdf) / th.sum(mask_surface)

    param_loss = get_param_loss(transformed_params, type_annotation)
    primitive_count_loss = get_primitive_count_loss(transformed_params, temperature)
    overlap_loss = get_batched_overlap_loss(primitive_sdfs, output_sdf, sigmoid_func, scale_factor)

    shape_unoverlap_loss = get_batched_shape_unoverlap_loss(primitive_sdfs, output_sdf, sigmoid_func, scale_factor)

    total_loss = AlgConf.LOSS_OCC_ALPHA * loss_shape_occ + \
            + AlgConf.LOSS_PRIMITIVE_COUNT_ALPHA * primitive_count_loss + \
            + AlgConf.LOSS_OVERLAP_ALPHA * overlap_loss + \
            + AlgConf.LOSS_SHAPE_UNOVERLAP_ALPHA * shape_unoverlap_loss \
            + AlgConf.LOSS_PARAM_REGULARIZATION_ALPHA * param_loss 
            # + AlgConf.LOSS_sdf_ALPHA * loss_shape_sdf + \
    if not AlgConf.SKIP_SURFACE:
        total_loss = total_loss + AlgConf.LOSS_SURFACE_ADJ_OCC_ALPHA * loss_surface_adj_occ + \
            + AlgConf.LOSS_SURFACE_SDF_ALPHA * loss_surface_sdf 
    return total_loss