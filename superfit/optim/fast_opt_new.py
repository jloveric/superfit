
import time
import torch as th
import numpy as np
import superfit.symbolic as sps
import cubvh
from collections import defaultdict
from .param_conversion import params_from_variables
from ..symbolic.utils import gather_primitives
from ..utils.config import AlgorithmConfig as AlgConf
from ..utils.stats import Stats
from ..utils.logger import logger
from ..utils.mesh_sdf import sdf_to_mesh
from .utils import (perform_batched_stochastic_precondition, exponential_temperature_schedule, 
                    recompute_sdf_from_BVH, get_mask_scaled_aabb, quick_sample_points)
from .curvature import get_points_and_weights
from .measures import get_iou
from .main_opt import make_optimizer, get_scale_factor
from .compile_function import CompiledOps
from .losses import compute_reflection_loss, compute_semantic_loss

SMU_K = 0.05

def run_optimization_loop_fast(init_opt_program, target_mesh, target, sketcher, 
                          variable_list, tensor_list, param_groups,
                          compiled_ops: CompiledOps = None, 
                          render_mode: bool = False, render_iter: int = 0,
                          post_prune: bool = False,
                          *args, **kwargs):
    ## Prelims
    opt_program = init_opt_program
    has_temp = isinstance(opt_program, (sps.SuperFrustumPackedBatchedStochasticSU, 
                                        sps.SolidSFPackedBatchedStochasticSU,
                                        )
                        )

    device = sketcher.device
    prim_params = opt_program.get_arg(0)
    n_prims = prim_params.shape[0]

    min_temp = th.tensor([AlgConf.MIN_TEMP_VAL], device=device)
    max_temp = th.tensor([AlgConf.MAX_TEMP_VAL], device=device)

    # Scale factor calculation - precompute all
    start = np.log(AlgConf.SCALE_FACTOR_START)
    end = np.log(AlgConf.SCALE_FACTOR_END)
    scale_factors = np.exp(np.arange(start, end, (end-start)/float(AlgConf.N_ITERS))).tolist()

    best_params = None
    best_shape_iou = th.tensor([-1.0], device=device)
    best_surface_iou = th.tensor([-1.0], device=device)
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

    ## Process Input:
    st = time.time()
    surface_sampled_points, curvature_weights = get_points_and_weights(target_mesh, sketcher, n_points=AlgConf.N_SURFACE_POINTS)
    curvature_weights = AlgConf.CURVATURE_WEIGHTS_SCALE * curvature_weights

    BVH = cubvh.cuBVH(target_mesh.vertices, target_mesh.faces)
    
    # Pre-allocate base_coords once
    base_coords = sketcher.get_base_coords()
    base_coords = base_coords.unsqueeze(0)#.expand(1, base_coords.shape[0], base_coords.shape[1])

    logger.debug(f"Creating BVH: {time.time() - st:.3f}s")
    if target_mask is not None:
        base_coords = base_coords[:, target_mask, :]

    # Pre-compute size for base_coords
    base_coords_size = base_coords.shape[1]

    ##  ----- OPTIM -- 
    logger.info("Starting optimization loop")
    optim = make_optimizer(param_groups)
    start_time = time.time()
    
    # Pre-allocate coordinate buffers to avoid repeated allocations
    surface_adj_points = None
    batched_surface_adj_points = None
    hard_target_surface_adj = None
    hard_target_surface_adj_fl = None
    output_sdf = None
    
    # Pre-batch surface sampled points once (doesn't change)
    batched_surface_sampled_points = surface_sampled_points.unsqueeze(0)# .expand(1, surface_sampled_points.shape[0], surface_sampled_points.shape[1])
    surface_sampled_points_size = surface_sampled_points.shape[0]

    # Render Mode
    if render_mode:
        render_params = defaultdict(list)

    if post_prune:
        stochastic_precondition_n_iters = 0
        start_temp_decay = True
    
    if AlgConf.SEMANTIC_LOSS:
        sem_points, sem_points_labels = kwargs.get("sem_points", None), kwargs.get("sem_points_labels", None)
        n_sem_classes = kwargs.get("n_sem_classes", None)

    if AlgConf.TVERSKY_MODE:
        pos_weight = AlgConf.TVERSKY_BETA * hard_target_fl           # for t(x) == 1
        neg_weight = AlgConf.TVERSKY_ALPHA * (1 - hard_target_fl)    # for t(x) == 0
        tversky_weights = pos_weight + neg_weight            # [N]
    else:
        tversky_weights = None
        tversky_weights_surface_adj = None

    i = 0
    best_iter = 0
    while (i >= 0):

        ### ITERATION CONFIG
        # optim.zero_grad()
        for variable in variable_list:
            variable.grad = None

        scale_factor = get_scale_factor(i, scale_factors)
        
        if start_temp_decay:
            temperature = exponential_temperature_schedule(i-decay_start_iter, base_iters, max_temp, min_temp, device=device)
        else:
            temperature = max_temp

        # Get transformed parameters
        # transformed_params = params_from_variables(variable_list, tensor_list)
        transformed_params = compiled_ops.param_from_variables(variable_list)
        # HACK
        if not AlgConf.SMOOTHEN:
            transformed_params[1] = transformed_params[1] * 0.0

        transformed_params.append(temperature)
        
        # Renew surface points if needed (including first iteration)
        if i % AlgConf.RENEW_PTS_ITER == 0:
            # Use in-place operations where possible
            perturbations = (th.rand_like(surface_sampled_points) - 0.5) * AlgConf.SURFACE_ADJ_PERTURBATION_SCALE
            surface_adj_points = surface_sampled_points + perturbations
            if AlgConf.BIDIR and output_sdf is not None:
                print("Sampling on pred mesh")
                with th.no_grad():
                    _, full_output_sdf = compiled_ops.compiled_assembly_execution(sketcher.get_base_coords().unsqueeze(0), transformed_params)
                    pred_mesh = sdf_to_mesh(full_output_sdf[0].detach(), sketcher)
                n_orig_points = int(surface_adj_points.shape[0] * AlgConf.BIDIR_SAMPLE_RATIO) 
                n_new_points = surface_adj_points.shape[0] - n_orig_points
                _pred_sampled_points = quick_sample_points(pred_mesh, sketcher, n_points=n_new_points)
                _pred_sampled_points = _pred_sampled_points + perturbations[n_orig_points:]
                surface_adj_points = th.cat([surface_adj_points[:n_orig_points], _pred_sampled_points], dim=0)


            surface_sampled_sdf = recompute_sdf_from_BVH(surface_adj_points, BVH, mode="watertight")
            hard_target_surface_adj = (surface_sampled_sdf <= 0.0)
            hard_target_surface_adj_fl = hard_target_surface_adj.float()
            # Pre-batch once
            batched_surface_adj_points = surface_adj_points.unsqueeze(0)# .expand(1, surface_adj_points.shape[0], surface_adj_points.shape[1])
            # TBD: Add points from Program Surface.
            if AlgConf.TVERSKY_MODE:
                surface_sampled_occ_fl = (surface_sampled_sdf <= 0.0).float()
                pos_weight = AlgConf.TVERSKY_BETA * surface_sampled_occ_fl
                neg_weight = AlgConf.TVERSKY_ALPHA * (1 - surface_sampled_occ_fl)
                tversky_weights_surface_adj = pos_weight + neg_weight 
        
        # Concatenate coordinates more efficiently
        # Pre-compute sizes
        surface_adj_size = batched_surface_adj_points.shape[1]
        
        # Concatenate all coordinates
        all_coords = th.cat([base_coords, batched_surface_adj_points, batched_surface_sampled_points], dim=1).contiguous()
        all_sizes = [base_coords_size, surface_adj_size, surface_sampled_points_size]
        
        # Stochastic preconditioning
        if i < stochastic_precondition_n_iters:
            all_coords = perform_batched_stochastic_precondition(all_coords, i-decay_start_iter, stochastic_precondition_n_iters, AlgConf.STOCHASTIC_PRECONDITION_INIT_VAL_LOWER)
            
        ## MAIN FORWARD
        primitive_sdfs, output_sdf = compiled_ops.compiled_assembly_execution(all_coords, transformed_params)
        # primitive_sdfs, output_sdf = opt_functions(all_coords, *transformed_params)
        output_sdf = output_sdf[0]
        mask = (output_sdf <= AlgConf.LOSS_BAND).float()
        mask_sum = mask.sum()
        if not mask_sum > 0:
            if i % AlgConf.LOG_FREQUENCY == 0:
                logger.warning("No valid points")
            i += 1
            transformed_params = transformed_params[:-1]
            continue
        
        # Use slicing instead of detach().clone() where gradients aren't needed
        # Only detach for statistics computation
        size0 = all_sizes[0]
        size1 = all_sizes[1]
        
        # Split output efficiently
        output_shape_sdf = output_sdf[:size0]
        output_surface_adj_sdf = output_sdf[size0:size0+size1]
        output_for_occ = output_sdf[:size0+size1]
        output_surface_sdf = output_sdf[size0+size1:]
        
        # Compute occupancy more efficiently
        output_tanh = th.tanh(output_for_occ * scale_factor)
        output_for_occ_occ = th.sigmoid(-output_tanh * scale_factor)
        output_shape_occ = output_for_occ_occ[:size0]
        output_surface_adj_occ = output_for_occ_occ[size0:size0+size1]
        
        # Split masks
        mask_shape = mask[:size0]
        mask_surface_adj = mask[size0:size0+size1]
        mask_surface = mask[size0+size1:]
        
        if AlgConf.GRADUAL_LOSS_WEIGHTS and (i - decay_start_iter) < base_iters:
            update_loss_lambda(i-decay_start_iter, base_iters)
        ##  LOSSES - optimized computations
        total_loss = compiled_ops.compiled_loss_function(output_shape_occ, hard_target_fl, 
                                                output_surface_adj_occ, hard_target_surface_adj_fl, 
                                                output_surface_sdf,
                                                primitive_sdfs, output_sdf, 
                                                mask_shape, mask_surface, mask_surface_adj,
                                                transformed_params, 
                                                scale_factor, curvature_weights)
        
        if AlgConf.TVERSKY_MODE:
            alpha = min((i - decay_start_iter) / (base_iters), 1.0) ** 2
            delta_shape = (output_shape_occ - hard_target_fl) ** 2
            delta_shape = delta_shape * tversky_weights * alpha
            loss_shape_occ = 0.5 * th.sum(mask_shape * delta_shape) / th.sum(mask_shape)
            delta_surface_adj = (output_surface_adj_occ - hard_target_surface_adj_fl) ** 2
            delta_surface_adj = delta_surface_adj * tversky_weights_surface_adj * alpha
            loss_surface_adj_occ = 0.5 * th.sum(mask_surface_adj * delta_surface_adj) / th.sum(mask_surface_adj)
            total_loss = total_loss + AlgConf.LOSS_OCC_ALPHA * loss_shape_occ + AlgConf.LOSS_SURFACE_ADJ_OCC_ALPHA * loss_surface_adj_occ
        
        if AlgConf.SEMANTIC_LOSS:
            # First gather points. 
            point_soft_assoc, sem_output_sdf = compiled_ops.point2prim_soft(sem_points, transformed_params, smu_k=SMU_K, scale_factor=scale_factor)

            sem_mask = (sem_output_sdf[0] <= AlgConf.LOSS_BAND)# .float()
            n_points = sem_mask.sum().item()
            if n_points == 0:
                continue
            semantic_loss = compute_semantic_loss(point_soft_assoc, sem_mask, sem_points_labels, n_sem_classes, transformed_params)
            total_loss = total_loss + semantic_loss * AlgConf.SEMANTIC_LOSS_ALPHA
        
        if AlgConf.LOSS_SURFACE_ADJ_SDF:
            surf_adj_sdf_delta = (output_surface_adj_sdf - surface_sampled_sdf) ** 2
            surf_adj_sdf_delta = surf_adj_sdf_delta * (1 + curvature_weights)
            mask_surface_adj_sum = mask_surface_adj.sum()
            loss_surface_adj_sdf = 0.5 * (mask_surface_adj * surf_adj_sdf_delta).sum() / (mask_surface_adj_sum + 1e-8)
            total_loss = total_loss + AlgConf.LOSS_SURFACE_ADJ_SDF_ALPHA * loss_surface_adj_sdf

        total_loss.backward()
        
        optim.step()

        # Statistics - only compute when needed
        log_iter = (i % AlgConf.LOG_FREQUENCY == 0)
        
        with th.no_grad():
            # Only detach for stats computation
            output_shape_sdf_detached = output_shape_sdf.detach()
            hard_output_shape = (output_shape_sdf_detached <= 0.0)
            shape_iou = get_iou(hard_output_shape, hard_target)
            
            output_surface_adj_sdf_detached = output_surface_adj_sdf.detach()
            hard_output_surface_adj = (output_surface_adj_sdf_detached <= 0.0)
            surface_adj_iou = get_iou(hard_output_surface_adj, hard_target_surface_adj)
            sdf_error = th.abs(output_surface_sdf.detach()).mean()
            Stats.record("iter_shape_iou", shape_iou.item(), log=False, as_list=True)
            Stats.record("iter_surface_adj_iou", surface_adj_iou.item(), log=False, as_list=True)
            Stats.record("iter_total_loss", total_loss.item(), log=False, as_list=True)

            if log_iter:
                logger.info(f"iter_{i} : shape_iou: {shape_iou.item():.3f}")
                logger.info(f"iter_{i} : surface_adj_iou: {surface_adj_iou.item():.3f}")
                logger.info(f"iter_{i} : total_loss: {total_loss.item():.5f}")
            
        # STOPPING DESIGN:
        shape_improve = (shape_iou > best_shape_iou + AlgConf.MIN_IMPROVEMENT)
        surface_improve = (surface_adj_iou > best_surface_iou + AlgConf.MIN_IMPROVEMENT)

        act_shape_improve = (shape_iou > best_shape_iou)
        act_surface_improve = (surface_adj_iou > best_surface_iou)
        if surface_improve:
            best_surface_iou = surface_adj_iou
        if shape_improve:
            best_shape_iou = shape_iou
            
        if shape_improve or surface_improve:
            if log_iter:
                logger.info(f"best_shape_iou: {shape_iou.item():.3f}")
                logger.info(f"best_total_loss: {total_loss.item():.5f}")
            iterations_without_improvement = 0
            best_iter = i
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
            any_stop = stopping_criteria_2 or stopping_criteria_3
            if any_stop and start_temp_decay:
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
                max_iter = max(i + base_iters * 2.0, AlgConf.MAX_ITER)
                logger.info(f"---- new max_iter: {max_iter}, new iter_limit: {iter_limit}  ----")

        
        if has_temp:
            transformed_params = transformed_params[:-1]
        if render_mode and (i % render_iter) == 0:
            for pos, param in enumerate(transformed_params):
                render_params[pos].append(param.detach().cpu())
        if log_iter:
            cur_time = time.time()
            iteration_rate = (cur_time - start_time) / (i + 1e-10)
            logger.info(f"Iteration rate: {iteration_rate:.3f} seconds per iteration | Best Iter: {best_iter} | Iterations without improvement: {iterations_without_improvement}")
            logger.info(f"Iteration {i}, Shape IOU: {shape_iou.item():.3f} | Surface Adj IOU: {surface_adj_iou.item():.3f} | Surface SDF Error: {sdf_error.item():.5f}")
            logger.info(f"Best Shape IOU: {best_shape_iou.item():.3f} | Best Surface Adj IOU: {best_surface_iou.item():.3f}")
            
            logger.info(f"Total Loss: {total_loss.item():.5f} | Temperature: {temperature.item():.3f} | Scale Factor: {scale_factor:.3f} | N Prims: {n_prims}")
        i += 1
    
    logger.info(f"Optimization stopped after {i} iterations - iterations_without_improvement {iterations_without_improvement}")

    # Final results
    if best_params is None:
        best_program = orig_program.tensor()
    else:
        best_program = opt_program.inject_tensor_list(best_params)
    
    Stats.record("time_total", time.time() - start_time)
    Stats.record("n_iters", i)
    Stats.record("iterations_without_improvement", iterations_without_improvement)
    if render_mode:
        for pos, param in render_params.items():
            render_params[pos] = th.stack(param)
        Stats.record("render_params", render_params, log=False)
    return best_program


def update_loss_lambda(i, base_iters, init_val=1.0):
    final_val = 0
    frac = min(i / (base_iters), 1.0)
    alpha = init_val * frac + final_val * (1 - frac)
    # AlgConf.LOSS_OCC_ALPHA = AlgConf.REAL_LOSS_OCC_ALPHA * alpha
    # AlgConf.LOSS_SURFACE_ADJ_OCC_ALPHA = AlgConf.REAL_LOSS_SURFACE_ADJ_OCC_ALPHA * alpha
    # AlgConf.LOSS_SURFACE_SDF_ALPHA = AlgConf.REAL_LOSS_SURFACE_SDF_ALPHA * alpha
    # AlgConf.LOSS_SURFACE_ADJ_SDF_ALPHA = AlgConf.REAL_LOSS_SURFACE_ADJ_SDF_ALPHA * alpha
    # AlgConf.LOSS_PARAM_REGULARIZATION_ALPHA = AlgConf.REAL_LOSS_PARAM_REGULARIZATION_ALPHA * alpha
    # AlgConf.LOSS_PRIMITIVE_COUNT_ALPHA = AlgConf.REAL_LOSS_PRIMITIVE_COUNT_ALPHA * alpha
    AlgConf.LOSS_OVERLAP_ALPHA = AlgConf.REAL_LOSS_OVERLAP_ALPHA * alpha
    AlgConf.LOSS_SHAPE_UNOVERLAP_ALPHA = AlgConf.REAL_LOSS_SHAPE_UNOVERLAP_ALPHA * alpha
    
