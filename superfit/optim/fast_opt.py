
import time
import torch as th
import numpy as np
import geolipi.symbolic as gls
import superfit.symbolic as sps
import cubvh
from .param_conversion import params_from_variables
from ..symbolic.utils import gather_primitives
from ..utils.config import AlgorithmConfig as AlgConf
from .utils import (perform_batched_stochastic_precondition, exponential_temperature_schedule, 
                    recompute_sdf_from_BVH, get_mask_scaled_aabb)
from .curvature import get_points_and_weights
from .measures import get_iou
from .main_opt import make_optimizer, get_scale_factor
from .compile_function import _params_from_variables_fast_sf

def run_optimization_loop_fast(init_opt_program, target_mesh, target, sketcher, 
                          variable_list, tensor_list, param_groups,
                          compiled_ops):
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
    best_obj = th.tensor([-1.0], device=device)
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
    print("Processing targets", time.time() - st)
    
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
    base_coords = base_coords.unsqueeze(0).expand(1, base_coords.shape[0], base_coords.shape[1])

    print("creating BVH", time.time() - st)
    if target_mask is not None:
        base_coords = base_coords[:, target_mask, :]

    # Pre-compute size for base_coords
    base_coords_size = base_coords.shape[1]

    ##  ----- OPTIM -- 
    print("Starting optimization loop")
    optim = make_optimizer(param_groups)
    start_time = time.time()
    
    # Pre-allocate coordinate buffers to avoid repeated allocations
    surface_adj_points = None
    batched_surface_adj_points = None
    hard_target_surface_adj = None
    hard_target_surface_adj_fl = None
    
    # Pre-batch surface sampled points once (doesn't change)
    batched_surface_sampled_points = surface_sampled_points.unsqueeze(0).expand(1, surface_sampled_points.shape[0], surface_sampled_points.shape[1])
    surface_sampled_points_size = surface_sampled_points.shape[0]
    
    i = 0
    while (i < 10_000):

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
        transformed_params = _params_from_variables_fast_sf(variable_list)
        transformed_params.append(temperature)
        
        # Renew surface points if needed (including first iteration)
        if i % AlgConf.RENEW_PTS_ITER == 0:
            # Use in-place operations where possible
            perturbations = (th.rand_like(surface_sampled_points) - 0.5) * AlgConf.SURFACE_ADJ_PERTURBATION_SCALE
            surface_adj_points = surface_sampled_points + perturbations
            surface_sampled_sdf = recompute_sdf_from_BVH(surface_adj_points, BVH, mode="watertight")
            hard_target_surface_adj = (surface_sampled_sdf <= 0.0)
            hard_target_surface_adj_fl = hard_target_surface_adj.float()
            # Pre-batch once
            batched_surface_adj_points = surface_adj_points.unsqueeze(0).expand(1, surface_adj_points.shape[0], surface_adj_points.shape[1])
        
        # Concatenate coordinates more efficiently
        # Pre-compute sizes
        surface_adj_size = batched_surface_adj_points.shape[1]
        
        # Concatenate all coordinates
        all_coords = th.cat([base_coords, batched_surface_adj_points, batched_surface_sampled_points], dim=1)
        all_sizes = [base_coords_size, surface_adj_size, surface_sampled_points_size]
        
        # Stochastic preconditioning
        if i < stochastic_precondition_n_iters:
            all_coords = perform_batched_stochastic_precondition(all_coords, i-decay_start_iter, stochastic_precondition_n_iters)
            
        ## MAIN FORWARD
        primitive_sdfs, output_sdf = compiled_ops.compiled_assembly_execution(all_coords, *transformed_params)
        # primitive_sdfs, output_sdf = opt_functions(all_coords, *transformed_params)
        output_sdf = output_sdf[0]
        mask = (output_sdf <= AlgConf.LOSS_BAND).float()
        mask_sum = mask.sum()
        if not mask_sum > 0:
            if i % AlgConf.LOG_FREQUENCY == 0:
                print("No valid points")
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

        ##  LOSSES - optimized computations
        total_loss = compiled_ops.compiled_loss_function(output_shape_occ, hard_target_fl, 
        # total_loss = compute_total_loss(output_shape_occ, hard_target_fl, 
                                                output_surface_adj_occ, hard_target_surface_adj_fl, 
                                                output_surface_sdf,
                                                primitive_sdfs, output_sdf, 
                                                mask_shape, mask_surface, mask_surface_adj,
                                                transformed_params, temperature, 
                                                scale_factor, curvature_weights)
        # params = [output_shape_occ, hard_target_fl, output_surface_adj_occ, hard_target_surface_adj_fl, 
        #                                         output_surface_sdf,
        #                                         primitive_sdfs, output_sdf, 
        #                                         mask_shape, mask_surface, mask_surface_adj,
        #                                         temperature,  curvature_weights]
        total_loss.backward()
        
        optim.step()

        # Statistics - only compute when needed
        should_log = (i % AlgConf.LOG_FREQUENCY == 0)
        cur_iter_stats = {}
        
        with th.no_grad():
            # Only detach for stats computation
            output_shape_sdf_detached = output_shape_sdf.detach()
            hard_output_shape = (output_shape_sdf_detached <= 0.0)
            shape_iou = get_iou(hard_output_shape, hard_target)
            
            output_surface_adj_sdf_detached = output_surface_adj_sdf.detach()
            hard_output_surface_adj = (output_surface_adj_sdf_detached <= 0.0)
            surface_adj_iou = get_iou(hard_output_surface_adj, hard_target_surface_adj)
            sdf_error = th.abs(output_surface_sdf.detach()).mean()

            if should_log:
                cur_iter_stats["shape_iou"] = shape_iou.item()
                cur_iter_stats["total_loss"] = total_loss.item()
            
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
            if should_log:
                best_stats = {x: y for x, y in cur_iter_stats.items()}
            iterations_without_improvement = 0
            best_obj = shape_iou
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
                print("===========Stopping due to stopping criteria===========")
                print(f"cur_iter: {i}, iter_limit: {iter_limit}, max_iter: {max_iter}")
                print(f"iterations_without_improvement: {iterations_without_improvement}, saturation patience: {AlgConf.SAT_PATIENCE}")
                break
            if any_stop and not start_temp_decay:
                print("===========Starting Temp Decay===========")
                start_temp_decay = True
                decay_start_iter = i
                iterations_without_improvement = 0
                stochastic_precondition_n_iters = i + stochastic_precondition_n_iters
                iter_limit = i + base_iters
                max_iter = max(iter_limit, AlgConf.MAX_ITER)
                print(f"---- new max_iter: {max_iter}, new iter_limit: {iter_limit}  ----")

        
        i += 1
        if has_temp:
            transformed_params = transformed_params[:-1]

        if should_log:
            cur_time = time.time()
            iteration_rate = (cur_time - start_time) / i
            print(f"Iteration rate: {iteration_rate:.3f} seconds per iteration")
            print(f"Iteration {i}, Shape IOU: {shape_iou.item():.3f} | Surface Adj IOU: {surface_adj_iou.item():.3f} | Surface SDF Error: {sdf_error.item():.5f}")
            print(f"Total Loss: {total_loss.item():.5f} | Temperature: {temperature.item():.3f} | Scale Factor: {scale_factor:.3f} | N Prims: {n_prims}")
            print(f"Iterations without improvement: {iterations_without_improvement}, new_iou: {shape_iou.item():.3f}, best_iou: {best_shape_iou.item():.3f} | new_surface_iou: {surface_adj_iou.item():.3f}, best_surface_iou: {best_surface_iou.item():.3f}")
    
    print(f"Optimization stopped after {i} iterations - iterations_without_improvement {iterations_without_improvement}")

    # Final results
    if best_params is None:
        best_program = orig_program.tensor()
        best_stats = {}
    else:
        best_program = opt_program.inject_tensor_list(best_params)
        if 'best_stats' not in locals():
            best_stats = {}
    
    return best_program, best_stats

