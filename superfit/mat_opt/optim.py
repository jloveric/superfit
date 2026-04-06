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
import torch as th
import sysl.symbolic as sls
import geolipi.symbolic as gls
import time
from superfit.optim.utils import perform_stochastic_precondition, sample_surface_proximal_points
from superfit.utils.mesh_preprocess import quick_sample_points
from superfit.optim.param_conversion import params_from_variables, transform_to_tunable
from sysl.torch_compute.evaluate_mat_expr import recursive_evaluate_mat_expr
from superfit.utils.config import AlgorithmConfig as AlgConf
from .color_utils import material_loss_rgb_oklab_huber, laplacian_loss_flat
from .utils import query_materials_from_surface_cubvh
from superfit.utils.logger import logger
from superfit.utils.stats import Stats
from superfit.optim.primitive_registry import HANDLER_REGISTRY
import superfit.symbolic as sps
from ..symbolic.utils import INVERSE_PRIM_MAP

N_POINTS = 100_000
LAPLACIAN_LOSS_WEIGHT = 1e-5
RESAMPLE_INTERVAL = 10
LOSS_IMPROVEMENT_THRESHOLD = 0.005

def optimize_color(target_mesh, sample_mesh, new_expr, sketcher, verbose=True):

    # 2. Convert the geom expression into an optimizable form. 
    version = getattr(sps, INVERSE_PRIM_MAP[AlgConf.PRIM_TYPE])
    handler = HANDLER_REGISTRY[version]
    assert handler is not None, f"No handler found for {new_expr.base_class}"

    # 3. Set up Optim etc. 
    new_expr = new_expr.tensor()
    tensor_list = new_expr.gather_tensor_list(type_annotate=True, index_annotate=True)
    variable_list = transform_to_tunable(tensor_list, handler)
    opt_var_list = [x for ind, x in enumerate(variable_list)]
    type_annotation = [tuple(x[1:]) for x in tensor_list]

    material_params = []
    other_params = []
    for i, type_annot in enumerate(type_annotation):
        if issubclass(type_annot[0], (sls.Material, sls.SphericalRGBGrid3D)):
            material_params.append(opt_var_list[i])
        else:
            other_params.append(opt_var_list[i])
    
    Stats.record("n_material_params", len(material_params))
    Stats.record("n_other_params", len(other_params))

    optimization_config = {
        'base_iters': 350,
        "saturation_patience": 100,
    }
    param_groups = [
        {'params': material_params, 'lr': 0.05},
        # {'params': other_params, 'lr': 0.0},
    ]

    # 3. Optimize the color and MR
    if verbose:
        logger.info("Starting color optimization...")
    
    best_program, best_obj = run_color_optimization_loop(
        param_groups=param_groups,
        opt_program=new_expr,
        target_mesh=target_mesh,
        sample_mesh=sample_mesh,
        sketcher=sketcher,
        variable_list=variable_list,
        tensor_list=tensor_list,
        type_annotation=type_annotation,
        verbose=verbose,
        handler=handler,
        **optimization_config,
    )
    
    if verbose:
        final_obj = best_obj.item() if isinstance(best_obj, th.Tensor) else best_obj
        logger.info(f"Color optimization complete. Final objective: {final_obj:.6f}")

    # 4. Return the optimized color and MR
    return best_program, best_obj

def run_color_optimization_loop(
        param_groups,
        opt_program,
        target_mesh,
        sample_mesh,
        sketcher,
        variable_list,
        tensor_list,
        type_annotation,
        verbose,
        base_iters,
        saturation_patience,
        stochastic_precondition=True,
        handler=None,
        ):

    optim = th.optim.Adam(param_groups)
    device = sketcher.device

    best_params = None
    best_obj = th.tensor([1.0], device=device)
    iterations_without_improvement = 0
    i = 0
    orig_program = gls.GLFunction.from_state(opt_program.state())
    decay_start_iter = 0
    
    Stats.record("base_iters", base_iters)
    Stats.record("saturation_patience", saturation_patience)
    Stats.record("stochastic_precondition", stochastic_precondition)
    
    start_time = time.time()
    logger.info("Starting color optimization loop")
    
    while (i < base_iters) or (iterations_without_improvement < saturation_patience):
        optim.zero_grad()
        if i % RESAMPLE_INTERVAL == 0:
            # 1. From the mesh generate the targets -> Positions, Color + MR
            points, mat_targets = get_targets_v2(target_mesh, sample_mesh, sketcher)

        # Get transformed parameters
        transformed_params = params_from_variables(variable_list, tensor_list, handler)
        input_points = points.clone()
        if stochastic_precondition:
            input_points = perform_stochastic_precondition(input_points, sketcher, i-decay_start_iter, base_iters/2)
            
        coords = sketcher.make_homogenous_coords(input_points)
        cur_expr = opt_program.inject_tensor_list(transformed_params)
        outputs = recursive_evaluate_mat_expr(cur_expr, sketcher, coords=coords)
        
        mat_props = outputs[..., 1:]
        loss_mat = material_loss_rgb_oklab_huber(mat_props, mat_targets)
        # loss_tv = tv_loss_flat(cur_expr)
        loss_lap = laplacian_loss_flat(cur_expr)
        total_loss = loss_mat + loss_lap * LAPLACIAN_LOSS_WEIGHT

        # Backward pass
        total_loss.backward()
        # Handle NaN gradients
        nan_detected = False
        opt_var_list = [x for ind, x in enumerate(variable_list)]
        for opt_ind, opt_var in enumerate(opt_var_list):
            if not (opt_var.grad is None):
                if th.isnan(opt_var.grad).any():
                    nan_detected = True
                    opt_var.grad[th.isnan(opt_var.grad)] = 0.0
        if nan_detected:
            logger.warning(f"NaN detected in gradients at iteration {i}")
        
        optim.step()
        
        # Evaluation
        new_obj = loss_mat
        # Update best parameters
        # Any improve its good. 
        mat_improve =  (loss_mat < best_obj - LOSS_IMPROVEMENT_THRESHOLD)
        if loss_mat < best_obj:
            best_params = [x.detach() for x in transformed_params]
        if mat_improve:
            best_obj = loss_mat
            best_params = [x.detach() for x in transformed_params]
            iterations_without_improvement = 0
            
        else:
            iterations_without_improvement += 1
        
        # Record stats every iteration (as lists for history tracking)
        with th.no_grad():
            Stats.record("iter_loss_mat", loss_mat.item(), log=False, as_list=True)
            Stats.record("iter_loss_lap", loss_lap.item(), log=False, as_list=True)
            Stats.record("iter_total_loss", total_loss.item(), log=False, as_list=True)
            Stats.record("iter_best_obj", best_obj.item(), log=False, as_list=True)
        
        # Logging at intervals
        log_iter = (i % 10 == 0)
        if verbose and log_iter:
            logger.info(f"iter_{i} : loss_mat: {loss_mat.item():.5f}")
            logger.info(f"iter_{i} : loss_lap: {loss_lap.item():.5f}")
            logger.info(f"iter_{i} : total_loss: {total_loss.item():.5f}")
            logger.info(f"iter_{i} : best_obj: {best_obj.item():.5f}")
            logger.info(f"iter_{i} : iterations_without_improvement: {iterations_without_improvement}")
            
        i += 1
        if i == 1000:
            logger.warning("Optimization stopped after 1000 iterations")
            break
    
    # Final results
    Stats.record("time_total", time.time() - start_time)
    Stats.record("n_iters", i)
    Stats.record("iterations_without_improvement", iterations_without_improvement)
    final_best_obj = best_obj.item() if isinstance(best_obj, th.Tensor) else best_obj
    Stats.record("final_best_obj", final_best_obj)
    
    logger.info(f"Optimization stopped after {i} iterations - iterations_without_improvement {iterations_without_improvement}")
    if verbose:
        logger.info(f"Best objective achieved: {final_best_obj:.6f}")
    
    if best_params is None:
        best_program = orig_program
    else:
        best_program = opt_program.inject_tensor_list(best_params)
        
    
    return best_program, best_obj

def get_targets(target_mesh, sketcher, use_base_coords=False):
    if use_base_coords:
        surface_sampled_points = sketcher.get_base_coords()
    else:
        surface_sampled_points, sampled_normals = quick_sample_points(target_mesh, sketcher, n_points=N_POINTS)
    colors, mr = query_materials_from_surface_cubvh(target_mesh, surface_sampled_points)
    mat_targets = th.cat([colors, mr], dim=1)
    return surface_sampled_points, mat_targets

def get_targets_v2(target_mesh, sample_mesh, sketcher):
    # Get the mesh and sample on that 
    surface_sampled_points = quick_sample_points(sample_mesh, sketcher, n_points=N_POINTS)

    surface_proximal_samples = sample_surface_proximal_points(sample_mesh, n_points=N_POINTS//2, jitter_sigma=AlgConf.SURFACE_ADJ_PERTURBATION_SCALE * 0.1)
    surface_proximal_samples = th.from_numpy(surface_proximal_samples).float().to(sketcher.device)
    surface_sampled_points = th.cat([surface_sampled_points, surface_proximal_samples], dim=0)
    colors, mr = query_materials_from_surface_cubvh(target_mesh, surface_sampled_points)
    mat_targets = th.cat([colors, mr], dim=1)
    return surface_sampled_points, mat_targets
