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
import os
import numpy as np
import distinctipy
import time
from PIL import Image
import superfit.symbolic as sps
from sysl.utils import recursive_sm_to_smg, recursive_gls_to_sysl
from sysl.shader import evaluate_to_shader
from geolipi.utils import frames_to_animation
from sysl.shader_runtime.offline_render import render_multipass
from superfit.optim.primitive_registry import HANDLER_REGISTRY
from superfit.optim.expr_conversion import convert_to_unbatched, convert_to_unpacked
from .config import AlgorithmConfig as AlgConf
from .logger import logger
from ..symbolic.utils import fetch_singular_expr_eval, INVERSE_PRIM_MAP

LOG_FREQ = 10

RENDER_SETTINGS = {
    "variables": {
        "_AA": 1,
        "_ADD_FLOOR_PLANE": False,
        "_RAYCAST_MAX_STEPS": 150,
        # Camera Params
        "cameraAngleX": np.pi/4,
        "cameraAngleY": 3 * np.pi/4,
        "cameraDistance": 3.0,
    }

}


def get_expr_at_iter(param_dict, handler, cur_iter):
    n_params = len(param_dict.keys())
    cur_params = [param_dict[i][cur_iter] for i in range(n_params)]
    # Create Batched Expr
    batched_expr = handler.packed_batched_stochastic_su_class(*cur_params)
    #  convert to simple expr:
    unbatched_packed_expr = convert_to_unbatched(batched_expr, handler)
    expression = convert_to_unpacked(unbatched_packed_expr, handler)
    return expression

def render_via_param_seq(in_param_dict, handler, frame_sampling_rate=10, 
                         camera_movement_magnitude=0.3, camera_movement_frequency=1.0, 
                         version="v4"):
    param_dict = {x:y[::frame_sampling_rate] for x, y in in_param_dict.items()}
    some_param = param_dict[0]
    n_params = len(param_dict.keys())

    n_iters = some_param.shape[0]
    n_prims = some_param.shape[1]
    renders = []
    colors = distinctipy.get_colors(n_prims+2)
    start_time = time.time()
    for cur_iter in range(n_iters):
        # Settings:
        render_settings = {x:y for x, y in RENDER_SETTINGS.items()}
        render_settings['render_mode'] = version
        # Update camera settings
        render_settings = update_camera_settings(
            render_settings, cur_iter, n_iters,
            movement_magnitude=camera_movement_magnitude,
            movement_frequency=camera_movement_frequency
        )

        expression = get_expr_at_iter(param_dict, handler, cur_iter)
        mat_expr, _ = recursive_gls_to_sysl(expression.sympy(), version=version, colors=colors)
        singular_expr = fetch_singular_expr_eval(mat_expr.tensor(), temperature=1.0, relaxed_eval=True).sympy()
        expr_smg = recursive_sm_to_smg(singular_expr)

        shader_info = evaluate_to_shader(expr_smg, mode="multipass", post_process_shader=["part_outline_nobg"], settings=render_settings)
        # shader_info = evaluate_to_shader(expr_smg, mode="singlepass", settings=render_settings)
        image = render_multipass(shader_info, size=(256, 256))
        image = Image.fromarray(image)
        renders.append(image)
        cur_time = time.time()
        if cur_iter % LOG_FREQ == 0:
            iteration_rate = (cur_time - start_time) / (cur_iter + 1e-10)
            logger.info(f"Iteration rate: {iteration_rate:.3f} seconds per iteration | Iteration {cur_iter} | Total Iterations: {n_iters} | Time taken: {(cur_time - start_time):.2f} seconds")
    return renders

def extract_param_seq(info_dict, resfit_idx=0):
    param_dict = {int(x.split(".")[-1]):y for x, y in info_dict.items() if f"iter_{resfit_idx}.optimization.render_params" in x}
    return param_dict

def update_camera_settings(render_settings, cur_iter, total_iters, 
                          movement_magnitude=0.1, movement_frequency=1.0):
    """
    Update camera settings to create smooth motion during optimization.
    
    Args:
        render_settings: Dictionary containing camera settings to modify
        cur_iter: Current iteration number
        total_iters: Total number of iterations
        movement_magnitude: Controls the amplitude of camera movement (default: 0.1)
        movement_frequency: Controls how fast the camera moves (default: 1.0)
    
    Returns:
        Modified render_settings dictionary
    """
    # Normalize iteration to [0, 1] range
    progress = cur_iter / max(total_iters - 1, 1)
    
    # Create circular motion for horizontal angle (cameraAngleY)
    # Full circle over the course of optimization
    angle_y_offset = 2 * np.pi * progress * movement_frequency
    render_settings["variables"]["cameraAngleY"] = 3 * np.pi/4 + movement_magnitude * np.sin(angle_y_offset)
    
    # Add slight vertical variation (cameraAngleX)
    # Smaller amplitude for vertical movement
    angle_x_offset = 2 * np.pi * progress * movement_frequency * 0.5
    render_settings["variables"]["cameraAngleX"] = np.pi/6 + movement_magnitude * 0.3 * np.cos(angle_x_offset)
    
    # Slow zoom in and out effect (cameraDistance)
    # Zoom in and out over the course of optimization
    distance_offset = 2 * np.pi * progress * movement_frequency
    base_distance = 3.0
    distance_variation = movement_magnitude * 0.5 * np.sin(distance_offset)
    render_settings["variables"]["cameraDistance"] = base_distance + distance_variation
    
    return render_settings


def generate_renders(info_dict, save_dir, save_name,mode="all", save_seperately=True):
    version = getattr(sps, AlgConf.PRIM_TYPE)
    handler = HANDLER_REGISTRY[version]
    if mode == "all":
        n_iters = info_dict["n_iters"]
        all_renders = []
        for resfit_idx in range(n_iters):
            param_dict = extract_param_seq(info_dict, resfit_idx)
            renders = render_via_param_seq(param_dict, handler)
            all_renders.extend(renders)
            if save_seperately:
                cur_save_name = f"{save_name}_{resfit_idx}"
                save_path = os.path.join(save_dir, cur_save_name)
                save_renders(renders, save_path)

        if not save_seperately:
            save_path = os.path.join(save_dir, save_name)
            save_renders(all_renders, save_path)
    else:
        resfit_idx = mode
        param_dict = extract_param_seq(info_dict, resfit_idx)
        renders = render_via_param_seq(param_dict, handler)
        all_renders = renders
        save_path = os.path.join(save_dir, save_name)
        save_renders(all_renders, save_path)


def save_renders(renders, save_path):
    logger.info(f"Saving renders to {save_path}")
    frames_to_animation(renders,
        save_path,
        fps=30,
        format="mp4",
        mp4_quality="high")
    return save_path
