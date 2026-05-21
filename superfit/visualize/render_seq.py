# SPDX-FileCopyrightText: 2026 Aditya Ganeshan
# SPDX-License-Identifier: MIT

"""
Optimization replay rendering and session-based multipass animation helpers.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import distinctipy
import numpy as np
from PIL import Image

import superfit.symbolic as sps
from geolipi.utils import frames_to_animation
from sysl.shader import evaluate_to_shader
from sysl.shader_runtime.offline_render import MultipassSession, render_multipass
from sysl.utils import recursive_gls_to_sysl, recursive_sm_to_smg

from superfit.optim.expr_conversion import convert_to_unbatched, convert_to_unpacked
from superfit.optim.primitive_registry import HANDLER_REGISTRY
from superfit.symbolic.utils import fetch_singular_expr_eval
from superfit.utils.config import AlgorithmConfig as AlgConf
from superfit.utils.logger import logger

LOG_FREQ = 10
FRAME_SAMPLING_RATE = 1
CAMERA_MOVEMENT_MAGNITUDE = 0.3
CAMERA_MOVEMENT_FREQUENCY = 1.0
RENDER_SETTINGS = {
    "variables": {
        "_AA": 4,
        "_ADD_FLOOR_PLANE": False,
        "_RAYCAST_MAX_STEPS": 150,
        "cameraAngleX": np.pi / 8,
        "cameraAngleY": 3 * np.pi / 4,
        "cameraDistance": 4.0,
        "cameraOrigin": (0.0, -1.0, 0.0),
        "resolution": (1024, 1024),
    }
}


def get_expr_at_iter(param_dict, handler, cur_iter):
    n_params = len(param_dict.keys())
    cur_params = [param_dict[i][cur_iter] for i in range(n_params)]
    batched_expr = handler.packed_batched_stochastic_su_class(*cur_params)
    unbatched_packed_expr = convert_to_unbatched(batched_expr, handler)
    expression = convert_to_unpacked(unbatched_packed_expr, handler)
    return expression


def render_via_param_seq(
    in_param_dict,
    handler,
    frame_sampling_rate=FRAME_SAMPLING_RATE,
    camera_movement_magnitude=CAMERA_MOVEMENT_MAGNITUDE,
    camera_movement_frequency=CAMERA_MOVEMENT_FREQUENCY,
    version="v4",
):
    param_dict = {x: y[::frame_sampling_rate] for x, y in in_param_dict.items()}
    some_param = param_dict[0]
    n_iters = some_param.shape[0]
    n_prims = some_param.shape[1]
    renders = []
    colors = distinctipy.get_colors(n_prims + 2, rng=0)
    start_time = time.time()
    for cur_iter in range(n_iters):
        render_settings = update_camera_settings(
            {
                "render_mode": version,
                "variables": dict(RENDER_SETTINGS["variables"]),
            },
            cur_iter,
            n_iters,
            movement_magnitude=camera_movement_magnitude,
            movement_frequency=camera_movement_frequency,
        )

        expression = get_expr_at_iter(param_dict, handler, cur_iter)
        mat_expr, _ = recursive_gls_to_sysl(expression.sympy(), version=version, colors=colors)
        singular_expr = fetch_singular_expr_eval(
            mat_expr.tensor(device="cpu"),
            temperature=1.0,
            relaxed_eval=True,
            device="cpu",
        ).sympy()
        expr_smg = recursive_sm_to_smg(singular_expr)

        shader_info = evaluate_to_shader(
            expr_smg,
            mode="multipass",
            post_process_shader=["part_outline_nobg"],
            settings=render_settings,
        )
        image = render_multipass(shader_info)
        renders.append(Image.fromarray(image))
        cur_time = time.time()
        if cur_iter % LOG_FREQ == 0:
            iteration_rate = (cur_time - start_time) / (cur_iter + 1e-10)
            logger.info(
                f"Iteration rate: {iteration_rate:.3f} seconds per iteration | "
                f"Iteration {cur_iter} | Total Iterations: {n_iters} | "
                f"Time taken: {(cur_time - start_time):.2f} seconds"
            )
    return renders


def extract_param_seq(info_dict, resfit_idx=0):
    param_dict = {
        int(x.split(".")[-1]): y
        for x, y in info_dict.items()
        if f"iter_{resfit_idx}.optimization.render_params" in x
    }
    return param_dict


def update_camera_settings(
    render_settings,
    cur_iter,
    total_iters,
    movement_magnitude=0.1,
    movement_frequency=1.0,
):
    progress = cur_iter / max(total_iters - 1, 1)
    variables = render_settings["variables"]
    angle_y_offset = 2 * np.pi * progress * movement_frequency
    base_angle_y = variables["cameraAngleY"]
    variables["cameraAngleY"] = base_angle_y + movement_magnitude * np.sin(angle_y_offset)
    angle_x_offset = 2 * np.pi * progress * movement_frequency * 0.5
    base_angle_x = variables["cameraAngleX"]
    variables["cameraAngleX"] = base_angle_x + movement_magnitude * 0.3 * np.cos(angle_x_offset)
    distance_offset = 2 * np.pi * progress * movement_frequency
    base_distance = variables["cameraDistance"]
    distance_variation = movement_magnitude * 0.5 * np.sin(distance_offset)
    variables["cameraDistance"] = base_distance + distance_variation
    return render_settings


def generate_renders(info_dict, save_dir, save_name, mode="all", save_separately=True):
    version = getattr(sps, AlgConf.PRIM_TYPE)
    handler = HANDLER_REGISTRY[version]
    if mode == "all":
        n_iters = info_dict["n_iters"]
        all_renders = []
        for resfit_idx in range(n_iters):
            param_dict = extract_param_seq(info_dict, resfit_idx)
            renders = render_via_param_seq(param_dict, handler)
            all_renders.extend(renders)
            if save_separately:
                cur_save_name = f"{save_name}_{resfit_idx}"
                save_path = os.path.join(save_dir, cur_save_name)
                save_renders(renders, save_path)
        if not save_separately:
            save_path = os.path.join(save_dir, save_name)
            save_renders(all_renders, save_path)
    else:
        resfit_idx = mode
        param_dict = extract_param_seq(info_dict, resfit_idx)
        renders = render_via_param_seq(param_dict, handler)
        save_path = os.path.join(save_dir, save_name)
        save_renders(renders, save_path)


def save_renders(renders, save_path, fps: int = 30):
    logger.info(f"Saving renders to {save_path}")
    return frames_to_animation(
        renders, save_path, fps=fps, format="mp4", mp4_quality="high"
    )


def apply_variable_overrides(
    shader_info: Sequence[Dict[str, Any]],
    overrides: Mapping[str, Any],
) -> None:
    for pass_def in shader_info:
        uniforms = pass_def.get("uniforms", {})
        for var_name, value in overrides.items():
            if var_name in uniforms:
                uniforms[var_name]["init_value"] = value


_apply_variable_overrides = apply_variable_overrides


def render_shader_frames(
    frame_inputs: Iterable[tuple[Sequence[Dict[str, Any]], Mapping[str, Any]]],
    setup_env: bool = True,
    log_freq: Optional[int] = LOG_FREQ,
) -> List[Image.Image]:
    frames: List[Image.Image] = []
    start_time = time.time()
    with MultipassSession(setup_env=setup_env) as session:
        for i, (shader_info, overrides) in enumerate(frame_inputs):
            apply_variable_overrides(shader_info, overrides)
            frames.append(Image.fromarray(session.render(shader_info)))
            if log_freq is not None and i % log_freq == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = elapsed / (i + 1e-10)
                logger.info(
                    f"render_sequence: frame {i} | {rate*1000:.1f} ms/frame | "
                    f"{(i + 1) / max(elapsed, 1e-9):.2f} fps | elapsed {elapsed:.2f}s"
                )
    total = time.time() - start_time
    n = len(frames)
    if n:
        logger.info(
            f"render_sequence: rendered {n} frames in {total:.2f}s "
            f"({n / max(total, 1e-9):.2f} fps avg, {total / n * 1000:.1f} ms/frame)"
        )
    return frames


def render_sequence(
    shader_info: Sequence[Dict[str, Any]],
    frame_overrides: Iterable[Mapping[str, Any]],
    setup_env: bool = True,
    log_freq: Optional[int] = LOG_FREQ,
) -> List[Image.Image]:
    return render_shader_frames(
        ((shader_info, overrides) for overrides in frame_overrides),
        setup_env=setup_env,
        log_freq=log_freq,
    )
