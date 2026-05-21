# SPDX-FileCopyrightText: 2026 Aditya Ganeshan
# SPDX-License-Identifier: MIT

"""High-level assembly video generators."""

from __future__ import annotations

import _pickle as cPickle
import io
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import distinctipy
import imageio.v2 as imageio
import numpy as np
import sympy as sp
import torch as th
from PIL import Image

import geolipi.symbolic as gls
import superfit.symbolic as sps
import sysl.symbolic as sls
from sysl.shader import evaluate_to_shader
from sysl.utils import recursive_gls_to_sysl, recursive_sm_to_smg

from superfit.optim.primitive_registry import HANDLER_REGISTRY
from superfit.symbolic.utils import fetch_singular_expr_eval, n_prims_in_expr
from superfit.utils.config import AlgorithmConfig as AlgConf
from superfit.utils.io import get_best_expr
from superfit.utils.logger import logger

from .config import CameraConfig, VideoDefaults, load_defaults
from .render_seq import (
    extract_param_seq,
    get_expr_at_iter,
    render_sequence,
    render_shader_frames,
    save_renders,
)

MODE_OUTPUT_STEMS = {
    "spiral": "spiral",
    "explode_spiral": "explode_spiral",
    "color_reveal": "color_reveal",
    "explode_color_compact": "explode_color_compact",
    "explode_pause_fit_back": "explode_pause_fit_back",
    "opt_seq": "opt_seq",
    "combine": "opt_then_explode",
    "legacy_opt": "opt_video",
}


def asset_name_from_path(pkl_path: str) -> str:
    return os.path.splitext(os.path.basename(pkl_path))[0]


class _TorchCPUUnpickler(cPickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: th.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_info_dict(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as handle:
        return _TorchCPUUnpickler(handle).load()


def _load_asset(
    pkl_path: str,
    *,
    cfg: VideoDefaults,
    prog_type: str = "pruned_program",
    temperature: float = 10000.0,
    version: str = "v4",
) -> Dict[str, Any]:
    info_dict = load_info_dict(pkl_path)
    expr = get_best_expr(info_dict, prog_type=prog_type)
    expr_in = fetch_singular_expr_eval(
        expr.tensor(device="cpu"),
        temperature=temperature,
        relaxed_eval=True,
        remove_marker=True,
        device="cpu",
    ).sympy()
    expr_in = recursive_sm_to_smg(expr_in)

    n_prims = n_prims_in_expr(expr_in)
    colors = distinctipy.get_colors(n_prims, pastel_factor=0.1, rng=cfg.color_seed)
    mat_expr, _ = recursive_gls_to_sysl(expr_in, version=version, ind=0, colors=colors)
    return {
        "info_dict": info_dict,
        "expr": expr_in,
        "colors": colors,
        "mat_expr": mat_expr,
        "asset_name": asset_name_from_path(pkl_path),
        "n_prims": n_prims,
    }


def _defaults(mode: str, defaults: Optional[VideoDefaults]) -> VideoDefaults:
    return defaults if defaults is not None else load_defaults(mode)


def _save_video(
    frames: Sequence[Image.Image],
    save_dir: str,
    asset_name: str,
    stem: str,
    fps: int,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    return save_renders(frames, os.path.join(save_dir, f"{asset_name}_{stem}"), fps=fps)


def _shader(mat_expr, cfg: VideoDefaults, post_process: str, **variables):
    shader_vars = cfg.camera.base_variables(cfg.aa, cfg.render_size)
    shader_vars.update(variables)
    return evaluate_to_shader(
        mat_expr,
        mode="multipass",
        post_process_shader=[post_process],
        settings={"variables": shader_vars},
    )


def _progress(i: int, n_frames: int) -> float:
    return i / max(n_frames - 1, 1) if n_frames > 1 else 0.0


def smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def ease_out_quint(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return 1.0 - (1.0 - x) ** 5


def lerp_color(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    return tuple(s + t * (e - s) for s, e in zip(start, end))


def spiral_angle_progress(
    progress: float,
    *,
    ease_duration_frac: float = 0.35,
    ease_rotation_frac: float = 0.18,
    ease_out_duration_frac: float = 0.0,
    ease_power: float = 3.0,
) -> float:
    progress = max(0.0, min(1.0, float(progress)))
    if ease_out_duration_frac <= 1e-9:
        t = max(1e-6, min(0.95, float(ease_duration_frac)))
        eased_at_t = max(0.0, min(1.0, float(ease_rotation_frac)))
        power = max(2.0, float(ease_power))
        if progress <= t:
            return eased_at_t * (progress / t) ** power
        return eased_at_t + (1.0 - eased_at_t) * (progress - t) / (1.0 - t)

    accel = max(0.0, min(0.95, float(ease_duration_frac)))
    decel = max(0.0, min(0.95, float(ease_out_duration_frac)))
    if accel + decel > 0.95:
        scale = 0.95 / (accel + decel)
        accel *= scale
        decel *= scale
    cruise = 1.0 - accel - decel
    total_area = 0.5 * accel + cruise + 0.5 * decel
    if total_area <= 1e-9:
        return progress

    def smoothstep_integral(s: float) -> float:
        return s**3 - 0.5 * s**4

    if progress < accel:
        area = accel * smoothstep_integral(progress / max(accel, 1e-9))
    elif progress < accel + cruise:
        area = 0.5 * accel + (progress - accel)
    else:
        s = (progress - accel - cruise) / max(decel, 1e-9)
        area = 0.5 * accel + cruise + decel * (s - smoothstep_integral(s))
    return max(0.0, min(1.0, area / total_area))


def spiral_camera_overrides(
    progress: float,
    camera: CameraConfig,
    *,
    n_rots: Optional[float] = None,
    distance: Optional[float] = None,
    spiral_ease_duration_frac: float = 0.35,
    spiral_ease_rotation_frac: float = 0.18,
    spiral_ease_out_duration_frac: float = 0.0,
    spiral_ease_power: float = 3.0,
) -> Dict[str, Any]:
    n_rots = camera.n_rots if n_rots is None else n_rots
    distance = camera.distance if distance is None else distance
    theta = 2 * np.pi * n_rots * spiral_angle_progress(
        progress,
        ease_duration_frac=spiral_ease_duration_frac,
        ease_rotation_frac=spiral_ease_rotation_frac,
        ease_out_duration_frac=spiral_ease_out_duration_frac,
        ease_power=spiral_ease_power,
    )
    return {
        "cameraAngleY": camera.angle_y + camera.angular_radius * np.cos(theta),
        "cameraAngleX": camera.angle_x + camera.angular_radius * np.sin(theta),
        "cameraDistance": distance + camera.zoom_amplitude * np.sin(camera.z_rate * theta),
        "cameraOrigin": camera.origin,
    }


def _uniform_scaled_vec3(offset_tuple, scale_uniform):
    return gls.Vec3(
        *[
            gls.BinaryOperator(sp.Float(float(c)), scale_uniform, "mul")
            for c in offset_tuple
        ]
    )


def _uniformize_translations(expr, scale_uniform):
    if isinstance(expr, gls.Translate3D):
        child = expr.get_arg(0)
        if isinstance(child, gls.GLFunction):
            child = _uniformize_translations(child, scale_uniform)
        return gls.Translate3D(child, _uniform_scaled_vec3(expr.get_arg(1), scale_uniform))
    if not isinstance(expr, gls.GLFunction):
        return expr
    return expr.func(
        *[
            _uniformize_translations(arg, scale_uniform)
            if isinstance(arg, gls.GLFunction)
            else arg
            for arg in expr.get_args()
        ]
    )


def _uniformize_albedos(expr, base_color, counter=None, uniforms=None):
    if counter is None:
        counter = [0]
    if uniforms is None:
        uniforms = []
    if isinstance(expr, sls.MaterialV4):
        i = counter[0]
        color = gls.UniformVec3(base_color, base_color, (1.0, 1.0, 1.0), f"color_{i}")
        uniforms.append(color)
        counter[0] += 1
        return expr.func(color, *list(expr.get_args())[1:]), uniforms
    if isinstance(expr, gls.GLFunction):
        args = []
        for arg in expr.get_args():
            if isinstance(arg, gls.GLFunction):
                arg, uniforms = _uniformize_albedos(arg, base_color, counter, uniforms)
            args.append(arg)
        return expr.func(*args), uniforms
    return expr, uniforms


def _exploded_material_expr(asset: Dict[str, Any], cfg: VideoDefaults):
    scale = gls.UniformFloat(
        (1.0,),
        (1.0,),
        (1.0 + cfg.explosion_amount,),
        "explosion_scale",
    )
    exploded_expr = _uniformize_translations(asset["expr"], scale)
    return recursive_gls_to_sysl(
        exploded_expr, version="v4", ind=0, colors=asset["colors"]
    )[0]


def _outline_shaders(mat_expr, cfg: VideoDefaults, **variables):
    return (
        _shader(mat_expr, cfg, "shape_outline_nobg", outline_nhbd=0, **variables),
        _shader(mat_expr, cfg, "shape_outline_nobg", outline_nhbd=1, **variables),
        _shader(mat_expr, cfg, "part_outline_nobg", outline_nhbd=1, **variables),
    )


def _clip_camera_pullback(progress: float, prep_frac: float, post_frac: float) -> float:
    progress = max(0.0, min(1.0, float(progress)))
    prep_frac = max(0.0, min(0.49, float(prep_frac)))
    post_frac = max(0.0, min(0.49, float(post_frac)))
    if prep_frac + post_frac > 0.98:
        scale = 0.98 / (prep_frac + post_frac)
        prep_frac *= scale
        post_frac *= scale
    if progress < prep_frac:
        return smoothstep01(progress / prep_frac) if prep_frac > 1e-9 else 1.0
    if progress > 1.0 - post_frac:
        return smoothstep01((1.0 - progress) / post_frac) if post_frac > 1e-9 else 0.0
    return 1.0


def _active_clip_progress(progress: float, prep_frac: float, post_frac: float) -> float:
    middle = 1.0 - prep_frac - post_frac
    if middle < 1e-9 or progress < prep_frac or progress > 1.0 - post_frac:
        return 0.0
    return (progress - prep_frac) / middle


def _bulge_envelope(
    progress: float,
    *,
    rise_frac: float,
    fall_frac: float,
    sharpness: float,
    drift: float,
) -> float:
    def logistic(x):
        return 1.0 / (1.0 + np.exp(-x))

    floor = max(
        logistic(-sharpness * rise_frac) * logistic(sharpness * (1.0 - fall_frac)),
        logistic(sharpness * (1.0 - rise_frac))
        * logistic(-sharpness * fall_frac)
        * (1.0 + drift),
    )
    rise = logistic(sharpness * (progress - rise_frac))
    fall = logistic(sharpness * ((1.0 - fall_frac) - progress))
    return max(0.0, rise * fall * (1.0 + drift * progress) - floor)


def _apply_pullback(overrides: Dict[str, Any], amount: float, pullback: float) -> None:
    overrides["cameraDistance"] = overrides["cameraDistance"] + pullback * max(
        0.0, min(1.0, float(amount))
    )


def _fit_back_timeline(t_s: float, cfg: VideoDefaults, hold_s: float) -> Tuple[float, float]:
    prep_s = min(max(cfg.explode_s * cfg.explosion_camera_prep_frac, 0.0), cfg.explode_s)
    active_explode = max(cfg.explode_s - prep_s, 1e-9)
    post_s = min(max(cfg.compact_s * cfg.explosion_camera_post_frac, 0.0), cfg.compact_s)

    if t_s < cfg.explode_s:
        if t_s < prep_s:
            return 1.0, smoothstep01(t_s / prep_s) if prep_s > 1e-9 else 1.0
        return 1.0 + cfg.explosion_amount * ease_out_quint((t_s - prep_s) / active_explode), 1.0

    if t_s < cfg.explode_s + hold_s:
        return 1.0 + cfg.explosion_amount, 1.0

    t_compact = t_s - cfg.explode_s - hold_s
    explosion_scale = 1.0 + cfg.explosion_amount * (1.0 - smoothstep01(t_compact / cfg.compact_s))
    if t_compact < cfg.compact_s - post_s:
        return explosion_scale, 1.0
    u = (t_compact - (cfg.compact_s - post_s)) / post_s if post_s > 1e-9 else 1.0
    return explosion_scale, smoothstep01(1.0 - max(0.0, min(1.0, u)))


def _skip_uniform_subtree(expr) -> bool:
    return isinstance(expr, sls.Material)


def _uniformize_leaves(expr, prefix, counter=None, descriptors=None):
    if counter is None:
        counter = [0]
    if descriptors is None:
        descriptors = []
    if not isinstance(expr, gls.GLFunction) or _skip_uniform_subtree(expr):
        return expr, descriptors

    args = []
    for arg in expr.args:
        if isinstance(arg, gls.UniformVariable):
            args.append(arg)
        elif isinstance(arg, gls.GLFunction):
            arg, descriptors = _uniformize_leaves(arg, prefix, counter, descriptors)
            args.append(arg)
        elif isinstance(arg, sp.Tuple) and len(arg) in (1, 2, 3, 4):
            name = f"{prefix}_{counter[0]}"
            vals = tuple(float(x) for x in arg)
            cls = {
                1: gls.UniformFloat,
                2: gls.UniformVec2,
                3: gls.UniformVec3,
                4: gls.UniformVec4,
            }[len(vals)]
            uniform = cls(vals, vals, vals, name)
            descriptors.append((name, len(vals)))
            counter[0] += 1
            args.append(uniform)
        else:
            args.append(arg)
    return expr.func(*args), descriptors


def _leaf_values(expr, out=None):
    if out is None:
        out = []
    if isinstance(expr, gls.GLFunction) and not _skip_uniform_subtree(expr):
        for arg in expr.args:
            if isinstance(arg, gls.GLFunction):
                _leaf_values(arg, out)
            elif isinstance(arg, sp.Tuple) and len(arg) in (1, 2, 3, 4):
                vals = tuple(float(x) for x in arg)
                out.append(vals[0] if len(vals) == 1 else vals)
    return out


def _topology_key(expr, out=None):
    if out is None:
        out = []
    if isinstance(expr, gls.GLFunction):
        out.append(type(expr).__name__)
        if _skip_uniform_subtree(expr):
            return tuple(out)
        for arg in expr.args:
            if isinstance(arg, gls.UniformVariable):
                out.append(("uniform", type(arg).__name__))
            elif isinstance(arg, gls.GLFunction):
                _topology_key(arg, out)
            elif isinstance(arg, sp.Tuple) and len(arg) in (1, 2, 3, 4):
                out.append(("tuple", len(arg)))
            else:
                out.append(("other", type(arg).__name__))
    return tuple(out)


def _opt_camera_lerp(frame_idx: int, total_frames: int, camera: CameraConfig) -> Dict[str, Any]:
    p = frame_idx / max(total_frames - 1, 1)
    return {
        "cameraAngleX": camera.opt_pan_start_x
        + p * (camera.opt_pan_end_x - camera.opt_pan_start_x),
        "cameraAngleY": camera.opt_pan_start_y
        + p * (camera.opt_pan_end_y - camera.opt_pan_start_y),
        "cameraDistance": camera.distance,
        "cameraOrigin": camera.origin,
    }


def generate_spiral_video(
    pkl_path: str,
    save_dir: str,
    *,
    save_name: Optional[str] = None,
    defaults: Optional[VideoDefaults] = None,
    mat_expr=None,
    asset_name: Optional[str] = None,
) -> str:
    cfg = _defaults("spiral", defaults)
    name = asset_name or asset_name_from_path(pkl_path)
    if mat_expr is None:
        asset = _load_asset(pkl_path, cfg=cfg)
        mat_expr = asset["mat_expr"]
        name = asset["asset_name"]

    shader_info = _shader(mat_expr, cfg, "part_outline_nobg")

    def frame_overrides():
        for i in range(cfg.spiral_n_frames):
            yield spiral_camera_overrides(
                _progress(i, cfg.spiral_n_frames),
                cfg.camera,
                n_rots=cfg.spiral_n_rots,
                spiral_ease_duration_frac=cfg.spiral_ease_duration_frac,
                spiral_ease_rotation_frac=cfg.spiral_ease_rotation_frac,
                spiral_ease_out_duration_frac=cfg.spiral_ease_out_duration_frac,
                spiral_ease_power=cfg.spiral_ease_power,
            )

    frames = render_sequence(shader_info, frame_overrides())
    return _save_video(frames, save_dir, name, save_name or MODE_OUTPUT_STEMS["spiral"], cfg.fps)


def generate_explode_spiral_video(
    pkl_path: str,
    save_dir: str,
    *,
    save_name: Optional[str] = None,
    defaults: Optional[VideoDefaults] = None,
) -> str:
    cfg = _defaults("explode_spiral", defaults)
    asset = _load_asset(pkl_path, cfg=cfg)
    n_frames = int(round(cfg.fps * cfg.explode_duration_s))
    shader_info = _shader(
        _exploded_material_expr(asset, cfg),
        cfg,
        "part_outline_nobg",
        cameraDistance=cfg.camera.explode_distance,
        explosion_scale=1.0,
    )

    def frame_overrides():
        for i in range(n_frames):
            progress = _progress(i, n_frames)
            middle_progress = _active_clip_progress(
                progress,
                cfg.explosion_camera_prep_frac,
                cfg.explosion_camera_post_frac,
            )
            bulge = _bulge_envelope(
                middle_progress,
                rise_frac=cfg.rise_frac,
                fall_frac=cfg.fall_frac,
                sharpness=cfg.sharpness,
                drift=cfg.drift,
            )
            overrides = dict(cfg.camera.base_variables(cfg.aa))
            overrides["cameraDistance"] = cfg.camera.explode_distance
            overrides["explosion_scale"] = float(1.0 + cfg.explosion_amount * bulge)
            _apply_pullback(
                overrides,
                _clip_camera_pullback(
                    progress,
                    cfg.explosion_camera_prep_frac,
                    cfg.explosion_camera_post_frac,
                ),
                cfg.explosion_camera_pullback,
            )
            yield overrides

    frames = render_sequence(shader_info, frame_overrides())
    return _save_video(
        frames,
        save_dir,
        asset["asset_name"],
        save_name or MODE_OUTPUT_STEMS["explode_spiral"],
        cfg.fps,
    )


def generate_color_reveal_video(
    pkl_path: str,
    save_dir: str,
    *,
    save_name: Optional[str] = None,
    defaults: Optional[VideoDefaults] = None,
) -> str:
    cfg = _defaults("color_reveal", defaults)
    asset = _load_asset(pkl_path, cfg=cfg)
    mat_expr = recursive_gls_to_sysl(
        asset["expr"], version="v4", ind=0, colors=asset["colors"]
    )[0]
    mat_expr, _ = _uniformize_albedos(mat_expr, cfg.base_color)
    shader_shape0, shader_shape1, shader_part1 = _outline_shaders(mat_expr, cfg)
    n_frames = int(round(cfg.fps * cfg.color_reveal_duration_s))

    def frame_inputs():
        for i in range(n_frames):
            progress = _progress(i, n_frames)
            color_t = smoothstep01(
                (progress - cfg.color_t0) / max(cfg.color_t1 - cfg.color_t0, 1e-9)
            )
            overrides = spiral_camera_overrides(
                progress,
                cfg.camera,
                n_rots=cfg.color_reveal_n_rots,
                spiral_ease_duration_frac=cfg.color_reveal_spiral_ease_duration_frac,
                spiral_ease_rotation_frac=cfg.color_reveal_spiral_ease_rotation_frac,
                spiral_ease_power=cfg.color_reveal_spiral_ease_power,
            )
            for j, target in enumerate(asset["colors"]):
                overrides[f"color_{j}"] = lerp_color(cfg.base_color, target, color_t)
            if progress >= cfg.part_switch_t:
                yield shader_part1, overrides
            elif progress < cfg.outline_on_t:
                yield shader_shape0, overrides
            else:
                yield shader_shape1, overrides

    frames = render_shader_frames(frame_inputs())
    return _save_video(
        frames,
        save_dir,
        asset["asset_name"],
        save_name or MODE_OUTPUT_STEMS["color_reveal"],
        cfg.fps,
    )


def generate_explode_color_compact_video(
    pkl_path: str,
    save_dir: str,
    *,
    save_name: Optional[str] = None,
    defaults: Optional[VideoDefaults] = None,
) -> str:
    cfg = _defaults("explode_color_compact", defaults)
    asset = _load_asset(pkl_path, cfg=cfg)
    mat_expr, _ = _uniformize_albedos(_exploded_material_expr(asset, cfg), cfg.base_color)
    shader_shape0, shader_shape1, shader_part1 = _outline_shaders(
        mat_expr, cfg, explosion_scale=1.0
    )
    n_frames = int(round(cfg.fps * (cfg.explode_s + cfg.color_s + cfg.compact_s)))

    def frame_inputs():
        for i in range(n_frames):
            t_s = i / cfg.fps
            if t_s < cfg.explode_s + cfg.color_s:
                explosion_scale, cam_pull = _fit_back_timeline(t_s, cfg, cfg.color_s)
                color_t = (
                    0.0
                    if t_s < cfg.explode_s
                    else smoothstep01((t_s - cfg.explode_s) / cfg.color_s)
                )
                shader_info = shader_shape0 if t_s < cfg.outline_full_s else shader_shape1
            else:
                explosion_scale, cam_pull = _fit_back_timeline(t_s, cfg, cfg.color_s)
                color_t = 1.0
                shader_info = shader_part1

            overrides = dict(cfg.camera.base_variables(cfg.aa))
            overrides["explosion_scale"] = float(explosion_scale)
            _apply_pullback(overrides, cam_pull, cfg.explosion_camera_pullback)
            for j, target in enumerate(asset["colors"]):
                overrides[f"color_{j}"] = lerp_color(cfg.base_color, target, color_t)
            yield shader_info, overrides

    frames = render_shader_frames(frame_inputs())
    return _save_video(
        frames,
        save_dir,
        asset["asset_name"],
        save_name or MODE_OUTPUT_STEMS["explode_color_compact"],
        cfg.fps,
    )


def generate_explode_pause_fit_back_video(
    pkl_path: str,
    save_dir: str,
    *,
    save_name: Optional[str] = None,
    defaults: Optional[VideoDefaults] = None,
) -> str:
    cfg = _defaults("explode_pause_fit_back", defaults)
    asset = _load_asset(pkl_path, cfg=cfg)
    shader_info = _shader(
        _exploded_material_expr(asset, cfg),
        cfg,
        "part_outline_nobg",
        explosion_scale=1.0,
        outline_nhbd=cfg.outline_nhbd,
    )
    n_frames = int(round(cfg.fps * (cfg.explode_s + cfg.pause_s + cfg.compact_s)))

    def frame_overrides():
        for i in range(n_frames):
            explosion_scale, cam_pull = _fit_back_timeline(i / cfg.fps, cfg, cfg.pause_s)
            overrides = dict(cfg.camera.base_variables(cfg.aa))
            overrides["explosion_scale"] = float(explosion_scale)
            _apply_pullback(overrides, cam_pull, cfg.explosion_camera_pullback)
            yield overrides

    frames = render_sequence(shader_info, frame_overrides())
    return _save_video(
        frames,
        save_dir,
        asset["asset_name"],
        save_name or MODE_OUTPUT_STEMS["explode_pause_fit_back"],
        cfg.fps,
    )


def _build_smg_from_params(pdict, handler, step, palette):
    expression = get_expr_at_iter(pdict, handler, step)
    mat_expr, _ = recursive_gls_to_sysl(expression.sympy(), version="v4", colors=palette)
    singular_expr = fetch_singular_expr_eval(
        mat_expr.tensor(device="cpu"),
        temperature=1.0,
        relaxed_eval=True,
        device="cpu",
    ).sympy()
    return recursive_sm_to_smg(singular_expr)


def _max_prims_in_opt_plan(param_dicts, handler, plan_by_iter, iters_with_params) -> int:
    max_prims = 1
    for resfit_idx in iters_with_params:
        for step in plan_by_iter[resfit_idx]:
            expr = get_expr_at_iter(param_dicts[resfit_idx], handler, step)
            max_prims = max(max_prims, n_prims_in_expr(expr.sympy()))
    return max_prims


def generate_opt_sequence_video(
    pkl_path: str,
    save_dir: str,
    *,
    save_name: Optional[str] = None,
    defaults: Optional[VideoDefaults] = None,
    info_dict=None,
) -> str:
    cfg = _defaults("opt_seq", defaults)
    info_dict = load_info_dict(pkl_path) if info_dict is None else info_dict
    asset_name = asset_name_from_path(pkl_path)
    n_iters = info_dict["n_iters"]

    if np.isscalar(cfg.time_per_iter):
        time_per_iter = [float(cfg.time_per_iter)] * n_iters
    else:
        time_per_iter = [float(t) for t in cfg.time_per_iter]

    param_dicts = {r: extract_param_seq(info_dict, r) for r in range(n_iters)}
    skip = set(cfg.skip_iters)
    iters_with_params = [r for r in range(n_iters) if param_dicts[r] and r not in skip]
    if not iters_with_params:
        raise RuntimeError(
            f"No 'iter_<r>.optimization.render_params.<i>' keys found in {pkl_path}. "
            "Re-run optimisation with render_params enabled."
        )

    plan_by_iter = {}
    for r in iters_with_params:
        n_steps = param_dicts[r][0].shape[0]
        n_frames = max(1, int(round(time_per_iter[r] * cfg.fps)))
        plan_by_iter[r] = (
            [n_steps - 1]
            if n_frames == 1
            else np.linspace(0, n_steps - 1, n_frames).round().astype(int).tolist()
        )

    version = getattr(sps, AlgConf.PRIM_TYPE)
    handler = HANDLER_REGISTRY[version]
    max_prims = _max_prims_in_opt_plan(
        param_dicts, handler, plan_by_iter, iters_with_params
    )
    palette = [cfg.base_color] * (max_prims + 8)
    total_frames = sum(len(v) for v in plan_by_iter.values())
    logger.info(
        f"opt_seq: palette length {len(palette)} "
        f"(max primitives {max_prims} over planned steps)"
    )

    def compile_for_topology(smg_step, prefix):
        uniform_expr, descriptors = _uniformize_leaves(smg_step, prefix)
        return (
            _shader(
                uniform_expr,
                cfg,
                "shape_outline_nobg",
                outline_nhbd=cfg.outline_nhbd,
            ),
            descriptors,
        )

    n_compiles = 0

    def frame_inputs():
        nonlocal n_compiles
        frame_idx = 0
        for resfit_idx in iters_with_params:
            topo_cache = {}
            epoch = 0
            for step in plan_by_iter[resfit_idx]:
                smg_step = _build_smg_from_params(
                    param_dicts[resfit_idx], handler, step, palette
                )
                key = _topology_key(smg_step)
                if key not in topo_cache:
                    topo_cache[key] = compile_for_topology(
                        smg_step, prefix=f"optvar_{resfit_idx}_e{epoch}"
                    )
                    n_compiles += 1
                    epoch += 1

                shader_info, descriptors = topo_cache[key]
                overrides = _opt_camera_lerp(frame_idx, total_frames, cfg.camera)
                for (name, _dim), value in zip(descriptors, _leaf_values(smg_step)):
                    overrides[name] = value
                frame_idx += 1
                yield shader_info, overrides

    frames = render_shader_frames(frame_inputs())
    logger.info(
        f"opt_seq: rendered {len(frames)} frames, {n_compiles} shader compiles "
        f"(vs {len(frames)} per-step recompile)"
    )
    return _save_video(
        frames,
        save_dir,
        asset_name,
        save_name or MODE_OUTPUT_STEMS["opt_seq"],
        cfg.fps,
    )


def read_video_frames(path: str) -> List[Image.Image]:
    reader = imageio.get_reader(path)
    try:
        return [Image.fromarray(frame) for frame in reader]
    finally:
        reader.close()


def concatenate_videos(
    video_paths: Sequence[str],
    save_path: str,
    *,
    fps: int = 30,
) -> str:
    frames: List[Image.Image] = []
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing video for concatenation: {path}")
        part = read_video_frames(path)
        logger.info(f"Loaded {len(part)} frames from {path}")
        frames.extend(part)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    out = save_renders(frames, save_path, fps=fps)
    logger.info(f"Saved concatenated video ({len(frames)} frames) to {out}")
    return out


def generate_combined_video(
    save_dir: str,
    asset_name: str,
    *,
    save_name: Optional[str] = None,
    segments: Optional[Sequence[str]] = None,
    fps: int = 30,
) -> str:
    segments = segments or ("opt_seq", "explode_color_compact", "spiral")
    paths = [os.path.join(save_dir, f"{asset_name}_{seg}.mp4") for seg in segments]
    stem = save_name or MODE_OUTPUT_STEMS["combine"]
    return concatenate_videos(paths, os.path.join(save_dir, f"{asset_name}_{stem}"), fps=fps)
