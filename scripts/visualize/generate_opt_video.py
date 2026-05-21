# SPDX-FileCopyrightText: 2026 Aditya Ganeshan
# SPDX-License-Identifier: MIT

"""
Unified CLI for SuperFit assembly video generation.

Modes map to ``superfit.visualize.video_generators`` functions. Use
``legacy_opt`` for the original per-resfit-iter optimisation replay path.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

from superfit.visualize.config import load_defaults
from superfit.visualize.render_seq import generate_renders
from superfit.visualize.video_generators import (
    MODE_OUTPUT_STEMS,
    asset_name_from_path,
    generate_color_reveal_video,
    generate_combined_video,
    generate_explode_color_compact_video,
    generate_explode_pause_fit_back_video,
    generate_explode_spiral_video,
    generate_opt_sequence_video,
    generate_spiral_video,
    load_info_dict,
)
from superfit.utils.logger import logger

VIDEO_MODES = (
    "spiral",
    "explode_spiral",
    "color_reveal",
    "explode_color_compact",
    "explode_pause_fit_back",
    "opt_seq",
    "combine",
    "legacy_opt",
)


def _add_common_video_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input_pkl",
        "--input_path",
        dest="input_pkl",
        type=str,
        required=True,
        help="Path to primitive_assembly.pkl",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=VIDEO_MODES,
        help="Video generation mode",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory for output MP4(s)",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Output stem (without asset prefix); defaults per mode",
    )
    parser.add_argument("--fps", type=int, default=None, help="Override default FPS")
    parser.add_argument("--aa", type=int, default=None, help="Anti-aliasing level (_AA)")
    parser.add_argument(
        "--color_seed",
        type=int,
        default=None,
        help="Seed for distinct per-primitive colors",
    )
    parser.add_argument(
        "--render_size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Render resolution (default 1024 1024)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate assembly videos from primitive_assembly.pkl"
    )
    _add_common_video_args(parser)

    # legacy_opt
    parser.add_argument(
        "--legacy_mode",
        type=str,
        default="all",
        help="legacy_opt only: 'all' or resfit iter index",
    )
    parser.add_argument(
        "--no_save_separately",
        dest="save_separately",
        action="store_false",
        default=True,
        help="legacy_opt only: one combined video instead of per-iter files",
    )

    # timing / opt_seq
    parser.add_argument(
        "--time_per_iter",
        type=float,
        default=None,
        help="opt_seq: seconds of video per resfit iter",
    )
    parser.add_argument(
        "--skip_iters",
        type=int,
        nargs="*",
        default=None,
        help="opt_seq: resfit iter indices to skip",
    )

    # explode_spiral
    parser.add_argument(
        "--explode_duration_s",
        type=float,
        default=None,
        help="explode_spiral: clip length in seconds",
    )
    parser.add_argument(
        "--spiral_n_frames",
        type=int,
        default=None,
        help="spiral: number of rendered frames",
    )
    parser.add_argument(
        "--spiral_n_rots",
        type=float,
        default=None,
        help="spiral: number of full camera rotations",
    )
    parser.add_argument(
        "--spiral_ease_duration_frac",
        type=float,
        default=None,
        help="spiral: fraction of clip spent accelerating",
    )
    parser.add_argument(
        "--spiral_ease_out_duration_frac",
        type=float,
        default=None,
        help="spiral: fraction of clip spent decelerating",
    )
    parser.add_argument(
        "--pause_s",
        type=float,
        default=None,
        help="explode_pause_fit_back: pause length between explode and fit-back",
    )
    parser.add_argument(
        "--explosion_camera_pullback",
        type=float,
        default=None,
        help="Explosion modes: extra camera distance while exploded",
    )

    # combine
    parser.add_argument(
        "--combine_segments",
        type=str,
        nargs="*",
        default=None,
        help="combine: segment stems, e.g. opt_seq explode_color_compact spiral",
    )

    # camera / visualize config (passed through to load_defaults)
    parser.add_argument("--origin", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--angle_x", type=float, default=None)
    parser.add_argument("--angle_y", type=float, default=None)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--explode_distance", type=float, default=None)
    parser.add_argument("--opt_pan_start_x", type=float, default=None)
    parser.add_argument("--opt_pan_start_y", type=float, default=None)
    parser.add_argument("--opt_pan_end_x", type=float, default=None)
    parser.add_argument("--opt_pan_end_y", type=float, default=None)

    return parser.parse_args()


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.fps is not None:
        overrides["fps"] = args.fps
    if args.aa is not None:
        overrides["aa"] = args.aa
    if args.color_seed is not None:
        overrides["color_seed"] = args.color_seed
    if args.render_size is not None:
        overrides["render_size"] = tuple(args.render_size)
    if args.time_per_iter is not None:
        overrides["time_per_iter"] = args.time_per_iter
    if args.skip_iters is not None:
        overrides["skip_iters"] = args.skip_iters
    if args.explode_duration_s is not None:
        overrides["explode_duration_s"] = args.explode_duration_s
    if args.spiral_n_frames is not None:
        overrides["spiral_n_frames"] = args.spiral_n_frames
    if args.spiral_n_rots is not None:
        overrides["spiral_n_rots"] = args.spiral_n_rots
    if args.spiral_ease_duration_frac is not None:
        overrides["spiral_ease_duration_frac"] = args.spiral_ease_duration_frac
    if args.spiral_ease_out_duration_frac is not None:
        overrides["spiral_ease_out_duration_frac"] = args.spiral_ease_out_duration_frac
    if args.pause_s is not None:
        overrides["pause_s"] = args.pause_s
    if args.explosion_camera_pullback is not None:
        overrides["explosion_camera_pullback"] = args.explosion_camera_pullback
    if args.combine_segments is not None:
        overrides["combine_segments"] = tuple(args.combine_segments)
    if args.origin is not None:
        overrides["origin"] = tuple(args.origin)
    for key in (
        "angle_x",
        "angle_y",
        "distance",
        "explode_distance",
        "opt_pan_start_x",
        "opt_pan_start_y",
        "opt_pan_end_x",
        "opt_pan_end_y",
    ):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    return overrides


def _parse_legacy_mode(mode_str: str):
    if mode_str == "all":
        return "all"
    mode_int = int(mode_str)
    if mode_int < 0:
        raise ValueError("legacy_mode must be 'all' or a non-negative integer")
    return mode_int


def main(args: argparse.Namespace) -> None:
    if not os.path.exists(args.input_pkl):
        raise FileNotFoundError(f"File not found: {args.input_pkl}")
    os.makedirs(args.save_dir, exist_ok=True)

    defaults = load_defaults(args.mode, _build_overrides(args))
    asset_name = asset_name_from_path(args.input_pkl)
    stem = args.save_name or MODE_OUTPUT_STEMS.get(args.mode, args.mode)

    logger.info(f"Mode: {args.mode} | input: {args.input_pkl} | save_dir: {args.save_dir}")

    if args.mode == "spiral":
        generate_spiral_video(
            args.input_pkl, args.save_dir, save_name=stem, defaults=defaults
        )
    elif args.mode == "explode_spiral":
        generate_explode_spiral_video(
            args.input_pkl, args.save_dir, save_name=stem, defaults=defaults
        )
    elif args.mode == "color_reveal":
        generate_color_reveal_video(
            args.input_pkl, args.save_dir, save_name=stem, defaults=defaults
        )
    elif args.mode == "explode_color_compact":
        generate_explode_color_compact_video(
            args.input_pkl, args.save_dir, save_name=stem, defaults=defaults
        )
    elif args.mode == "explode_pause_fit_back":
        generate_explode_pause_fit_back_video(
            args.input_pkl, args.save_dir, save_name=stem, defaults=defaults
        )
    elif args.mode == "opt_seq":
        generate_opt_sequence_video(
            args.input_pkl, args.save_dir, save_name=stem, defaults=defaults
        )
    elif args.mode == "combine":
        segments = defaults.combine_segments
        generate_combined_video(
            args.save_dir,
            asset_name,
            save_name=stem,
            segments=segments,
            fps=defaults.fps,
        )
    elif args.mode == "legacy_opt":
        info_dict = load_info_dict(args.input_pkl)
        n_iters = info_dict.get("n_iters", 0)
        if n_iters == 0:
            raise ValueError("n_iters not found or is 0 in pkl file")
        legacy_mode = _parse_legacy_mode(args.legacy_mode)
        if isinstance(legacy_mode, int) and legacy_mode >= n_iters:
            raise ValueError(
                f"legacy_mode {legacy_mode} out of range (0-{n_iters - 1})"
            )
        generate_renders(
            info_dict,
            args.save_dir,
            stem,
            mode=legacy_mode,
            save_separately=args.save_separately,
        )
    else:
        raise ValueError(f"Unhandled mode: {args.mode}")

    logger.info(f"Done. Outputs in {args.save_dir}")


if __name__ == "__main__":
    main(parse_args())
