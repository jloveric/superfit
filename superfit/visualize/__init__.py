# SPDX-FileCopyrightText: 2026 Aditya Ganeshan
# SPDX-License-Identifier: MIT

"""
Video rendering and optimization replay utilities for SuperFit assemblies.
"""

from .render_seq import (
    apply_variable_overrides,
    extract_param_seq,
    generate_renders,
    get_expr_at_iter,
    render_shader_frames,
    render_sequence,
    render_via_param_seq,
    save_renders,
    update_camera_settings,
    _apply_variable_overrides,
)
from .video_generators import (
    generate_color_reveal_video,
    generate_combined_video,
    generate_explode_color_compact_video,
    generate_explode_pause_fit_back_video,
    generate_explode_spiral_video,
    generate_opt_sequence_video,
    generate_spiral_video,
    asset_name_from_path,
    concatenate_videos,
    load_info_dict,
    spiral_angle_progress,
)
from .config import CameraConfig, VideoDefaults, load_defaults

__all__ = [
    "CameraConfig",
    "VideoDefaults",
    "load_defaults",
    "apply_variable_overrides",
    "extract_param_seq",
    "generate_renders",
    "get_expr_at_iter",
    "render_shader_frames",
    "render_sequence",
    "render_via_param_seq",
    "save_renders",
    "update_camera_settings",
    "_apply_variable_overrides",
    "asset_name_from_path",
    "load_info_dict",
    "generate_spiral_video",
    "generate_explode_spiral_video",
    "generate_color_reveal_video",
    "generate_explode_color_compact_video",
    "generate_explode_pause_fit_back_video",
    "generate_opt_sequence_video",
    "generate_combined_video",
    "concatenate_videos",
    "spiral_angle_progress",
]
