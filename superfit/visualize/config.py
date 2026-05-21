# SPDX-FileCopyrightText: 2026 Aditya Ganeshan
# SPDX-License-Identifier: MIT

"""
Shared configuration objects for visualize rendering and video generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CameraConfig:
    """Camera settings passed into shader ``settings['variables']``."""

    angle_x: float = np.pi / 8
    angle_y: float = 3 * np.pi / 4
    distance: float = 4.0
    origin: Tuple[float, float, float] = (0.0, -1.0, 0.0)

    # Spiral motion (NeRF/LLFF-style wobble about look-at axis).
    angular_radius: float = 0.12
    zoom_amplitude: float = 0.20
    n_rots: float = 2.0
    z_rate: float = 0.5

    # Presets used by specific video modes.
    preview_distance: float = 3.0
    preview_origin: Tuple[float, float, float] = (1.0, 0.0, -1.5)
    explode_distance: float = 5.5

    # Optimisation-sequence linear pan endpoints.
    opt_pan_start_x: float = np.pi / 80
    opt_pan_start_y: float = 10 * np.pi / 8
    opt_pan_end_x: float = np.pi / 8
    opt_pan_end_y: float = 3 * np.pi / 4

    def base_variables(
        self,
        aa: int = 1,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        variables = {
            "_AA": aa,
            "_RAYCAST_MAX_STEPS": 200,
            "cameraAngleX": self.angle_x,
            "cameraAngleY": self.angle_y,
            "cameraDistance": self.distance,
            "cameraOrigin": self.origin,
            "_FOCAL_LENGTH": 2.25,
        }
        if resolution is not None:
            variables["resolution"] = tuple(resolution)
        return variables


@dataclass
class VideoDefaults:
    """Per-mode default knobs for video generation."""

    aa: int = 4
    render_size: Tuple[int, int] = (1024, 1024)
    fps: int = 30
    base_color: Tuple[float, float, float] = (0.55, 0.55, 0.55)
    color_seed: Optional[int] = 0
    camera: CameraConfig = field(default_factory=CameraConfig)

    # spiral
    spiral_n_frames: int = 60
    spiral_n_rots: float = 2.0
    spiral_ease_duration_frac: float = 0.35
    spiral_ease_rotation_frac: float = 0.18
    spiral_ease_out_duration_frac: float = 0.0
    spiral_ease_power: float = 3.0

    # explode_spiral
    explode_duration_s: float = 3.5
    explosion_amount: float = 1.0
    explosion_camera_pullback: float = 1.0
    explosion_camera_prep_frac: float = 0.15
    explosion_camera_post_frac: float = 0.15
    rise_frac: float = 0.15
    fall_frac: float = 0.15
    sharpness: float = 18.0
    drift: float = 0.10

    # color_reveal
    color_reveal_duration_s: float = 4.0
    color_reveal_n_rots: float = 1.0
    color_reveal_spiral_ease_duration_frac: float = 0.35
    color_reveal_spiral_ease_rotation_frac: float = 0.18
    color_reveal_spiral_ease_power: float = 3.0
    outline_on_t: float = 0.05
    color_t0: float = 0.20
    color_t1: float = 0.70
    part_switch_t: float = 0.78

    # explode_color_compact
    explode_s: float = 1.0
    color_s: float = 0.5
    pause_s: float = 0.5
    compact_s: float = 0.5
    outline_full_s: float = 0.20

    # opt_seq
    time_per_iter: float = 1.0
    outline_nhbd: int = 1
    skip_iters: Sequence[int] = field(default_factory=list)

    # combine
    combine_segments: Sequence[str] = ("opt_seq", "explode_color_compact", "spiral")


# Script-level defaults keyed by CLI mode name.
MODE_DEFAULTS: Dict[str, VideoDefaults] = {
    "spiral": VideoDefaults(),
    "explode_spiral": VideoDefaults(),
    "color_reveal": VideoDefaults(),
    "explode_color_compact": VideoDefaults(
        explosion_camera_prep_frac=0.25,
        explosion_camera_post_frac=0.35,
    ),
    "explode_pause_fit_back": VideoDefaults(
        explosion_camera_prep_frac=0.25,
        explosion_camera_post_frac=0.35,
    ),
    "opt_seq": VideoDefaults(),
    "combine": VideoDefaults(),
    "legacy_opt": VideoDefaults(),
}


def load_defaults(mode: str, overrides: Optional[Dict[str, Any]] = None) -> VideoDefaults:
    """Return defaults for ``mode``, optionally patched with flat overrides."""
    if mode not in MODE_DEFAULTS:
        raise ValueError(f"Unknown mode {mode!r}. Expected one of {sorted(MODE_DEFAULTS)}")
    defaults = MODE_DEFAULTS[mode]
    if not overrides:
        return defaults

    flat = dict(overrides)
    camera_keys = {f.name for f in fields(CameraConfig)}
    defaults_keys = {f.name for f in fields(VideoDefaults)}

    camera_updates = {k: flat.pop(k) for k in list(flat) if k in camera_keys}
    video_updates = {k: flat[k] for k in flat if k in defaults_keys}
    camera = replace(defaults.camera, **camera_updates)
    return replace(defaults, camera=camera, **video_updates)
