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
import json
import os
from pathlib import Path

USE_CUDA = True
CLEAN_UP_DELTA = 0.005
MIN_VOLUME_LIMIT = 0.0001

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_ENV_VAR = "SUPERFIT_CONFIG"
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "superfit_config.json"


def _load_local_config() -> dict:
    config_path = os.environ.get(_CONFIG_ENV_VAR)
    path = Path(config_path).expanduser() if config_path else _DEFAULT_CONFIG_PATH

    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"SuperFit config must be a JSON object: {path}")
    return config


def _resolve_base_path(config: dict, key: str, default: Path) -> str:
    value = config.get(key)
    path = Path(os.path.expandvars(str(value))).expanduser() if value else default
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return str(path)


_LOCAL_CONFIG = _load_local_config()

# Explicit base paths. Override with superfit_config.json, or set SUPERFIT_CONFIG
# to point at another JSON file with data_base, project_base, and outputs_base.
DATA_BASE = _resolve_base_path(_LOCAL_CONFIG, "data_base", _REPO_ROOT / "data")
PROJECT_BASE = _resolve_base_path(_LOCAL_CONFIG, "project_base", _REPO_ROOT)
OUTPUTS_BASE = _resolve_base_path(_LOCAL_CONFIG, "outputs_base", _REPO_ROOT / "outputs")

# Derived paths.
TOY4K_PATH_PREFIX = f"{DATA_BASE}/toys4k_obj_files"
PARTOBJAVERSE_BASE = f"{DATA_BASE}/partobjaverse"
PARTOBJAVERSE_MESH_DIR = f"{PARTOBJAVERSE_BASE}/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh"
PARTOBJAVERSE_INSTANCE_DIR = f"{PARTOBJAVERSE_BASE}/PartObjaverse-Tiny_instance_gt"
AOT_ARTIFACT_DIR = f"{DATA_BASE}/project_sf/aot"
SAVE_DIR_BASE = OUTPUTS_BASE
TOY4K_FILE_PATH = "superfit/dataset/new_testset.csv"
SEMANTIC_LOC = f"{PROJECT_BASE}/PartField/"
