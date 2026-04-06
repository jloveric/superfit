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
USE_CUDA = True
CLEAN_UP_DELTA = 0.005
MIN_VOLUME_LIMIT = 0.0001

# Explicit base paths (edit for your machine/cluster).
DATA_BASE = "/users/aganesh8/data/aganesh8/data"
PROJECT_BASE = "/users/aganesh8/data/aganesh8/projects/project_sf"
OUTPUTS_BASE = "/oscar/data/dritchi1/aganesh8/data/project_sf/outputs"

# Derived paths.
TOY4K_PATH_PREFIX = f"{DATA_BASE}/toys4k_obj_files"
PARTOBJAVERSE_BASE = f"{DATA_BASE}/partobjaverse"
PARTOBJAVERSE_MESH_DIR = f"{PARTOBJAVERSE_BASE}/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh"
PARTOBJAVERSE_INSTANCE_DIR = f"{PARTOBJAVERSE_BASE}/PartObjaverse-Tiny_instance_gt"
AOT_ARTIFACT_DIR = f"{DATA_BASE}/project_sf/aot"
SAVE_DIR_BASE = OUTPUTS_BASE
TOY4K_CSV_FILE = "superfit/dataset/new_testset.csv"
SEMANTIC_LOC = f"{PROJECT_BASE}/PartField/"