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
from dataclasses import dataclass, field

import dataclasses
import random

import numpy as np
import json
import torch as th
from typing import Dict, Any

@dataclass
class AlgorithmConfig:

    ## MISC
    DEFAULT_LOGITS_RESTART_VALUES: list[float] = (1.9, -1.9)
    # PRUNE_METRIC: str = "surface_iou"
    PRUNE_METRIC: str = "surface_iou_wt_curvature"
    N_SURFACE_POINTS_EVAL: int = 100_000
    MPS_MIN_IMPROVEMENT: float = 0.001
    PRUNE_RESOLUTION: int = 128
    DECOMPOSE_RESOLUTION: int = 256
    OPT_RESOLUTION: int = 64
    # SKETCHER:
    DATA_RESOLUTION: int = 256

    RESFIT_MAX_ITER: int = 10
    MPS_LEN_WEIGHT: float = -1e-3
    MPS_STOP_IOU: float = 0.99

    # DECOMPOSE: - 
    # HACK: Set with main_setting as dicts are not supported at Init. 
    DECOMPOSE_SIZE_LIMIT = None
    DECOMPOSE_MODE: str = None
    DECOMPOSE_CONFIG: Dict[str, Any] = None

    MIN_VOLUME_LIMIT_FOR_REINIT: float = 2e-5
    
    EARLY_STOP: bool = True
    EARLY_STOP_ITER: int = 1

    # OPTIM
    OPTIMIZER: str = "ADAM"
    WEIGHT_DECAY: float = 1e-5
    OPT_EPSILON: float = 1e-9
    OPT_LR_RATE: float = 0.01
    EXISTENCE_LR_MULTIPLIER: float = 5.0
    MIN_TEMP_VAL: float = 0.1
    MAX_TEMP_VAL: float = 1.0

    SCALE_FACTOR_START: float = 10.0
    SCALE_FACTOR_END: float = 15.0
    N_ITERS: int = 400
    SAT_PATIENCE: int = 100
    LOSS_BAND: float = 0.05
    MIN_IMPROVEMENT: float = 0.0001
    OPT_STOPPING_IOU: float = 0.99
    MAX_ITER: int = 1000

    LOWER_SP: bool = True
    STOCHASTIC_PRECONDITION_INIT_VAL: float = 2 * np.sqrt(3) * 0.01
    STOCHASTIC_PRECONDITION_INIT_VAL_LOWER: float = 2 * np.sqrt(3) * 0.01
    
    DO_PRUNE: bool = True

    PRIM_TYPE: str = "VarAxisSF"
    # PRIM_TYPE: str = "SuperFrustum"
    SMOOTHEN: bool = True
    
    # OTHER:
    TARGET_MODE: str = "dilated"
    TARGET_MODE_DILATION: float = 0.15
    RENEW_PTS_ITER: int = 100
    N_SURFACE_POINTS: int = 100_000
    LOG_FREQUENCY: int = 50

    # Compilation 
    OPT_DTYPE: str = th.float32
    AOT_ARTIFACT_FILE: str = None
    SAVE_JIT_CACHE: bool = True
    OVERWRITE_JIT_CACHE: bool = False
    TORCH_COMPILE: bool = True
    USE_CUSTOM_OP: bool = False
    COMPILED_FUNCTIONS: str = None

    USE_CURVATURE_WEIGHTS: bool = True
    INTERNAL_CURVATURE_WEIGHTS: bool = True
    CURVATURE_WEIGHTS_SCALE: float = 1.0
    SURFACE_ADJ_PERTURBATION_SCALE: float = 0.05 # 0.05
    BIDIR: bool = True
    BIDIR_RESOLUTION: int = 128
    BIDIR_SAMPLE_RATIO: float = 0.75
    
    # Reconstruction Losses:
    LOSS_OCC_ALPHA: float = 2.0
    LOSS_SURFACE_ADJ_OCC_ALPHA: float = 2.0
    LOSS_SURFACE_SDF_ALPHA: float = 1.0
    LOSS_SURFACE_ADJ_SDF_ALPHA: float = 1.0

    # Quality Losses:
    STOCHASTIC_DROPOUT: bool = True
    LOSS_PRIMITIVE_COUNT_ALPHA: float = 2e-3
    LOSS_OVERLAP_ALPHA: float = 1e-1
    LOSS_SHAPE_UNOVERLAP_ALPHA: float = 1e-1
    LOSS_PARAM_REGULARIZATION_ALPHA: float = 1e-5

    # Tversky Loss:
    TVERSKY_MODE: bool = False
    TVERSKY_ALPHA: float = 1.0
    TVERSKY_BETA: float = 0.0

    # Semantic Loss:
    SEMANTIC_LOSS: bool = False
    SEMANTIC_LOSS_ALPHA: float = 1.0
    SEMANTIC_LOSS_BAND: float = 0.01

    # Reflection Loss:
    REFLECTION_LOSS: bool = False
    REFLECTION_LOSS_ALPHA: float = 1.0

    
    # SEED CONFIGURATION
    RANDOM_SEED: int = 42  # Seed for optimization (set once at start)
    EVAL_SEED: int = 12345  # Seed for evaluation (reset at start of each eval call)
    USE_DETERMINISTIC: bool = False  # Enable PyTorch deterministic mode (may impact performance)
    
    OPT_POST_PRUNE: bool = False
    OLD_MESH_PROCESS: bool = False

    # RENDER CONFIGURATION
    RENDER_MODE: bool = True
    # Subsample if its an issue.
    RENDER_ITER: int = 5

    @staticmethod
    def save_to_file(file_path):
        # convert to string
        configurations = {}
        for key, value in AlgorithmConfig.__dict__.items():
            # skip dunder and private attributes
            if key.startswith("__"):
                continue

            # skip callables and modules/classes
            if callable(value) or isinstance(value, type):
                continue

            # ensure JSON-serializable (convert dataclass instance to dict if needed)
            try:
                if dataclasses.is_dataclass(value):
                    value = dataclasses.asdict(value)
                json.dumps(value)  # test serializability
                configurations[key] = value
            except (TypeError, ValueError):
                # skip non-serializable values
                continue

        with open(file_path, "w") as f:
            json.dump(configurations, f, indent=4)

    
def main_setting():
    AlgorithmConfig.COMPILED_FUNCTIONS = None
    AlgorithmConfig.OLD_MESH_PROCESS = False
    AlgorithmConfig.DECOMPOSE_MODE = "MSD"
    AlgorithmConfig.DECOMPOSE_SIZE_LIMIT = 20
    AlgorithmConfig.DECOMPOSE_CONFIG = {
        "min_eroded_part_size_ratio": 0.005,
        "min_part_size_ratio": 0.0005,
        "size_limit": 20,
        "max_msd_iter": 5,
    }


def apply_data_resolution(data_resolution: int) -> None:
    AlgorithmConfig.DATA_RESOLUTION = data_resolution
    AlgorithmConfig.PRUNE_RESOLUTION = max(32, data_resolution // 2)
    AlgorithmConfig.DECOMPOSE_RESOLUTION = data_resolution
    AlgorithmConfig.OPT_RESOLUTION = max(32, data_resolution // 4)

def fast_test_override():
    AlgorithmConfig.N_ITERS = 10
    AlgorithmConfig.MAX_ITER = 20
    AlgorithmConfig.SAT_PATIENCE = 5
    AlgorithmConfig.RESFIT_MAX_ITER = 2
    AlgorithmConfig.SAVE_JIT_CACHE = False
    AlgorithmConfig.TORCH_COMPILE = False
    AlgorithmConfig.COMPILED_FUNCTIONS = None
    AlgorithmConfig.PRUNE_RESOLUTION: int = 64
    AlgorithmConfig.DECOMPOSE_RESOLUTION: int = 128
    AlgorithmConfig.OPT_RESOLUTION: int = 32
    # SKETCHER:
    AlgorithmConfig.DATA_RESOLUTION: int = 128

def low_cost_mode():
    AlgorithmConfig.N_SURFACE_POINTS = 75_000
    AlgorithmConfig.OPT_RESOLUTION = 32
    # AlgorithmConfig.N_ITERS: int = 350
    # AlgorithmConfig.SAT_PATIENCE: int = 100
    # AlgorithmConfig.MAX_ITER: int = 1200
    # AlgorithmConfig.OPT_LR_RATE: float = 0.01

def low_cost_mode_v2():
    AlgorithmConfig.N_ITERS: int = 250
    AlgorithmConfig.SAT_PATIENCE: int = 100
    AlgorithmConfig.MAX_ITER: int = 1000
    AlgorithmConfig.OPT_LR_RATE: float = 0.01

def cvpr_submission_settings():
    AlgorithmConfig.RESFIT_MAX_ITER = 10
    AlgorithmConfig.DECOMPOSE_MODE = "MSD"
    AlgorithmConfig.DECOMPOSE_SIZE_LIMIT = 20
    AlgorithmConfig.DECOMPOSE_CONFIG = {
        "min_eroded_part_size_ratio": 0.005,
        "min_part_size_ratio": 0.001,
        "size_limit": 20,
        "max_mps_iter": 7,
    }
    AlgorithmConfig.RENEW_PTS_ITER: int = 100
    AlgorithmConfig.N_SURFACE_POINTS: int = 100_000
    AlgorithmConfig.TARGET_MODE: str = "dilated"
    AlgorithmConfig.TARGET_MODE_DILATION: float = 0.2
    AlgorithmConfig.OPT_RESOLUTION: int = 64

    AlgorithmConfig.N_ITERS: int = 400
    AlgorithmConfig.SAT_PATIENCE: int = 100
    AlgorithmConfig.MAX_ITER: int = 1600
    AlgorithmConfig.OPT_LR_RATE: float = 0.01

    AlgorithmConfig.USE_CURVATURE_WEIGHTS = True
    AlgorithmConfig.TVERSKY_MODE = False
    AlgorithmConfig.DO_PRUNE = True

    # Next: With Quality and Length Losses. 
    AlgorithmConfig.LOSS_PARAM_REGULARIZATION_ALPHA = 1e-8
    AlgorithmConfig.LOSS_PRIMITIVE_COUNT_ALPHA: float = 2e-3
    AlgorithmConfig.LOSS_OVERLAP_ALPHA: float = 2e-2
    AlgorithmConfig.LOSS_SHAPE_UNOVERLAP_ALPHA: float = 2e-2
    # NO SDF LOSS on outside points.
    AlgorithmConfig.LOSS_SURFACE_ADJ_SDF_ALPHA: float = 0.0
    AlgorithmConfig.INTERNAL_CURVATURE_WEIGHTS = False

def high_cost_mode():
    AlgorithmConfig.RENEW_PTS_ITER: int = 100
    AlgorithmConfig.N_SURFACE_POINTS: int = 200_000
    AlgorithmConfig.TARGET_MODE: str = "dilated"
    AlgorithmConfig.TARGET_MODE_DILATION: float = 0.1
    AlgorithmConfig.OPT_RESOLUTION: int = 256

    AlgorithmConfig.N_ITERS: int = 500
    AlgorithmConfig.SAT_PATIENCE: int = 150
    AlgorithmConfig.MAX_ITER: int = 2500


def initialize_seeds(seed: int = None, use_deterministic: bool = None):
    """
    Initialize random seeds for optimization (set once at start of optimization).
    This allows the RNG state to evolve naturally during optimization.
    
    Args:
        seed: Random seed to use. If None, uses AlgorithmConfig.RANDOM_SEED
        use_deterministic: Whether to enable PyTorch deterministic mode. 
                          If None, uses AlgorithmConfig.USE_DETERMINISTIC
    """
    if seed is None:
        seed = AlgorithmConfig.RANDOM_SEED
    if use_deterministic is None:
        use_deterministic = AlgorithmConfig.USE_DETERMINISTIC
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    
    # PyTorch deterministic mode (if requested)
    if use_deterministic:
        th.use_deterministic_algorithms(True)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
    
    return seed


def reset_eval_seeds(seed: int = None):
    """
    Reset random seeds for evaluation (called at start of each evaluation call).
    This ensures evaluations always use the same set of randomly sampled points
    for fair comparison across different models/configs.
    
    Args:
        seed: Random seed to use. If None, uses AlgorithmConfig.EVAL_SEED
    """
    if seed is None:
        seed = AlgorithmConfig.EVAL_SEED
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random (reset both CPU and GPU)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    
    return seed

def set_config_ablation(ablation: int, fastmode: bool = True):
    """
    Apply ablation-specific config overrides on top of main_setting().

    Args:
        ablation: Ablation number (0 = baseline, no extra overrides).
        fastmode: If False, disables FastMode and TORCH_COMPILE.
    """
    if not fastmode:
        AlgorithmConfig.TORCH_COMPILE = False

    AlgorithmConfig.USE_CUSTOM_OP = False

    if ablation == 0:
        pass
    if ablation == 1:
        AlgorithmConfig.PRIM_TYPE = "Cuboid"
    elif ablation == 2:
        AlgorithmConfig.PRIM_TYPE = "VarAxisSQ"
    elif ablation == 3:
        AlgorithmConfig.PRIM_TYPE = "VarAxisSPP"
    elif ablation == 4:
        AlgorithmConfig.PRIM_TYPE = "VarAxisSG"
    elif ablation == 5:
        # Original Submission
        AlgorithmConfig.PRIM_TYPE = "SuperFrustum"
        cvpr_submission_settings()
    elif ablation == 6:
        # Solid Primitives. 
        AlgorithmConfig.PRIM_TYPE = "SolidSF"
    elif ablation == 7:
        AlgorithmConfig.PRIM_TYPE = "VarAxisSF"
    elif ablation == 8:
        AlgorithmConfig.PRIM_TYPE = "VarAxisSF"
        AlgorithmConfig.USE_CUSTOM_OP = True
        AlgorithmConfig.TORCH_COMPILE = False
