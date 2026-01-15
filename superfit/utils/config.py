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
    OPT_RESOLUTION: int = 128
    # SKETCHER:
    DATA_RESOLUTION: int = 256

    MPS_MAX_ITER: int = 5
    MPS_LEN_WEIGHT: float = -1e-3
    DATA_DECIMATION: float = 0.5
    MPS_STOP_IOU: float = 0.99

    # DECOMPOSE:
    DECOMPOSE_SIZE_LIMIT = None
    DECOMPOSE_MODE: str = None
    DECOMPOSE_CONFIG: Dict[str, Any] = None


    MIN_VOLUME_LIMIT_FOR_REINIT: float = 2e-5
    CLEANUP_V1: bool = True
    EARLY_STOP_ITER: int = 1


    ### INVERSION RELATED
    OPT_MIN_TRANSLATE: float = -0.9999
    OPT_MAX_TRANSLATE: float = 0.9999
    OPT_MIN_SCALE: float = 0.00001
    OPT_MAX_SCALE: float = 1.9999

    # OPTIM
    OPTIMIZER: str = "ADAM"
    WEIGHT_DECAY: float = 1e-5
    OPT_EPSILON: float = 1e-9
    OPT_LR_RATE: float = 0.01
    MIN_TEMP_VAL: float = 0.1
    MAX_TEMP_VAL: float = 1.0

    SCALE_FACTOR_START: float = 10.0
    SCALE_FACTOR_END: float = 15.0
    N_ITERS: int = 500
    SAT_PATIENCE: int = 150
    LOSS_BAND: float = 0.05
    MIN_IMPROVEMENT: float = 0.0001
    OPT_STOPPING_IOU: float = 0.99
    MAX_ITER: int = 2500
    STOCHASTIC_PRECONDITION_INIT_VAL: float = 2 * np.sqrt(3) * 0.01

    # OTHER:
    # TARGET_MODE: str = "bboxed"
    TARGET_MODE: str = None
    TARGET_MODE_DILATION: float = 0.1
    RENEW_PTS_ITER: int = 100
    N_SURFACE_POINTS: int = 200_000
    LOG_FREQUENCY: int = 50


    # LOSS Weights:
    SKIP_SURFACE: str = False
    SURFACE_ADJ_PERTURBATION_SCALE: float = 0.05 # 0.05
    STOCHASTIC_DROPOUT: bool = True
    LOSS_OCC_ALPHA: float = 1.0
    LOSS_SURFACE_ADJ_OCC_ALPHA: float = 5.0
    LOSS_SURFACE_SDF_ALPHA: float = 0.5
    LOSS_PCOUNT_PHASIC: bool = False
    LOSS_PRIMITIVE_COUNT_ALPHA: float = 5e-4
    LOSS_PARAM_REGULARIZATION_ALPHA: float = 1e-5
    LOSS_PQUAL_PHASIC: bool = False
    LOSS_OVERLAP_ALPHA: float = 8.0
    LOSS_SHAPE_UNOVERLAP_ALPHA: float = 0.0


    SEMANTIC_LOSS: bool = False
    LOSS_SEMANTIC_PR_TO_PO_ALPHA: float = 10.0
    LOSS_SEMANTIC_PR_TO_PR_ALPHA: float = 1.0

    TVERSKY_MODE: bool = False
    TVERSKY_ALPHA: float = 2.0
    TVERSKY_BETA: float = 1.0

    OPT_HACK_GRAD = False
    GRAD_LOWER_RATE_V2 = 0.001
    GRAD_INCREASE_RATE = 10
    GRAD_LOWER_RATE = 0.01

    DO_PRUNE: bool = True
    OPT_HALF: bool = False

    USE_CURVATURE_WEIGHTS: bool = False
    CURVATURE_WEIGHTS_SCALE: float = 1.0

    ## LAST RUN CONFIGS:
    RUN_LAST_OPT: bool = False
    LAST_RUN_LOSS_OCC_ALPHA: float = 1.0
    LAST_RUN_LOSS_SURFACE_ADJ_OCC_ALPHA: float = 2.0
    LAST_RUN_LOSS_SURFACE_SDF_ALPHA: float = 0.2
    LAST_RUN_LOSS_PRIMITIVE_COUNT_ALPHA: float = 0
    LAST_RUN_LOSS_PARAM_REGULARIZATION_ALPHA: float = 0
    LAST_RUN_LOSS_OVERLAP_ALPHA: float = 0
    LAST_RUN_LOSS_SHAPE_UNOVERLAP_ALPHA: float = 0
    LAST_RUN_MAX_TEMP_VAL: float = 0.1

    PRIM_TYPE: str = "SuperFrustum"
    SMOOTHEN: bool = True
    CUBOID_MODE: bool = False
    NEO_SMPL_MODE: bool = False
    FREEZE_PREV_PRIMS: bool = False
    
    OPT_DTYPE: str = th.float32
    AOT_ARTIFACT_DIR: str = "../aot"
    AOT_ARTIFACT_FILE: str = None
    SAVE_JIT_CACHE: bool = True
    OVERWRITE_JIT_CACHE: bool = False
    TorchCompile: bool = True
    FastMode: bool = True
    
    # SEED CONFIGURATION
    RANDOM_SEED: int = 42  # Seed for optimization (set once at start)
    EVAL_SEED: int = 12345  # Seed for evaluation (reset at start of each eval call)
    USE_DETERMINISTIC: bool = False  # Enable PyTorch deterministic mode (may impact performance)
    
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

    AlgorithmConfig.RENEW_PTS_ITER: int = 100
    AlgorithmConfig.N_SURFACE_POINTS: int = 100_000
    AlgorithmConfig.TARGET_MODE: str = "dilated"
    # AlgorithmConfig.TARGET_MODE: str = None
    AlgorithmConfig.TARGET_MODE_DILATION: float = 0.2
    AlgorithmConfig.OPT_RESOLUTION: int = 128


    AlgorithmConfig.N_ITERS: int = 400
    AlgorithmConfig.SAT_PATIENCE: int = 100
    AlgorithmConfig.MAX_ITER: int = 1600
    AlgorithmConfig.OPT_LR_RATE: float = 0.01

    AlgorithmConfig.USE_CURVATURE_WEIGHTS = True
    AlgorithmConfig.DO_PRUNE = True
    AlgorithmConfig.OPT_HACK_GRAD = False
    AlgorithmConfig.TVERSKY_MODE = False

    # Next: With Quality and Length Losses. 
    AlgorithmConfig.LOSS_PARAM_REGULARIZATION_ALPHA = 1e-8
    AlgorithmConfig.STOCHASTIC_DROPOUT: bool = True
    AlgorithmConfig.LOSS_PRIMITIVE_COUNT_ALPHA: float = 2e-3
    AlgorithmConfig.LOSS_OVERLAP_ALPHA: float = 2e-2
    AlgorithmConfig.LOSS_SHAPE_UNOVERLAP_ALPHA: float = 2e-2


    AlgorithmConfig.MPS_MAX_ITER = 10
    AlgorithmConfig.DECOMPOSE_MODE = "MSD_NEW"
    AlgorithmConfig.DECOMPOSE_SIZE_LIMIT = 20
    AlgorithmConfig.DECOMPOSE_CONFIG = {
        "min_eroded_part_size_ratio": 0.005,
        "min_part_size_ratio": 0.0005,
        "size_limit": 20,
        "max_mps_iter": 7,
    }

def low_cost_mode():
    AlgorithmConfig.RENEW_PTS_ITER: int = 100
    AlgorithmConfig.N_SURFACE_POINTS: int = 100_000
    AlgorithmConfig.TARGET_MODE: str = "dilated"
    AlgorithmConfig.TARGET_MODE_DILATION: float = 0.2
    AlgorithmConfig.OPT_RESOLUTION: int = 64

    # AlgorithmConfig.N_ITERS: int = 350
    # AlgorithmConfig.SAT_PATIENCE: int = 100
    # AlgorithmConfig.MAX_ITER: int = 1200
    # AlgorithmConfig.OPT_LR_RATE: float = 0.01

def low_cost_mode_v2():
    # AlgorithmConfig.RENEW_PTS_ITER: int = 100
    # AlgorithmConfig.N_SURFACE_POINTS: int = 100_000
    # AlgorithmConfig.TARGET_MODE: str = "dilated"
    # AlgorithmConfig.TARGET_MODE_DILATION: float = 0.2
    # AlgorithmConfig.OPT_RESOLUTION: int =64

    # AlgorithmConfig.N_ITERS: int = 250
    # AlgorithmConfig.SAT_PATIENCE: int = 100
    # AlgorithmConfig.MAX_ITER: int = 1000
    # AlgorithmConfig.OPT_LR_RATE: float = 0.02
    ...

def medium_cost_mode():
    AlgorithmConfig.RENEW_PTS_ITER: int = 100
    AlgorithmConfig.N_SURFACE_POINTS: int = 100_000
    AlgorithmConfig.TARGET_MODE: str = "dilated"
    AlgorithmConfig.TARGET_MODE_DILATION: float = 0.2
    AlgorithmConfig.OPT_RESOLUTION: int = 128

    AlgorithmConfig.N_ITERS: int = 400
    AlgorithmConfig.SAT_PATIENCE: int = 100
    AlgorithmConfig.MAX_ITER: int = 1600
    AlgorithmConfig.OPT_LR_RATE: float = 0.01


def high_cost_mode():
    AlgorithmConfig.RENEW_PTS_ITER: int = 100
    AlgorithmConfig.N_SURFACE_POINTS: int = 200_000
    AlgorithmConfig.TARGET_MODE: str = "dilated"
    AlgorithmConfig.TARGET_MODE_DILATION: float = 0.1
    AlgorithmConfig.OPT_RESOLUTION: int = 256

    AlgorithmConfig.N_ITERS: int = 500
    AlgorithmConfig.SAT_PATIENCE: int = 150
    AlgorithmConfig.MAX_ITER: int = 2500


def check_config():
    if AlgorithmConfig.FastMode:
        assert AlgorithmConfig.PRIM_TYPE == "SuperFrustum"


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
    
    # Open3D random (if available)
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except ImportError:
        pass
    
    return seed