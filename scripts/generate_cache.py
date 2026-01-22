import os
import argparse
import torch as th
import geolipi.symbolic as gls
import superfit.symbolic as sps
from geolipi.torch_compute import Sketcher
from superfit.utils.config import AlgorithmConfig as AlgConf
from superfit.utils.logger import logger
from superfit.utils.constants import AOT_ARTIFACT_DIR
from superfit.optim.compile_function import compile_cached_with_dummy_opt
from superfit.optim.primitive_registry import HANDLER_REGISTRY
import superfit.utils.config as config_options

# Constants from prim_initialize.py
ROUNDNESS_INIT_VAL = 0.4
ONION_INIT_VAL = 0.4
SCALE_INIT_VAL = 0.7
BULGE_INIT_VAL = 0.01
SMOOTH_INIT_VAL = 0.01
VARAXIS_INIT_VAL = 2.5


def create_simple_expression(sketcher):
    """
    Create a simple expression with 2 primitives for cache generation.
    """
    version = getattr(sps, AlgConf.PRIM_TYPE)
    
    # Create two simple primitives with basic parameters
    # Primitive 1: centered at (-0.3, 0, 0)
    scale1 = (0.3, 0.3, 0.3)
    center1 = (-0.3, 0.0, 0.0)
    rotation1 = (0.0, 0.0, 0.0)
    
    # Primitive 2: centered at (0.3, 0, 0)
    scale2 = (0.3, 0.3, 0.3)
    center2 = (0.3, 0.0, 0.0)
    rotation2 = (0.0, 0.0, 0.0)
    
    # Create primitives based on type
    if issubclass(version, sps.SuperFrustum):
        prim1 = version(
            scale1, 
            (ROUNDNESS_INIT_VAL,), 
            (SMOOTH_INIT_VAL,), 
            (SCALE_INIT_VAL,), 
            (BULGE_INIT_VAL,), 
            (ONION_INIT_VAL,)
        )
        prim2 = version(
            scale2, 
            (ROUNDNESS_INIT_VAL,), 
            (SMOOTH_INIT_VAL,), 
            (SCALE_INIT_VAL,), 
            (BULGE_INIT_VAL,), 
            (ONION_INIT_VAL,)
        )
    elif issubclass(version, sps.SolidSF):
        init_logits = (0.0, 0.0, 0.0, 0.0)
        prim1 = version(
            scale1, 
            (ROUNDNESS_INIT_VAL,), 
            (SMOOTH_INIT_VAL,), 
            (SCALE_INIT_VAL,), 
            (BULGE_INIT_VAL,), 
            (ONION_INIT_VAL,),
            init_logits
        )
        prim2 = version(
            scale2, 
            (ROUNDNESS_INIT_VAL,), 
            (SMOOTH_INIT_VAL,), 
            (SCALE_INIT_VAL,), 
            (BULGE_INIT_VAL,), 
            (ONION_INIT_VAL,),
            init_logits
        )
    elif issubclass(version, sps.VarAxisSF):
        init_logits = (VARAXIS_INIT_VAL, -VARAXIS_INIT_VAL, -VARAXIS_INIT_VAL)
        prim1 = version(
            scale1, 
            (ROUNDNESS_INIT_VAL,), 
            (SMOOTH_INIT_VAL,), 
            (SCALE_INIT_VAL,), 
            (BULGE_INIT_VAL,), 
            (ONION_INIT_VAL,),
            init_logits
        )
        prim2 = version(
            scale2, 
            (ROUNDNESS_INIT_VAL,), 
            (SMOOTH_INIT_VAL,), 
            (SCALE_INIT_VAL,), 
            (BULGE_INIT_VAL,), 
            (ONION_INIT_VAL,),
            init_logits
        )
    else:
        raise ValueError(f"Unsupported primitive type: {version}")
    
    # Apply rotation and translation
    prim1 = gls.AxisAngleRotate3D(prim1, rotation1)
    prim1 = gls.Translate3D(prim1, center1)
    
    prim2 = gls.AxisAngleRotate3D(prim2, rotation2)
    prim2 = gls.Translate3D(prim2, center2)
    
    # Wrap with PrimitiveMarker
    prim1 = sps.PrimitiveMarker(prim1)
    prim2 = sps.PrimitiveMarker(prim2)
    
    # Combine with SmoothUnion
    if AlgConf.SMOOTHEN:
        expr = gls.SmoothUnion(prim1, prim2, (SMOOTH_INIT_VAL,))
    else:
        expr = gls.SmoothUnion(prim1, prim2, (0.0,))
    
    return expr


def generate_cache_for_ablation(ablation: int):
    """
    Generate cache for a specific ablation number.
    """
    logger.info(f"Generating cache for ablation {ablation}")
    
    # Setup config based on ablation (similar to generate_on_testset.py)
    config_options.main_setting()
    
    if ablation == 0:
        pass
    elif ablation == 6:
        config_options.low_cost_mode()
    elif ablation == 7:
        AlgConf.PRIM_TYPE = "VarAxisSF"
    
    # Set the artifact file path (same as in generate_on_testset.py line 56)
    AlgConf.AOT_ARTIFACT_FILE = os.path.join(AOT_ARTIFACT_DIR, f"aot_artifact_{ablation}.pt")
    
    # Ensure artifact directory exists
    artifact_dir = os.path.dirname(AlgConf.AOT_ARTIFACT_FILE)
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir, exist_ok=True)
        logger.info(f"Created artifact directory: {artifact_dir}")
    
    # Create sketcher
    sketcher = Sketcher(resolution=AlgConf.OPT_RESOLUTION, n_dims=3, dtype=AlgConf.OPT_DTYPE)
    
    # Create simple expression with 2 primitives
    logger.info("Creating simple expression with 2 primitives")
    expr = create_simple_expression(sketcher)
    
    # Convert to tensor format
    expr = expr.tensor(dtype=AlgConf.OPT_DTYPE)
    
    # Get handler (same as in entry.py)
    version = getattr(sps, AlgConf.PRIM_TYPE)
    handler = HANDLER_REGISTRY[version]
    assert handler is not None, f"No handler found for {AlgConf.PRIM_TYPE}"
    
    logger.info(f"Using handler: {handler.__class__.__name__}")
    logger.info(f"Artifact file: {AlgConf.AOT_ARTIFACT_FILE}")
    
    # Generate cache using compile_cached_with_dummy_opt
    logger.info("Compiling with dummy optimization to generate cache...")
    AlgConf.SAVE_JIT_CACHE = True
    AlgConf.OVERWRITE_JIT_CACHE = True
    compiled_ops = compile_cached_with_dummy_opt(expr, sketcher, handler, torch_compile=True)
    
    logger.info(f"Cache generation complete! Artifacts saved to: {AlgConf.AOT_ARTIFACT_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Generate cache for different ablations")
    parser.add_argument("--ablation", type=int, required=True, help="Ablation number")
    
    args = parser.parse_args()
    
    generate_cache_for_ablation(args.ablation)


if __name__ == "__main__":
    main()
