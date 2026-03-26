# Code to initialize primitive from a given sdf volume. 
import torch as th
import numpy as np
import geolipi.symbolic as gls
import superfit.symbolic as sps
from superfit.utils.config import AlgorithmConfig as AlgConf
from superfit.symbolic.utils import inject_stochastic_prim
from superfit.symbolic.utils import gather_primitives
from superfit.utils.logger import logger

ROUNDNESS_INIT_VAL = 0.4
ONION_INIT_VAL = 0.4
SCALE_INIT_VAL = 0.7
BULGE_INIT_VAL = 0.02
SMOOTH_INIT_VAL = 0.01
TRAPEZE_INIT_VAL = 0.5
TAPER_BULGE_INIT_VAL = 0.01
ROT2D_INIT_VAL = 0.5
VARAXIS_INIT_VAL = AlgConf.DEFAULT_LOGITS_RESTART_VALUES[0]
EXTRUSSION_INIT_VAL = 0.1

# MAX_INIT_PROB = 0.95
MAX_INIT_PROB = 0.95


def initialize_sp_prims(prim_params, sketcher):
    """
    Initialize a primitive given parameters and settings.
    
    Args:
        prim_params: dict with 'center', 'rotation', 'scale'
        n_dims: 2 or 3 for 2D or 3D primitives
        gumbel_sf: boolean flag for using Gumbel soft max primitives
    
    Returns:
        Configured primitive
    """
    center = prim_params['center']
    rotation = prim_params['rotation']
    scale = prim_params['scale']
    version = getattr(sps, AlgConf.PRIM_TYPE)

    if issubclass(version, sps.Cuboid):
        primitive = version(scale)
    elif issubclass(version, sps.SuperQuadric):
        primitive = version(scale, (SCALE_INIT_VAL,), (SCALE_INIT_VAL,))
    elif issubclass(version, sps.SuperFrustum):
        taper_amount = prim_params.get("taper", (SCALE_INIT_VAL,))
        onion_amount = prim_params.get("onion_amount", (ONION_INIT_VAL,))
        roundness = prim_params.get("roundness", (ROUNDNESS_INIT_VAL,))
        primitive = version(scale, roundness, (SMOOTH_INIT_VAL,), taper_amount, (BULGE_INIT_VAL,), onion_amount)
    elif issubclass(version, sps.SuperGeon):
        taper_amount = prim_params.get("taper", (SCALE_INIT_VAL,))
        onion_amount = prim_params.get("onion_amount", (ONION_INIT_VAL,))
        roundness = prim_params.get("roundness", (ROUNDNESS_INIT_VAL,))
        primitive = version(scale, roundness, (SMOOTH_INIT_VAL,), 
                            taper_amount, (BULGE_INIT_VAL,), onion_amount,
                            (TRAPEZE_INIT_VAL,), (TAPER_BULGE_INIT_VAL,), (ROT2D_INIT_VAL,))
    elif issubclass(version, sps.SPProto):
        taper_amount = prim_params.get("taper", (SCALE_INIT_VAL,))
        onion_amount = prim_params.get("onion_amount", (ONION_INIT_VAL,))
        r = prim_params.get("roundness", ROUNDNESS_INIT_VAL)
        if isinstance(r, th.Tensor):
            r = r.squeeze().item()
        roundness = (r, r, r, r)
        primitive = version(scale, roundness, (SMOOTH_INIT_VAL,), 
                            onion_amount, (EXTRUSSION_INIT_VAL, EXTRUSSION_INIT_VAL * 0.1))
    elif issubclass(version, sps.SolidSF):
        taper_amount = prim_params.get("taper", (SCALE_INIT_VAL,))
        onion_amount = prim_params.get("onion_amount", (ONION_INIT_VAL,))
        roundness = prim_params.get("roundness", (ROUNDNESS_INIT_VAL,))
        init_logits = (0.0, 0.0, 0.0, 0.0)
        primitive = version(scale, roundness, (SMOOTH_INIT_VAL,), taper_amount, (BULGE_INIT_VAL,), onion_amount, init_logits)
    elif issubclass(version, sps.VarAxisSF):
        taper_amount = prim_params.get("taper", (SCALE_INIT_VAL,))
        onion_amount = prim_params.get("onion_amount", (ONION_INIT_VAL,))
        roundness = prim_params.get("roundness", (ROUNDNESS_INIT_VAL,))
        roundness = prim_params.get("roundness", (ROUNDNESS_INIT_VAL,))
        # Highly likely to be type 1. 
        init_logits = (VARAXIS_INIT_VAL, -VARAXIS_INIT_VAL, -VARAXIS_INIT_VAL)
        primitive = version(scale, roundness, (SMOOTH_INIT_VAL,), taper_amount, (BULGE_INIT_VAL,), onion_amount, init_logits)
    elif issubclass(version, sps.VarAxisSG):
        init_logits = (VARAXIS_INIT_VAL, -VARAXIS_INIT_VAL, -VARAXIS_INIT_VAL)
        taper_amount = prim_params.get("taper", (SCALE_INIT_VAL,))
        onion_amount = prim_params.get("onion_amount", (ONION_INIT_VAL,))
        roundness = prim_params.get("roundness", (ROUNDNESS_INIT_VAL,))
        primitive = version(scale, roundness, (SMOOTH_INIT_VAL,), 
                            taper_amount, (BULGE_INIT_VAL,), onion_amount,
                            (TRAPEZE_INIT_VAL,), (TAPER_BULGE_INIT_VAL,), (ROT2D_INIT_VAL,), init_logits)
    elif issubclass(version, sps.VarAxisSPP):
        init_logits = (VARAXIS_INIT_VAL, -VARAXIS_INIT_VAL, -VARAXIS_INIT_VAL)
        taper_amount = prim_params.get("taper", (SCALE_INIT_VAL,))
        onion_amount = prim_params.get("onion_amount", (ONION_INIT_VAL,))
        r = prim_params.get("roundness", ROUNDNESS_INIT_VAL)
        if isinstance(r, th.Tensor):
            r = r.squeeze().item()
        roundness = (r, r, r, r)
        primitive = version(scale, roundness, (SMOOTH_INIT_VAL,), 
                            onion_amount, (EXTRUSSION_INIT_VAL, EXTRUSSION_INIT_VAL), init_logits)
    elif issubclass(version, sps.VarAxisSQ):
        init_logits = (VARAXIS_INIT_VAL, -VARAXIS_INIT_VAL, -VARAXIS_INIT_VAL)
        epsilon_1 = prim_params.get("epsilon_1", (SCALE_INIT_VAL,))
        epsilon_2 = prim_params.get("epsilon_2", (SCALE_INIT_VAL,))
        primitive = version(scale, epsilon_1, epsilon_2, init_logits)
    else:
        raise ValueError(f"Unsupported version: {version}")
    primitive = gls.AxisAngleRotate3D(primitive, rotation)
    primitive = gls.Translate3D(primitive, center)
    
    return primitive

def get_init_prim_program(primitive_fits, sketcher, init_program=None, 
                    logits_keep_drop=(10.0, -0.0)):
    expr = init_program
    # HERE WE WILL ALSO GET Negative - partition them. 
    expr_list = []
    for prim_parms in primitive_fits:
        primitive = initialize_sp_prims(prim_parms, sketcher)
        primitive = sps.PrimitiveMarker(primitive)
        if AlgConf.STOCHASTIC_DROPOUT:
            primitive = inject_stochastic_prim(primitive, logits_keep_drop)
        expr_list.append(primitive)
    
    for cur_expr in expr_list:
        if expr is None:
            expr = cur_expr
        else:
            if AlgConf.SMOOTHEN:
                expr = gls.SmoothUnion(expr, cur_expr, (SMOOTH_INIT_VAL,))
            else:
                expr = gls.SmoothUnion(expr, cur_expr, (0.0,))
    # size correction:
    primitives = gather_primitives(expr)
    if len(primitives) < 3:
        if len(primitives) == 1:
            logger.info(f"Adding a second primitive to the expression because it has only {len(primitives)} primitives.")
            primitives.append(primitives[0])
            primitives.append(primitives[0])
        elif len(primitives) == 2:
            logger.info(f"Adding a second primitive to the expression because it has only {len(primitives)} primitives.")
            primitives.append(primitives[0])
        
        for cur_expr in primitives:
            if expr is None:
                expr = cur_expr
            else:
                if AlgConf.SMOOTHEN:
                    expr = gls.SmoothUnion(expr, cur_expr, (SMOOTH_INIT_VAL,))
                else:
                    expr = gls.SmoothUnion(expr, cur_expr, (0.0,))

    
    return expr


def simple_cleanup_volumetric(all_parts, all_indices, size_limit=50):
    volumes = [(part<=0).float().sum() for part in all_parts]
    volumes, all_parts = zip(*sorted(zip(volumes, all_parts), key=lambda x: -x[0]))
    # _, all_indices = zip(*sorted(zip(volumes, all_indices), key=lambda x: -x[0]))
    # all_indices = all_indices[:size_limit]
    return all_parts[:size_limit], all_indices
    

def get_delta(n_prims):
    inverse_rate = n_prims
    desired_prob = 0.9 ** (1 / inverse_rate)
    desired_prob = min(desired_prob, MAX_INIT_PROB) # Should we?
    delta = np.log(desired_prob/ (1 - desired_prob))
    return delta