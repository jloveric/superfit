import torch as th
import geolipi.symbolic as gls
import superfit.symbolic as sps
from ...utils.config import AlgorithmConfig as AlgConf
from ..primitive_registry import PrimitiveHandler, register_handler
from ...torch_compute.batched_others import (
    batched_cuboid_packed_stochastic_eval,
    batched_sq_packed_stochastic_eval,
    batched_varaxis_sq_packed_stochastic_eval,
)
from .utils import (
    build_transform_constants,
    make_packed_param_to_var,
    make_packed_var_to_param,
    make_param_to_var_dispatcher,
    make_var_to_param_dispatcher,
    make_param_from_variables_fast,
)

# SQ: [0:3] translate | [3:6] size | [6:7] epsilon_1 | [7:8] epsilon_2 | [8:11] rotation
_SQ_TC = build_transform_constants([
    gls.Translate3D, gls.Translate3D, gls.Translate3D,
    "sp_sq_size", "sp_sq_size", "sp_sq_size",
    "sp_sq_scale", "sp_sq_scale",
])
sq_packed_param_to_var = make_packed_param_to_var(_SQ_TC)
sq_packed_var_to_param = make_packed_var_to_param(_SQ_TC)
param_to_var_sq = make_param_to_var_dispatcher(sq_packed_param_to_var)
var_to_param_sq = make_var_to_param_dispatcher(sq_packed_var_to_param)
_params_from_variables_fast_sq = make_param_from_variables_fast(sq_packed_var_to_param)

# Cuboid: [0:3] translate | [3:6] size | [6:9] rotation
_CUBOID_TC = build_transform_constants([
    gls.Translate3D, gls.Translate3D, gls.Translate3D,
    "sp_size", "sp_size", "sp_size",
])
cuboid_packed_param_to_var = make_packed_param_to_var(_CUBOID_TC)
cuboid_packed_var_to_param = make_packed_var_to_param(_CUBOID_TC)
param_to_var_cuboid = make_param_to_var_dispatcher(cuboid_packed_param_to_var)
var_to_param_cuboid = make_var_to_param_dispatcher(cuboid_packed_var_to_param)
_params_from_variables_fast_cuboid = make_param_from_variables_fast(cuboid_packed_var_to_param)


# ========================== Unpack helpers ==================================

def unpack_params_sq(params):
    assert params.shape[-1] == 5
    skew_vec = params[..., :3]
    epsilon_1 = params[..., 3:4]
    epsilon_2 = params[..., 4:5]
    return skew_vec, epsilon_1, epsilon_2

def unpack_params_cuboid(params):
    assert params.shape[-1] == 3
    size = params[..., :3]
    return (size,)

def unpack_params_varaxis_sq(params):
    assert params.shape[-1] == 8
    skew_vec = params[..., :3]
    epsilon_1 = params[..., 3:4]
    epsilon_2 = params[..., 4:5]
    axis_logits = params[..., 5:8]
    return skew_vec, epsilon_1, epsilon_2, axis_logits

def reinit_params_varaxis_sq(prim_expr, prim_param):
    val = AlgConf.DEFAULT_LOGITS_RESTART_VALUES[0]
    if isinstance(prim_expr, sps.SuperQuadricY):
        log_reinit = (val, -val, -val)
        log_reinit_param = [th.Tensor(log_reinit).to(prim_param.device),]
    elif isinstance(prim_expr, sps.SuperQuadricZ):
        log_reinit = (-val, val, -val)
        log_reinit_param = [th.Tensor(log_reinit).to(prim_param.device),]
    elif isinstance(prim_expr, sps.SuperQuadricX):
        log_reinit = (-val, -val, val)
        log_reinit_param = [th.Tensor(log_reinit).to(prim_param.device),]
    elif isinstance(prim_expr, sps.VarAxisSQ):
        log_reinit_param = []
    else:
        raise ValueError(f"Unsupported primitive type: {prim_expr}")
    return log_reinit_param




def get_param_loss_cuboid(transformed_params):
    prim_params, su_ops, logits, temperature = transformed_params
    su_loss = su_ops.sum()
    temperature = transformed_params[-1]
    logits = transformed_params[-2]
    # SHould this also have GMBL noise? 
    soft = th.softmax(logits / temperature, dim=-1)
    prim_size_loss = prim_params[:, 3:6].sum(dim=-1)
    loss = su_loss + (prim_size_loss * soft[:, 0]).sum()
    return loss
    

def get_param_loss_sq(transformed_params):
    prim_params, su_ops, logits, temperature = transformed_params
    su_loss = su_ops.sum()
    temperature = transformed_params[-1]
    logits = transformed_params[-2]
    # SHould this also have GMBL noise? 
    soft = th.softmax(logits / temperature, dim=-1)
    prim_size_loss = prim_params[:, 3:6].sum(dim=-1)
    loss = su_loss + (prim_size_loss * soft[:, 0]).sum()
    return loss
    

# ========================== Handlers ========================================

@register_handler
class SQHandler(PrimitiveHandler):
    base_class = sps.SuperQuadric
    packed_class = sps.SQPacked
    packed_batched_class = sps.SQPackedBatched
    packed_batched_stochastic_class = sps.SQPackedBatchedStochastic
    packed_batched_su_class = sps.SQPackedBatchedSU
    packed_batched_stochastic_su_class = sps.SQPackedBatchedStochasticSU
    batched_param_size = 11

    unpack_params = unpack_params_sq
    param_to_var = param_to_var_sq
    var_to_param = var_to_param_sq
    param_from_variables_fast = _params_from_variables_fast_sq
    batched_eval_function = batched_sq_packed_stochastic_eval
    point2prim_hard = None
    point2prim_soft = None
    reinit_params = None
    PARAM_IND_TO_NAME = {
        0: "sp_size",
        1: "sp_roundness_e1",
        2: "sp_roundness_e2",
    }
    get_param_loss = get_param_loss_sq

@register_handler
class CuboidHandler(PrimitiveHandler):
    base_class = sps.Cuboid
    packed_class = sps.CuboidPacked
    packed_batched_class = sps.CuboidPackedBatched
    packed_batched_stochastic_class = sps.CuboidPackedBatchedStochastic
    packed_batched_su_class = sps.CuboidPackedBatchedSU
    packed_batched_stochastic_su_class = sps.CuboidPackedBatchedStochasticSU
    batched_param_size = 9

    unpack_params = unpack_params_cuboid
    param_to_var = param_to_var_cuboid
    var_to_param = var_to_param_cuboid
    param_from_variables_fast = _params_from_variables_fast_cuboid
    batched_eval_function = batched_cuboid_packed_stochastic_eval
    point2prim_hard = None
    point2prim_soft = None
    reinit_params = None
    PARAM_IND_TO_NAME = {
        0: "sp_size",
    }
    get_param_loss = get_param_loss_cuboid
@register_handler
class VarAxisSQHandler(PrimitiveHandler):
    base_class = sps.VarAxisSQ
    packed_class = sps.VarAxisSQPacked
    packed_batched_class = sps.VarAxisSQPackedBatched
    packed_batched_stochastic_class = sps.VarAxisSQPackedBatchedStochastic
    packed_batched_su_class = sps.VarAxisSQPackedBatchedSU
    packed_batched_stochastic_su_class = sps.VarAxisSQPackedBatchedStochasticSU
    batched_param_size = 14

    unpack_params = unpack_params_varaxis_sq
    param_to_var = param_to_var_sq
    var_to_param = var_to_param_sq
    param_from_variables_fast = _params_from_variables_fast_sq
    batched_eval_function = batched_varaxis_sq_packed_stochastic_eval
    point2prim_hard = None
    point2prim_soft = None
    reinit_params = reinit_params_varaxis_sq
    PARAM_IND_TO_NAME = {
        0: "sp_size",
    }
    get_param_loss = get_param_loss_sq