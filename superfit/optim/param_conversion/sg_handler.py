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
import torch as th
import geolipi.symbolic as gls
import superfit.symbolic as sps
from ..primitive_registry import PrimitiveHandler, register_handler
from ...utils.config import AlgorithmConfig as AlgConf
from ..losses import get_param_loss_sf
from ...torch_compute.batched_sg import (
    batched_sg_packed_stochastic_eval,
    batched_varaxis_sg_packed_stochastic_eval,
)
from .utils import (
    build_transform_constants,
    make_packed_param_to_var,
    make_packed_var_to_param,
    make_param_to_var_dispatcher,
    make_var_to_param_dispatcher,
    make_param_from_variables_fast,
)

_SG_TC = build_transform_constants([
    gls.Translate3D, gls.Translate3D, gls.Translate3D,
    "sp_size", "sp_size", "sp_size",
    "sp_roundness", "sp_dilate_3d", "sp_taper", "sp_bulge", "sp_onion_ratio",
    "sp_trapeze", "sp_taper_bulge",
])
sg_packed_param_to_var = make_packed_param_to_var(_SG_TC)
sg_packed_var_to_param = make_packed_var_to_param(_SG_TC)
param_to_var_sg = make_param_to_var_dispatcher(sg_packed_param_to_var)
var_to_param_sg = make_var_to_param_dispatcher(sg_packed_var_to_param)
sg_param_from_variables_fast = make_param_from_variables_fast(sg_packed_var_to_param)

SG_PARAM_IND_TO_NAME = {
    0: "sp_size",
    1: "sp_roundness",
    2: "sp_dilate_3d",
    3: "sp_taper",
    4: "sp_bulge",
    5: "sp_onion_ratio",
    6: "sp_trapeze",
    7: "sp_taper_bulge",
}


def unpack_params_sg(params):
    assert params.shape[-1] == 11
    size = params[..., :3]
    roundness = params[..., 3:4]
    dilate_3d = params[..., 4:5]
    taper = params[..., 5:6]
    bulge = params[..., 6:7]
    onion = params[..., 7:8]
    trapeze = params[..., 8:9]
    taper_bulge = params[..., 9:10]
    rot2d = params[..., 10:11]
    return size, roundness, dilate_3d, taper, bulge, onion, trapeze, taper_bulge, rot2d


def unpack_params_var_axis_sg(params):
    assert params.shape[-1] == 14
    sg_params = unpack_params_sg(params[..., :11])
    logits = params[..., 11:14]
    return sg_params + (logits,)

def reinit_params_varaxis_sg(prim_expr, prim_param):
    val = AlgConf.DEFAULT_LOGITS_RESTART_VALUES[0]
    if isinstance(prim_expr, sps.SuperGeonY):
        log_reinit = (val, -val, -val)
        log_reinit_param = [th.Tensor(log_reinit).to(prim_param.device),]
    elif isinstance(prim_expr, sps.SuperGeonZ):
        log_reinit = (-val, val, -val)
        log_reinit_param = [th.Tensor(log_reinit).to(prim_param.device),]
    elif isinstance(prim_expr, sps.SuperGeonX):
        log_reinit = (-val, -val, val)
        log_reinit_param = [th.Tensor(log_reinit).to(prim_param.device),]
    elif isinstance(prim_expr, sps.VarAxisSG):
        log_reinit_param = []
    else:
        raise ValueError(f"Unsupported primitive type: {prim_expr}")
    return log_reinit_param



@register_handler
class SGHandler(PrimitiveHandler):
    base_class = sps.SuperGeon
    packed_class = sps.SuperGeonPacked
    packed_batched_class = sps.SGPackedBatched
    packed_batched_stochastic_class = sps.SGPackedBatchedStochastic
    packed_batched_su_class = sps.SGPackedBatchedSU
    packed_batched_stochastic_su_class = sps.SGPackedBatchedStochasticSU
    batched_param_size = 17
    unpack_params = unpack_params_sg
    param_to_var = param_to_var_sg
    var_to_param = var_to_param_sg
    param_from_variables_fast = sg_param_from_variables_fast
    batched_eval_function = batched_sg_packed_stochastic_eval
    point2prim_hard = None
    point2prim_soft = None
    reinit_params = None
    PARAM_IND_TO_NAME = SG_PARAM_IND_TO_NAME
    get_param_loss = get_param_loss_sf

@register_handler
class VarAxisSGHandler(PrimitiveHandler):
    base_class = sps.VarAxisSG
    packed_class = sps.VarAxisSGPacked
    packed_batched_class = sps.VarAxisSGPackedBatched
    packed_batched_stochastic_class = sps.VarAxisSGPackedBatchedStochastic
    packed_batched_su_class = sps.VarAxisSGPackedBatchedSU
    packed_batched_stochastic_su_class = sps.VarAxisSGPackedBatchedStochasticSU
    batched_param_size = 20
    unpack_params = unpack_params_var_axis_sg
    param_to_var = param_to_var_sg
    var_to_param = var_to_param_sg
    param_from_variables_fast = sg_param_from_variables_fast
    batched_eval_function = batched_varaxis_sg_packed_stochastic_eval
    point2prim_hard = None
    point2prim_soft = None
    reinit_params = reinit_params_varaxis_sg
    PARAM_IND_TO_NAME = SG_PARAM_IND_TO_NAME
    get_param_loss = get_param_loss_sf