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
from .utils import (
    build_transform_constants,
    make_packed_param_to_var,
    make_packed_var_to_param,
    make_param_to_var_dispatcher,
    make_var_to_param_dispatcher,
    make_param_from_variables_fast,
)
from ...torch_compute.batched_spproto import (
    batched_spp_packed_stochastic_eval,
    batched_varaxis_spp_packed_stochastic_eval,
)

_SPPROTO_TC = build_transform_constants([
    gls.Translate3D, gls.Translate3D, gls.Translate3D,
    "sp_size", "sp_size", "sp_size",
    "sp_roundness", "sp_roundness", "sp_roundness", "sp_roundness",
    "sp_dilate_3d",
    "sp_onion_ratio",
    "sp_extrussion", "sp_extrussion",
])
spproto_packed_param_to_var = make_packed_param_to_var(_SPPROTO_TC)
spproto_packed_var_to_param = make_packed_var_to_param(_SPPROTO_TC)
param_to_var_ext_spproto = make_param_to_var_dispatcher(spproto_packed_param_to_var)
var_to_param_ext_spproto = make_var_to_param_dispatcher(spproto_packed_var_to_param)
spproto_param_from_variables_fast = make_param_from_variables_fast(spproto_packed_var_to_param)

SPPROTO_PARAM_IND_TO_NAME = {
    0: "sp_size",
    1: "sp_roundness",
    2: "sp_dilate_3d",
    3: "sp_onion_ratio",
    4: "sp_extrussion",
}


def unpack_params_spproto(params):
    assert params.shape[-1] == 11
    size = params[..., :3]
    roundness = params[..., 3:7]
    dilate_3d = params[..., 7:8]
    onion = params[..., 8:9]
    extrusion = params[..., 9:11]
    return size, roundness, dilate_3d, onion, extrusion


def unpack_params_var_axis_spp(params):
    assert params.shape[-1] == 14
    spp_params = unpack_params_spproto(params[..., :11])
    logits = params[..., 11:14]
    return spp_params + (logits,)


@register_handler
class SPProtoHandler(PrimitiveHandler):
    base_class = sps.SPProto
    packed_class = sps.SPProtoPacked
    packed_batched_class = sps.SPProtoPackedBatched
    packed_batched_stochastic_class = sps.SPProtoPackedBatchedStochastic
    packed_batched_su_class = sps.SPProtoPackedBatchedSU
    packed_batched_stochastic_su_class = sps.SPProtoPackedBatchedStochasticSU
    batched_param_size = 17
    unpack_params = unpack_params_spproto
    param_to_var = param_to_var_ext_spproto
    var_to_param = var_to_param_ext_spproto
    param_from_variables_fast = spproto_param_from_variables_fast
    batched_eval_function = batched_spp_packed_stochastic_eval
    point2prim_hard = None
    point2prim_soft = None
    reinit_params = None
    PARAM_IND_TO_NAME = SPPROTO_PARAM_IND_TO_NAME    
    get_param_loss = get_param_loss_sf

    
def reinit_params_varaxis_spp(prim_expr, prim_param):
    val = AlgConf.DEFAULT_LOGITS_RESTART_VALUES[0]
    if isinstance(prim_expr, sps.SPProtoY):
        log_reinit = (val, -val, -val)
        log_reinit_param = [th.Tensor(log_reinit).to(prim_param.device),]
    elif isinstance(prim_expr, sps.SPProtoZ):
        log_reinit = (-val, val, -val)
        log_reinit_param = [th.Tensor(log_reinit).to(prim_param.device),]
    elif isinstance(prim_expr, sps.SPProtoX):
        log_reinit = (-val, -val, val)
        log_reinit_param = [th.Tensor(log_reinit).to(prim_param.device),]
    elif isinstance(prim_expr, sps.VarAxisSPP):
        log_reinit_param = []
    else:
        raise ValueError(f"Unsupported primitive type: {prim_expr}")
    return log_reinit_param


@register_handler
class VarAxisSPPHandler(PrimitiveHandler):
    base_class = sps.VarAxisSPP
    packed_class = sps.VarAxisSPPPacked
    packed_batched_class = sps.VarAxisSPPPackedBatched
    packed_batched_stochastic_class = sps.VarAxisSPPPackedBatchedStochastic
    packed_batched_su_class = sps.VarAxisSPPPackedBatchedSU
    packed_batched_stochastic_su_class = sps.VarAxisSPPPackedBatchedStochasticSU
    batched_param_size = 20
    unpack_params = unpack_params_var_axis_spp
    param_to_var = param_to_var_ext_spproto
    var_to_param = var_to_param_ext_spproto
    param_from_variables_fast = spproto_param_from_variables_fast
    batched_eval_function = batched_varaxis_spp_packed_stochastic_eval
    point2prim_hard = None
    point2prim_soft = None
    reinit_params = reinit_params_varaxis_spp
    PARAM_IND_TO_NAME = SPPROTO_PARAM_IND_TO_NAME
    get_param_loss = get_param_loss_sf
