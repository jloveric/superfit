import torch as th
import superfit.symbolic as sps
import geolipi.symbolic as gls
from sysl.torch_compute.mat_combinators import sdf_geom_only_smooth_union, sdf_smooth_union
from ..primitive_registry import PrimitiveHandler, register_handler
from .utils import (
    build_transform_constants,
    make_packed_param_to_var,
    make_packed_var_to_param,
    make_param_to_var_dispatcher,
    make_var_to_param_dispatcher,
    make_param_from_variables_fast,
)
from ...torch_compute.batched_sf import (sample_gumbel, batched_sf_packed_eval,
                                                batched_sf_packed_stochastic_eval,
                                                batched_varaxis_sf_packed_stochastic_eval, 
                                                batched_solid_sf_packed_stochastic_eval)
from ...torch_compute.compile_friendly import _sdf_smooth_union_pair
from sysl.torch_compute.mat_combinators import mix, EPSILON
from ...utils.config import AlgorithmConfig as AlgConf
from ..losses import get_param_loss_sf


# Packed layout -- first 11 dims transformed, rest pass-through.
#   [0:3] translate | [3:6] size | [6:7] roundness | [7:8] dilate | [8:9] taper | [9:10] bulge | [10:11] onion
#   SF (14): [11:14] rotation | VarAxis (17): + [14:17] logits | SolidSF (18): + [14:18] logits
_SF_TC = build_transform_constants([
    gls.Translate3D, gls.Translate3D, gls.Translate3D,
    "sp_size", "sp_size", "sp_size",
    "sp_roundness", "sp_dilate_3d", "sp_taper", "sp_bulge", "sp_onion_ratio",
])
sf_packed_param_to_var = make_packed_param_to_var(_SF_TC)
sf_packed_var_to_param = make_packed_var_to_param(_SF_TC)
param_to_var_sf = make_param_to_var_dispatcher(sf_packed_param_to_var)
var_to_param_sf = make_var_to_param_dispatcher(sf_packed_var_to_param)
_params_from_variables_fast_sf = make_param_from_variables_fast(sf_packed_var_to_param)
param_to_var_ext_sf = make_param_to_var_dispatcher(sf_packed_param_to_var)
var_to_param_ext_sf = make_var_to_param_dispatcher(sf_packed_var_to_param)
_ext_sf_param_from_variables_fast = make_param_from_variables_fast(sf_packed_var_to_param)

PARAM_IND_TO_NAME = {
    0: "sp_size",
    1: "sp_roundness",
    2: "sp_dilate_3d",
    3: "sp_taper",
    4: "sp_bulge",
    5: "sp_onion_ratio",
}


# Unpack
def unpack_params_sf(params):
    assert params.shape[-1] == 8
    size = params[..., :3]
    roundness = params[..., 3:4]
    dilate_3d = params[..., 4:5]
    scale = params[..., 5:6]
    bulge_ratio = params[..., 6:7]
    onion_ratio = params[..., 7:8]
    return size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio

# Custom function for param match:

def point2prim_hard(coords, all_params):
    # B, N
    params, su_vals, logits, temperature = all_params 
    outputs = batched_sf_packed_eval(coords, params)
    g = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)

    # Unpack weights explicitly
    w0 = w[:, 0:1]
    w1 = w[:, 1:2]

    # Equivalent to weighted sum of [outputs, 1.0]
    primitive_sdfs = outputs * w0 + w1

    K = primitive_sdfs.shape[0]

    prim_with_ids = []
    for i in range(K):
        ind_field = th.zeros_like(primitive_sdfs[i]) + i
        updated_prim = th.stack([primitive_sdfs[i], ind_field], dim=-1)
        prim_with_ids.append(updated_prim)

    out = prim_with_ids[0]
    for i in range(1, K):
        k_reshaped = su_vals[i-1].unsqueeze(-1)
        out = sdf_geom_only_smooth_union(out, prim_with_ids[i], k_reshaped)
    
    return out


def point2prim_soft_distance(coords, all_params):
    # B, N
    params, su_vals, logits, temperature = all_params 
    outputs = batched_sf_packed_eval(coords, params)
    g = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)

    # Unpack weights explicitly
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]

    # Equivalent to weighted sum of [outputs, 1.0]
    primitive_sdfs = outputs * w0 + w1

    # Shape: B, N
    tau = temperature ** 2.0
    K = primitive_sdfs.shape[0]
    # Do a distance based point association:
    logits = -primitive_sdfs.T / tau
    logits = logits - logits.max(dim=-1, keepdim=True).values  # stability
    probs = th.softmax(logits, dim=-1)

    out = primitive_sdfs[0]
    for i in range(1, K):
        k_reshaped = su_vals[i-1].unsqueeze(-1)
        out = _sdf_smooth_union_pair(out, primitive_sdfs[i], k_reshaped)

    return probs, out


def point2prim_soft_smu(coords, all_params, smu_k=0.01, scale_factor=1.0):
    # B, N
    params, su_vals, logits, temperature = all_params 
    outputs = batched_sf_packed_eval(coords, params)
    g = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)

    # Unpack weights explicitly
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]

    # Equivalent to weighted sum of [outputs, 1.0]
    primitive_sdfs = outputs * w0 + w1

    # Shape: B, N
    # tau = temperature ** 2.0
    K = primitive_sdfs.shape[0]
    # Do a distance based point association:
    # probs = th.softmax(logits, dim=-1)

    out = primitive_sdfs[0]
    empty_distr = th.zeros((primitive_sdfs.shape[1], K+1)).to(primitive_sdfs.device).float()
    # Add the last one
    cur_distr = empty_distr.clone()
    cur_distr[:, 0] = 1.0
    for i in range(1, K):
        k_reshaped = su_vals[i-1].unsqueeze(-1)
        var_a, var_b = out, primitive_sdfs[i]
        h = th.clamp(0.5 + 0.5 * (var_b - var_a) / (k_reshaped + EPSILON), min=0.0, max=1.0)
        out = mix(var_b, var_a, h) - k_reshaped * h * (1.0 - h);

        h_2 = th.clamp(0.5 + 0.5 * (var_b- var_a) / (smu_k + EPSILON), min=0.0, max=1.0)
        h_2 = h_2.squeeze().unsqueeze(-1)
        new_distr = empty_distr.clone()
        new_distr[:, i] = 1.0
        # out = _sdf_smooth_union_pair(out, primitive_sdfs[i], k_reshaped)
        cur_distr = mix(new_distr, cur_distr, h_2)
    
    output_tanh = th.tanh(out * scale_factor)
    output_for_occ_occ = th.sigmoid(-output_tanh * scale_factor)
    cur_distr[:, -1] = 1.0 - output_for_occ_occ * 1.0

    return cur_distr, out

@register_handler
class SFHandler(PrimitiveHandler):
    base_class = sps.SuperFrustum
    packed_class = sps.SuperFrustumPacked
    packed_batched_class = sps.SuperFrustumPackedBatched
    packed_batched_stochastic_class = sps.SuperFrustumPackedBatchedStochastic
    packed_batched_su_class = sps.SuperFrustumPackedBatchedSU
    packed_batched_stochastic_su_class = sps.SuperFrustumPackedBatchedStochasticSU
    batched_param_size = 14
    
    unpack_params = unpack_params_sf
    param_to_var = param_to_var_sf
    var_to_param = var_to_param_sf
    param_from_variables_fast = _params_from_variables_fast_sf
    batched_eval_function = batched_sf_packed_stochastic_eval
    point2prim_hard = point2prim_hard
    point2prim_soft = point2prim_soft_smu
    PARAM_IND_TO_NAME = PARAM_IND_TO_NAME
    get_param_loss = get_param_loss_sf

def unpack_params_solid_sf(params):
    assert params.shape[-1] == 12
    sf_params = unpack_params_sf(params[..., :8])
    logits = params[..., 8:12]
    return sf_params + (logits,)

def unpack_params_var_axis_sf(params):
    assert params.shape[-1] == 11
    sf_params = unpack_params_sf(params[..., :8])
    logits = params[..., 8:11]
    return sf_params + (logits,)


# Reinit helpers

def reinit_params_solid_sf(prim_expr, prim_param):
    if isinstance(prim_expr, sps.SolidSF):
        log_reinit_param = []
    else:
        raise ValueError(f"Unsupported primitive type: {prim_expr}")
    return log_reinit_param

def reinit_params_varaxis_sf(prim_expr, prim_param):
    val = AlgConf.DEFAULT_LOGITS_RESTART_VALUES[0]
    if isinstance(prim_expr, sps.SuperFrustumY):
        log_reinit = (val, -val, -val)
        log_reinit_param = [th.Tensor(log_reinit).to(prim_param.device),]
    elif isinstance(prim_expr, sps.SuperFrustumZ):
        log_reinit = (-val, val, -val)
        log_reinit_param = [th.Tensor(log_reinit).to(prim_param.device),]
    elif isinstance(prim_expr, sps.SuperFrustumX):
        log_reinit = (-val, -val, val)
        log_reinit_param = [th.Tensor(log_reinit).to(prim_param.device),]
    elif isinstance(prim_expr, sps.VarAxisSF):
        log_reinit_param = []
    else:
        raise ValueError(f"Unsupported primitive type: {prim_expr}")
    return log_reinit_param



@register_handler
class VarAxisSFHandler(PrimitiveHandler):
    base_class = sps.VarAxisSF
    packed_class = sps.VarAxisSFPacked
    packed_batched_class = sps.VarAxisSFPackedBatched
    packed_batched_stochastic_class = sps.VarAxisSFPackedBatchedStochastic
    packed_batched_su_class = sps.VarAxisSFPackedBatchedSU
    packed_batched_stochastic_su_class = sps.VarAxisSFPackedBatchedStochasticSU
    batched_param_size = 17
    unpack_params = unpack_params_var_axis_sf
    param_to_var = param_to_var_ext_sf
    var_to_param = var_to_param_ext_sf
    param_from_variables_fast = _ext_sf_param_from_variables_fast
    batched_eval_function = batched_varaxis_sf_packed_stochastic_eval
    point2prim_hard = None
    point2prim_soft = None
    reinit_params = reinit_params_varaxis_sf
    get_param_loss = get_param_loss_sf



@register_handler
class SolidSFHandler(PrimitiveHandler):
    base_class = sps.SolidSF
    packed_class = sps.SolidSFPacked
    packed_batched_class = sps.SolidSFPackedBatched
    packed_batched_stochastic_class = sps.SolidSFPackedBatchedStochastic
    packed_batched_su_class = sps.SolidSFPackedBatchedSU
    packed_batched_stochastic_su_class = sps.SolidSFPackedBatchedStochasticSU
    batched_param_size = 18
    unpack_params = unpack_params_solid_sf
    param_to_var = param_to_var_ext_sf
    var_to_param = var_to_param_ext_sf
    param_from_variables_fast = _ext_sf_param_from_variables_fast
    batched_eval_function = batched_solid_sf_packed_stochastic_eval
    point2prim_hard = None
    point2prim_soft = None
    reinit_params = reinit_params_solid_sf
    get_param_loss = get_param_loss_sf