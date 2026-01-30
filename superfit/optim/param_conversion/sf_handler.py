import torch as th
import superfit.symbolic as sps
import geolipi.symbolic as gls
from ..primitive_registry import PrimitiveHandler, register_handler
from .utils import process_param_to_var, process_var_to_param, su_param_to_var, su_var_to_param
from ...torch_compute.batched_primitives import batched_sf_packed_stochastic_eval
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

# Param<->var functions
def split_sf_packed(param):
    param_0 = param[..., :3]
    param_1 = param[..., 3:6]
    param_2 = param[..., 6:9]
    param_3 = param[..., 9:10]
    param_4 = param[..., 10:11]
    param_5 = param[..., 11:12]
    param_6 = param[..., 12:13]
    param_7 = param[..., 13:14]
    return param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7

def sf_packed_param_to_var(param: th.Tensor) -> th.Tensor:
    """
    param -> variable  (inverse squash)
    Uses algebraic-tanh inverse:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        => v = pu / sqrt(1 - pu^2), with tiny safety.
    """
    p0, p1, p2, p3, p4, p5, p6, p7 = split_sf_packed(param)

    sym_list   = [gls.Translate3D, "sp_size", "sp_roundness", "sp_dilate_3d", "sp_scale_opp", "sp_bulge", "sp_onion_ratio"]
    param_list = [p0,              p2,         p3,              p4,              p5,              p6,              p7]
    
    v_list = process_param_to_var(sym_list, param_list)

    v0, v2, v3, v4, v5, v6, v7 = v_list
    v1 = p1.detach().clone()                            # pass-through (non-optim)
    # Concatenate and return a LEAF tensor for the optimizer
    vcat = th.cat([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1)
    vcat = vcat.detach().requires_grad_(True)
    return vcat

def sf_packed_var_to_param(variable: th.Tensor) -> th.Tensor:
    """
    variable -> param  (forward squash)
    Algebraic-tanh squash with margin:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        param  = p_unit * scale + offset
    """
    v0, v1, v2, v3, v4, v5, v6, v7 = split_sf_packed(variable)

    sym_list = [gls.Translate3D, "sp_size", "sp_roundness", "sp_dilate_3d", "sp_scale_opp", "sp_bulge", "sp_onion_ratio"]
    v_parts  = [v0,              v2,         v3,              v4,              v5,              v6,              v7]
    p_list = process_var_to_param(sym_list, v_parts)

    p0, p2, p3, p4, p5, p6, p7 = p_list
    p1 = v1  # pass-through

    return th.cat([p0, p1, p2, p3, p4, p5, p6, p7], dim=-1)

def param_to_var_sf(param: th.Tensor, local_ind: int) -> th.Tensor:

    if local_ind == 0:
        variable = sf_packed_param_to_var(param)
    elif local_ind == 1:
        variable = su_param_to_var(param)
    elif local_ind == 2:
        variable = th.autograd.Variable(param, requires_grad=True)
    else:
        raise ValueError(f"Unsupported local index: {local_ind}")
    return variable

def var_to_param_sf(variable: th.Tensor, local_ind: int) -> th.Tensor:
    if local_ind == 0:
        param = sf_packed_var_to_param(variable)
    elif local_ind == 1:
        param = su_var_to_param(variable)
    elif local_ind == 2:
        param = variable
    else:
        raise ValueError(f"Unsupported local index: {local_ind}")
    return param
### Handlers

def _params_from_variables_fast_sf(tensor_list):
    # TODO: write a faster version...
    param_list = []
    variable = tensor_list[0]
    param = sf_packed_var_to_param(variable)
    param_list.append(param)
    variable = tensor_list[1]
    mul, extra = 0.25, 0.25
    param = th.tanh(variable) * mul + extra
    param_list.append(param)
    variable = tensor_list[2]
    param = variable
    param_list.append(param)
    return param_list

PARAM_IND_TO_NAME = {
    0: "sp_size",
    1: "sp_roundness",
    2: "sp_dilate_3d",
    3: "sp_scale_opp",
    4: "sp_bulge",
    5: "sp_onion_ratio",
}

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
    
    PARAM_IND_TO_NAME = PARAM_IND_TO_NAME


