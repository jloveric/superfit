import torch as th
import geolipi.symbolic as gls
import superfit.symbolic as sps
from .sf_handler import unpack_params_sf, split_sf_packed
from ..primitive_registry import PrimitiveHandler, register_handler
from .utils import process_param_to_var, process_var_to_param, su_param_to_var, su_var_to_param
from ...torch_compute.batched_primitives import batched_varaxis_sf_packed_stochastic_eval, batched_solid_sf_packed_stochastic_eval

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

def split_solid_sf_packed(param):
    param_sf = split_sf_packed(param[..., :14])
    logits = param[..., 14:18]
    return param_sf + (logits,)

def split_varaxis_sf_packed(param):
    param_sf = split_sf_packed(param[..., :14])
    logits = param[..., 14:17]
    return param_sf + (logits,)

def solid_sf_packed_param_to_var(param: th.Tensor) -> th.Tensor:
    """
    param -> variable  (inverse squash)
    Uses algebraic-tanh inverse:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        => v = pu / sqrt(1 - pu^2), with tiny safety.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8 = split_solid_sf_packed(param)

    sym_list   = [gls.Translate3D, "sp_size", "sp_roundness", "sp_dilate_3d", "sp_scale_opp", "sp_bulge", "sp_onion_ratio", "sp_logits"]
    param_list = [p0,              p2,         p3,              p4,              p5,              p6,              p7,              p8]
    v_list = process_param_to_var(sym_list, param_list)

    v0, v2, v3, v4, v5, v6, v7, v8 = v_list
    v1 = p1.detach().clone()                            # pass-through (non-optim)

    # Concatenate and return a LEAF tensor for the optimizer
    vcat = th.cat([v0, v1, v2, v3, v4, v5, v6, v7, v8], dim=-1)
    vcat = vcat.detach().requires_grad_(True)
    return vcat

def varaxis_sf_packed_param_to_var(param: th.Tensor) -> th.Tensor:
    """
    param -> variable  (inverse squash)
    Uses algebraic-tanh inverse:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        => v = pu / sqrt(1 - pu^2), with tiny safety.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8 = split_varaxis_sf_packed(param)

    sym_list   = [gls.Translate3D, "sp_size", "sp_roundness", "sp_dilate_3d", "sp_scale_opp", "sp_bulge", "sp_onion_ratio", "sp_logits"]
    param_list = [p0,              p2,         p3,              p4,              p5,              p6,              p7,              p8]
    v_list = process_param_to_var(sym_list, param_list)

    v0, v2, v3, v4, v5, v6, v7, v8 = v_list
    v1 = p1.detach().clone()                            # pass-through (non-optim)

    # Concatenate and return a LEAF tensor for the optimizer
    vcat = th.cat([v0, v1, v2, v3, v4, v5, v6, v7, v8], dim=-1)
    vcat = vcat.detach().requires_grad_(True)
    return vcat

def solid_sf_packed_var_to_param(variable: th.Tensor) -> th.Tensor:
    """
    variable -> param  (forward squash)
    Algebraic-tanh squash with margin:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        param  = p_unit * scale + offset
    """
    v0, v1, v2, v3, v4, v5, v6, v7, v8 = split_solid_sf_packed(variable)

    sym_list = [gls.Translate3D, "sp_size", "sp_roundness", "sp_dilate_3d", "sp_scale_opp", "sp_bulge", "sp_onion_ratio", "sp_logits"]
    v_parts  = [v0,              v2,         v3,              v4,              v5,              v6,              v7,              v8]
    p_list = process_var_to_param(sym_list, v_parts)

    p0, p2, p3, p4, p5, p6, p7, p8 = p_list
    p1 = v1  # pass-through
    #
    return th.cat([p0, p1, p2, p3, p4, p5, p6, p7, v8], dim=-1)
    # return th.cat([p0, p1, p2, p3, p4, p5, p6, p7, p8], dim=-1)

def varaxis_sf_packed_var_to_param(variable: th.Tensor) -> th.Tensor:
    """
    variable -> param  (forward squash)
    Algebraic-tanh squash with margin:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        param  = p_unit * scale + offset
    """
    v0, v1, v2, v3, v4, v5, v6, v7, v8 = split_varaxis_sf_packed(variable)

    sym_list = [gls.Translate3D, "sp_size", "sp_roundness", "sp_dilate_3d", "sp_scale_opp", "sp_bulge", "sp_onion_ratio", "sp_logits"]
    v_parts  = [v0,              v2,         v3,              v4,              v5,              v6,              v7,              v8]
    p_list = process_var_to_param(sym_list, v_parts)

    p0, p2, p3, p4, p5, p6, p7, p8 = p_list
    p1 = v1  # pass-through

    return th.cat([p0, p1, p2, p3, p4, p5, p6, p7, v8], dim=-1)
    # return th.cat([p0, p1, p2, p3, p4, p5, p6, p7, p8], dim=-1)


def param_to_var_sf(param: th.Tensor, local_ind: int) -> th.Tensor:

    if local_ind == 0:
        variable = solid_sf_packed_param_to_var(param)
    elif local_ind == 1:
        variable = su_param_to_var(param)
    elif local_ind == 2:
        variable = th.autograd.Variable(param, requires_grad=True)
    else:
        raise ValueError(f"Unsupported local index: {local_ind}")
    return variable

def var_to_param_sf(variable: th.Tensor, local_ind: int) -> th.Tensor:
    if local_ind == 0:
        param = solid_sf_packed_var_to_param(variable)
    elif local_ind == 1:
        param = su_var_to_param(variable)
    elif local_ind == 2:
        param = variable
    else:
        raise ValueError(f"Unsupported local index: {local_ind}")
    return param

def _solid_sf_param_from_variables_fast(tensor_list):
    # TODO: write a faster version...
    param_list = []
    variable = tensor_list[0]
    param = solid_sf_packed_var_to_param(variable)
    param_list.append(param)
    variable = tensor_list[1]
    mul, extra = 1.0, 1.0
    param = th.tanh(variable) * mul + extra
    param_list.append(param)
    variable = tensor_list[2]
    param_list.append(variable)
    return param_list

def _varaxis_sf_param_from_variables_fast(tensor_list):
    # TODO: write a faster version...
    param_list = []
    variable = tensor_list[0]
    param = varaxis_sf_packed_var_to_param(variable)
    param_list.append(param)
    variable = tensor_list[1]
    mul, extra = 1.0, 1.0
    param = th.tanh(variable) * mul + extra
    param_list.append(param)
    variable = tensor_list[2]
    param_list.append(variable)
    return param_list
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
    param_to_var = param_to_var_sf
    var_to_param = var_to_param_sf
    param_from_variables_fast = _solid_sf_param_from_variables_fast
    batched_eval_function = batched_solid_sf_packed_stochastic_eval

# Param to var classes.




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
    # Same functions will work.
    param_to_var = param_to_var_sf
    var_to_param = var_to_param_sf
    param_from_variables_fast = _varaxis_sf_param_from_variables_fast
    batched_eval_function = batched_varaxis_sf_packed_stochastic_eval
    