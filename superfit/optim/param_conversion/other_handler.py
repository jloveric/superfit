import torch as th
import geolipi.symbolic as gls
import superfit.symbolic as sps
from ..primitive_registry import PrimitiveHandler, register_handler
from .utils import (process_param_to_var, process_var_to_param, su_param_to_var, su_var_to_param)

def unpack_params_sq(params):
    assert params.shape[-1] == 5
    skew_vec = params[..., :3]
    epsilon_1 = params[..., 3:4]
    epsilon_2 = params[..., 4:5]
    return skew_vec, epsilon_1, epsilon_2

def split_sq_packed(param):
    param_0 = param[..., :3]
    param_1 = param[..., 3:6]
    param_2 = param[..., 6:9]
    param_3 = param[..., 9:10]
    param_4 = param[..., 10:11]
    return param_0, param_1, param_2, param_3, param_4

# Processing function.

def sq_packed_param_to_var(param: th.Tensor) -> th.Tensor:
    """
    param -> variable  (inverse squash)
    Uses algebraic-tanh inverse:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        => v = pu / sqrt(1 - pu^2), with tiny safety.
    """
    p0, p1, p2, p3, p4 = split_sq_packed(param)

    sym_list   = [gls.Translate3D, "sp_size", "sp_roundness", "sp_roundness", ]
    param_list = [p0,              p2,         p3,              p4          ]
    
    v_list = process_param_to_var(sym_list, param_list)

    v0, v2, v3, v4 = v_list
    v1 = p1.detach().clone()                            # pass-through (non-optim)
    # Concatenate and return a LEAF tensor for the optimizer
    vcat = th.cat([v0, v1, v2, v3, v4], dim=-1)
    vcat = vcat.detach().requires_grad_(True)
    return vcat
#  Param to var

def sq_packed_var_to_param(variable: th.Tensor) -> th.Tensor:
    """
    variable -> param  (forward squash)
    Algebraic-tanh squash with margin:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        param  = p_unit * scale + offset
    """
    v0, v1, v2, v3, v4 = split_sq_packed(variable)
    sym_list = [gls.Translate3D, "sp_size", "sp_roundness", "sp_roundness", ]
    v_parts  = [v0,              v2,         v3,              v4]
    p_list = process_var_to_param(sym_list, v_parts)
    p0, p2, p3, p4 = p_list
    p1 = v1  # pass-through
    return th.cat([p0, p1, p2, p3, p4], dim=-1)

# Param to var
def param_to_var_sq(param: th.Tensor, local_ind: int) -> th.Tensor:
    if local_ind == 0:
        variable = sq_packed_param_to_var(param)
    elif local_ind == 1:
        variable = su_param_to_var(param)
    elif local_ind == 2:
        variable = th.autograd.Variable(param, requires_grad=True)
    else:
        raise ValueError(f"Unsupported local index: {local_ind}")
    return variable

# Var to param
def var_to_param_sq(variable: th.Tensor, local_ind: int) -> th.Tensor:
    if local_ind == 0:
        param = sq_packed_var_to_param(variable)
    elif local_ind == 1:
        param = su_var_to_param(variable)
    elif local_ind == 2:
        param = variable
    else:
        raise ValueError(f"Unsupported local index: {local_ind}")
    return param


@register_handler
class SQHandler(PrimitiveHandler):
    base_class = sps.SuperQuadric
    packed_class = sps.SQPacked
    packed_batched_class = sps.SQPackedBatched
    packed_batched_stochastic_class = sps.SQPackedBatchedStochastic
    packed_batched_su_class = sps.SQPackedBatchedSU
    packed_batched_stochastic_su_class = sps.SQPackedBatchedStochasticSU
    
    unpack_params = unpack_params_sq
    param_to_var = param_to_var_sq
    var_to_param = var_to_param_sq


## CUBOID

def unpack_params_cuboid(params):
    assert params.shape[-1] == 3
    size = params[..., :3]
    return size

def split_cuboid_packed(param):
    param_0 = param[..., :3]
    param_1 = param[..., 3:6]
    param_2 = param[..., 6:9]
    return param_0, param_1, param_2

def cuboid_packed_param_to_var(param: th.Tensor) -> th.Tensor:
    """
    param -> variable  (inverse squash)
    Uses algebraic-tanh inverse:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        => v = pu / sqrt(1 - pu^2), with tiny safety.
    """
    p0, p1, p2 = split_cuboid_packed(param)
    sym_list   = [gls.Translate3D, "sp_size", ]
    param_list = [p0,              p2          ]
    v_list = process_param_to_var(sym_list, param_list)
    v0, v2 = v_list
    v1 = p1.detach().clone()                            # pass-through (non-optim)
    # Concatenate and return a LEAF tensor for the optimizer
    vcat = th.cat([v0, v1, v2], dim=-1)
    vcat = vcat.detach().requires_grad_(True)
    return vcat

def cuboid_packed_var_to_param(variable: th.Tensor) -> th.Tensor:
    """
    variable -> param  (forward squash)
    Algebraic-tanh squash with margin:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        param  = p_unit * scale + offset
    """
    v0, v1, v2 = split_cuboid_packed(variable)
    sym_list = [gls.Translate3D, "sp_size", ]
    v_parts  = [v0,              v2          ]
    p_list = process_var_to_param(sym_list, v_parts)
    p0, p2 = p_list
    p1 = v1  # pass-through
    return th.cat([p0, p1, p2], dim=-1)

# Param to var
def param_to_var_cuboid(param: th.Tensor, local_ind: int) -> th.Tensor:
    if local_ind == 0:
        variable = cuboid_packed_param_to_var(param)
    elif local_ind == 1:
        variable = su_param_to_var(param)
    elif local_ind == 2:
        variable = th.autograd.Variable(param, requires_grad=True)
    else:
        raise ValueError(f"Unsupported local index: {local_ind}")
    return variable

# Var to param
def var_to_param_cuboid(variable: th.Tensor, local_ind: int) -> th.Tensor:
    if local_ind == 0:
        param = cuboid_packed_var_to_param(variable)
    elif local_ind == 1:
        param = su_var_to_param(variable)
    elif local_ind == 2:
        param = variable
    else:
        raise ValueError(f"Unsupported local index: {local_ind}")
    return param
# Handler
@register_handler
class CuboidHandler(PrimitiveHandler):
    base_class = sps.Cuboid
    packed_class = sps.CuboidPacked
    packed_batched_class = sps.CuboidPackedBatched
    packed_batched_stochastic_class = sps.CuboidPackedBatchedStochastic
    packed_batched_su_class = sps.CuboidPackedBatchedSU
    packed_batched_stochastic_su_class = sps.CuboidPackedBatchedStochasticSU
    
    unpack_params = unpack_params_cuboid
    param_to_var = param_to_var_cuboid
    var_to_param = var_to_param_cuboid
    # For CUboid loss param will be a problem