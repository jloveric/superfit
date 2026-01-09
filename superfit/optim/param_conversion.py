import torch as th
import geolipi.symbolic as gls
import superfit.symbolic as sps
from ..symbolic.utils import split_tapered_packed_param, gather_primitives
from ..symbolic.utils import gather_smooth_union_ops, generate_from_sm_ops_and_primitives
from ..utils.config import AlgorithmConfig as AlgConf

VARIABLE_TRANSFORM_PARAM = {
    gls.SmoothUnion: (1.0, 1.0) ,
}

RANGE_DICT = {
    gls.SmoothUnion: (AlgConf.OPT_MIN_SCALE, AlgConf.OPT_MAX_SCALE / 2.0),
    gls.Translate3D: (-AlgConf.OPT_MAX_TRANSLATE, AlgConf.OPT_MAX_TRANSLATE),
    "neo_size":      (0.01,   1.98),
    "neo_roundness": (AlgConf.OPT_MIN_SCALE,   AlgConf.OPT_MAX_SCALE / 2.0),
    "neo_dilate_3d": (0.0,   AlgConf.OPT_MAX_SCALE / 2.0),
    "neo_scale_opp": (0.01,   1.995),
    "neo_bulge":     (AlgConf.OPT_MIN_SCALE,   0.995),
    "neo_onion_ratio": (AlgConf.OPT_MIN_SCALE,   0.975),
    "neo_sq_scale": (AlgConf.OPT_MIN_SCALE, AlgConf.OPT_MAX_SCALE/2.0),
    "neo_logits": (-2.0, 2.0),
}

# Small margins for stability (avoid exact ±1 and sqrt(0))
_EPS_OUT   = 1e-4
_SAFE_SQRT = 1e-12


def split_ntco_packed_param(param):
    param_0 = param[..., :3]
    param_1 = param[..., 3:6]
    param_2 = param[..., 6:9]
    param_3 = param[..., 9:10]
    param_4 = param[..., 10:11]
    param_5 = param[..., 11:12]
    param_6 = param[..., 12:13]
    param_7 = param[..., 13:14]
    return param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7

def split_solid_ntco_packed_param(param):
    param_0 = param[..., :3]
    param_1 = param[..., 3:6]
    param_2 = param[..., 6:9]
    param_3 = param[..., 9:10]
    param_4 = param[..., 10:11]
    param_5 = param[..., 11:12]
    param_6 = param[..., 12:13]
    param_7 = param[..., 13:14]
    param_8 = param[..., 14:18]
    return param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8


def ntco_packed_param_to_var(param: th.Tensor) -> th.Tensor:
    """
    param -> variable  (inverse squash)
    Uses algebraic-tanh inverse:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        => v = pu / sqrt(1 - pu^2), with tiny safety.
    """
    p0, p1, p2, p3, p4, p5, p6, p7 = split_ntco_packed_param(param)

    sym_list   = [gls.Translate3D, "neo_size", "neo_roundness", "neo_dilate_3d", "neo_scale_opp", "neo_bulge", "neo_onion_ratio"]
    param_list = [p0,              p2,         p3,              p4,              p5,              p6,              p7]

    v_list = []
    for sym, p in zip(sym_list, param_list):
        pmin, pmax = RANGE_DICT[sym]
        # 1) Clamp physical range
        p = th.clamp(p, pmin, pmax)
        # 2) Affine to (-1, 1) (unit domain)
        scale  = 0.5 * (pmax - pmin)
        offset = 0.5 * (pmax + pmin)
        pu = (p - offset) / scale                      # ideally in [-1, 1]
        # 3) Undo output margin and invert algebraic-tanh
        pu = pu / (1 - _EPS_OUT)                       # map back to (-1,1)
        pu = th.clamp(pu, -1 + 1e-6, 1 - 1e-6)         # safety for sqrt
        v  = pu / th.sqrt(th.clamp(1 - pu * pu, min=_SAFE_SQRT))
        v_list.append(v)

    v0, v2, v3, v4, v5, v6, v7 = v_list
    v1 = p1.detach().clone()                            # pass-through (non-optim)

    # Concatenate and return a LEAF tensor for the optimizer
    vcat = th.cat([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1)
    vcat = vcat.detach().requires_grad_(True)
    return vcat

def ntco_packed_var_to_param(variable: th.Tensor) -> th.Tensor:
    """
    variable -> param  (forward squash)
    Algebraic-tanh squash with margin:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        param  = p_unit * scale + offset
    """
    v0, v1, v2, v3, v4, v5, v6, v7 = split_ntco_packed_param(variable)

    sym_list = [gls.Translate3D, "neo_size", "neo_roundness", "neo_dilate_3d", "neo_scale_opp", "neo_bulge", "neo_onion_ratio"]
    v_parts  = [v0,              v2,         v3,              v4,              v5,              v6,              v7]

    p_list = []
    for sym, v in zip(sym_list, v_parts):
        pmin, pmax = RANGE_DICT[sym]
        scale  = 0.5 * (pmax - pmin)
        offset = 0.5 * (pmax + pmin)
        # Algebraic-tanh squash (polynomial tails), with margin from ±1
        p_unit = (1 - _EPS_OUT) * (v / th.sqrt(1 + v * v))
        p      = p_unit * scale + offset
        # Final safety clamp against drift
        p      = th.clamp(p, pmin, pmax)
        p_list.append(p)

    p0, p2, p3, p4, p5, p6, p7 = p_list
    p1 = v1  # pass-through

    return th.cat([p0, p1, p2, p3, p4, p5, p6, p7], dim=-1)




def solid_ntco_packed_param_to_var(param: th.Tensor) -> th.Tensor:
    """
    param -> variable  (inverse squash)
    Uses algebraic-tanh inverse:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        => v = pu / sqrt(1 - pu^2), with tiny safety.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8 = split_solid_ntco_packed_param(param)

    sym_list   = [gls.Translate3D, "neo_size", "neo_roundness", "neo_dilate_3d", "neo_scale_opp", "neo_bulge", "neo_onion_ratio", "neo_logits"]
    param_list = [p0,              p2,         p3,              p4,              p5,              p6,              p7,              p8]

    v_list = []
    for sym, p in zip(sym_list, param_list):
        pmin, pmax = RANGE_DICT[sym]
        # 1) Clamp physical range
        p = th.clamp(p, pmin, pmax)
        # 2) Affine to (-1, 1) (unit domain)
        scale  = 0.5 * (pmax - pmin)
        offset = 0.5 * (pmax + pmin)
        pu = (p - offset) / scale                      # ideally in [-1, 1]
        # 3) Undo output margin and invert algebraic-tanh
        pu = pu / (1 - _EPS_OUT)                       # map back to (-1,1)
        pu = th.clamp(pu, -1 + 1e-6, 1 - 1e-6)         # safety for sqrt
        v  = pu / th.sqrt(th.clamp(1 - pu * pu, min=_SAFE_SQRT))
        v_list.append(v)

    v0, v2, v3, v4, v5, v6, v7, v8 = v_list
    v1 = p1.detach().clone()                            # pass-through (non-optim)

    # Concatenate and return a LEAF tensor for the optimizer
    vcat = th.cat([v0, v1, v2, v3, v4, v5, v6, v7, v8], dim=-1)
    vcat = vcat.detach().requires_grad_(True)
    return vcat


def solid_ntco_packed_var_to_param(variable: th.Tensor) -> th.Tensor:
    """
    variable -> param  (forward squash)
    Algebraic-tanh squash with margin:
        p_unit = (1-eps) * v / sqrt(1 + v^2)
        param  = p_unit * scale + offset
    """
    v0, v1, v2, v3, v4, v5, v6, v7, v8 = split_solid_ntco_packed_param(variable)

    sym_list = [gls.Translate3D, "neo_size", "neo_roundness", "neo_dilate_3d", "neo_scale_opp", "neo_bulge", "neo_onion_ratio", "neo_logits"]
    v_parts  = [v0,              v2,         v3,              v4,              v5,              v6,              v7,              v8]

    p_list = []
    for sym, v in zip(sym_list, v_parts):
        pmin, pmax = RANGE_DICT[sym]
        scale  = 0.5 * (pmax - pmin)
        offset = 0.5 * (pmax + pmin)
        # Algebraic-tanh squash (polynomial tails), with margin from ±1
        p_unit = (1 - _EPS_OUT) * (v / th.sqrt(1 + v * v))
        p      = p_unit * scale + offset
        # Final safety clamp against drift
        p      = th.clamp(p, pmin, pmax)
        p_list.append(p)

    p0, p2, p3, p4, p5, p6, p7, p8 = p_list
    p1 = v1  # pass-through

    return th.cat([p0, p1, p2, p3, p4, p5, p6, p7, p8], dim=-1)



def transform_to_tunable(variable_list):

    params = []
    parsed_variables = []
    for cur_var in variable_list:
        param, inverted_variable = invert_variable(cur_var)
        params.append(param)
        parsed_variables.append(inverted_variable)

    return parsed_variables


def params_from_variables(variable_list, tensor_list):
    params = []
    for ind, inverted_variable in enumerate(variable_list):
        info = tensor_list[ind]
        cur_var = (inverted_variable, info[1], info[2], info[3])
        param = revert_variable(cur_var)
        params.append(param)

    return params


## SUPPORT - Translate, Rotate, Smooth Union, StochasticPrimitive
def invert_variable(variable_info_set):
    param, command_symbol, var_type, local_ind = variable_info_set
    if issubclass(command_symbol, gls.SmoothUnion):
        clip_min, clip_max = RANGE_DICT[command_symbol]
        param = th.clip(param, clip_min, clip_max)
        mul, extra = VARIABLE_TRANSFORM_PARAM[command_symbol]
        variable = th.atanh((param - extra) / mul)
        variable = th.autograd.Variable(variable, requires_grad=True)
    elif issubclass(command_symbol, (sps.StochasticPrimitive)):
        variable = th.autograd.Variable(param, requires_grad=True)
    elif issubclass(command_symbol, sps.SuperFrustumPackedBatchedSU):
        # Parameters are prim Param and SU
        if local_ind == 0:
            variable = ntco_packed_param_to_var(param)
        elif local_ind == 1:
            command_symbol = gls.SmoothUnion
            clip_min, clip_max = RANGE_DICT[command_symbol]
            param = th.clip(param, clip_min, clip_max)
            mul, extra = VARIABLE_TRANSFORM_PARAM[command_symbol]
            variable = th.atanh((param - extra) / mul)
            variable = th.autograd.Variable(variable, requires_grad=True)
        else:
            raise ValueError(f"Unsupported local index: {local_ind}")
    elif issubclass(command_symbol, sps.SuperFrustumPackedBatchedStochasticSU):
        # Parameters are prim Param, SU, Logits
        if local_ind == 0:
            variable = ntco_packed_param_to_var(param)
        elif local_ind == 1:
            command_symbol = gls.SmoothUnion
            clip_min, clip_max = RANGE_DICT[command_symbol]
            param = th.clip(param, clip_min, clip_max)
            mul, extra = VARIABLE_TRANSFORM_PARAM[command_symbol]
            variable = th.atanh((param - extra) / mul)
            variable = th.autograd.Variable(variable, requires_grad=True)
        elif local_ind == 2:
            variable = th.autograd.Variable(param, requires_grad=True)
        else:
            raise ValueError(f"Unsupported local index: {local_ind}")
    elif issubclass(command_symbol, sps.SolidSFPackedBatchedSU):
        # Parameters are prim Param and SU
        if local_ind == 0:
            variable = solid_ntco_packed_param_to_var(param)
        elif local_ind == 1:
            command_symbol = gls.SmoothUnion
            clip_min, clip_max = RANGE_DICT[command_symbol]
            param = th.clip(param, clip_min, clip_max)
            mul, extra = VARIABLE_TRANSFORM_PARAM[command_symbol]
            variable = th.atanh((param - extra) / mul)
            variable = th.autograd.Variable(variable, requires_grad=True)
        else:
            raise ValueError(f"Unsupported local index: {local_ind}")
    elif issubclass(command_symbol, sps.SolidSFPackedBatchedStochasticSU):
        # Parameters are prim Param, SU, Logits
        if local_ind == 0:
            variable = solid_ntco_packed_param_to_var(param)
        elif local_ind == 1:
            command_symbol = gls.SmoothUnion
            clip_min, clip_max = RANGE_DICT[command_symbol]
            param = th.clip(param, clip_min, clip_max)
            mul, extra = VARIABLE_TRANSFORM_PARAM[command_symbol]
            variable = th.atanh((param - extra) / mul)
            variable = th.autograd.Variable(variable, requires_grad=True)
        elif local_ind == 2:
            variable = th.autograd.Variable(param, requires_grad=True)
        else:
            raise ValueError(f"Unsupported local index: {local_ind}")
    else:
        raise ValueError(f"Unsupported command symbol: {command_symbol}")
    return param, variable

def revert_variable(variable_info_set):
    variable, command_symbol, var_type, local_ind = variable_info_set
    if issubclass(command_symbol, gls.SmoothUnion):
        mul, extra = VARIABLE_TRANSFORM_PARAM[command_symbol]
        param = th.tanh(variable) * mul + extra
    elif issubclass(command_symbol, (sps.StochasticPrimitive)):
        param = variable
    elif issubclass(command_symbol, sps.SuperFrustumPackedBatchedSU):
        if local_ind == 0:
            param = ntco_packed_var_to_param(variable)
        elif local_ind == 1:
            command_symbol = gls.SmoothUnion
            mul, extra = VARIABLE_TRANSFORM_PARAM[command_symbol]
            param = th.tanh(variable) * mul + extra
        else:
            raise ValueError(f"Unsupported local index: {local_ind}")
    elif issubclass(command_symbol, sps.SuperFrustumPackedBatchedStochasticSU):
        if local_ind == 0:
            param = ntco_packed_var_to_param(variable)
        elif local_ind == 1:
            command_symbol = gls.SmoothUnion
            mul, extra = VARIABLE_TRANSFORM_PARAM[command_symbol]
            param = th.tanh(variable) * mul + extra
        elif local_ind == 2:
            param = variable
        else:
            raise ValueError(f"Unsupported local index: {local_ind}")
    elif issubclass(command_symbol, sps.SolidSFPackedBatchedSU):
        if local_ind == 0:
            param = solid_ntco_packed_var_to_param(variable)
        elif local_ind == 1:
            command_symbol = gls.SmoothUnion
            mul, extra = VARIABLE_TRANSFORM_PARAM[command_symbol]
            param = th.tanh(variable) * mul + extra
        else:
            raise ValueError(f"Unsupported local index: {local_ind}")
    elif issubclass(command_symbol, sps.SolidSFPackedBatchedStochasticSU):
        if local_ind == 0:
            param = solid_ntco_packed_var_to_param(variable)
        elif local_ind == 1:
            command_symbol = gls.SmoothUnion
            mul, extra = VARIABLE_TRANSFORM_PARAM[command_symbol]
            param = th.tanh(variable) * mul + extra
        elif local_ind == 2:
            param = variable
        else:
            raise ValueError(f"Unsupported local index: {local_ind}")
    else:
        raise ValueError(f"Unsupported command symbol: {command_symbol}")
    return param


def convert_to_unbatched(program):
    if isinstance(program, (sps.SuperFrustumPackedBatchedStochasticSU,
                            sps.SolidSFPackedBatchedStochasticSU)):
        prim_params = program.get_arg(0)
        sm_ops = program.get_arg(1)
        logits = program.get_arg(2)
        n_prims = prim_params.shape[0]
        if issubclass(program.__class__, sps.SuperFrustumPackedBatchedStochasticSU):
            prim_class = sps.SuperFrustumPacked
        elif issubclass(program.__class__, sps.SolidSFPackedBatchedStochasticSU):
            prim_class = sps.SolidSFPacked
        else:
            raise ValueError(f"Unsupported primitive class: {program.__class__}")
        new_prims = []
        for i in range(n_prims):
            cur_prim = prim_params[i]
            new_prim = sps.PrimitiveMarker(sps.StochasticPrimitive(prim_class(cur_prim), logits[i]))
            new_prims.append(new_prim)
        out_program = generate_from_sm_ops_and_primitives(sm_ops, new_prims)
    elif isinstance(program, (sps.SuperFrustumPackedBatchedSU,
                              sps.SolidSFPackedBatchedSU)):
        prim_params = program.get_arg(0)
        sm_ops = program.get_arg(1)
        new_prims = []
        n_prims = prim_params.shape[0]
        if issubclass(program.__class__, sps.SuperFrustumPackedBatchedSU):
            prim_class = sps.SuperFrustumPacked
        elif issubclass(program.__class__, sps.SolidSFPackedBatchedSU):
            prim_class = sps.SolidSFPacked
        else:
            raise ValueError(f"Unsupported primitive class: {program.__class__}")
        for i in range(n_prims):
            cur_prim = prim_params[i]
            new_prim = sps.PrimitiveMarker(prim_class(cur_prim))
            new_prims.append(new_prim)
        out_program = generate_from_sm_ops_and_primitives(sm_ops, new_prims)
    else:
        raise ValueError(f"Unsupported command symbol: {program}")
    return out_program



def convert_to_batched(program):
    primitives = gather_primitives(program)
    sm_ops = gather_smooth_union_ops(program)
    if sm_ops:
        sm_ops = th.stack(sm_ops, dim=0)
    else:
        sm_ops = None
    if AlgConf.STOCHASTIC_DROPOUT: 
        logits = []
        prim_params = []
        prim_class = None
        for sub_expr in primitives:
            sub_expr = sub_expr.args[0]
            if isinstance(sub_expr, sps.StochasticPrimitive):
                stochastic_expr = sub_expr
                log_param = stochastic_expr.get_arg(1)
                logits.append(log_param)
                prim_expr = stochastic_expr.get_arg(0)
                prim_param = prim_expr.get_arg(0)
                prim_params.append(prim_param)
                prim_class = prim_expr.__class__
            elif isinstance(sub_expr, (sps.SuperFrustumPacked, 
                                       sps.SolidSFPacked,)):
                prim_expr = sub_expr
                prim_param = prim_expr.get_arg(0)
                prim_params.append(prim_param)
                logit_fake = th.Tensor(AlgConf.DEFAULT_LOGITS_RESTART_VALUES).to(prim_param.device)
                logits.append(logit_fake)
                prim_class = prim_expr.__class__
        logits = th.stack(logits, dim=0)
        prim_params = th.stack(prim_params, dim=0)
        if issubclass(prim_class, sps.SuperFrustumPacked):
            batched_class = sps.SuperFrustumPackedBatchedStochasticSU
        elif issubclass(prim_class, sps.SolidSFPacked):
            batched_class = sps.SolidSFPackedBatchedStochasticSU
        else:
            raise ValueError(f"Unsupported primitive class: {prim_class}")
        if sm_ops is not None:
            out_program = batched_class(prim_params, sm_ops, logits)
        else:
            # TODO: THIS IS MESSY AND WRONG.
            out_program = batched_class(prim_params, logits)
    else:
        prim_params = []
        prim_class = None
        for sub_expr in primitives:
            prim_expr = sub_expr.args[0]
            prim_param = prim_expr.get_arg(0)
            if prim_param in prim_expr.lookup_table:
                prim_param = prim_expr.lookup_table[prim_param]
            prim_params.append(prim_param)
            prim_class = prim_expr.__class__
        prim_params = th.stack(prim_params, dim=0)
        if issubclass(prim_class, sps.SuperFrustumPacked):
            batched_class = sps.SuperFrustumPackedBatchedSU
        elif issubclass(prim_class, sps.SolidSFPacked):
            batched_class = sps.SolidSFPackedBatchedSU
        else:
            raise ValueError(f"Unsupported primitive class: {prim_class}")
        if sm_ops is not None:
            out_program = batched_class(prim_params, sm_ops)
        else:
            out_program = batched_class(prim_params)
    return out_program