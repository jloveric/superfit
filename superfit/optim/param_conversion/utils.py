import torch as th
from typing import NamedTuple
import geolipi.symbolic as gls
import superfit.symbolic as sps
from ...utils.config import AlgorithmConfig as AlgConf

### Range
OPT_MIN_TRANSLATE: float = -0.9999
OPT_MAX_TRANSLATE: float = 0.9999
OPT_MIN_SCALE: float = 0.00001
OPT_MAX_SCALE: float = 1.9999

RANGE_DICT = {
    gls.SmoothUnion: (OPT_MIN_SCALE, OPT_MAX_SCALE / 4.0),
    gls.Translate3D: (-OPT_MAX_TRANSLATE, OPT_MAX_TRANSLATE),
    "sp_size":      (OPT_MIN_SCALE,   OPT_MAX_SCALE),
    "sp_roundness": (0.0,   OPT_MAX_SCALE / 2.0),
    "sp_dilate_3d": (0.0,   OPT_MAX_SCALE / 4.0),
    "sp_taper": (OPT_MIN_SCALE,   OPT_MAX_SCALE),
    "sp_bulge":     (-OPT_MAX_SCALE,   OPT_MAX_SCALE),
    "sp_onion_ratio": (0,   OPT_MAX_SCALE/2.0),
    "sp_logits": (-2.5, 2.5),
    "sp_extrussion": (0.0,   OPT_MAX_SCALE / 2.0),
    "sp_trapeze": (-OPT_MAX_TRANSLATE, OPT_MAX_TRANSLATE),
    "sp_taper_bulge": (-OPT_MAX_TRANSLATE, OPT_MAX_TRANSLATE),
    "sp_sq_size": (0.01, OPT_MAX_SCALE),
    "sp_sq_scale": (0.1, OPT_MAX_SCALE * 2.0),
}

# Small margins for stability (avoid exact ±1 and sqrt(0))
_EPS_OUT   = 1e-4
_SAFE_SQRT = 1e-12

# SmoothUnion tanh scale/offset for param_from_variables_fast
SU_MUL, SU_EXTRA = 0.25, 0.25


class TransformConstants(NamedTuple):
    n_transform: int
    pmin: th.Tensor
    pmax: th.Tensor
    scale: th.Tensor
    offset: th.Tensor


def build_transform_constants(sym_list):
    """Build precomputed constant vectors from a list of RANGE_DICT keys."""
    pmin = th.tensor([RANGE_DICT[s][0] for s in sym_list])
    pmax = th.tensor([RANGE_DICT[s][1] for s in sym_list])
    return TransformConstants(
        n_transform=len(sym_list),
        pmin=pmin,
        pmax=pmax,
        scale=0.5 * (pmax - pmin),
        offset=0.5 * (pmax + pmin),
    )


def make_packed_param_to_var(tc):
    """Factory: returns a batched param->var function with baked-in constants."""
    def packed_param_to_var(param):
        p_transformed = param[..., :tc.n_transform]
        p_rest        = param[..., tc.n_transform:]
        pmin   = tc.pmin.to(p_transformed)
        pmax   = tc.pmax.to(p_transformed)
        scale  = tc.scale.to(p_transformed)
        offset = tc.offset.to(p_transformed)
        p  = th.clamp(p_transformed, pmin, pmax)
        pu = (p - offset) / scale
        pu = pu / (1 - _EPS_OUT)
        pu = th.clamp(pu, -1 + 1e-6, 1 - 1e-6)
        v  = pu / th.sqrt(th.clamp(1 - pu * pu, min=_SAFE_SQRT))
        v_rest = p_rest.detach().clone()
        vcat   = th.cat([v, v_rest], dim=-1)
        return vcat.detach().requires_grad_(True)
    return packed_param_to_var


def make_packed_var_to_param(tc):
    """Factory: returns a batched var->param function with baked-in constants."""
    def packed_var_to_param(variable):
        v_transformed = variable[..., :tc.n_transform]
        v_rest        = variable[..., tc.n_transform:]
        scale  = tc.scale.to(v_transformed)
        offset = tc.offset.to(v_transformed)
        pmin   = tc.pmin.to(v_transformed)
        pmax   = tc.pmax.to(v_transformed)
        p_unit = (1 - _EPS_OUT) * (v_transformed / th.sqrt(1 + v_transformed * v_transformed))
        p      = p_unit * scale + offset
        p      = th.clamp(p, pmin, pmax)
        return th.cat([p, v_rest], dim=-1)
    return packed_var_to_param


def make_param_to_var_dispatcher(packed_p2v_fn):
    """Factory: returns a local_ind dispatcher for param->var."""
    def param_to_var(param, local_ind):
        if local_ind == 0:
            return packed_p2v_fn(param)
        elif local_ind == 1:
            return su_param_to_var(param)
        elif local_ind == 2:
            return th.autograd.Variable(param, requires_grad=True)
        else:
            raise ValueError(f"Unsupported local index: {local_ind}")
    return param_to_var


def make_var_to_param_dispatcher(packed_v2p_fn):
    """Factory: returns a local_ind dispatcher for var->param."""
    def var_to_param(variable, local_ind):
        if local_ind == 0:
            return packed_v2p_fn(variable)
        elif local_ind == 1:
            return su_var_to_param(variable)
        elif local_ind == 2:
            return variable
        else:
            raise ValueError(f"Unsupported local index: {local_ind}")
    return var_to_param


def make_param_from_variables_fast(packed_v2p_fn):
    """Factory: returns a branchless param_from_variables_fast."""
    def param_from_variables_fast(tensor_list):
        return [
            packed_v2p_fn(tensor_list[0]),
            th.tanh(tensor_list[1]) * SU_MUL + SU_EXTRA,
            tensor_list[2],
        ]
    return param_from_variables_fast


def process_param_to_var(sym_list, param_list):
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
    return v_list

def process_var_to_param(sym_list, v_parts):
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
    return p_list

def su_param_to_var(param):
    pmin, pmax = RANGE_DICT[gls.SmoothUnion]
    mul  = 0.5 * (pmax - pmin)
    extra = 0.5 * (pmax + pmin)
    param = th.clip(param, pmin, pmax)
    variable = th.atanh((param - extra) / mul)
    variable = th.autograd.Variable(variable, requires_grad=True)
    return variable

def su_var_to_param(variable):
    pmin, pmax = RANGE_DICT[gls.SmoothUnion]
    mul  = 0.5 * (pmax - pmin)
    extra = 0.5 * (pmax + pmin)
    param = th.tanh(variable) * mul + extra
    return param