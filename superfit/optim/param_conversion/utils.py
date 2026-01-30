import torch as th
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
    "sp_size":      (0.01,   1.98),
    "sp_roundness": (OPT_MIN_SCALE,   OPT_MAX_SCALE / 2.0),
    "sp_dilate_3d": (0.0,   OPT_MAX_SCALE / 4.0),
    "sp_scale_opp": (0.01,   1.995),
    "sp_bulge":     (OPT_MIN_SCALE,   0.995),
    "sp_onion_ratio": (OPT_MIN_SCALE,   0.975),
    "sp_sq_scale": (OPT_MIN_SCALE, OPT_MAX_SCALE/2.0),
    "sp_logits": (-2.5, 2.5),
}

# Small margins for stability (avoid exact ±1 and sqrt(0))
_EPS_OUT   = 1e-4
_SAFE_SQRT = 1e-12


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