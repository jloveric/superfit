import torch as th
import torch.nn.functional as F
from geolipi.torch_compute.constants import EPSILON
from geolipi.torch_compute.maps import PRIMITIVE_MAP
import geolipi.torch_compute.transforms as transform_bank
import superfit.symbolic as sps
from geolipi.torch_compute.sdf_functions_2d import sdf2d_trapezoid
from geolipi.torch_compute.transforms import get_affine_rotate_2D
from .primitives import map_arc_bulge, varaxis_P
# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
IDENTITY_MAT_2D = th.eye(2, dtype=th.float32)
MAX_INNER_BULGE = 0.9

def saturate(x: th.Tensor) -> th.Tensor:
    return x.clamp(0.0, 1.0)


def sd2_point_segment(p: th.Tensor, a: th.Tensor, e: th.Tensor, inv_e2: th.Tensor) -> th.Tensor:
    """
    p: (...,2), a: (...,2) or (2,), e: (...,2) or (2,), inv_e2: (...) or scalar
    returns d2: (...)
    """
    pa = p - a
    t = saturate((pa * e).sum(dim=-1) * inv_e2)
    d = pa - e * t[..., None]
    return (d * d).sum(dim=-1)


# ---------------------------------------------------------
# 1) sdBulgedRightEdgeFast (Torch)
# ---------------------------------------------------------

def sd_bulged_right_edge_fast(
    p: th.Tensor,          # (...,2)
    A: th.Tensor,          # (...,2) or (2,)
    B: th.Tensor,          # (...,2) or (2,)
    bulge: th.Tensor,      # (...) or scalar
    bulge_eps: float = 1e-6,
) -> th.Tensor:
    """
    Torch port of GLSL sdBulgedRightEdgeFast.
    Returns (...,2): [distance, c_like]
    """
    e = B - A
    e2 = (e * e).sum(dim=-1)

    # Degenerate fallback
    deg = e2 < 1e-16

    # Safe L/invL for all lanes
    L = th.sqrt(th.clamp_min(e2, 1e-16))
    invL = 1.0 / L

    t = e * invL[..., None]
    nIn = th.stack((e[..., 1], -e[..., 0]), dim=-1) * invL[..., None]

    q = p - A
    u = (q * nIn).sum(dim=-1)
    v = (q * t).sum(dim=-1)

    local = th.stack((u, v - 0.5 * L), dim=-1)

    # Straight-edge fast path
    straight = bulge.abs() < bulge_eps
    zc_st = local[..., 1].clamp(min=-0.5 * L, max=0.5 * L)
    dvec_st = th.stack((local[..., 0], local[..., 1] - zc_st), dim=-1)
    d_st = th.linalg.norm(dvec_st, dim=-1)
    c_st = local[..., 0]

    # Curved path
    # map_arc_bulge expects (...,2), z scalar/tensor broadcastable, bulge broadcastable
    m = map_arc_bulge(local, L, bulge)  # (...,2)
    zc_cv = m[..., 1].clamp(min=-0.5 * L, max=0.5 * L)
    dvec_cv = th.stack((m[..., 0], m[..., 1] - zc_cv), dim=-1)
    d_cv = th.linalg.norm(dvec_cv, dim=-1)
    c_cv = -m[..., 0]

    d = th.where(straight, d_st, d_cv)
    c_like = th.where(straight, c_st, c_cv)

    # Degenerate override
    d_deg = th.linalg.norm(p - A, dim=-1)
    c_deg = th.ones_like(d_deg)

    d = th.where(deg, d_deg, d)
    c_like = th.where(deg, c_deg, c_like)

    return th.stack((d, c_like), dim=-1)


# ---------------------------------------------------------
# 2) sdTaperTrapezoidOnionBulgeFast (Torch)
# ---------------------------------------------------------

def sd_taper_trapezoid_onion_bulge_fast(
    p: th.Tensor,           # (...,2)
    inner: th.Tensor,       # (...) or scalar
    h: th.Tensor,           # (...) or scalar
    x3: th.Tensor,          # (...) or scalar
    onion_ratio: th.Tensor, # (...) or scalar
    bulge: th.Tensor,       # (...) or scalar
) -> th.Tensor:
    """
    Torch port of GLSL sdTaperTrapezoidOnionBulgeFast.
    Returns signed distance (...), negative inside.
    """
    one_minus_onion = 1.0 - onion_ratio
    xL = -inner * one_minus_onion
    xTL = -inner + (x3 + inner) * onion_ratio
    yB = -h
    yT = h

    Lb = th.stack((xL, yB), dim=-1)
    Lt = th.stack((xTL, yT), dim=-1)
    eL = Lt - Lb

    e2L = (eL * eL).sum(dim=-1)
    inv_e2L = 1.0 / th.clamp_min(e2L, 1e-12)

    # Left segment d2
    d2_left = sd2_point_segment(p, Lb, eL, inv_e2L)

    # Bottom segment [xL, 0] at yB
    xb = p[..., 0].clamp(min=xL, max=th.zeros_like(xL))
    dyb = p[..., 1] - yB
    d2_bottom = (p[..., 0] - xb).square() + dyb.square()

    # Top segment [xTL, x3] at yT
    xt = p[..., 0].clamp(min=xTL, max=x3)
    dyt = p[..., 1] - yT
    d2_top = (p[..., 0] - xt).square() + dyt.square()

    # Right bulged edge
    A = th.stack((th.zeros_like(yB), yB), dim=-1)
    B = th.stack((x3, yT), dim=-1)
    right_res = sd_bulged_right_edge_fast(p, A, B, bulge)   # (...,2)
    d2_right = right_res[..., 0].square()

    d2 = th.minimum(th.minimum(d2_left, d2_bottom), th.minimum(d2_top, d2_right))

    # Inside test
    cL = eL[..., 1] * (p[..., 0] - Lb[..., 0]) - eL[..., 0] * (p[..., 1] - Lb[..., 1])
    cL = -cL

    cB = yB - p[..., 1]
    cT = p[..., 1] - yT
    cS = right_res[..., 1]

    m = th.maximum(th.maximum(cL, cB), th.maximum(cT, cS))
    d = th.sqrt(th.clamp_min(d2, 0.0))
    return th.where(m <= 0.0, -d, d)


# ---------------------------------------------------------
# 3) InnerSuperGeonFast (Torch)
# ---------------------------------------------------------

def inner_supergeon_fast_eval(
    p: th.Tensor,             # (...,3)
    size: th.Tensor,          # (...,3) or (3,)
    roundness: th.Tensor,     # (...) or scalar
    onion_ratio: th.Tensor,   # (...) or scalar
    trapeze: th.Tensor,   # (...) or scalar
    taper: th.Tensor,         # (...,1) or (1,)
    taper_bulge: th.Tensor,   # (...,1) or (1,)
    dilate_3d: th.Tensor,     # (...) or scalar
) -> th.Tensor:
    """
    Torch port of GLSL InnerSuperGeonFast.
    """
    size_h = size * 0.5
    taper = taper.squeeze(-1)
    taper_bulge = taper_bulge.squeeze(-1)

    # Swap p.xy and size.xy when trapezoider >= 0
    sw = (trapeze >= 0.0)[..., None]  # (...,1) bool
    pxy = th.where(sw, p[..., [1, 0]], p[..., [0, 1]])
    sxy = th.where(sw, size_h[..., [1, 0]], size_h[..., [0, 1]])

    min_size = th.minimum(size_h[..., 0], size_h[..., 1])
    trap_amount = 1.0 - trapeze.abs()

    r = roundness * min_size
    sxy_r = sxy - r[..., None]

    # Trapezoid2D then roundness subtraction
    # trapezoid2d(pxy, a, b, h) corresponds to Trapezoid2D(p.xy, sxy_r.x, sxy_r.x*trap_amount, sxy_r.y)
    sdf2d = sdf2d_trapezoid(
        pxy,
        sxy_r[..., 0],
        sxy_r[..., 0] * trap_amount,
        sxy_r[..., 1]
    ) - r

    pos_2d = th.stack((sdf2d, p[..., 2]), dim=-1)

    half_height = size_h[..., 2]
    x3 = -(1.0 - taper) * min_size

    inv_h = 1.0 / th.clamp_min(half_height, 1e-8)
    max_bulge = min_size * (1.0 - onion_ratio) * inv_h * th.minimum(
        th.ones_like(taper), taper
    )
    max_bulge = th.minimum(
        th.ones_like(taper) * MAX_INNER_BULGE, max_bulge
    )
    min_bulge = th.full_like(max_bulge, 0.5)

    bulge_scale = th.where(taper_bulge > 0.0, max_bulge, min_bulge)
    bulge = taper_bulge.clamp(-1.0, 1.0) * bulge_scale

    sd = sd_taper_trapezoid_onion_bulge_fast(
        pos_2d, min_size, half_height, x3, onion_ratio.squeeze(-1), bulge.squeeze(-1)
    )

    return sd - dilate_3d

def euler_rotate_2d(p: th.Tensor, angle: th.Tensor) -> th.Tensor:
    matrix = get_affine_rotate_2D(IDENTITY_MAT_2D.clone().to(p.device), angle)
    out_p = th.matmul(p, matrix)
    return out_p
# ---------------------------------------------------------
# Full SuperGeon eval (Torch)
# ---------------------------------------------------------

def supergeon_eval(
    coords: th.Tensor,        # (...,3)
    size: th.Tensor,          # (...,3) or (3,)
    roundness: th.Tensor,     # (...) or scalar
    dilate_3d: th.Tensor,     # (...) or scalar
    taper: th.Tensor,         # (...,1) or (1,)
    bulge: th.Tensor,         # (...) or scalar
    onion_ratio: th.Tensor,   # (...) or scalar
    trapeze: th.Tensor,       # (...) or scalar   (trapezoider)
    taper_bulge: th.Tensor,   # (...,1) or (1,)
    rot2d: th.Tensor,        # (...) or scalar
    bulge_eps: float = 0.0,
) -> th.Tensor:
    """
    Torch port of GLSL SuperGeon:
      if bulge != 0: mapArcBulge(p.xz, size.z, bulge), else x = -x
      rotate xy
      InnerSuperGeonFast(...)
    """
    # Bulge mapping branch
    has_bulge = (bulge.abs() > bulge_eps)

    # Curved path: map xz
    new_xz = map_arc_bulge(coords[..., [0, 2]], size[..., 2], bulge)
    p_curved = th.stack((new_xz[..., 0], coords[..., 1], new_xz[..., 1]), dim=-1)

    # Straight path: flip x
    p_straight = th.stack((-coords[..., 0], coords[..., 1], coords[..., 2]), dim=-1)

    new_p = th.where(has_bulge[..., None], p_curved, p_straight)

    # Rotate xy
    new_xy = euler_rotate_2d(new_p[..., :2], rot2d)
    new_p = th.stack((new_xy[..., 0], new_xy[..., 1], new_p[..., 2]), dim=-1)

    # Inner eval
    return inner_supergeon_fast_eval(
        new_p, size, roundness, onion_ratio, trapeze, taper, taper_bulge, dilate_3d
    )

def varaxis_sg_eval(
    coords: th.Tensor,        # (...,3)
    size: th.Tensor,          # (...,3) or (3,)
    roundness: th.Tensor,     # (...) or scalar
    dilate_3d: th.Tensor,     # (...) or scalar
    taper: th.Tensor,         # (...,1) or (1,)
    bulge: th.Tensor,         # (...) or scalar
    onion_ratio: th.Tensor,   # (...) or scalar
    trapeze: th.Tensor,       # (...) or scalar   (trapezoider)
    taper_bulge: th.Tensor,   # (...,1) or (1,)
    rot2d: th.Tensor,        # (...) or scalar
    logits: th.Tensor,        # (...) or scalar
    temperature: float = 1.0,
    bulge_eps: float = 0.0,
) -> th.Tensor:
    """
    Torch port of GLSL VarAxisSG:
    """

    # --- ST gumbel-softmax: one-hot forward, soft backward
    P = varaxis_P(coords, logits, temperature)
    # --- apply permutation to coords and size
    coords_p = coords @ P  # (...,3)
    size_p = size @ P      # (...,3) typically (3,)
    return supergeon_eval(coords, size, roundness, dilate_3d, taper, bulge, onion_ratio, trapeze, taper_bulge, rot2d, bulge_eps)  

def sg_y_eval(coords: th.Tensor, size: th.Tensor, *args, **kwargs) -> th.Tensor:
    return supergeon_eval(coords, size, *args, **kwargs)

def sg_z_eval(coords: th.Tensor, size: th.Tensor, *args, **kwargs) -> th.Tensor:
    return supergeon_eval(coords[..., [1, 2, 0]], size[..., [1, 2, 0]], *args, **kwargs)

def sg_x_eval(coords: th.Tensor, size: th.Tensor, *args, **kwargs) -> th.Tensor:
    return supergeon_eval(coords[..., [2, 0, 1]], size[..., [2, 0, 1]], *args, **kwargs)

function_map = {
    sps.SuperGeon: supergeon_eval,
    sps.VarAxisSG: varaxis_sg_eval,
    sps.SuperGeonY: sg_y_eval,
    sps.SuperGeonZ: sg_z_eval,
    sps.SuperGeonX: sg_x_eval,
}
PRIMITIVE_MAP.update(function_map)

