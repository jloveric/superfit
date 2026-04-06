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
import torch.nn.functional as F
from .primitives import map_arc_bulge
import superfit.symbolic as sps
from geolipi.torch_compute.maps import PRIMITIVE_MAP
from geolipi.torch_compute.constants import EPSILON
# from geolipi.torch_compute.transforms import axis_angle_to_rotation_matrix
from .compile_friendly import axis_angle_to_rotation_matrix
from superfit.symbolic.utils import sample_gumbel
from .batched_sf import smooth_union_k_way, common_transform_coords
from .super_geon import sdf2d_trapezoid, MAX_INNER_BULGE

SQRT_EPS = 1e-12
def saturate(x: th.Tensor) -> th.Tensor:
    return x.clamp(0.0, 1.0)

def sd2_point_segment_batched(
    p: th.Tensor,      # (B,N,2)
    a: th.Tensor,      # (B,1,2)
    e: th.Tensor,      # (B,1,2)
    inv_e2: th.Tensor  # (B,1)
) -> th.Tensor:        # (B,N)
    pa = p - a
    t = saturate((pa * e).sum(dim=-1) * inv_e2)
    d = pa - e * t[..., None]
    return (d * d).sum(dim=-1)

def batched_supergeon_outer_transform(
    transformed_coords: th.Tensor,  # (B,M,3)
    size: th.Tensor,                # (B,3)
    bulge: th.Tensor,               # (B,) or (B,1)
    rot2d: th.Tensor,               # (B,) or (B,1)
    bulge_eps: float = 0.0,
) -> th.Tensor:
    """
    Applies:
      1) if |bulge|>eps: map_arc_bulge on xz
         else: x -> -x
      2) rotate xy by rot2d
    Returns:
      new_p: (B,M,3)
    """
    B, M, _ = transformed_coords.shape
    bulge = bulge.view(B, 1)   # (B,1)
    rot2d = rot2d.view(B, 1)   # (B,1)

    has_bulge = bulge.abs() > bulge_eps   # (B,1) bool

    # Curved path (reuse your existing map_arc_bulge implementation)
    # NOTE: pass size[...,2:3] so it broadcasts over M.
    new_xz = map_arc_bulge(
        transformed_coords[..., (0, 2)],  # (B,M,2)
        size[..., 2:3],                   # (B,1)
        bulge                             # (B,1)
    )  # (B,M,2)

    p_curved = th.stack(
        (new_xz[..., 0], transformed_coords[..., 1], new_xz[..., 1]), dim=-1
    )  # (B,M,3)

    # Straight path: x flip
    p_straight = th.stack(
        (-transformed_coords[..., 0], transformed_coords[..., 1], transformed_coords[..., 2]), dim=-1
    )  # (B,M,3)

    # Select path per primitive
    new_p = th.where(has_bulge[:, None, :], p_curved, p_straight)  # (B,M,3)

    # Rotate xy after bulging/x-flip (batched: B,2,2 matrix and bmm)
    c = th.cos(-rot2d).squeeze(-1)   # (B,)
    s = th.sin(-rot2d).squeeze(-1)   # (B,)
    R = th.stack((th.stack((c, -s), dim=-1), th.stack((s, c), dim=-1)), dim=1)  # (B, 2, 2)
    new_xy = th.bmm(new_p[..., :2], R)  # (B, M, 2) @ (B, 2, 2) -> (B, M, 2)
    new_p = th.stack((new_xy[..., 0], new_xy[..., 1], new_p[..., 2]), dim=-1)
    return new_p

def batched_sdf2d_trapezoid(
    transformed_coords: th.Tensor,  # (B,M,3)
    size: th.Tensor,                # (B,3)
    roundness: th.Tensor,           # (B,) or (B,1)
    trapeze: th.Tensor,             # (B,) or (B,1)
):
    """
    Returns:
      sdf2d: (B,M)
      z: (B,M)
      min_size: (B,)
      half_height: (B,)   # size_h[...,2]
      pxy_used: (B,M,2)   # after trapeze-dependent swap
    """
    B, M, _ = transformed_coords.shape
    roundness = roundness.view(B)
    trapeze = trapeze.view(B)

    p = transformed_coords
    size_h = size * 0.5  # (B,3)

    # Swap xy when trapeze >= 0 (matches your prior logic)
    sw = (trapeze >= 0.0)[:, None, None]  # (B,1,1)
    pxy = th.where(sw, p[..., [1, 0]], p[..., [0, 1]])  # (B,M,2)

    sxy = size_h[..., :2]                                   # (B,2)
    sxy_sw = th.where((trapeze >= 0.0)[:, None], sxy[:, [1, 0]], sxy[:, [0, 1]])  # (B,2)

    min_size = th.minimum(size_h[..., 0], size_h[..., 1])  # (B,)
    trap_amount = 1.0 - trapeze.abs()                       # (B,)

    r = roundness * min_size                                # (B,)
    sxy_r = sxy_sw - r[:, None]                             # (B,2)

    # Trapezoid2D(pxy, a, b, h) then subtract roundness
    sdf2d = sdf2d_trapezoid(
        pxy,
        sxy_r[:, 0:1],                         # a: (B,1)
        (sxy_r[:, 0] * trap_amount)[:, None], # b: (B,1)
        sxy_r[:, 1:2],                         # h: (B,1)
    ) - r[:, None]                             # (B,M)

    z = p[..., 2]                               # (B,M)
    half_height = size_h[..., 2]                # (B,)
    return sdf2d, z, min_size, half_height, pxy


def sd_bulged_right_edge_batched(
    p: th.Tensor,       # (B,N,2)
    A: th.Tensor,       # (B,2)
    Bp: th.Tensor,      # (B,2)
    bulge: th.Tensor,   # (B,) or (B,1)
    bulge_eps: float = 1e-6,
) -> th.Tensor:
    """
    Returns (B,N,2): [distance, c_like]
      - distance: shortest distance to bulged edge
      - c_like: inside-test proxy (<=0 means inside for right edge convention)
    """
    B, N, _ = p.shape
    bulge = bulge.view(B, 1)  # (B,1)

    e = (Bp - A)                            # (B,2)
    e2 = (e * e).sum(dim=-1, keepdim=True)  # (B,1)
    deg = e2 < SQRT_EPS

    L = th.sqrt(th.clamp_min(e2, SQRT_EPS))   # (B,1)
    invL = 1.0 / L

    t = e * invL                            # (B,2)
    nIn = th.stack((e[:, 1], -e[:, 0]), dim=-1) * invL  # (B,2)

    q = p - A[:, None, :]                   # (B,N,2)
    u = (q * nIn[:, None, :]).sum(dim=-1)   # (B,N)
    v = (q * t[:, None, :]).sum(dim=-1)     # (B,N)

    local = th.stack((u, v - 0.5 * L), dim=-1)  # (B,N,2)

    # straight edge path
    zc_st = th.minimum(th.maximum(local[..., 1], -0.5 * L), 0.5 * L)
    dvec_st = th.stack((local[..., 0], local[..., 1] - zc_st), dim=-1)
    d_st = th.linalg.vector_norm(dvec_st, dim=-1)
    c_st = local[..., 0]

    # curved path via map_arc_bulge
    m = map_arc_bulge(local, L, bulge)  # (B,N,2)
    zc_cv = th.minimum(th.maximum(m[..., 1], -0.5 * L), 0.5 * L)
    dvec_cv = th.stack((m[..., 0], m[..., 1] - zc_cv), dim=-1)
    d_cv = th.linalg.vector_norm(dvec_cv, dim=-1)
    c_cv = -m[..., 0]

    straight = bulge.abs() < bulge_eps  # (B,1)
    d = th.where(straight, d_st, d_cv)
    c_like = th.where(straight, c_st, c_cv)

    # degenerate override
    d_deg = th.linalg.vector_norm(p - A[:, None, :], dim=-1)
    c_deg = th.ones_like(d_deg)

    d = th.where(deg, d_deg, d)
    c_like = th.where(deg, c_deg, c_like)

    return th.stack((d, c_like), dim=-1)
def sd_taper_trapezoid_onion_bulged_batched(
    pos_2d: th.Tensor,     # (B,N,2) ; p=(sdf2d,z)
    inner: th.Tensor,      # (B,) or (B,1)
    half_height: th.Tensor,# (B,) or (B,1)
    x3: th.Tensor,         # (B,) or (B,1)
    onion_ratio: th.Tensor,# (B,) or (B,1)
    bulge: th.Tensor,      # (B,) or (B,1)
) -> th.Tensor:            # (B,N)
    B, N, _ = pos_2d.shape
    inner = inner.view(B, 1)
    half_height = half_height.view(B, 1)
    x3 = x3.view(B, 1)
    onion_ratio = onion_ratio.view(B, 1)
    bulge = bulge.view(B, 1)

    xL  = -inner * (1.0 - onion_ratio)
    xTL = -inner + (x3 + inner) * onion_ratio
    yB  = -half_height
    yT  = half_height

    # Key points/edges
    Lb = th.cat((xL,  yB), dim=1)   # (B,2)
    Lt = th.cat((xTL, yT), dim=1)   # (B,2)
    eL = Lt - Lb                     # (B,2)

    # ---- distances to 3 straight boundaries ----
    d2_left = sd2_point_segment_batched(
        pos_2d, Lb[:, None, :], eL[:, None, :],
        (1.0 / th.clamp_min((eL * eL).sum(dim=-1, keepdim=True), SQRT_EPS))
    )  # (B,N)

    px = pos_2d[..., 0]
    py = pos_2d[..., 1]

    xb = th.minimum(th.maximum(px, xL), th.zeros_like(xL))
    d2_bottom = (px - xb).square() + (py - yB).square()

    xt = th.minimum(th.maximum(px, xTL), x3)
    d2_top = (px - xt).square() + (py - yT).square()

    # ---- right bulged boundary ----
    Rb = th.cat((th.zeros_like(yB), yB), dim=1)  # (B,2)
    Rt = th.cat((x3, yT), dim=1)                 # (B,2)
    right_res = sd_bulged_right_edge_batched(pos_2d, Rb, Rt, bulge)  # (B,N,2)
    d2_right = right_res[..., 0].square()

    d2 = th.minimum(th.minimum(d2_left, d2_bottom), th.minimum(d2_top, d2_right))

    # ---- inside test ----
    # Left half-plane (same convention as your GLSL)
    cL = eL[:, 1:2] * (px - Lb[:, 0:1]) - eL[:, 0:1] * (py - Lb[:, 1:2])
    cL = -cL

    cB = yB - py
    cT = py - yT
    cS = right_res[..., 1]

    m = th.maximum(th.maximum(cL, cB), th.maximum(cT, cS))
    d = th.sqrt(th.clamp_min(d2, SQRT_EPS))
    return th.where(m <= 0.0, -d, d)

def batched_super_geon_eval(
    transformed_coords: th.Tensor,  # (B,M,3)
    size: th.Tensor,                # (B,3)
    roundness: th.Tensor,           # (B,1) or (B,)
    dilate_3d: th.Tensor,           # (B,1) or (B,)
    taper: th.Tensor,               # (B,1) or (B,)
    bulge: th.Tensor,               # (B,1) or (B,)       # outer bulge (pre-warp)
    onion_ratio: th.Tensor,         # (B,1) or (B,)
    trapeze: th.Tensor,             # (B,1) or (B,)
    taper_bulge: th.Tensor,         # (B,1) or (B,)       # inner bulge controller
    rot2d: th.Tensor,               # (B,1) or (B,)
    bulge_eps: float = 0.0,
) -> th.Tensor:                     # (B,M)
    """
    Batched SuperGeon:
      1) outer transform: bulge warp on xz OR x-flip, then xy rotation
      2) 2D trapezoid sdf in xy (with roundness + trapeze swap)
      3) tapered-onion-bulged trapezoid sdf in (sdf2d, z)
      4) subtract dilate_3d

    Requires these helpers to be defined:
      - batched_supergeon_outer_transform
      - batched_sdf2d_trapezoid
      - sd_taper_trapezoid_onion_bulged_batched
    """
    # ---------- shape normalization ----------
    B, M, _ = transformed_coords.shape
    # roundness   = roundness.view(B, 1)
    # dilate_3d   = dilate_3d.view(B, 1)
    # taper       = taper.view(B, 1)
    # bulge       = bulge.view(B, 1)
    # onion_ratio = onion_ratio.view(B, 1)
    # trapeze     = trapeze.view(B, 1)
    # taper_bulge = taper_bulge.view(B, 1)
    # rot2d       = rot2d.view(B, 1)

    # ---------- 1) outer transform ----------
    p = batched_supergeon_outer_transform(
        transformed_coords=transformed_coords,  # (B,M,3)
        size=size,                              # (B,3)
        bulge=bulge,                            # (B,1)
        rot2d=rot2d,                            # (B,1)
        bulge_eps=bulge_eps,
    )  # (B,M,3)

    # ---------- 2) sdf2d in xy via trapezoid ----------
    sdf2d, z, min_size, half_height, _ = batched_sdf2d_trapezoid(
        transformed_coords=p,    # (B,M,3)
        size=size,               # (B,3)
        roundness=roundness,     # (B,1)
        trapeze=trapeze,         # (B,1)
    )
    # sdf2d: (B,M), z: (B,M), min_size:(B,), half_height:(B,)

    # ---------- 3) inner taper + bulge params ----------
    # x3 = -(1 - taper) * min_size
    x3 = -(1.0 - taper.squeeze(-1)) * min_size  # (B,)

    # inner bulge (from taper_bulge) following your prior logic
    inv_h = 1.0 / th.clamp_min(half_height, bulge_eps)  # (B,)
    max_bulge = (
        min_size * (1.0 - onion_ratio.squeeze(-1)) * inv_h
        * th.minimum(th.ones_like(taper.squeeze(-1)), taper.squeeze(-1))
    )  # (B,)
    max_bulge = th.minimum(
        th.ones_like(taper.squeeze(-1)) * MAX_INNER_BULGE, max_bulge
    )
    min_bulge = th.full_like(max_bulge, 0.5)
    bulge_scale = th.where(taper_bulge.squeeze(-1) > 0.0, max_bulge, min_bulge)
    bulge_inner = taper_bulge.squeeze(-1).clamp(-1.0, 1.0) * bulge_scale  # (B,)

    # (sdf2d, z)-space points
    pos_2d = th.stack((sdf2d, z), dim=-1)  # (B,M,2)

    sd = sd_taper_trapezoid_onion_bulged_batched(
        pos_2d=pos_2d,                         # (B,M,2)
        inner=min_size,                        # (B,)
        half_height=half_height,               # (B,)
        x3=x3,                                 # (B,)
        onion_ratio=onion_ratio.squeeze(-1),   # (B,)
        bulge=bulge_inner,                     # (B,)
    )  # (B,M)

    # ---------- 4) final dilate ----------
    return sd - dilate_3d  # (B,M)

def unpacked_params_sg(params):
    translate = params[...,  :3]
    size = params[..., 3:6]
    roundness = params[..., 6:7]
    dilate_3d = params[..., 7:8]
    taper = params[..., 8:9]
    bulge = params[..., 9:10]
    onion_ratio = params[..., 10:11]
    trapeze = params[..., 11:12]
    taper_bulge = params[..., 12:13]
    rot2d = params[..., 13:14]
    rotate = params[..., 14:17]
    return translate, size, roundness, dilate_3d, taper, bulge, onion_ratio, trapeze, taper_bulge, rot2d, rotate

def unpacked_params_varsg(params):
    translate = params[...,  :3]
    size = params[..., 3:6]
    roundness = params[..., 6:7]
    dilate_3d = params[..., 7:8]
    taper = params[..., 8:9]
    bulge = params[..., 9:10]
    onion_ratio = params[..., 10:11]
    trapeze = params[..., 11:12]
    taper_bulge = params[..., 12:13]
    rot2d = params[..., 13:14]
    logits = params[..., 14:17]
    rotate = params[..., 17:20]
    return translate, size, roundness, dilate_3d, taper, bulge, onion_ratio, trapeze, taper_bulge, rot2d, logits, rotate

def batched_sg_packed_eval(coords, params):
    """
    coords: (B, M, 3)
    params: (B, 13+) -> [tx,ty,tz, rx,ry,rz, sx,sy,sz, round, dilate, scale, bulge]
    returns: (B, M)
    """
    # -------- unpack once, keep shapes broadcast-friendly
    unpacked_params = unpacked_params_sg(params)
    translate, size, roundness, dilate_3d, taper, bulge, onion_ratio, trapeze, taper_bulge, rot2d, rotate = unpacked_params
    transformed_coords = common_transform_coords(coords, translate, rotate)
    sdf_eval = batched_super_geon_eval(transformed_coords, size, roundness, dilate_3d, taper, bulge, onion_ratio, trapeze, taper_bulge, rot2d)
    return sdf_eval
    
def batched_varaxis_sg_packed_eval(coords: th.Tensor,
                                              params: th.Tensor,
                                              temperature: float):
    """
    coords: (B, M, 3)
    params: (B, K) where last 3 entries are logits for {xyz, yzx, zxy}
            and size is at params[..., 6:9] in the base part.
    temperature: float

    Forward: hard per-batch axis permutation (xyz / yzx / zxy)
    Backward: straight-through (grad flows as if soft mixture)
    """
    unpacked_params = unpacked_params_varsg(params)
    translate, size, roundness, dilate_3d, taper, bulge, onion_ratio, trapeze, taper_bulge, rot2d, logits, rotate = unpacked_params
    
    transformed_coords = common_transform_coords(coords, translate, rotate)
    sdf_y = batched_super_geon_eval(transformed_coords, size, roundness, dilate_3d, taper, bulge, onion_ratio, trapeze, taper_bulge, rot2d)
    new_coords = transformed_coords.clone()[:, :, [1, 2, 0]]
    new_size = size.clone()[:, [1, 2, 0]]
    sdf_z = batched_super_geon_eval(new_coords, new_size, roundness, dilate_3d, taper, bulge, onion_ratio, trapeze, taper_bulge, rot2d)
    new_coords = transformed_coords.clone()[:, :, [2, 0, 1]]
    new_size = size.clone()[:, [2, 0, 1]]
    sdf_x = batched_super_geon_eval(new_coords, new_size, roundness, dilate_3d, taper, bulge, onion_ratio, trapeze, taper_bulge, rot2d)
    # IDEA - as the SDF field becomes more disparate across the axes, we should use a sharper distribution..
    g  = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    # div_t = deviation_based_temperature(roundness, dilate_3d, scale, bulge_ratio, onion_ratio)
    # w  = th.softmax((logits/div_t + g) / temperature, dim=-1)  # (..., 2)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)
    wy = w[:, 0:1]
    wz = w[:, 1:2]
    wx = w[:, 2:3]
    out = wy * sdf_y + wz * sdf_z + wx * sdf_x
    return out

def batched_sg_packed_su_eval(coords, params, su_vals):
    output = batched_sg_packed_eval(coords, params)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)

def batched_sg_packed_stochastic_eval(coords, params, logits, temperature):
    # B, N
    outputs = batched_sg_packed_eval(coords, params)
    g = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)

    # Unpack weights explicitly
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]

    # Equivalent to weighted sum of [outputs, 1.0]
    out = outputs * w0 + w1
    return out

def batched_sg_packed_stochastic_su_eval(coords, params, su_vals, logits, temperature):
    output = batched_sg_packed_stochastic_eval(coords, params, logits, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)

def batched_varaxis_sg_packed_stochastic_eval(coords, params, logits, temperature):
    # B, N
    outputs = batched_varaxis_sg_packed_eval(coords, params, temperature)
    g  = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]
    out = outputs * w0 + w1
    return out

def batched_varaxis_sg_packed_su_eval(coords, params, su_vals, temperature):
    output = batched_varaxis_sg_packed_eval(coords, params, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)


def batched_varaxis_sg_packed_stochastic_su_eval(coords, params, su_vals, logits, temperature):
    output = batched_varaxis_sg_packed_stochastic_eval(coords, params, logits, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)

function_map = {
    sps.SGPackedBatched: batched_sg_packed_eval,
    sps.SGPackedBatchedStochastic: batched_sg_packed_stochastic_eval,
    sps.SGPackedBatchedSU: batched_sg_packed_su_eval,
    sps.SGPackedBatchedStochasticSU: batched_sg_packed_stochastic_su_eval,
    # VarAxisSG batched variants
    sps.VarAxisSGPackedBatched: batched_varaxis_sg_packed_eval,
    sps.VarAxisSGPackedBatchedStochastic: batched_varaxis_sg_packed_stochastic_eval,
    sps.VarAxisSGPackedBatchedSU: batched_varaxis_sg_packed_su_eval,
    sps.VarAxisSGPackedBatchedStochasticSU: batched_varaxis_sg_packed_stochastic_su_eval,
}
PRIMITIVE_MAP.update(function_map)

