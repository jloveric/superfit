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
from .compile_friendly import axis_angle_to_rotation_matrix, _sdf_smooth_union_pair
from superfit.symbolic.utils import sample_gumbel

def smooth_union_k_way(output, su_vals):
    K = output.shape[0]
    out = output[0]
    for i in range(1, K):
        k_reshaped = su_vals[i-1].unsqueeze(-1)
        out = _sdf_smooth_union_pair(out, output[i], k_reshaped)
    return out

def common_transform_coords(coords, translate, rotate):
    R = axis_angle_to_rotation_matrix(rotate)                  # (B,3,3)
    p_local = coords - translate.unsqueeze(1)                  # (B,M,3)
    transformed_coords = th.matmul(p_local, R.transpose(-1, -2))  # (B,M,3)
    return transformed_coords

def unpacked_params_sf(params):

    translate   = params[...,  :3]      # (B,3)
    size        = params[...,  3:6]     # (B,3)
    roundness   = params[...,  6:7]    # (B,1)
    dilate_3d   = params[..., 7:8]    # (B,1)
    scale       = params[..., 8:9]    # (B,1)
    bulge_ratio = params[..., 9:10]    # (B,1)
    onion_ratio = params[..., 10:11]    # (B,1)
    rotate      = params[...,  11:14]     # (B,3) axis-angle

    return translate, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, rotate

def unpacked_params_varsf(params):

    translate   = params[...,  :3]      # (B,3)
    size        = params[...,  3:6]     # (B,3)
    roundness   = params[...,  6:7]    # (B,1)
    dilate_3d   = params[..., 7:8]    # (B,1)
    scale       = params[..., 8:9]    # (B,1)
    bulge_ratio = params[..., 9:10]    # (B,1)
    onion_ratio = params[..., 10:11]    # (B,1)
    logits      = params[..., 11:14]    # (B,3)
    rotate      = params[...,  14:17]     # (B,3) axis-angle

    return translate, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, logits, rotate

def sd_taper_trapezoid_onion_exact_batched(pos_2d: th.Tensor,
                                     inner: th.Tensor,         # (B,) or (B,1)
                                     half_height: th.Tensor,   # (B,) or (B,1)
                                     x3: th.Tensor,             # (B,) or (B,1)
                                     onion_ratio: th.Tensor     # (B,) or (B,1)
                                     ) -> th.Tensor:           # (B,N) output
    """
    Batched signed distance to convex taper trapezoid.

    Inputs:
      pos_2d: (B,N,2) — input points
      inner: (B,) or (B,1)
      half_height: (B,) or (B,1)
      x3: (B,) or (B,1)

    Output:
      Signed distance: (B,N)
    """
    B, N, _ = pos_2d.shape
    inner = inner.view(B, 1)
    half_height = half_height.view(B, 1)
    x3 = x3.view(B, 1)
    onion_ratio = onion_ratio.view(B, 1)
    zero = th.zeros_like(inner)

    # Vertices A: shape (B,4,2)
    A = th.stack((
        th.cat([-inner + (x3 + inner) * onion_ratio, +half_height], dim=1),  # p0
        th.cat([-inner * (1 - onion_ratio), -half_height], dim=1),  # p1
        th.cat([zero,  -half_height], dim=1),   # p2
        th.cat([x3,    +half_height], dim=1),   # p3
    ), dim=1)  # (B,4,2)

    # Edges A->B
    Bv = A.roll(shifts=-1, dims=1)       # (B,4,2)
    E = Bv - A                            # (B,4,2)

    # Expand inputs for pairwise segment distances
    P  = pos_2d.unsqueeze(2)             # (B,N,1,2)
    A_ = A.unsqueeze(1)                  # (B,1,4,2)
    E_ = E.unsqueeze(1)                  # (B,1,4,2)
    PA = P - A_                          # (B,N,4,2)

    denom = (E_ * E_).sum(dim=-1).clamp_min(1e-18)  # (B,1,4)
    t = ((PA * E_).sum(dim=-1) / denom).clamp(0.0, 1.0)  # (B,N,4)
    closest = A_ + t.unsqueeze(-1) * E_                  # (B,N,4,2)
    dists = (P - closest).norm(dim=-1)                   # (B,N,4)
    dmin = dists.min(dim=-1).values                      # (B,N)

    # Inside test: left-of-all-edges (CCW polygon)
    cross = E_[..., 0] * PA[..., 1] - E_[..., 1] * PA[..., 0]  # (B,N,4)
    inside = (cross >= 0).all(dim=-1)                          # (B,N)

    return th.where(inside, -dmin, dmin)  # (B,N)



def batched_sf_packed_eval_part_2(transformed_coords, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio):

    # -------- xz bulge map (avoid extra stacks until needed)
    # Use your faster map_arc_bulge that returns (...,2)
    new_p_xz = map_arc_bulge(
        transformed_coords[..., (0, 2)],   # (B,M,2): take x,z
        size[..., 2:3],                    # (B,1) -> broadcast
        bulge_ratio                        # (B,1)
    )
    # Replace x,z with mapped values
    transformed_coords = th.stack(
        (new_p_xz[..., 0], transformed_coords[..., 1], new_p_xz[..., 1]), dim=-1
    )  # (B,M,3)

    # -------- rounded-rectangle SDF in xy (no tiny temporaries)
    xy = transformed_coords[..., :2]                  # (B,M,2)
    z  = transformed_coords[..., 2]                   # (B,M)

    # inner = 0.5 * min(size.x, size.y), h = 0.5 * size.z
    inner = 0.5 * size[..., :2].amin(dim=-1)         # (B,)
    h     = 0.5 * size[..., 2]                       # (B,)

    r = (roundness.squeeze(-1) * inner).unsqueeze(-1)   # (B,1)

    bounds = (size[..., :2] * 0.5).unsqueeze(1)      # (B,1,2)
    q = xy.abs() - bounds + r.unsqueeze(-1)                        # (B,M,2)

    # outside + inside - r  (no stacks; boolean-friendly clamps)
    q_pos = th.clamp_min(q, 0.0)
    outside = th.linalg.vector_norm(q_pos, dim=-1)           # (B,M)
    m = th.maximum(q[..., 0], q[..., 1])                     # (B,M)
    inside = th.clamp_max(m, 0.0)                            # (B,M)
    sdf2d = outside + inside - r                 # (B,M)

    # -------- trapezoid in (sdf2d, z)-space (avoid extra stack)
    # x3 = - (1 - scale) * inner
    x3 = - (1.0 - scale.squeeze(-1)) * inner                 # (B,)

    pos_2d = th.stack((sdf2d, z), dim=-1)                    # (B,M,2)
    sd = sd_taper_trapezoid_onion_exact_batched(                   # (B,M)
        pos_2d, inner, h, x3, onion_ratio
    )

    return sd - dilate_3d  # (B,1) broadcasts over (B,M)

# @th.jit.script
def batched_sf_packed_eval(coords, params):
    """
    coords: (B, M, 3)
    params: (B, 13+) -> [tx,ty,tz, rx,ry,rz, sx,sy,sz, round, dilate, scale, bulge]
    returns: (B, M)
    """
    # -------- unpack once, keep shapes broadcast-friendly
    unpacked_params = unpacked_params_sf(params)
    translate, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, rotate = unpacked_params
    transformed_coords = common_transform_coords(coords, translate, rotate)
    sdf_eval = batched_sf_packed_eval_part_2(transformed_coords, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio)
    return sdf_eval
    
def batched_sf_packed_su_eval(coords, params, su_vals):
    output = batched_sf_packed_eval(coords, params)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)

def batched_sf_packed_stochastic_eval(coords, params, logits, temperature):
    # B, N
    outputs = batched_sf_packed_eval(coords, params)
    g = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)

    # Unpack weights explicitly
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]

    # Equivalent to weighted sum of [outputs, 1.0]
    out = outputs * w0 + w1
    return out

def batched_sf_packed_stochastic_su_eval(coords, params, su_vals, logits, temperature):
    output = batched_sf_packed_stochastic_eval(coords, params, logits, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)


def batched_solid_sf_packed_eval(coords, params, temperature):
    unpacked_params = unpacked_params_sf(params)
    translate, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, rotate = unpacked_params
    logits = params[..., 14:18]

    
    g  = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)
    w_cube = w[..., 0:1]
    w_sphere = w[..., 1:2]
    w_cylinder = w[..., 2:3]
    w_cone = w[..., 3:4]
    size_xy = (size[..., 0:1] + size[..., 1:2])/2.0
    size_cyl = th.cat((size_xy, size_xy, size[..., 2:3]), dim=-1)
    size = (w_cube) * size +  (w_cylinder + w_cone) * size_cyl
    roundness = (w_cylinder + w_cone)
    dilate_3d = w_sphere * dilate_3d
    scale = (w_cube + w_sphere + w_cylinder)
    bulge_ratio = 0 * bulge_ratio
    onion_ratio = (w_cube + w_cylinder + w_cone) * onion_ratio
    new_params = th.cat([translate, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, rotate], dim=-1)

    out_eval = batched_sf_packed_eval(coords, new_params)
    return out_eval

def batched_solid_sf_packed_su_eval(coords, params, su_vals, temperature):
    output = batched_solid_sf_packed_eval(coords, params, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)

def batched_solid_sf_packed_stochastic_eval(coords, params, logits, temperature):
    # B, N
    outputs = batched_solid_sf_packed_eval(coords, params, temperature)
    g  = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]
    out = outputs * w0 + w1
    return out

def batched_solid_sf_packed_stochastic_su_eval(coords, params, su_vals, logits, temperature):
    output = batched_solid_sf_packed_stochastic_eval(coords, params, logits, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)
# Other option - just update the tables used in the origin code. 


def batched_varaxis_sf_packed_eval_st(coords: th.Tensor,
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
    
    unpacked_params = unpacked_params_sf(params)
    translate, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, rotate = unpacked_params
    logits = params[..., 14:17]

    # ----- straight-through gumbel-softmax: one-hot in forward, soft in backward
    transformed_coords = common_transform_coords(coords, translate, rotate)

    # ----- permutation matrices A such that: new = old @ A  (matches gather old[..., perm])
    # 0: xyz -> [0,1,2]
    # 1: yzx -> [1,2,0]
    # 2: zxy -> [2,0,1]
    # Each A has A[perm[j], j] = 1
    y = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)  # (B,3)
    P_stack = coords.new_tensor([
        [[1,0,0],[0,1,0],[0,0,1]],  # xyz  perm [0,1,2]
        [[0,0,1],[1,0,0],[0,1,0]],  # yzx  perm [1,2,0]  (new=[y,z,x])
        [[0,1,0],[0,0,1],[1,0,0]],  # zxy  perm [2,0,1]  (new=[z,x,y])
    ])  # (3,3,3)

    # Build per-batch permutation matrix (B,3,3). Forward it's exactly one of the above.
    P = th.einsum('bk,kij->bij', y, P_stack)  # (B,3,3)

    # ----- apply permutation to coords and size (single matmul each)
    coords_p = th.bmm(transformed_coords, P)  # (B,M,3)

    size_p = th.bmm(size.unsqueeze(1), P).squeeze(1)  # (B,3)

    return batched_sf_packed_eval_part_2(coords_p, size_p, roundness, dilate_3d, scale, bulge_ratio, onion_ratio)

def batched_varaxis_sf_packed_eval(coords: th.Tensor,
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
    unpacked_params = unpacked_params_varsf(params)
    translate, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, logits, rotate = unpacked_params
    
    
    transformed_coords = common_transform_coords(coords, translate, rotate)
    sdf_y = batched_sf_packed_eval_part_2(transformed_coords, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio)
    new_coords = transformed_coords.clone()[:, :, [1, 2, 0]]
    new_size = size.clone()[:, [1, 2, 0]]
    sdf_z = batched_sf_packed_eval_part_2(new_coords, new_size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio)
    new_coords = transformed_coords.clone()[:, :, [2, 0, 1]]
    new_size = size.clone()[:, [2, 0, 1]]
    sdf_x = batched_sf_packed_eval_part_2(new_coords, new_size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio)
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

def deviation_based_temperature(roundness, dilate_3d, scale, bulge_ratio, onion_ratio):
    scale_deviation = th.abs(scale.clone().detach() - 1)
    # Bulge deviation has a more drastic effect on the SDF.
    bulge_deviation = th.abs(bulge_ratio.clone().detach()) * 2.0
    deviation = scale_deviation + bulge_deviation
    div_t = 0.05 + (1 - th.tanh(10 * deviation - 2.0)) * 0.475
    return div_t

def batched_varaxis_sf_packed_stochastic_eval(coords, params, logits, temperature):
    # B, N
    outputs = batched_varaxis_sf_packed_eval(coords, params, temperature)
    g  = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]
    out = outputs * w0 + w1
    return out

def batched_varaxis_sf_packed_su_eval(coords, params, su_vals, temperature):
    output = batched_varaxis_sf_packed_eval(coords, params, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)


def batched_varaxis_sf_packed_stochastic_su_eval(coords, params, su_vals, logits, temperature):
    output = batched_varaxis_sf_packed_stochastic_eval(coords, params, logits, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)

function_map = {
    sps.SuperFrustumPackedBatched: batched_sf_packed_eval,
    sps.SuperFrustumPackedBatchedStochastic: batched_sf_packed_stochastic_eval,
    sps.SuperFrustumPackedBatchedSU: batched_sf_packed_su_eval,
    sps.SuperFrustumPackedBatchedStochasticSU: batched_sf_packed_stochastic_su_eval,
    # SolidSF batched variants
    sps.SolidSFPackedBatched: batched_solid_sf_packed_eval,
    sps.SolidSFPackedBatchedStochastic: batched_solid_sf_packed_stochastic_eval,
    sps.SolidSFPackedBatchedSU: batched_solid_sf_packed_su_eval,
    sps.SolidSFPackedBatchedStochasticSU: batched_solid_sf_packed_stochastic_su_eval,
    # VarAxisSF batched variants
    sps.VarAxisSFPackedBatched: batched_varaxis_sf_packed_eval,
    sps.VarAxisSFPackedBatchedStochastic: batched_varaxis_sf_packed_stochastic_eval,
    sps.VarAxisSFPackedBatchedSU: batched_varaxis_sf_packed_su_eval,
    sps.VarAxisSFPackedBatchedStochasticSU: batched_varaxis_sf_packed_stochastic_su_eval,
}
PRIMITIVE_MAP.update(function_map)

