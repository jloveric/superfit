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
from geolipi.torch_compute.constants import EPSILON

def sample_gumbel(shape, eps=1e-10, device=None, dtype=None):
    U = th.rand(shape, device=device, dtype=dtype)
    return -th.log(-th.log(U + eps) + eps)

def map_arc_bulge(points, z, bulge, eps=1e-5):
    """
    Python / PyTorch port of GLSL mapArcBulge.
    Works with (..., 2) or (..., 3) points; returns (..., 2).
    """
    p = points[:, :, :2]                                  # ensure (..., 2)
    px, py = p[:, :, 0], p[:, :, 1]
    px = px * th.sign(bulge)

    half_z = 0.5 * z
    theta_top = th.clamp_min(th.abs(bulge) * (th.pi * 0.5), eps) # cheaper than clamp(min=)

    # center and radius (center at (center_pos, 0))
    center_pos = half_z / th.tan(theta_top)
    dx = px - center_pos
    dy = py
    radius = th.sqrt(th.square(center_pos) + th.square(half_z))

    # angle wrt arc center (needed for inside mapping and region tests)
    point_angle = th.atan2(dy, -dx)                     # atan2(py, center_pos - px)

    # inside region mapping
    angle_ratio = th.clamp(point_angle / theta_top, -1.0, 1.0)
    new_y = angle_ratio * half_z
    new_x = th.sqrt(th.square(dx) + th.square(dy)) - radius
    inside_point = th.stack((new_x, new_y), dim=-1)

    # precompute sin/cos once; note ||(s,c)|| == 1 so no normalization needed
    s = th.sin(theta_top)
    c = th.cos(theta_top)

    # --- top (+z/2) ---
    # tangent t_top = ( s,  c); normal n_top = (-c, s); end_top = (0, +half_z)
    along_top =  px * s + (py - half_z) * c
    perp_top  = -px * c + (py - half_z) * s
    above_point = th.stack((perp_top, half_z + along_top), dim=-1)

    # --- bottom (-z/2) ---
    # tangent t_bot = (-s,  c); normal n_bot = (-c,-s); end_bot = (0, -half_z)
    along_bot = -px * s + (py + half_z) * c
    perp_bot  = -px * c - (py + half_z) * s
    below_point = th.stack((perp_bot, -half_z + along_bot), dim=-1)

    # --- region masks (boolean) ---
    mask_above = (point_angle >  theta_top)             # (...,)
    mask_below = (point_angle < -theta_top)             # (...,)

    # --- mix with th.where (no dtype casts, no multiplies) ---
    out = th.where(mask_above[:, :, None], above_point, inside_point)
    out = th.where(mask_below[:, :, None], below_point, out)
    out = out * th.sign(bulge)[:, :, None]
    return out

def old_axis_angle_to_rotation_matrix(axis_angle: th.Tensor) -> th.Tensor:
    """
    Convert an axis-angle vector (..., 3) to a rotation matrix (..., 3, 3)
    using the Rodrigues' rotation formula.
    """
    B = axis_angle.shape[0]
    theta = th.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)  # (..., 1)
    axis = axis_angle / theta  # normalized axis (..., 3)

    x, y, z = axis.unbind(-1)  # each (...,)
    zero = th.zeros_like(x)
    K = th.stack([
        zero, -z,    y,
         z,  zero,  -x,
        -y,   x,   zero
    ], dim=-1).reshape((B, 3, 3))  # (..., 3, 3)

    I = th.eye(3, device=axis.device, dtype=axis.dtype).expand(K.shape)
    sin = th.sin(theta)[:, None]
    cos = th.cos(theta)[:, None]

    R = I + sin * K + (1 - cos) * (K @ K)  # Rodrigues' formula

    return R

def axis_angle_to_rotation_matrix(axis_angle: th.Tensor, eps: float = 1e-8) -> th.Tensor:

    if axis_angle.ndim != 2 or axis_angle.shape[-1] != 3:
        raise ValueError(f"Expected (B, 3), got {tuple(axis_angle.shape)}")

    B = axis_angle.shape[0]
    theta = th.linalg.norm(axis_angle, dim=-1, keepdim=True)  # (B, 1)
    axis = axis_angle / theta.clamp_min(eps)
    x, y, z = axis.unbind(-1)

    zero = th.zeros_like(x)
    K = th.stack([

        zero, -z,    y,

         z,  zero,  -x,

        -y,   x,   zero

    ], dim=-1).reshape(B, 3, 3)
    I = th.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).expand(B, 3, 3)
    sin = th.sin(theta).unsqueeze(-1)   # (B,1,1)
    cos = th.cos(theta).unsqueeze(-1)   # (B,1,1)
    R = I + sin * K + (1.0 - cos) * (K @ K)

    return R


def _sdf_smooth_union_pair(sdf_a: th.Tensor, sdf_b: th.Tensor, k: th.Tensor) -> th.Tensor:
    """
    Smooth union of two SDFs (exactly your formula).
    sdf_a, sdf_b: same shape (broadcasting okay).
    k: scalar/tensor broadcastable to sdf_a/sdf_b.

    k *= 4.0;
    float h = max( k-abs(a-b), 0.0 )/k;
    return min(a,b) - h*h*k*(1.0/4.0);
    """
    # k *= 4.0
    # h = th.clamp(k - th.abs(sdf_a - sdf_b), min=0.0) / k
    # sdf = th.minimum(sdf_a, sdf_b) - h * h * k / 4.0
    h = th.clamp(0.5 + 0.5 * (sdf_b - sdf_a) / (k + EPSILON), min=0.0, max=1.0)
    sdf = th.lerp(sdf_b, sdf_a, h) - k * h * (1.0 - h)
    return sdf

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
    cross = E_[:, :, :, 0] * PA[:, :, :, 1] - E_[:, :, :, 1] * PA[:, :, :, 0]  # (B,N,4)
    inside = (cross >= 0).all(dim=-1)                          # (B,N)

    return th.where(inside, -dmin, dmin)  # (B,N)


# @th.jit.script
def batched_sf_packed_eval(coords, params):
    """
    coords: (B, M, 3)
    params: (B, 13+) -> [tx,ty,tz, rx,ry,rz, sx,sy,sz, round, dilate, scale, bulge]
    returns: (B, M)
    """
    # -------- unpack once, keep shapes broadcast-friendly

    translate   = params[:,  :3]      # (B,3)
    rotate      = params[:,  3:6]     # (B,3) axis-angle
    size        = params[:,  6:9]     # (B,3)
    roundness   = params[:,  9:10]    # (B,1)
    dilate_3d   = params[:, 10:11]    # (B,1)
    scale       = params[:, 11:12]    # (B,1)
    bulge_ratio = params[:, 12:13]    # (B,1)
    onion_ratio = params[:, 13:14]    # (B,1)
    # -------- rigid transform: R @ (x - t)   (avoid 4x4, bmm, einsum, w-divide)
    # Your original new_transform = R * T with T translating by -t yields R(x - t).
    R = axis_angle_to_rotation_matrix(rotate)                  # (B,3,3)
    p_local = coords - translate.unsqueeze(1)                  # (B,M,3)
    transformed_coords = th.matmul(p_local, R.transpose(-1, -2))  # (B,M,3)

    # -------- xz bulge map (avoid extra stacks until needed)
    # Use your faster map_arc_bulge that returns (...,2)
    new_p_xz = map_arc_bulge(
        transformed_coords[:, :, (0, 2)],   # (B,M,2): take x,z
        size[:, 2:3],                    # (B,1) -> broadcast
        bulge_ratio                        # (B,1)
    )
    # Replace x,z with mapped values
    transformed_coords = th.stack(
        (new_p_xz[:, :, 0], transformed_coords[:, :, 1], new_p_xz[:, :, 1]), dim=-1
    )  # (B,M,3)

    # -------- rounded-rectangle SDF in xy (no tiny temporaries)
    xy = transformed_coords[:, :, :2]                  # (B,M,2)
    z  = transformed_coords[:, :, 2]                   # (B,M)

    # inner = 0.5 * min(size.x, size.y), h = 0.5 * size.z
    inner = 0.5 * size[:, :2].amin(dim=-1)         # (B,)
    h     = 0.5 * size[:, 2]                       # (B,)

    r = (roundness.squeeze(-1) * inner).unsqueeze(-1)   # (B,1)

    bounds = (size[:, :2] * 0.5).unsqueeze(1)      # (B,1,2)
    q = xy.abs() - bounds + r.unsqueeze(-1)                        # (B,M,2)

    # outside + inside - r  (no stacks; boolean-friendly clamps)
    q_pos = th.clamp_min(q, 0.0)
    outside = th.linalg.vector_norm(q_pos, dim=-1)           # (B,M)
    m = th.maximum(q[:,:, 0], q[:,:, 1])                     # (B,M)
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

def batched_sf_packed_stochastic_eval(coords, params, logits, temperature):
    # B, N
    outputs = batched_sf_packed_eval(coords, params)
    g  = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)

    # Unpack weights explicitly
    w0 = w[:, 0:1]
    w1 = w[:, 1:2]

    # Equivalent to weighted sum of [outputs, 1.0]
    out = outputs * w0 + w1
    return out

def batched_sf_packed_stochastic_su_eval(coords, params, su_vals, logits, temperature):
    output = batched_sf_packed_stochastic_eval(coords, params, logits, temperature)

    K = output.shape[0]

    out = output[0]
    for i in range(1, K):
        k_reshaped = su_vals[i-1].unsqueeze(-1)
        out = _sdf_smooth_union_pair(out, output[i], k_reshaped)
    

    return (output, out)