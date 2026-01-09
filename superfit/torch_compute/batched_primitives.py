import torch as th
from .primitives import map_arc_bulge
import superfit.symbolic as sps
from geolipi.torch_compute.maps import PRIMITIVE_MAP
from geolipi.torch_compute.constants import EPSILON
from geolipi.torch_compute.transforms import axis_angle_to_rotation_matrix
from superfit.symbolic.utils import sample_gumbel


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
    cross = E_[..., 0] * PA[..., 1] - E_[..., 1] * PA[..., 0]  # (B,N,4)
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

    translate   = params[...,  :3]      # (B,3)
    rotate      = params[...,  3:6]     # (B,3) axis-angle
    size        = params[...,  6:9]     # (B,3)
    roundness   = params[...,  9:10]    # (B,1)
    dilate_3d   = params[..., 10:11]    # (B,1)
    scale       = params[..., 11:12]    # (B,1)
    bulge_ratio = params[..., 12:13]    # (B,1)
    onion_ratio = params[..., 13:14]    # (B,1)
    # -------- rigid transform: R @ (x - t)   (avoid 4x4, bmm, einsum, w-divide)
    # Your original new_transform = R * T with T translating by -t yields R(x - t).
    R = axis_angle_to_rotation_matrix(rotate)                  # (B,3,3)
    p_local = coords - translate.unsqueeze(1)                  # (B,M,3)
    transformed_coords = th.matmul(p_local, R.transpose(-1, -2))  # (B,M,3)

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


def batched_sf_packed_su_eval(coords, params, su_vals):
    output = batched_sf_packed_eval(coords, params)
    K = output.shape[0]
    out = output[0]
    for i in range(1, K):
        k_reshaped = su_vals[i-1].unsqueeze(-1)
        out = _sdf_smooth_union_pair(out, output[i], k_reshaped)
    return (output, out)

def batched_sf_packed_stochastic_eval(coords, params, logits, temperature):
    # B, N
    outputs = batched_sf_packed_eval(coords, params)
    g  = sample_gumbel(logits.shape, device=logits.device)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)

    # Unpack weights explicitly
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]

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


def batched_solid_sf_packed_eval(coords, params, temperature):
    translate = params[..., :3]
    rotate = params[..., 3:6]
    size = params[..., 6:9]
    roundness = params[..., 9:10]
    dilate_3d = params[..., 10:11]
    scale = params[..., 11:12]
    bulge_ratio = params[..., 12:13]
    onion_ratio = params[..., 13:14]
    logits = params[..., 14:18]

    
    g  = sample_gumbel(logits.shape, device=logits.device)
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
    new_params = th.cat([translate, rotate, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio], dim=-1)

    out_eval = batched_sf_packed_eval(coords, new_params)
    return out_eval

def batched_solid_sf_packed_su_eval(coords, params, su_vals, temperature):
    output = batched_solid_sf_packed_eval(coords, params, temperature)
    K = output.shape[0]
    out = output[0]
    for i in range(1, K):
        k_reshaped = su_vals[i-1].unsqueeze(-1)
        out = _sdf_smooth_union_pair(out, output[i], k_reshaped)
    return (output, out)

def batched_solid_sf_packed_stochastic_eval(coords, params, logits, temperature):
    # B, N
    outputs = batched_solid_sf_packed_eval(coords, params, temperature)
    g  = sample_gumbel(logits.shape, device=logits.device)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]
    out = outputs * w0 + w1
    return out

def batched_solid_sf_packed_stochastic_su_eval(coords, params, su_vals, logits, temperature):
    output = batched_solid_sf_packed_stochastic_eval(coords, params, logits, temperature)
    K = output.shape[0]
    out = output[0]
    for i in range(1, K):
        k_reshaped = su_vals[i-1].unsqueeze(-1) * (temperature ** 2)
        out = _sdf_smooth_union_pair(out, output[i], k_reshaped)
    return (output, out)
# Other option - just update the tables used in the origin code. 
function_map = {
    sps.SuperFrustumPackedBatched: batched_sf_packed_eval,
    sps.SuperFrustumPackedBatchedStochastic: batched_sf_packed_stochastic_eval,
    sps.SolidSFPackedBatched: batched_solid_sf_packed_eval,
    sps.SolidSFPackedBatchedStochastic: batched_solid_sf_packed_stochastic_eval,
    sps.SuperFrustumPackedBatchedSU: batched_sf_packed_su_eval,
    sps.SuperFrustumPackedBatchedStochasticSU: batched_sf_packed_stochastic_su_eval,
    sps.SolidSFPackedBatchedSU: batched_solid_sf_packed_su_eval,
    sps.SolidSFPackedBatchedStochasticSU: batched_solid_sf_packed_stochastic_su_eval,

}
PRIMITIVE_MAP.update(function_map)

