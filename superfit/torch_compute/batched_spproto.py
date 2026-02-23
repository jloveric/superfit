import torch as th
import torch.nn.functional as F
from .primitives import map_arc_bulge
import superfit.symbolic as sps
from geolipi.torch_compute.maps import PRIMITIVE_MAP
from geolipi.torch_compute.constants import EPSILON
# from geolipi.torch_compute.transforms import axis_angle_to_rotation_matrix
from .compile_friendly import axis_angle_to_rotation_matrix
from .batched_sf import smooth_union_k_way, common_transform_coords
from superfit.symbolic.utils import sample_gumbel

def batched_sp_proto_eval(
    transformed_coords: th.Tensor,  # (B,M,3)
    size: th.Tensor,                # (B,3)
    roundness: th.Tensor,           # (B,4)
    dilate_3d: th.Tensor,           # (B,1) or (B,)
    onion_ratio: th.Tensor,         # (B,1) or (B,)
    extrussion: th.Tensor,          # (B,2)
    on_eps: float = 1e-8,
) -> th.Tensor:
    """
    Batched SPProto port (faithful to GLSL + single-primitive torch port).
    Returns: (B,M)
    """
    B, M, _ = transformed_coords.shape

    # Normalize per-primitive scalars to (B,)
    onion_ratio = onion_ratio.view(B)
    dilate_3d = dilate_3d.view(B)

    # ---- unpack coords ----
    q2 = transformed_coords[..., :2]     # (B,M,2) : p.xy
    z  = transformed_coords[..., 2]      # (B,M)
    
    size = size / 2.0
    sx = size[:, 0]                      # (B,)
    sy = size[:, 1]                      # (B,)
    sz = size[:, 2]                      # (B,)

    # ---- common scales ----
    min_size = th.minimum(sx, sy)  # (B,)
    halfZ = sz                     # (B,)

    # r4 = roundness * min_size
    r4 = roundness * min_size[:, None]   # (B,4)

    # ex = extrussion * min(min_size, halfZ)
    ex_scale = th.minimum(min_size, halfZ)        # (B,)
    ex = extrussion * ex_scale[:, None]           # (B,2)

    onion_amount = onion_ratio * min_size         # (B,)

    # ---- 2D rounded box (per-corner) ----
    # rx = (x>0)? r4.xy : r4.zw
    mask_x = (q2[..., 0] > 0.0)                   # (B,M)
    rxy = r4[:, None, 0:2]                        # (B,1,2)
    rzw = r4[:, None, 2:4]                        # (B,1,2)
    rx = th.where(mask_x[..., None], rxy, rzw)    # (B,M,2)

    # rc = (y>0)? rx.x : rx.y
    rc = th.where(q2[..., 1] > 0.0, rx[..., 0], rx[..., 1])   # (B,M)

    # a = abs(q2) - size.xy + rc
    a = q2.abs() - size[:, None, :2] + rc[..., None]          # (B,M,2)

    # d = min(max(a.x,a.y),0) + length(max(a,0)) - rc
    a_pos = th.clamp_min(a, 0.0)                               # (B,M,2)
    outside = th.linalg.vector_norm(a_pos, dim=-1)             # (B,M)
    inside = th.clamp_max(th.maximum(a[..., 0], a[..., 1]), 0.0)  # (B,M)
    d = inside + outside - rc                                  # (B,M)

    # ---- pre-extrude inset/outset ----
    thv = 0.5 * th.maximum(ex[:, 0], ex[:, 1]) + min_size - onion_amount   # (B,)
    d = th.abs(d + thv[:, None]) - thv[:, None]                              # (B,M)

    # ---- asymmetric extrusion rounding by z sign ----
    er = th.where(z < 0.0, ex[:, 0:1], ex[:, 1:2])   # (B,M), via broadcast from (B,1)
    h = halfZ[:, None] - er                           # (B,M)

    # ---- rounded extrusion ----
    qx = d + er
    qy = z.abs() - h

    i = th.clamp_max(th.maximum(qx, qy), 0.0)
    o = th.stack((th.clamp_min(qx, 0.0), th.clamp_min(qy, 0.0)), dim=-1)   # (B,M,2)
    d = i + th.linalg.vector_norm(o, dim=-1) - er                           # (B,M)

    # ---- optional onion ----
    onion_applied = th.abs(d + onion_amount[:, None]) - onion_amount[:, None]
    d = th.where((onion_amount > on_eps)[:, None], onion_applied, d)

    # ---- final dilate ----
    return d - dilate_3d[:, None]   # (B,M)

def unpacked_params_spp(params):
    translate = params[...,  :3]
    size = params[..., 3:6]
    roundness = params[..., 6:10]
    dilate_3d = params[..., 10:11]
    onion_ratio = params[..., 11:12]
    extrusion_ratio = params[..., 12:14]
    rotate = params[..., 14:17]
    return translate, size, roundness, dilate_3d, onion_ratio, extrusion_ratio, rotate

def unpacked_params_varspp(params):
    translate = params[...,  :3]
    size = params[..., 3:6]
    roundness = params[..., 6:10]
    dilate_3d = params[..., 10:11]
    onion_ratio = params[..., 11:12]
    extrusion_ratio = params[..., 12:14]
    logits = params[..., 14:17]
    rotate = params[..., 17:20]
    return translate, size, roundness, dilate_3d, onion_ratio, extrusion_ratio, logits, rotate

def batched_spp_packed_eval(coords, params):
    """
    coords: (B, M, 3)
    params: (B, 13+) -> [tx,ty,tz, rx,ry,rz, sx,sy,sz, round, dilate, scale, bulge]
    returns: (B, M)
    """
    # -------- unpack once, keep shapes broadcast-friendly
    unpacked_params = unpacked_params_spp(params)
    translate, size, roundness, dilate_3d, onion_ratio, extrusion_ratio, rotate = unpacked_params
    transformed_coords = common_transform_coords(coords, translate, rotate)
    sdf_eval = batched_sp_proto_eval(transformed_coords, size, roundness, dilate_3d, onion_ratio, extrusion_ratio)
    return sdf_eval
    
def batched_varaxis_spp_packed_eval(coords: th.Tensor,
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
    unpacked_params = unpacked_params_varspp(params)
    translate, size, roundness, dilate_3d, onion_ratio, extrusion_ratio, logits, rotate = unpacked_params
    

    transformed_coords = common_transform_coords(coords, translate, rotate)
    sdf_y = batched_sp_proto_eval(transformed_coords, size, roundness, dilate_3d, onion_ratio, extrusion_ratio)
    new_coords = transformed_coords.clone()[:, :, [1, 2, 0]]
    new_size = size.clone()[:, [1, 2, 0]]
    sdf_z = batched_sp_proto_eval(new_coords, new_size, roundness, dilate_3d, onion_ratio, extrusion_ratio)
    new_coords = transformed_coords.clone()[:, :, [2, 0, 1]]
    new_size = size.clone()[:, [2, 0, 1]]
    sdf_x = batched_sp_proto_eval(new_coords, new_size, roundness, dilate_3d, onion_ratio, extrusion_ratio)
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

def batched_spp_packed_su_eval(coords, params, su_vals):
    output = batched_spp_packed_eval(coords, params)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)

def batched_spp_packed_stochastic_eval(coords, params, logits, temperature):
    # B, N
    outputs = batched_spp_packed_eval(coords, params)
    g = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)

    # Unpack weights explicitly
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]

    # Equivalent to weighted sum of [outputs, 1.0]
    out = outputs * w0 + w1
    return out

def batched_spp_packed_stochastic_su_eval(coords, params, su_vals, logits, temperature):
    output = batched_spp_packed_stochastic_eval(coords, params, logits, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)

def batched_varaxis_spp_packed_stochastic_eval(coords, params, logits, temperature):
    # B, N
    outputs = batched_varaxis_spp_packed_eval(coords, params, temperature)
    g  = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]
    out = outputs * w0 + w1
    return out

def batched_varaxis_spp_packed_su_eval(coords, params, su_vals, temperature):
    output = batched_varaxis_spp_packed_eval(coords, params, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)


def batched_varaxis_spp_packed_stochastic_su_eval(coords, params, su_vals, logits, temperature):
    output = batched_varaxis_spp_packed_stochastic_eval(coords, params, logits, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)
function_map = {
    sps.SPProtoPackedBatched: batched_spp_packed_eval,
    sps.SPProtoPackedBatchedStochastic: batched_spp_packed_stochastic_eval,
    sps.SPProtoPackedBatchedSU: batched_spp_packed_su_eval,
    sps.SPProtoPackedBatchedStochasticSU: batched_spp_packed_stochastic_su_eval,
    # VarAxisSPP
    sps.VarAxisSPPPackedBatched: batched_varaxis_spp_packed_eval,
    sps.VarAxisSPPPackedBatchedStochastic: batched_varaxis_spp_packed_stochastic_eval,
    sps.VarAxisSPPPackedBatchedSU: batched_varaxis_spp_packed_su_eval,
    sps.VarAxisSPPPackedBatchedStochasticSU: batched_varaxis_spp_packed_stochastic_su_eval,
}
PRIMITIVE_MAP.update(function_map)

