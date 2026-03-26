import torch as th
import torch.nn.functional as F
from .primitives import map_arc_bulge
import superfit.symbolic as sps
from geolipi.torch_compute.maps import PRIMITIVE_MAP
from geolipi.torch_compute.constants import EPSILON
# from geolipi.torch_compute.transforms import axis_angle_to_rotation_matrix
from ..symbolic.utils import sample_gumbel
from .compile_friendly import axis_angle_to_rotation_matrix, _sdf_smooth_union_pair
from geolipi.torch_compute.sdf_functions_3d import sdf3d_box, sdf3d_inexact_super_quadrics
from .batched_sf import smooth_union_k_way, common_transform_coords


def batched_cuboid_packed_eval(coords, params):
    translate = params[...,  :3]
    size = params[..., 3:6] / 2.0
    rotate = params[..., 6:9]
    transformed_coords = common_transform_coords(coords, translate, rotate)
    sdf_eval = sdf3d_box(transformed_coords, size)
    return sdf_eval

def batched_sq_packed_eval(coords, params):
    translate = params[...,  :3]
    skew_vec = params[..., 3:6] / 2.0
    epsilon_1 = params[..., 6:7]
    epsilon_2 = params[..., 7:8]
    rotate = params[..., 8:11]
    transformed_coords = common_transform_coords(coords, translate, rotate)
    sdf_eval = sdf3d_inexact_super_quadrics(transformed_coords, skew_vec, epsilon_1, epsilon_2)
    return sdf_eval

def batched_cuboid_packed_su_eval(coords, params, su_vals):
    output = batched_cuboid_packed_eval(coords, params)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)

def batched_cuboid_packed_stochastic_eval(coords, params, logits, temperature):
    # B, N
    outputs = batched_cuboid_packed_eval(coords, params)
    g = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)

    # Unpack weights explicitly
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]

    # Equivalent to weighted sum of [outputs, 1.0]
    out = outputs * w0 + w1
    return out

def batched_cuboid_packed_stochastic_su_eval(coords, params, su_vals, logits, temperature):
    output = batched_cuboid_packed_stochastic_eval(coords, params, logits, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)

    
def batched_sq_packed_su_eval(coords, params, su_vals):
    output = batched_sq_packed_eval(coords, params)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)

def batched_sq_packed_stochastic_eval(coords, params, logits, temperature):
    # B, N
    outputs = batched_sq_packed_eval(coords, params)
    g = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)

    # Unpack weights explicitly
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]

    # Equivalent to weighted sum of [outputs, 1.0]
    out = outputs * w0 + w1
    return out


def batched_sq_packed_stochastic_su_eval(coords, params, su_vals, logits, temperature):
    output = batched_sq_packed_stochastic_eval(coords, params, logits, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)


def batched_varaxis_sq_packed_eval(coords: th.Tensor,
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
    translate = params[...,  :3]
    skew_vec = params[..., 3:6] / 2.0
    epsilon_1 = params[..., 6:7]
    epsilon_2 = params[..., 7:8]
    logits = params[..., 8:11]
    rotate = params[..., 11:14]
    
    transformed_coords = common_transform_coords(coords, translate, rotate)
    
    sdf_y = sdf3d_inexact_super_quadrics(transformed_coords, skew_vec, epsilon_1, epsilon_2)
    new_coords = transformed_coords.clone()[:, :, [1, 2, 0]]
    new_skew_vec = skew_vec.clone()[:, [1, 2, 0]]
    sdf_z = sdf3d_inexact_super_quadrics(new_coords, new_skew_vec, epsilon_1, epsilon_2)
    new_coords = transformed_coords.clone()[:, :, [2, 0, 1]]
    new_skew_vec = skew_vec.clone()[:, [2, 0, 1]]
    sdf_x = sdf3d_inexact_super_quadrics(new_coords, new_skew_vec, epsilon_1, epsilon_2)
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


def batched_varaxis_sq_packed_stochastic_su_eval(coords, params, su_vals, logits, temperature):
    output = batched_varaxis_sq_packed_stochastic_eval(coords, params, logits, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)

    
def batched_varaxis_sq_packed_su_eval(coords, params, su_vals, temperature):
    output = batched_varaxis_sq_packed_eval(coords, params, temperature)
    out = smooth_union_k_way(output, su_vals)
    return (output, out)

def batched_varaxis_sq_packed_stochastic_eval(coords, params, logits, temperature):
    # B, N
    outputs = batched_varaxis_sq_packed_eval(coords, params, temperature)
    g = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    w  = th.softmax((logits + g) / temperature, dim=-1)  # (..., 2)

    # Unpack weights explicitly
    w0 = w[..., 0:1]
    w1 = w[..., 1:2]

    # Equivalent to weighted sum of [outputs, 1.0]
    out = outputs * w0 + w1
    return out


function_map = {
    sps.CuboidPackedBatched: batched_cuboid_packed_eval,
    sps.CuboidPackedBatchedStochastic: batched_cuboid_packed_stochastic_eval,
    sps.CuboidPackedBatchedSU: batched_cuboid_packed_su_eval,
    sps.CuboidPackedBatchedStochasticSU: batched_cuboid_packed_stochastic_su_eval,
    # SQ
    sps.SQPackedBatched: batched_sq_packed_eval,
    sps.SQPackedBatchedStochastic: batched_sq_packed_stochastic_eval,
    sps.SQPackedBatchedSU: batched_sq_packed_su_eval,
    sps.SQPackedBatchedStochasticSU: batched_sq_packed_stochastic_su_eval,
    # VarAxisSQ
    sps.VarAxisSQPackedBatched: batched_varaxis_sq_packed_eval,
    sps.VarAxisSQPackedBatchedStochastic: batched_varaxis_sq_packed_stochastic_eval,
    sps.VarAxisSQPackedBatchedSU: batched_varaxis_sq_packed_su_eval,
    sps.VarAxisSQPackedBatchedStochasticSU: batched_varaxis_sq_packed_stochastic_su_eval,
}
PRIMITIVE_MAP.update(function_map)

