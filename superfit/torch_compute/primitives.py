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
# Add singular and batch. 
# and from here add the older ones. 
import torch as th
import torch.nn.functional as F
from geolipi.torch_compute.constants import EPSILON
from geolipi.torch_compute.maps import PRIMITIVE_MAP
import geolipi.torch_compute.transforms as transform_bank
from geolipi.torch_compute.sdf_functions_3d import sdf3d_box, sdf3d_inexact_super_quadrics
import superfit.symbolic as sps
from superfit.symbolic.utils import sample_gumbel

### HACK!!
IDENTITY_MAT = th.eye(4, dtype=th.float32)
BULGE_EPS = 1e-5

def unpack_sf_params(params: th.Tensor) -> th.Tensor:
    translate = params[..., :3]
    size = params[..., 3:6]
    roundness = params[..., 6:7]
    dilate_3d = params[..., 7:8]
    scale = params[..., 8:9]
    bulge_ratio = params[..., 9:10]
    onion_ratio = params[..., 10:11]
    rotate = params[..., 11:14]
    return translate, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, rotate

def  cuboid_eval(coords: th.Tensor, size: th.Tensor) -> th.Tensor:
    size = size / 2.0
    sdf_eval = sdf3d_box(coords, size)
    return sdf_eval

def superquadric_eval(coords: th.Tensor, size: th.Tensor, epsilon_1: th.Tensor, epsilon_2: th.Tensor) -> th.Tensor:
    size = size / 2.0
    sdf_eval = sdf3d_inexact_super_quadrics(coords, size, epsilon_1, epsilon_2)
    return sdf_eval


def varaxis_sq_eval(coords: th.Tensor, size: th.Tensor, 
            epsilon_1: th.Tensor, epsilon_2: th.Tensor, logits: th.Tensor, temperature=1.0) -> th.Tensor:

    # --- ST gumbel-softmax: one-hot forward, soft backward
    P = varaxis_P(coords, logits, temperature)
    # --- apply permutation to coords and size
    coords_p = coords @ P  # (...,3)
    size_p = size @ P      # (...,3) typically (3,)
    new_sdf = superquadric_eval(coords_p, size_p, epsilon_1, epsilon_2)
    return new_sdf

def sdf2d_half_rounded_box(points: th.Tensor, bounds: th.Tensor, radius: th.Tensor | float) -> th.Tensor:
    """
    points: [N,2]
    bounds: [2]   (half-extents)
    radius: scalar tensor or float
    returns: [N]
    """
    bounds = bounds/ 2.0

    q = points.abs() - bounds + radius
    outside = th.norm(th.clamp(q, min=0.0), dim=-1)
    inside  = th.clamp(th.maximum(q[...,0], q[...,1]), max=0.0)
    return outside + inside - radius

def sdf2d_half_rounded_box_fourway(points: th.Tensor,
                              bounds: th.Tensor,
                              radius: th.Tensor) -> th.Tensor:
    """
    points: [B, N, 2] sample locations.
    bounds: [B, 2] half-extents (sx, sy).
    radius: scalar, [B], or [B,1] uniform corner radius.
    returns: [B, N] signed distances.
    """
    # Broadcast to [B, 1, ...]
    radius = radius[..., None, :]
    bounds = bounds[..., None, :] / 2.0
    radius_xy = th.where(points[..., 0:1] > 0, radius[..., 0:2], radius[..., 2:4])
    radius_x = th.where(points[..., 1:2] > 0, radius_xy[..., 0:1], radius_xy[..., 1:2])
    q = th.abs(points) - bounds + radius_x
    length_q = th.norm(th.clamp(q, min=0.0), dim=-1)
    sd = th.clamp(th.max(q[..., 0], q[..., 1]), max=0.0) + length_q - radius_x[..., 0]
    return sd.squeeze(-1)


def sp_simple_eval(coords: th.Tensor, size, roundness, dilate_3d) -> th.Tensor:
    height = size[..., -1:]
    local_z = coords[..., 2]
    r = roundness * 0.5 * th.amin(size[..., :2], dim=-1)

    sdf2d = sdf2d_half_rounded_box(coords[..., :2], size[..., :2], r)
    hf = th.maximum(local_z - height, -local_z)  # z - height v
    d = th.stack([sdf2d, hf], dim=-1)
    max_d = th.maximum(d, th.zeros_like(d))
    sdf3d = th.minimum(th.maximum(d[..., 0], d[..., 1]), th.tensor(0.0, device=coords.device)) + th.linalg.norm(max_d, dim=-1)
    sdf3d = sdf3d - dilate_3d
    return sdf3d

def sp_original_eval(coords: th.Tensor, size, roundness, onion_2d, dilate_3d) -> th.Tensor:
    height = size[..., -1:]
    local_z = coords[..., 2]
    min_size = th.amin(size[..., :2], dim=-1, keepdim=True) * 0.5
    r = roundness * min_size
    onion_amount = (1.0 - onion_2d) * min_size

    sdf2d = sdf2d_half_rounded_box_fourway(coords[..., :2], size[..., :2], r)
    sdf2d = th.where(sdf2d > 0.0, sdf2d, th.abs(sdf2d) - onion_amount)
    hf = th.maximum(local_z - height, -local_z)  # z - height v
    d = th.stack([sdf2d, hf], dim=-1)
    max_d = th.maximum(d, th.zeros_like(d))
    sdf3d = th.minimum(th.maximum(d[..., 0], d[..., 1]), th.tensor(0.0, device=coords.device)) + th.linalg.norm(max_d, dim=-1)
    
    sdf3d = sdf3d - dilate_3d
    return sdf3d


def map_arc_bulge(points, z, bulge, eps=BULGE_EPS):
    """
    Python / PyTorch port of GLSL mapArcBulge.
    Works with (..., 2) or (..., 3) points; returns (..., 2).
    """
    p = points[..., :2]                                  # ensure (..., 2)
    # p[..., 0] = p[..., 0] * th.sign(bulge)
    px, py = p[..., 0], p[..., 1]
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
    out = th.where(mask_above[..., None], above_point, inside_point)
    out = th.where(mask_below[..., None], below_point, out)
    out = th.stack([out[..., 0] * th.sign(bulge), out[..., 1]], dim=-1)
    return out

def sd_taper_trapezoid_onion_exact(pos_2d: th.Tensor,
                             inner: th.Tensor,         # () scalar
                             half_height: th.Tensor,   # () scalar
                             x3: th.Tensor,
                             onion_ratio: th.Tensor) -> th.Tensor:  # (1,) or () scalar
    """
    Signed distance to convex trapezoid with CCW vertices:
      p0 = (-inner, +half_height)
      p1 = (-inner, -half_height)
      p2 = (0,      -half_height)
      p3 = (x3,     +half_height)

    Inputs:
      pos_2d: (N,2)
      inner:  ()
      half_height: ()
      x3: (1,) or ()

    Output:
      (N,) exact signed distance (negative inside)
    """
    # Make sure x3 behaves as a scalar
    x3s = x3.squeeze()  # -> () if (1,)

    # Build vertex array A: (4,2) on the correct device/dtype
    # Use arithmetic with scalars to inherit device/dtype from them
    A = th.stack((
        th.stack((-inner + (x3s + inner) * onion_ratio,           +half_height)),  # p0
        th.stack((-inner * (1 - onion_ratio),           -half_height)),  # p1
        th.stack(( th.zeros_like(inner),       -half_height)),  # p2  (0 with right dtype/device)
        th.stack(( x3s,             +half_height)),  # p3
    ), dim=0)  # (4,2)

    # Edges A->B
    B = A.roll(shifts=-1, dims=0)          # (4,2)
    E = (B - A)                            # (4,2)

    # Vectorized point-to-segment distances
    P  = pos_2d.unsqueeze(1)               # (N,1,2)
    A_ = A.unsqueeze(0)                    # (1,4,2)
    E_ = E.unsqueeze(0)                    # (1,4,2)
    PA = P - A_                            # (N,4,2)

    denom = (E_ * E_).sum(dim=-1).clamp_min(1e-18)   # (1,4)
    t = ((PA * E_).sum(dim=-1) / denom).clamp(0.0, 1.0)   # (N,4)
    closest = A_ + t.unsqueeze(-1) * E_               # (N,4,2)
    dists = (P - closest).norm(dim=-1)                # (N,4)
    dmin = dists.min(dim=1).values                    # (N,)

    # Inside test: left-of-all-edges for CCW polygon
    cross = E_[..., 0] * PA[..., 1] - E_[..., 1] * PA[..., 0]  # (N,4)
    inside = (cross >= 0).all(dim=1)                           # (N,)

    return th.where(inside, -dmin, dmin)

def sp_tapered_onion_eval(coords: th.Tensor,
                     size: th.Tensor,
                     roundness: th.Tensor | float,
                     dilate_3d: th.Tensor | float,
                     scale: th.Tensor | float,
                     onion_ratio: th.Tensor | float) -> th.Tensor:
    # Unpack coordinates
    xy = coords[..., :2]                # (N,2)
    z  = coords[..., 2]                 # (N,)

    # Size-derived scalars (broadcast automatically)
    inner = 0.5 * th.amin(size[..., :2], dim=-1)  # (), half of min(size.x, size.y)
    h     = 0.5 * size[..., 2]                    # (), half of size.z

    # Scalar radius per-sample (broadcasted)
    r = roundness * inner                          # (1,) * () -> (1,) -> broadcast to (N,)

    # 2D rounded-rectangle SDF (uses size.xy directly as in GLSL)
    sdf2d = sdf2d_half_rounded_box(xy, size[..., :2], r)   # (N,)

    # Map to 2D (x,y) = (sdf2d, z)
    pos_2d = th.stack([sdf2d, z], dim=-1)                  # (N,2)

    # Trapezoid parameters (broadcast automatically)
    x3 = -(1.0 - scale) * inner                            # (1,) * () -> (1,) -> broadcast

    # Exact trapezoid SDF in (sdf2d, z)-space
    sd = sd_taper_trapezoid_onion_exact(pos_2d, inner, h, x3, onion_ratio[0])    # (N,)

    # Final dilation
    return sd - dilate_3d


def superfrustum_eval(coords: th.Tensor, size: th.Tensor, 
            roundness: th.Tensor | float, dilate_3d: th.Tensor | float, 
            scale: th.Tensor | float, bulge_ratio: th.Tensor | float, 
            onion_ratio: th.Tensor | float) -> th.Tensor:
    new_p_xz = map_arc_bulge(coords[..., [0, 2]], size[..., 2:3], bulge_ratio)
    new_p = th.stack((new_p_xz[..., 0], coords[..., 1], new_p_xz[..., 1]), dim=-1)
    new_sdf = sp_tapered_onion_eval(new_p, size, roundness, dilate_3d, scale, onion_ratio)
    return new_sdf

### NTC Packed ###
def superfrustum_packed_eval(coords, params):
    unpacked_params = unpack_sf_params(params)
    translate, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, rotate = unpacked_params
    # NEED to make this transform correctly. 
    pad = th.ones_like(coords[..., -1:])
    points_homog = th.cat([coords, pad], dim=-1)
    
    translate_transform = transform_bank.get_affine_translate_3D(IDENTITY_MAT.clone().to(translate.device), translate)
    rotate_transform = transform_bank.get_affine_rotate_axis_angle_3D(IDENTITY_MAT.clone().to(rotate.device), rotate)
    new_transform = rotate_transform @ translate_transform
    tranformed_coords = th.einsum("ij,mj->mi", new_transform, points_homog)

    n_dims = 3
    tranformed_coords = tranformed_coords[..., :n_dims] / (tranformed_coords[..., n_dims : n_dims + 1] + EPSILON)
    
    out_eval = superfrustum_eval(tranformed_coords, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio)
    return out_eval


def solid_sf_eval(coords: th.Tensor, size: th.Tensor, 
            roundness: th.Tensor | float, dilate_3d: th.Tensor | float, 
            scale: th.Tensor | float, bulge_ratio: th.Tensor | float, 
            onion_ratio: th.Tensor | float, logits, temperature=1.0) -> th.Tensor:
    # 
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
    new_sdf = superfrustum_eval(coords, size, roundness, dilate_3d, scale, bulge_ratio,onion_ratio)
    return new_sdf

def solid_sf_packed_eval(coords, params, temperature):
    
    unpacked_params = unpack_sf_params(params)
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
    out_eval = superfrustum_packed_eval(coords, new_params)
    return out_eval

# Other option - just update the tables used in the origin code. 

def varaxis_P(coords, logits, temperature=1.0):

    y = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)  # (3,)


    # --- permutation matrices A s.t. new = old @ A  (works for (...,3) @ (3,3))
    # 0: xyz, 1: yzx, 2: zxy
    P_stack = coords.new_tensor([
        [[1,0,0],[0,1,0],[0,0,1]],  # xyz
        [[0,0,1],[1,0,0],[0,1,0]],  # yzx  (new = [y,z,x])
        [[0,1,0],[0,0,1],[1,0,0]],  # zxy  (new = [z,x,y])
    ])  # (3,3,3)

    # Mix to get chosen permutation matrix; forward it's exactly one of them.
    # P = (y @ P_stack.reshape(3, 9)).reshape(3, 3)   # (3,3)
    P = th.einsum('k,kij->ij', y, P_stack)  # (3,3)
    return P


def varaxis_sf_eval(coords: th.Tensor, size: th.Tensor, 
            roundness: th.Tensor | float, dilate_3d: th.Tensor | float, 
            scale: th.Tensor | float, bulge_ratio: th.Tensor | float, 
            onion_ratio: th.Tensor | float, logits: th.Tensor, temperature=1.0) -> th.Tensor:

    # --- ST gumbel-softmax: one-hot forward, soft backward
    P = varaxis_P(coords, logits, temperature)
    # --- apply permutation to coords and size
    coords_p = coords @ P  # (...,3)
    size_p = size @ P      # (...,3) typically (3,)
    new_sdf = superfrustum_eval(coords_p, size_p, roundness, dilate_3d, scale, bulge_ratio, onion_ratio)
    return new_sdf

def superfrustum_y_eval(coords: th.Tensor, size, *args, **kwargs) -> th.Tensor:
    return superfrustum_eval(coords, size, *args, **kwargs)

def superfrustum_z_eval(coords: th.Tensor, size: th.Tensor, *args, **kwargs) -> th.Tensor:
    return superfrustum_eval(coords[..., [1, 2, 0]], size[..., [1, 2, 0]], *args, **kwargs)

def superfrustum_x_eval(coords: th.Tensor, size: th.Tensor, *args, **kwargs) -> th.Tensor:
    return superfrustum_eval(coords[..., [2, 0, 1]], size[..., [2, 0, 1]], *args, **kwargs)

def superquadric_y_eval(coords: th.Tensor, size: th.Tensor, *args, **kwargs) -> th.Tensor:
    return superquadric_eval(coords, size, *args, **kwargs)

def superquadric_z_eval(coords: th.Tensor, size: th.Tensor, *args, **kwargs) -> th.Tensor:
    return superquadric_eval(coords[..., [1, 2, 0]], size[..., [1, 2, 0]], *args, **kwargs)

def superquadric_x_eval(coords: th.Tensor, size: th.Tensor, *args, **kwargs) -> th.Tensor:
    return superquadric_eval(coords[..., [2, 0, 1]], size[..., [2, 0, 1]], *args, **kwargs)

def sp_proto_y_eval(coords: th.Tensor, size: th.Tensor, *args, **kwargs) -> th.Tensor:
    return sp_proto_eval(coords, size, *args, **kwargs)

def sp_proto_z_eval(coords: th.Tensor, size: th.Tensor, *args, **kwargs) -> th.Tensor:
    return sp_proto_eval(coords[..., [1, 2, 0]], size[..., [1, 2, 0]], *args, **kwargs)

def sp_proto_x_eval(coords: th.Tensor, size: th.Tensor, *args, **kwargs) -> th.Tensor:
    return sp_proto_eval(coords[..., [2, 0, 1]], size[..., [2, 0, 1]], *args, **kwargs)


def sp_proto_eval(
    coords: th.Tensor,                 # (..., 3)
    size: th.Tensor,                   # (3,) or (..., 3) broadcastable to coords batch
    roundness: th.Tensor | float,      # (4,) / (...,4) / scalar-broadcastable
    dilate_3d: th.Tensor | float,      # scalar or broadcastable to (...,)
    onion_ratio: th.Tensor | float,    # scalar or broadcastable to (...,)
    extrussion: th.Tensor | float,     # (2,) / (...,2) / scalar-broadcastable
    on_eps: float = 1e-8,
) -> th.Tensor:
    """
    PyTorch port of GLSL SPProto.

    GLSL reference:
      float SPProto(vec3 p, vec3 size, vec4 roundness, float dilate_3d, float onion_ratio, vec2 extrussion)

    Returns:
      sdf (...,)
    """

    # ---- unpack coords ----
    q2 = coords[..., :2]          # (...,2) : p.xy
    z = coords[..., 2]            # (...,)


    # If size is (3,), expand logically via broadcasting; same for others.
    size = size / 2.0
    sx = size[..., 0]
    sy = size[..., 1]
    sz = size[..., 2]

    # ---- common scales ----
    min_size = th.minimum(sx, sy)     # (...,) or scalar-broadcasted
    halfZ = sz

    r4 = roundness * min_size[..., None]    # (...,4)
    ex_scale = th.minimum(min_size, halfZ)  # (...,)
    ex = extrussion * ex_scale[..., None]   # (...,2)

    onion_amount = onion_ratio * min_size   # (...,)

    # ---- 2D rounded box with per-corner selection ----
    # rx = (x>0)? r4.xy : r4.zw
    mask_x = (q2[..., 0] > 0.0)[..., None]                 # (...,1)
    rx = th.where(mask_x, r4[..., 0:2], r4[..., 2:4])      # (...,2)

    # rc = (y>0)? rx.x : rx.y
    rc = th.where(q2[..., 1] > 0.0, rx[..., 0], rx[..., 1])  # (...,)

    # a = abs(q2) - size.xy + rc
    a = th.abs(q2) - size[..., :2] + rc[..., None]         # (...,2)

    # d = min(max(a.x,a.y),0) + length(max(a,0)) - rc
    m = th.clamp_min(a, 0.0)
    d = th.minimum(th.maximum(a[..., 0], a[..., 1]), th.zeros_like(rc)) \
        + th.linalg.norm(m, dim=-1) - rc

    # ---- pre-extrude inset/outset ----
    thv = 0.5 * th.maximum(ex[..., 0], ex[..., 1]) + min_size - onion_amount
    d = th.abs(d + thv) - thv

    # ---- asymmetric extrusion rounding by z sign ----
    er = th.where(z < 0.0, ex[..., 0], ex[..., 1])
    h = halfZ - er

    # ---- rounded extrusion ----
    qx = d + er
    qy = th.abs(z) - h

    i = th.minimum(th.maximum(qx, qy), th.zeros_like(qx))
    o = th.stack((th.clamp_min(qx, 0.0), th.clamp_min(qy, 0.0)), dim=-1)
    d = i + th.linalg.norm(o, dim=-1) - er

    # ---- optional onion ----
    # GLSL:
    # if (onion_amount > ON_EPS) d = abs(d + onion_amount) - onion_amount;
    # vectorized equivalent:
    onion_applied = th.abs(d + onion_amount) - onion_amount
    d = th.where(onion_amount > on_eps, onion_applied, d)

    # ---- final dilate ----
    return d - dilate_3d


def varaxis_spp_eval( 
    coords: th.Tensor,                 # (..., 3)
    size: th.Tensor,                   # (3,) or (..., 3) broadcastable to coords batch
    roundness: th.Tensor | float,      # (4,) / (...,4) / scalar-broadcastable
    dilate_3d: th.Tensor | float,      # scalar or broadcastable to (...,)
    onion_ratio: th.Tensor | float,    # scalar or broadcastable to (...,)
    extrussion: th.Tensor | float,     # (2,) / (...,2) / scalar-broadcastable
    logits: th.Tensor, 
    on_eps: float = 1e-8,
    temperature=1.0) -> th.Tensor:

    # --- ST gumbel-softmax: one-hot forward, soft backward
    P = varaxis_P(coords, logits, temperature)
    # --- apply permutation to coords and size
    coords_p = coords @ P  # (...,3)
    size_p = size @ P      # (...,3) typically (3,)
    new_sdf = sp_proto_eval(coords_p, size_p, roundness, dilate_3d, onion_ratio, extrussion)
    return new_sdf

function_map = {
    sps.Cuboid: cuboid_eval,
    # sps.CuboidPacked: cuboid_packed_eval,
    sps.SuperQuadric: superquadric_eval,
    # sps.SQPacked: sq_packed_eval,
    sps.SuperFrustum: superfrustum_eval,
    sps.SuperFrustumPacked: superfrustum_packed_eval,
    sps.SPProto: sp_proto_eval,
    # sps.SPProtoPacked: sp_proto_packed_eval,
    sps.SolidSF: solid_sf_eval,
    sps.SolidSFPacked: solid_sf_packed_eval,

    sps.VarAxisSF: varaxis_sf_eval,
    sps.VarAxisSQ: varaxis_sq_eval,
    sps.VarAxisSPP: varaxis_spp_eval,

    sps.SuperFrustumY: superfrustum_y_eval,
    sps.SuperFrustumZ: superfrustum_z_eval,
    sps.SuperFrustumX: superfrustum_x_eval,
    sps.SuperQuadricX: superquadric_x_eval,
    sps.SuperQuadricY: superquadric_y_eval,
    sps.SuperQuadricZ: superquadric_z_eval,
    sps.SPProtoX: sp_proto_x_eval,
    sps.SPProtoY: sp_proto_y_eval,
    sps.SPProtoZ: sp_proto_z_eval,

}
PRIMITIVE_MAP.update(function_map)

