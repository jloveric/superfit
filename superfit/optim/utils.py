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
import numpy as np
import cubvh
import trimesh
from ..utils.config import AlgorithmConfig as AlgConf, reset_eval_seeds

def quick_sample_points(mesh, sketcher, n_points=10000):
    # Ensure normals exist (needed only if you want smooth normals)
    # Sample uniformly on the surface
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    points = th.from_numpy(points).float().to(sketcher.device)
    return points


def quick_sample_points_and_normals(mesh, sketcher, n_points=10000):
    # Ensure normals exist (needed only if you want smooth normals)
    # Sample uniformly on the surface
    points, faces = trimesh.sample.sample_surface_even(mesh, n_points)

    # Optionally compute normals:
    normals = mesh.face_normals[faces]
    # o3d_mesh = o3d.geometry.TriangleMesh()
    # o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    # o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    # o3d_mesh.compute_vertex_normals()
    # pcd = o3d_mesh.sample_points_uniformly(number_of_points=n_points)
    # points = np.asarray(pcd.points)
    points = th.from_numpy(points).float().to(sketcher.device)

    normals = np.asarray(normals)
    normals = th.from_numpy(normals).float().to(sketcher.device)
    normals = normals / th.norm(normals, dim=1, keepdim=True)

    return points, normals


def get_sdf_and_gradients(points, mesh):
    """
    Computes signed distances and gradients w.r.t. mesh surface.

    Args:
        points (np.ndarray or torch.Tensor): (N, 3) query points
        mesh (trimesh.Trimesh): triangle mesh
        sketcher: unused here, placeholder

    Returns:
        sdf (torch.Tensor): (N,) signed distance values
        gradients (torch.Tensor): (N, 3) normalized gradient vectors
        surface_points (torch.Tensor): (N, 3) closest points on the mesh surface
    """
    # Ensure input points are numpy array for cubvh
    BVH = cubvh.cuBVH(mesh.vertices, mesh.faces)  # input: numpy arrays
    distances, face_id, uvw = BVH.signed_distance(points, return_uvw=True, mode='raystab')

    # Convert to torch tensors

    # Convert mesh vertices to torch tensors once
    verts = th.tensor(mesh.vertices, dtype=th.float32).to(points.device)
    faces = th.tensor(mesh.faces, dtype=th.long).to(points.device)

    # Extract face vertex positions via indexing
    face_indices = faces[face_id]  # (N, 3)
    v0 = verts[face_indices[:, 0]]
    v1 = verts[face_indices[:, 1]]
    v2 = verts[face_indices[:, 2]]

    # Compute surface point by barycentric interpolation
    surface_points = uvw[:, 0:1] * v0 + uvw[:, 1:2] * v1 + uvw[:, 2:3] * v2  # (N, 3)

    # Gradient = normalized vector from surface point to query point
    vectors = points - surface_points
    gradients = th.nn.functional.normalize(vectors, dim=1)

    return distances, gradients

def recompute_sdf_from_BVH(points, BVH, mode='raystab'):
    """
    Computes signed distances and gradients w.r.t. mesh surface.

    Args:
        points (np.ndarray or torch.Tensor): (N, 3) query points
        mesh (trimesh.Trimesh): triangle mesh
        sketcher: unused here, placeholder

    Returns:
        sdf (torch.Tensor): (N,) signed distance values
        gradients (torch.Tensor): (N, 3) normalized gradient vectors
        surface_points (torch.Tensor): (N, 3) closest points on the mesh surface
    """
    # Ensure input points are numpy array for cubvh
    distances, _, _ = BVH.signed_distance(points, return_uvw=False, mode=mode)
    return distances

def exponential_temperature_schedule(step: int,
                                total_steps: int,
                                T_max: float = 1.0,
                                T_min: float = 0.9,
                                gamma: float = 2.0,
                                device: str | th.device = "cuda") -> th.Tensor:
    """
    Exponential-like temperature decay with a 'rate' (shape) parameter 'gamma'
    that *still hits exact endpoints*: T(0)=T_max, T(total_steps)=T_min.

    gamma > 1.0 -> slower early decay (more conservative)
    gamma < 1.0 -> faster early decay

    Returns a scalar tensor.
    """
    if step <= 0:
        return T_max
    if step >= total_steps:
        return T_min

    s = step / float(total_steps)
    w = (1.0 - s) ** gamma  # weight on log(T_max)
    logT = w * th.log(T_max) + (1.0 - w) * th.log(T_min)
    return th.exp(logT)




def perform_stochastic_precondition(base_coords, sketcher, i, base_iters):
    init_val = 2 * np.sqrt(3) * 0.01
    final_val = 0
    # linear interpolate from i to base iters
    frac = min(i / (base_iters), 1.0)
    alpha = init_val * (1 - frac) + final_val * frac
    noise = th.randn_like(base_coords) * alpha
    base_coords = base_coords + noise
    return base_coords


def get_mask_scaled_aabb(points: th.Tensor,
                                  mesh: trimesh.Trimesh,
                                  padding: float = 0.15) -> th.Tensor:
    """
    Return subset of points inside the scaled bounding box of a mesh.

    Args:
        points: (N, 3) th.Tensor of 3D coordinates
        mesh: trimesh.Trimesh object with .bounds
        scale: float, scaling factor applied about the bbox center

    Returns:    
        th.Tensor of shape (M,) where M <= N
    """
    # bmin, bmax = mesh.bounds  # (2, 3)
    # center = (bmin + bmax) / 2.0
    # extent = (bmax - bmin) * scale / 2.0

    # scaled_bmin = center - extent
    # scaled_bmax = center + extent

    # scaled_bmin = th.tensor(scaled_bmin, dtype=points.dtype, device=points.device)
    # scaled_bmax = th.tensor(scaled_bmax, dtype=points.dtype, device=points.device)

    # mask = (points >= scaled_bmin) & (points <= scaled_bmax)
    # inside_mask = mask.all(dim=1)

    bmin, bmax = mesh.bounds  # numpy arrays
    padded_bmin = bmin - padding
    padded_bmax = bmax + padding

    padded_bmin = th.tensor(padded_bmin, dtype=points.dtype, device=points.device)
    padded_bmax = th.tensor(padded_bmax, dtype=points.dtype, device=points.device)

    mask = (points >= padded_bmin) & (points <= padded_bmax)
    inside_mask = mask.all(dim=1)
    return inside_mask


def sample_surface_proximal_points(mesh: trimesh.Trimesh, n_points=10000, jitter_sigma=0.0):
    """
    Sample points uniformly near the surface of a mesh using trimesh.

    Args:
        mesh (trimesh.Trimesh): input mesh
        n_points (int): number of points to sample
        jitter_sigma (float): stddev of Gaussian noise to add along normals

    Returns:
        (N, 3) numpy array of sampled points
    """
    reset_eval_seeds()  # resets numpy/torch/python RNG for reproducibility

    # Sample surface points and retrieve face indices
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)

    if jitter_sigma > 0.0:
        normals = mesh.face_normals[face_indices]
        noise = np.random.uniform(-jitter_sigma, jitter_sigma, size=points.shape)
        points = points + normals * noise

    return points


def perform_batched_stochastic_precondition(base_coords, i, base_iters, init_val):
    final_val = 0
    frac = min(i / (base_iters), 1.0)
    alpha = init_val * (1 - frac) + final_val * frac
    # More efficient noise generation
    noise = th.randn_like(base_coords) * alpha
    base_coords = base_coords + noise
    return base_coords


def perform_batched_stochastic_precondition_with_curvature(base_coords, i, base_iters, cut_value, full_value, curvature_weights):
    final_val = 0
    frac = min(i / (base_iters), 1.0)
    alpha = 1 * (1 - frac) + final_val * frac
    # More efficient noise generation
    # for places with low curvature -> use what would happen with full value.
    # for places with high curvature -> use what would happen with cut value.
    max_value = max(th.max(curvature_weights), 1.0)
    curvature_ratio = (max_value - curvature_weights)
    # high curvature -> low curvature ratio -> more cutoff value.

    # noise = th.randn_like(base_coords) * alpha * (max_value - curvature_weights)
    noise = th.randn_like(base_coords) * (full_value * curvature_ratio + cut_value * (1 - curvature_ratio)) * alpha
    base_coords = base_coords + noise
    return base_coords

