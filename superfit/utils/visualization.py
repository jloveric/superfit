
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
import distinctipy
import trimesh
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from .mesh_sdf import sdf_to_mesh
from sysl.torch_compute.evaluate_mat_expr import recursive_evaluate_mat_expr
from ..symbolic.utils import fetch_singular_expr_eval
from sysl.utils import recursive_gls_to_sysl, recursive_sm_to_smg
from geolipi.torch_compute.evaluate_expression import recursive_evaluate



def heatmap_on_mesh(mesh, points, values, cmap_name="coolwarm", vmin=None, vmax=None,
                    k_neighbors=3, mesh_alpha=255):
    """
    Paint a scalar heatmap onto mesh vertices by projecting per-point values.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to color.
    points : np.ndarray, shape (N, 3)
        3D positions of sampled points that carry scalar values.
    values : np.ndarray, shape (N,)
        Scalar value at each point (e.g. probability, loss, distance).
    cmap_name : str
        Matplotlib colormap name (e.g. "coolwarm", "viridis", "plasma", "RdBu_r").
    vmin, vmax : float or None
        Clamp range for the colormap. None = use data min/max.
    k_neighbors : int
        Number of nearest sampled-points to average for each mesh vertex.
    mesh_alpha : int (0-255)
        Alpha channel for the vertex colors.

    Returns
    -------
    colored_mesh : trimesh.Trimesh
        A copy of the mesh with vertex colors set to the heatmap.
    """
    if vmin is None:
        vmin = float(values.min())
    if vmax is None:
        vmax = float(values.max())

    # Build KD-tree on the sampled points
    tree = cKDTree(points)

    # For each mesh vertex, find k nearest sampled points and average their values
    dists, idxs = tree.query(mesh.vertices, k=k_neighbors)

    if k_neighbors == 1:
        vertex_values = values[idxs]
    else:
        # Inverse-distance weighted average (with small epsilon to avoid div-by-zero)
        weights = 1.0 / (dists + 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        vertex_values = (values[idxs] * weights).sum(axis=1)

    # Normalize to [0, 1] for the colormap
    normed = np.clip((vertex_values - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)

    # Apply colormap
    cmap = plt.cm.get_cmap(cmap_name)
    rgba = (cmap(normed) * 255).astype(np.uint8)
    rgba[:, 3] = mesh_alpha

    colored_mesh = mesh.copy()
    colored_mesh.visual.vertex_colors = rgba
    return colored_mesh

def get_face_points(mesh):
    """
    Return the centroid (center point) of each face in a trimesh.Trimesh.

    Args:
        mesh (trimesh.Trimesh): input mesh

    Returns:
        (F, 3) numpy array of face centroids, where F is number of faces
    """
    # mesh.faces is (F, 3) with indices into mesh.vertices (V, 3)
    # Gather the vertices for each face: shape (F, 3, 3)
    face_vertices = mesh.vertices[mesh.faces]

    # Compute centroid by averaging along axis 1 (the 3 vertices of each face)
    centroids = face_vertices.mean(axis=1)

    return centroids


def expr_to_mesh_with_ids(expr, sketcher):
    sdf = recursive_evaluate(expr.tensor(), sketcher)
    input_mesh = sdf_to_mesh(sdf, sketcher)
    # create the id version:

    sampled_expr = fetch_singular_expr_eval(expr.sympy(), relaxed_eval=False)
    new_expr = recursive_sm_to_smg(sampled_expr.sympy())
    mat_expr, _ = recursive_gls_to_sysl(new_expr, ind=0, version="v1")
    face_points = get_face_points(input_mesh)
    # surface_samples_inp = sample_surface_proximal_points(input_mesh, n_points=100_000, jitter_sigma=0.0)
    surface_samples_inp = th.from_numpy(face_points).float().to(sketcher.device)
    outputs = recursive_evaluate_mat_expr(mat_expr.tensor(), sketcher, coords=surface_samples_inp)
    surface_output_sdf, surface_prim_ids = outputs[..., 0], outputs[..., 1]
    return input_mesh, surface_prim_ids


def color_mesh_with_ids(mesh, surface_prim_ids):
    if isinstance(surface_prim_ids, th.Tensor):
        surface_prim_ids = surface_prim_ids.cpu().numpy()
    n_labels = int(surface_prim_ids.max().item())
    colors = distinctipy.get_colors(n_labels)
    # Create a new mesh with ONLY geometry (no material, no texture)
    clean_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        process=False   # <- keep topology exactly as is
    )

    # Optional: recompute normals in case the viewer depends on them
    clean_mesh.fix_normals()
    for i in range(n_labels):
        cur_faces = surface_prim_ids == i
        clean_mesh.visual.face_colors[cur_faces, :3] = np.array(colors[i]) * 255
    return clean_mesh
