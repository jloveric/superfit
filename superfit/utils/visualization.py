import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


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
