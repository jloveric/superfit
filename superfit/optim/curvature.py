import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import igl
import trimesh
from matplotlib import cm
import torch as th
from ..utils.logger import logger


# ---------- Sampling: uniform (area-weighted) with barycentrics ----------
def sample_points_uniform_area(V, F, n, vertex_normals=None):
    """
    Uniformly sample 'n' points on a triangle mesh surface using area weighting.
    Returns:
      P        : (n,3) sampled points
      face_idx : (n,) indices of chosen faces
      bary     : (n,3) barycentric coords per sample (sum to 1)
      N        : (n,3) normals at samples (bary-interpolated vertex normals if provided,
                 otherwise per-face normals)
    """
    rand = np.random.random

    # face areas and sampling probs
    tri = V[F]                          # (m,3,3)
    face_areas = 0.5 * np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)
    probs = face_areas / (face_areas.sum() + 1e-18)

    face_idx = np.random.choice(F.shape[0], size=n, p=probs)

    # random barycentric (u,v,w) ~ uniform on triangle
    r1 = np.sqrt(rand(n))
    r2 = rand(n)
    b0 = 1.0 - r1
    b1 = r1 * (1.0 - r2)
    b2 = r1 * r2
    bary = np.stack([b0, b1, b2], axis=1)     # (n,3)

    # sampled points (n,3)
    tri_sel = V[F[face_idx]]                  # (n,3,3)
    P = (bary[:, :, None] * tri_sel).sum(axis=1)

    # normals at samples
    if vertex_normals is None:
        # per-face normals
        fn = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
        fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-18)
        N = fn[face_idx]
    else:
        vn = vertex_normals
        N = (vn[F[face_idx]] * bary[:, :, None]).sum(axis=1)
        N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-18)

    return P, face_idx, bary, N


# ---------- Interpolation: per-vertex scalar -> sampled points ----------
def interpolate_vertex_scalar_to_samples(F, face_idx, bary, s_vertex):
    """
    F        : (m,3) faces
    face_idx : (n,) sampled face indices
    bary     : (n,3) barycentrics
    s_vertex : (V,) per-vertex scalar
    returns  : (n,) scalar at samples via bary interpolation
    """
    tri = F[face_idx]            # (n,3)
    vals = s_vertex[tri]         # (n,3)
    return (vals * bary).sum(axis=1)


# ---------- libIGL: principal curvatures & curvedness ----------
def igl_principal_curvature(V, F):
    """
    Returns dictionary with:
      K1, K2 : (V,) principal curvatures
      H,  K  : (V,) mean and Gaussian curvature
      C      : (V,) curvedness = sqrt((k1^2 + k2^2)/2)
    """
    PD1, PD2, K1, K2, _ = igl.principal_curvature(V, F)
    H = 0.5 * (K1 + K2)
    K = K1 * K2
    C = np.sqrt(0.5 * (K1**2 + K2**2))
    return dict(K1=K1, K2=K2, H=H, K=K, C=C)


# ---------- libIGL: cotangent Laplacian & Voronoi mass ----------
def igl_cotan_and_mass(V, F):
    """
    Returns:
      L : (VxV) cotangent Laplacian (NOTE: libIGL uses NEGATIVE semidefinite L)
      M : (VxV) Voronoi/mixed-area mass matrix (diagonal SPD)
      ell : mean edge length (float)
    """
    L = igl.cotmatrix(V, F)  # negative semidefinite
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
    ell = igl.avg_edge_length(V, F)

    # In pyigl these are SciPy CSR already; if not, try a safe conversion:
    if not sp.issparse(L):
        L = sp.csr_matrix((L.data(), L.indices(), L.indptr()), shape=L.shape)
    if not sp.issparse(M):
        M = sp.csr_matrix((M.data(), M.indices(), M.indptr()), shape=M.shape)

    return L.tocsr(), M.tocsr(), float(ell)

import torch
import scipy.sparse as sp

# TODO: Ideally use cholesky solve later using CUPY. 
def heat_smooth_scalar_torch(C_vertex, L, M, sigma, ell=None, device=None, tol=1e-8, max_iter=500):
    """
    GPU-accelerated implicit heat-kernel smoothing for scalar field on a mesh.
    Equivalent to (M - tau * L) x = M * C, using iterative CG solver.

    Parameters
    ----------
    C_vertex : ndarray (N,)
        Scalar field per vertex.
    L : scipy.sparse.csr_matrix
        NEGATIVE Laplacian matrix (SPD).
    M : scipy.sparse.csr_matrix
        Mass matrix (SPD, typically lumped or cotangent).
    sigma : float
        Smoothing scale (sigma ~ geodesic radius).
    ell : unused
    device : str
        'cuda' or 'cpu'. Auto-detected if None.
    tol, max_iter : float, int
        CG tolerance and iteration limit.

    Returns
    -------
    x : ndarray (N,)
        Smoothed scalar field.
    """
    tau = float(sigma) ** 2
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert scipy sparse to torch sparse
    def sp_to_torch(sps):
        sps = sps.tocoo()
        indices = torch.tensor(np.array([sps.row, sps.col]), dtype=torch.long)
        values = torch.tensor(sps.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, sps.shape, device=device)

    L_t = sp_to_torch(L)
    M_t = sp_to_torch(M)
    C_t = torch.tensor(C_vertex, dtype=torch.float32, device=device)

    # Build operator (M - tau*L)
    A_t = M_t - L_t * tau
    b_t = torch.sparse.mm(M_t, C_t.unsqueeze(1)).squeeze(1)

    # Conjugate Gradient solve
    x_t = torch.zeros_like(C_t)
    r = b_t - torch.sparse.mm(A_t, x_t.unsqueeze(1)).squeeze(1)
    p = r.clone()
    rs_old = torch.dot(r, r)

    for _ in range(max_iter):
        Ap = torch.sparse.mm(A_t, p.unsqueeze(1)).squeeze(1)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        x_t += alpha * p
        r -= alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x_t.detach().cpu().numpy()

# ---------- Heat smoothing on mesh (implicit) ----------
def heat_smooth_scalar_igl(C_vertex, L, M, sigma, ell=None):
    """
    Implicit heat-kernel smoothing for scalar field on a mesh.
    With libIGL's NEGATIVE Laplacian L, the SPD system is:
        (M - tau * L) x = M * C
    where tau = sigma^2  (sigma ~ geodesic radius in length units).
    """
    if ell is None:
        # if not provided, sigma is already in world units
        pass
    tau = float(sigma)**2

    A = (M - L * tau)          # still sparse
    b = M @ C_vertex
    # Solve
    x = spla.spsolve(A.tocsr(), b)
    return x
    
def curvature_to_weight(C_pts, Cmin=5.0, Cmax=50.0, steepness=5.0):
    """
    Map absolute curvedness to [0,1] weights using a soft step.
    Cmin, Cmax: expected curvature thresholds (unit-cube meshes)
    steepness: controls sharpness of transition.
    """
    C_clamped = np.clip(C_pts, 0, None)
    # smooth logistic transition from Cmin->Cmax
    z = (C_clamped - Cmin) / (Cmax - Cmin + 1e-9)
    w = 1 / (1 + np.exp(-steepness * (z - 0.5)))  # sigmoid step
    return w

# ---------- Multi-scale curvedness ----------
def multiscale_curvedness_igl(V, F, sigmas=None, combine='meah'):
    """
    Compute per-vertex curvedness (from principal curvatures) and smooth it
    over multiple scales with heat diffusion. Returns:
      C_multi      : (V,) combined multi-scale curvedness
      C_scales     : list[(V,)] per-scale curvedness
      sigmas_used  : list[float]
    """
    curv = igl_principal_curvature(V, F)
    C0 = curv['C']

    L, M, ell = igl_cotan_and_mass(V, F)
    if sigmas is None:
        # Pick scales relative to mean edge length
        sigmas = [0.5*ell, 1.0*ell, 2.0*ell]
        # sigmas =[10.0,] * 100
        # sigmas = [0.15*ell]
    C_scales = []
    for s in sigmas:
        C_s = heat_smooth_scalar_torch(C0, L, M, s)
        logger.debug(f"sigma={s:.3f}, mean={C_s.mean():.3f}, std={C_s.std():.3f}, max={C_s.max():.3f}")
        C_scales.append(C_s)
    # C_scales = [heat_smooth_scalar_igl(C0, L, M, s, ell=ell) for s in sigmas]

    if combine == 'max':
        C_multi = np.maximum.reduce(C_scales)
    elif combine == 'mean':
        C_multi = np.mean(np.stack(C_scales, axis=0), axis=0)
    elif combine == 'sum':
        C_multi = np.sum(np.stack(C_scales, axis=0), axis=0)
    else:
        raise ValueError("combine must be 'max', 'mean', or 'sum'.")

    return C_multi, C_scales, sigmas, curv


def get_points_and_weights(target_mesh, sketcher, n_points=10000):
    mesh_tm = target_mesh
    V = mesh_tm.vertices.view(np.ndarray).astype(np.float64)
    F = mesh_tm.faces.view(np.ndarray).astype(np.int32)

    # (Optional) vertex normals for nicer point colors
    VN = igl.per_vertex_normals(V, F)
    VN = VN / (np.linalg.norm(VN, axis=1, keepdims=True) + 1e-18)

    # ----- 1) Multi-scale curvedness on vertices (libIGL) -----
    with th.autocast('cuda', dtype=th.float32):
        C_multi, C_scales, sigmas, curv_all = multiscale_curvedness_igl(V, F,
                                                                        sigmas=None,  # or [0.5*ell, 1.0*ell, 2.0*ell]
                                                                        combine='mean')

    P, f_idx, bary, Np = sample_points_uniform_area(V, F, n_points, vertex_normals=VN)

    # ----- 3) Interpolate the smoothed curvedness to samples -----
    C_pts = interpolate_vertex_scalar_to_samples(F, f_idx, bary, C_multi)

    C_norm = curvature_to_weight(C_pts, Cmin=0, Cmax=30, steepness=8)
    points = th.from_numpy(P).float().to(sketcher.device).to(sketcher.dtype)
    weights = th.from_numpy(C_norm).float().to(sketcher.device).to(sketcher.dtype)
    return points, weights