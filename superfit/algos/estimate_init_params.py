# Code to initialize primitive from a given sdf volume. 
import torch as th
import torch as th
from torch import tensor as T
from superfit.utils.mesh_sdf import sdf_to_mesh
import geolipi.symbolic as gls
import superfit.symbolic as sps
from superfit.utils.config import AlgorithmConfig as AlgConf
from superfit.utils.logger import logger
from superfit.symbolic.utils import inject_stochastic_prim

# =============================================================================
# NUMERICAL CONSTANTS
# =============================================================================
# Epsilon values for numerical stability (ordered by precision)
EPS_TIGHT = 1e-12      # For normalization and angle computations
EPS_MEDIUM = 1e-9      # General numerical stability
EPS_LOOSE = 1e-6       # For rotation/axis-angle conversions and clamping

# =============================================================================
# BINNING / SAMPLING DEFAULTS
# =============================================================================
DEFAULT_NBINS = 12                  # Number of bins for axis scoring
DEFAULT_NBINS_MESH = 20             # Number of bins for mesh-based axis selection
DEFAULT_MIN_PTS_PER_BIN = 24        # Minimum points per bin for statistics
DEFAULT_MIN_PTS_PER_BIN_MESH = 50   # Minimum points per bin (mesh version)
DEFAULT_N_SAMPLES = 4096            # Surface sampling count for mesh analysis
DEFAULT_MIN_REQUIRED_PTS = 9        # Minimum points to process a primitive

# =============================================================================
# AXIS SCORING WEIGHTS (for cylinder/extrusion axis selection)
# =============================================================================
W_ELONGATION = 1.0      # Weight for elongation term (fast scoring)
W_ANISOTROPY = 0.5      # Weight for cross-section anisotropy penalty
W_RAD_TREND = 0.5       # Weight for radius trend (taper) penalty
W_RAD_VARIANCE = 0.5    # Weight for radius residual variance penalty
W_SLAB_PENALTY = 0.25   # Penalty for slab-like (flat) configurations


# =============================================================================
# CLAMP BOUNDS
# =============================================================================
ELONGATION_CLAMP_MAX = 50.0         # Max elongation ratio
SLABNESS_CLAMP_MAX = 50.0           # Max slabness ratio
TAPER_RATIO_CLAMP = (0.25, 4.0)     # (min, max) for individual axis taper
TAPER_AMOUNT_CLAMP = (0.25, 1.75)   # (min, max) for combined taper amount
ONION_AMOUNT_CLAMP = (0.025, 0.975) # (min, max) for onion (hollow) ratio

# =============================================================================
# PERCENTILE / QUANTILE THRESHOLDS
# =============================================================================
Q_ENDCAPS_DEFAULT = (2.5, 97.5)     # Percentiles for robust endpoint estimation
Q_INNER_OUTER = (0.05, 0.95)        # Quantiles for inner/outer radius estimation
Q_IQR = (0.25, 0.75)                # Interquartile range bounds

# =============================================================================
# OTHER ALGORITHM PARAMETERS
# =============================================================================
ONION_BAND_FRAC_DEFAULT = 0.15      # Fraction of radius for onion band detection
SIZE_SHRINK_FACTOR = 0.95           # Shrink factor for bounding size estimates



def rotation_matrix_to_axis_angle(R: th.Tensor, variant: str = 'default', eps: float = EPS_LOOSE) -> th.Tensor:
    """
    Convert a 3x3 rotation matrix to an axis-angle vector (3-vector).
    Handles edge cases like identity and 180° rotation robustly.
    """
    device, dtype = R.device, R.dtype
    trace = R.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    cos_theta = th.clamp((trace - 1) / 2, -1.0, 1.0)
    theta = th.acos(cos_theta)

    # === Case 1: theta ≈ 0 (identity)
    if th.isclose(theta, th.tensor(0.0, device=device, dtype=dtype), atol=eps):
        return th.tensor([eps, 0.0, 0.0], device=device, dtype=dtype)

    # === Case 2: theta ≈ π (180 degree rotation)
    if th.isclose(theta, th.tensor(th.pi, device=device, dtype=dtype), atol=eps):
        R_plus = (R + th.eye(3, device=device, dtype=dtype)) / 2
        axis = th.sqrt(th.clamp(R_plus.diagonal(), min=0))
        # If multiple entries are zero, fallback
        if axis.norm() < eps:
            axis = th.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        axis = axis / axis.norm()
        return theta * axis

    # === Normal case
    skew = (R - R.transpose(-1, -2)) / (2 * th.sin(theta))
    axis = th.stack([
        skew[2, 1],
        skew[0, 2],
        skew[1, 0]
    ])

    # === Variant remapping
    if variant == 'flip':
        axis = -axis
    elif variant == 'xzy':
        axis = axis[[0, 2, 1]]
    elif variant == 'yzx':
        axis = axis[[1, 2, 0]]
    elif variant == 'zxy':
        axis = axis[[2, 0, 1]]
    elif variant == 'neg_x':
        axis = axis * th.tensor([-1.0, 1.0, 1.0], device=axis.device)
    elif variant == 'neg_y':
        axis = axis * th.tensor([1.0, -1.0, 1.0], device=axis.device)
    elif variant == 'neg_z':
        axis = axis * th.tensor([1.0, 1.0, -1.0], device=axis.device)
    elif variant != 'default':
        raise ValueError(f"Unknown variant '{variant}'.")

    return theta * axis
def _percentiles_1d(x, p_low, p_high):
    q = th.tensor([p_low/100.0, p_high/100.0], device=x.device, dtype=x.dtype)
    a, b = th.quantile(x, q, interpolation='linear')
    return a, b

def _right_handed_from_svd(V):
    # V: 3x3 whose columns are principal directions
    V = V.clone()
    if th.det(V) < 0:
        V[:, -1] *= -1.0
    return V
def pca_frame(points):
    """
    points: [K,3] (torch)
    Returns: center [3], V [3,3] (right-handed, columns = PCs), S [3] (singular values)
    """
    center = points.mean(dim=0)
    X = points - center
    U, S, Vh = th.linalg.svd(X, full_matrices=False)
    V = _right_handed_from_svd(Vh.T)  # columns are PCs
    return center, V, S

def _safe_normalize(v, eps=EPS_TIGHT):
    n = v.norm()
    return v / (n + eps)
def _frame_from_z(z):
    """Make an ONB (x,y,z) with given unit z (robust if z ~ e_x)."""
    z = _safe_normalize(z)
    tmp = th.tensor([1.0, 0.0, 0.0], device=z.device, dtype=z.dtype)
    if th.abs(th.dot(tmp, z)) > 0.95:
        tmp = th.tensor([0.0, 1.0, 0.0], device=z.device, dtype=z.dtype)
    x = tmp - th.dot(tmp, z) * z
    x = _safe_normalize(x)
    y = th.cross(z, x, dim=-1)
    return x, y, z
    

def choose_extrusion_axis_from_mesh_v1(mesh, nbins=DEFAULT_NBINS_MESH, n_samples=DEFAULT_N_SAMPLES, min_pts_per_bin=DEFAULT_MIN_PTS_PER_BIN_MESH, version="v1"):
    """
    Evaluate candidate sweep directions by cross-section consistency.

    Returns: best_dir (torch[3]), scores (dict[dir_key -> float])
    """

    device = 'cuda' if th.cuda.is_available() else 'cpu'

    # --- sample points from mesh surface ---
    pts, _ = mesh.sample(n_samples, return_index=True)   # trimesh API
    pts = th.from_numpy(pts.copy()).float().to(device)
    pts -= pts.mean(dim=0, keepdim=True)

    # --- candidate directions: PCA + normal covariance ---
    verts = th.from_numpy(mesh.vertices.copy()).float().to(device)
    verts -= verts.mean(dim=0)
    cov_v = verts.T @ verts
    _, V = th.linalg.eigh(cov_v)
    cand_dirs = [V[:, i] / V[:, i].norm() for i in range(3)]

    normals = th.from_numpy(mesh.face_normals.copy()).float().to(device)
    areas = th.from_numpy(mesh.area_faces.copy()).float().to(device)
    M = (normals.T * areas).T
    cov_n = M.T @ M
    evals_n, evecs_n = th.linalg.eigh(cov_n)
    cand_dirs.append(evecs_n[:, 0] / evecs_n[:, 0].norm())

    scores = {}
    for k, z_dir in enumerate(cand_dirs):
        z_dir = z_dir / (z_dir.norm() + 1e-9)
        x_dir, y_dir, _ = _frame_from_z(z_dir)

        # project points
        t = pts @ z_dir
        u = pts @ x_dir
        v = pts @ y_dir

        # bin along t
        qs = th.linspace(0, 1, nbins+1, device=device)
        edges = th.quantile(t, qs)
        slice_stats = []
        for b in range(nbins):
            m = (t >= edges[b]) & (t < edges[b+1]) if b < nbins-1 else (t >= edges[b]) & (t <= edges[b+1])
            if int(m.sum()) < min_pts_per_bin:
                continue
            uu, vv = u[m], v[m]
            uu -= uu.mean(); vv -= vv.mean()
            cov2d = th.stack([uu.var(), vv.var()])
            slice_stats.append(cov2d)
        if len(slice_stats) < 2:
            scores[f"cand_{k}"] = -1e6
            continue
    
        slice_stats = th.stack(slice_stats)  # [S,2]
        if version == "v1":
            # cross-section stability = low std of slice_stats
            # stability = 1.0 / (slice_stats.std(dim=0).mean() + 1e-6)
            # normalized stability: relative variation
            stability = 1.0 - slice_stats.std(dim=0).mean() / (slice_stats.mean(dim=0).mean() + 1e-6)
            circularity = 1.0 - th.abs(slice_stats.mean(0)[0] - slice_stats.mean(0)[1]) / (slice_stats.mean(0).sum() + 1e-6)
            # elong = (t.max() - t.min()) / (slice_stats.mean().sum().sqrt() + 1e-6)
            var_u, var_v = slice_stats.mean(dim=0)
            cross_radius = th.sqrt((var_u + var_v))   # average std ≈ radius
            elong = (t.max() - t.min()) / (cross_radius + 1e-6)
            score = 1.0 * stability  + 0.1 * elong + 1.0 * circularity
            
            logger.debug(f"stability: {stability.item():.6f}, elong: {elong.item():.6f}, circularity: {circularity.item():.6f}")
            scores[f"cand_{k}"] = score.item()
        scores[f"cand_{k}"] = float(score.item())


    best_idx = max(scores, key=lambda k: scores[k])
    best_dir = cand_dirs[int(best_idx.split("_")[1])]
    return best_dir, scores

def _axis_cyl_score_fast(
    X, z,
    nbins=DEFAULT_NBINS, min_pts_per_bin=DEFAULT_MIN_PTS_PER_BIN,
    w_elong=W_ELONGATION, w_aniso=W_ANISOTROPY, w_radtrend=W_RAD_TREND, w_radvar=W_RAD_VARIANCE,
    slab_penalty=W_SLAB_PENALTY, use_bin_medians=True, eps=EPS_TIGHT
):
    """
    Higher is better.

    Terms:
      + elongation          := IQR(t) / (IQR(u) + IQR(v))
      - aniso               := |var(u) - var(v)| / (var(u) + var(v))
      - rad_trend           := |slope| of r ~ a + b t   (fit on medians if binning)
      - rad_resid_norm      := median(|r - (a+bt)|)/median(r)
      - slabness (tie-break): penalize tiny IQR(t) vs in-plane spread
    """
    # Orthonormal frame and projections
    x, y, z = _frame_from_z(z)              # each [3]
    t = X @ z; u = X @ x; v = X @ y
    r = th.sqrt(u*u + v*v) + eps

    # --- robust spreads (IQR) ---
    def iqr(v):
        q = th.quantile(v, th.tensor([0.25, 0.75], device=v.device, dtype=v.dtype), interpolation='linear')
        return (q[1] - q[0]).clamp_min(eps)

    iqr_t = iqr(t); iqr_u = iqr(u); iqr_v = iqr(v)
    elongation = (iqr_t / (iqr_u + iqr_v + eps)).clamp_max(ELONGATION_CLAMP_MAX)

    # --- cross-section circularity (prefer var(u) ≈ var(v)) ---
    vu = th.var(u, unbiased=False) + eps
    vv = th.var(v, unbiased=False) + eps
    aniso = (th.abs(vu - vv) / (vu + vv)).clamp_max(1.0)

    # --- radius constancy along t: trend + residual ---
    # Option A: fit on per-bin medians (robust) to avoid local clutter
    if use_bin_medians and t.numel() >= nbins * min_pts_per_bin:
        qs = th.linspace(0, 1, nbins+1, device=t.device, dtype=t.dtype)
        edges = th.quantile(t, qs, interpolation='linear')
        tb, rb, counts = [], [], []
        for b in range(nbins):
            m = (t >= edges[b]) & (t < edges[b+1]) if b < nbins-1 else (t >= edges[b]) & (t <= edges[b+1])
            if int(m.sum().item()) < min_pts_per_bin: 
                continue
            tb.append(th.median(t[m]))
            rb.append(th.median(r[m]))
            counts.append(float(m.sum().item()))
        if len(tb) < 2:
            # fall back to unbinned
            tb = [th.median(t)]
            rb = [th.median(r)]
            counts = [1.0]
        tb = th.stack(tb); rb = th.stack(rb)
        w = th.tensor(counts, device=t.device, dtype=t.dtype)
        A = th.stack([th.ones_like(tb), tb], dim=1)  # [n,2]
        # Weighted least squares for a, b
        W = th.diag(w / (w.mean() + eps))
        AtW = A.T @ W
        coef = th.linalg.lstsq(AtW @ A, AtW @ rb).solution
        a, b = coef[0], coef[1]
        radtrend = th.abs(b) / (rb.median() + eps)
        rad_resid = (rb - (a + b*tb)).abs().median() / (rb.median() + eps)
    else:
        # Unbinned robust L2 fit via normal equations
        A = th.stack([th.ones_like(t), t], dim=1)      # [K,2]
        coef = th.linalg.lstsq(A, r).solution
        a, b = coef[0], coef[1]
        radtrend = th.abs(b) / (r.median() + eps)
        rad_resid = (r - (a + b*t)).abs().median() / (r.median() + eps)

    # --- slab/coin tie-break (prefer axis that isn't the thin thickness) ---
    inplane_iqr = (iqr_u + iqr_v).clamp_min(eps)
    slabness = (inplane_iqr / (iqr_t + eps)).clamp_max(SLABNESS_CLAMP_MAX)  # big => slab facing this z

    # Compose score (larger is better)
    score = (
        w_elong * elongation
        - w_aniso * aniso
        - w_radtrend * radtrend
        - w_radvar * rad_resid
        - slab_penalty * slabness
    )
    return score

def choose_extrusion_axis_from_pca_v2(
    X, V, nbins=DEFAULT_NBINS, min_pts_per_bin=DEFAULT_MIN_PTS_PER_BIN,
    w_elong=W_ELONGATION, w_aniso=W_ANISOTROPY, w_radtrend=W_RAD_TREND, w_radvar=W_RAD_VARIANCE,
    slab_penalty=W_SLAB_PENALTY, use_bin_medians=True
):
    """
    X: [K,3] centered points; V: [3,3] PCA axes (columns)
    Returns: j_best, scores (tensor[3]) with *higher is better*.
    """
    scores = []
    for j in range(3):
        z = _safe_normalize(V[:, j])
        s = _axis_cyl_score_fast(
            X, z,
            nbins=nbins, min_pts_per_bin=min_pts_per_bin,
            w_elong=w_elong, w_aniso=w_aniso,
            w_radtrend=w_radtrend, w_radvar=w_radvar,
            slab_penalty=slab_penalty,
            use_bin_medians=use_bin_medians
        )
        scores.append(s)
    scores = th.stack(scores)
    j_best = int(th.argmax(scores))
    return j_best, scores

def _signed_angle_about(axis, v_from, v_to, eps=EPS_TIGHT):
    # angle to rotate v_from to v_to about unit 'axis'
    axis = _safe_normalize(axis)
    v_from = v_from - th.dot(v_from, axis) * axis
    v_to   = v_to   - th.dot(v_to,   axis) * axis
    v_from = _safe_normalize(v_from, eps)
    v_to   = _safe_normalize(v_to,   eps)
    c = th.clamp(th.dot(v_from, v_to), -1.0, 1.0)
    s = th.dot(axis, th.cross(v_from, v_to, dim=-1))
    return th.atan2(s, c)


def generate_prim_initializations(
    all_parts,
    sketcher_3d,
    nbins: int = DEFAULT_NBINS,
    q_endcaps=Q_ENDCAPS_DEFAULT,
    min_pts_per_bin: int = DEFAULT_MIN_PTS_PER_BIN,
    alpha_scale: float = 0.1,
    n_augmented: int = 9,
    min_required_pts: int = DEFAULT_MIN_REQUIRED_PTS,
    gamma_thickness: float = 0.08,
    axis_selection_version: str = "v1",
    taper_use_lower_q: bool = True,
    onion_band_frac: float = ONION_BAND_FRAC_DEFAULT,
    verbose: bool = True,
):
    """
    v4: mesh-based axis selection + robust taper & onion estimates.
    (Cleaned and lightly optimized; logic preserved.)
    """
    primitive_fits = []
    points = sketcher_3d.get_base_coords().to(th.float32)  # [N,3]
    device, dtype = points.device, th.float32

    # --- cached tiny tensors / constants (avoid reallocations in loop) ---
    eps = EPS_MEDIUM
    eps6 = EPS_LOOSE
    q25q75 = th.tensor(list(Q_IQR), device=device, dtype=dtype)
    q005   = th.tensor(Q_INNER_OUTER[0], device=device, dtype=dtype)
    q095   = th.tensor(Q_INNER_OUTER[1], device=device, dtype=dtype)

    # Endcap percentiles as tensors
    q_lo, q_hi = float(q_endcaps[0]), float(q_endcaps[1])

    # local helpers that reuse cached constants
    def _robust_span(arr: th.Tensor) -> th.Tensor:
        lo, hi = _percentiles_1d(arr, q_lo, q_hi)
        return (hi - lo).clamp_min(eps6)

    def _fmt_num(x):
        return "None" if (x is None) else f"{float(x):.4f}"

    for i, part in enumerate(all_parts):
        # --- gather points for this part (adaptive dilation if too few) ---
        mask = (part <= 0)
        P = points[mask]
        if P.shape[0] < min_required_pts:
            delta = 0.0
            n_tries = 0
            # Use resolution once
            inv_res = 1.0 / sketcher_3d.resolution
            while P.shape[0] < min_required_pts and n_tries <= 10:
                delta += inv_res
                P = points[part <= delta]
                n_tries += 1
            if P.shape[0] < min_required_pts:
                if verbose:
                    logger.warning(f"No valid points for primitive {i} despite fixes")
                continue

        # --- PCA frame & centered points ---
        # stats-only; gradients not needed
        with th.no_grad():
            center, V, S = pca_frame(P)
        X = P - center

        # --- Try mesh-based sweep axis (with robust fallback) ---
        z_dir_mesh = None
        try:
            mesh = sdf_to_mesh(part, sketcher_3d)
            z_dir_m, scores = choose_extrusion_axis_from_mesh_v1(
                mesh, version=axis_selection_version
            )
            if verbose:
                logger.debug(f"Axis scores: {scores}")
            if (z_dir_m is not None) and (not th.isnan(z_dir_m).any()) and (z_dir_m.norm() >= eps6):
                z_dir_mesh = z_dir_m
        except Exception as e:
            if verbose:
                import traceback
                traceback.print_exc()
                logger.warning(f"Mesh extraction failed for primitive {i}: {e}")

        if z_dir_mesh is not None:
            z_dir_pca = _safe_normalize(z_dir_mesh.to(device=device, dtype=dtype))
            # PCA dominant as x_seed proxy, orthogonalized to z
            main_axis = V[:, th.argmax(S)].to(device=device, dtype=dtype)
            x_seed = _safe_normalize(main_axis - th.dot(main_axis, z_dir_pca) * z_dir_pca)
        else:
            j_best, _scores = choose_extrusion_axis_from_pca_v2(
                X, V, nbins=nbins, min_pts_per_bin=min_pts_per_bin,
            )
            z_dir_pca = _safe_normalize(V[:, j_best])
            x_seed = V[:, (j_best + 1) % 3]

        # --- In-plane frame from z and x_seed orientation ---
        x_dir0, y_dir0, _ = _frame_from_z(z_dir_pca)
        x_seed_proj = _safe_normalize(x_seed - th.dot(x_seed, z_dir_pca) * z_dir_pca)
        if th.dot(x_dir0, x_seed_proj) < 0:
            x_dir0 = -x_dir0
        # y_dir0 = cross(z, x) (keep coherent)
        y_dir0 = th.cross(z_dir_pca, x_dir0, dim=0)

        # --- Project once ---
        t = X @ z_dir_pca
        u = X @ x_dir0
        v = X @ y_dir0

        # --- robust caps & in-plane sizes ---
        t_min, t_max = _percentiles_1d(t, q_lo, q_hi)
        umin, umax   = _percentiles_1d(u, q_lo, q_hi)
        vmin, vmax   = _percentiles_1d(v, q_lo, q_hi)

        start_point = center + t_min * z_dir_pca
        end_point   = center + t_max * z_dir_pca

        size_u = (SIZE_SHRINK_FACTOR * (umax - umin)).clamp_min(eps6)
        size_v = (SIZE_SHRINK_FACTOR * (vmax - vmin)).clamp_min(eps6)

        # ----------------------
        # ROUNDNESS ESTIMATION
        # ----------------------
        # stats-only; gradients not needed
        with th.no_grad():
            r = th.sqrt(u * u + v * v)
            r_q = r[r > eps6]
            if r_q.numel() > 16:
                r_std = r_q.std(unbiased=False)
                r_mean = r_q.mean()
                roundness_raw = 1.0 - (r_std / (r_mean + eps))
                roundness_val = float(th.clamp(roundness_raw, 0.0, 1.0))
            else:
                roundness_val = 0.0
        roundness = th.tensor([roundness_val], device=device, dtype=dtype)

        # ---------------
        # TAPER ESTIMATE
        # ---------------
        with th.no_grad():
            tq = th.quantile(t, q25q75)
            t_lo, t_hi = tq[0], tq[1]
            m_lo = (t <= t_lo)
            m_hi = (t >= t_hi)

            # Base (x,y) at -z/2
            if taper_use_lower_q and m_lo.any():
                base_u = SIZE_SHRINK_FACTOR * _robust_span(u[m_lo])
                base_v = SIZE_SHRINK_FACTOR * _robust_span(v[m_lo])
                base_src = "lower25%"
            else:
                base_u, base_v = size_u, size_v
                base_src = "global"

            # Top (x,y) at +z/2
            if m_hi.any():
                top_u = SIZE_SHRINK_FACTOR * _robust_span(u[m_hi])
                top_v = SIZE_SHRINK_FACTOR * _robust_span(v[m_hi])
            else:
                top_u, top_v = size_u, size_v

            # Ratios & clamp
            taper_ratio_u = (top_u / (base_u + eps)).clamp(*TAPER_RATIO_CLAMP)
            taper_ratio_v = (top_v / (base_v + eps)).clamp(*TAPER_RATIO_CLAMP)
            taper_amount_val = float(th.clamp(0.5 * (taper_ratio_u + taper_ratio_v), *TAPER_AMOUNT_CLAMP))
        taper_amount = th.tensor([taper_amount_val], device=device, dtype=dtype)

        if verbose:
            logger.debug(
                f"[taper] base={base_src}  base_uv=({float(base_u):.4f},{float(base_v):.4f})  "
                f"top_uv=({float(top_u):.4f},{float(top_v):.4f})  taper={taper_amount_val:.3f}"
            )

        # ------------------------
        # ONION AMOUNT ESTIMATE
        # ------------------------
        with th.no_grad():
            # in-plane radius proxy
            var_u = th.var(u, unbiased=False)
            var_v = th.var(v, unbiased=False)
            R = 0.5 * (th.sqrt(var_u + eps) + th.sqrt(var_v + eps))
            band = th.clamp(onion_band_frac * R, min=eps6)

            abs_u, abs_v = th.abs(u), th.abs(v)
            mu = (abs_v <= band)  # u-axis band (v small)
            mv = (abs_u <= band)  # v-axis band (u small)

            # inner/outer from band
            def _inner_outer_from_band(vals: th.Tensor):
                if vals.numel() < 16:
                    return None, None
                rmin = th.quantile(vals, q005)
                rmax = th.quantile(vals, q095)
                return rmin, rmax

            rmin_u, rmax_u = _inner_outer_from_band(abs_u[mu])
            rmin_v, rmax_v = _inner_outer_from_band(abs_v[mv])

            # Cross-fallbacks
            if rmin_u is None or rmax_u is None:
                rmin_u, rmax_u = rmin_v, rmax_v
            if rmin_v is None or rmax_v is None:
                rmin_v, rmax_v = rmin_u, rmax_u

            if (rmin_u is None) or (rmin_v is None) or (rmax_u is None) or (rmax_v is None):
                onion_val = 0.1
            else:
                full_max_r = th.min(th.stack([rmax_u, rmax_v]))
                full_min_r = th.min(th.stack([rmin_u, rmin_v]))
                onion_raw = full_min_r / (full_max_r + eps)
                onion_val = float(th.clamp(onion_raw, *ONION_AMOUNT_CLAMP))
        onion_amount = th.tensor([onion_val], device=device, dtype=dtype)

        if verbose:
            logger.debug(
                f"[onion] band={float(band):.4f} | "
                f"u-band n={int(mu.sum())} v-band n={int(mv.sum())} | "
                f"rmin_u={_fmt_num(rmin_u)} rmax_u={_fmt_num(rmax_u)}  "
                f"rmin_v={_fmt_num(rmin_v)} rmax_v={_fmt_num(rmax_v)} | "
                f"onion={onion_val:.3f}"
            )

        # --- Rotation about (refined) z and final params ---
        # Keep original intent: compute theta wrt initial x_dir0/y_dir0,
        # then refine z using end-start segment.
        x0, y0, _ = _frame_from_z(z_dir_pca)
        theta = _signed_angle_about(z_dir_pca, x0, x_dir0)

        seg = end_point - start_point
        height = th.linalg.norm(seg)
        if height.abs() < eps:
            z_dir = z_dir_pca
            height = th.tensor(eps, device=device, dtype=dtype)
        else:
            z_dir = seg / height

        x_def, y_def, _ = _frame_from_z(z_dir)
        c, s = th.cos(theta), th.sin(theta)
        x_dir = c * x_def + s * y_def
        y_dir = -s * x_def + c * y_def

        Rm = th.stack((x_dir, y_dir, z_dir), dim=1).T  # [3,3]
        axis_angle = rotation_matrix_to_axis_angle(Rm)

        scale_out = th.stack((size_u, size_v, height)).to(device=device, dtype=dtype)

        primitive_fits.append({
            "center": center,
            "rotation": axis_angle,
            "scale": scale_out,
            "taper": taper_amount,
            "onion_amount": onion_amount,
            "roundness": roundness,
        })

    return primitive_fits