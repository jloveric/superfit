import torch
import numpy as np
import numpy as np
import torch
import trimesh
import torch.nn.functional as F
import numpy as np
import faiss


def fast_kmeans(features, K=20, use_pca=False, d_reduced=64, use_cosine=True):
    """
    features: torch.Tensor (N, F) on CPU or GPU
    K: number of clusters
    use_pca: bool — reduce dimensionality before clustering (default False)
    d_reduced: PCA dim if enabled
    use_cosine: bool — KMeans with cosine similarity (normalize features)
    
    returns:
        labels: (N,) int
        centroids: (K, dim)
    """

    feats = features.detach().float()
    N, D = feats.shape

    # Move to CPU for FAISS (fastest interoperability)
    feats = feats.cpu().numpy()

    # ---------------------------------------
    # (optional) PCA
    # ---------------------------------------
    if use_pca:
        pca = faiss.PCAMatrix(D, d_reduced)
        pca.train(feats)
        feats = pca.apply_py(feats)
        D = d_reduced

    # ---------------------------------------
    # cosine distance → normalize vectors
    # ---------------------------------------
    if use_cosine:
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
        feats = feats / norms
        metric = faiss.METRIC_INNER_PRODUCT  # maximizing similarity
        spherical = True
    else:
        metric = faiss.METRIC_L2
        spherical = False

    # ---------------------------------------
    # FAISS GPU K-means
    # ---------------------------------------
    clus = faiss.Kmeans(
        d=D,
        k=K,
        niter=20,
        verbose=False,
        gpu=True,
        spherical=spherical      # aligns with cosine clustering
    )

    clus.train(feats)

    # nearest centroid assignment
    _, labels = clus.index.search(feats, 1)
    labels = labels.reshape(-1)

    return labels, clus.centroids

@torch.no_grad()
def primitive_semantic_nmi_fast(
    point_features: torch.Tensor,
    prim_ids: torch.Tensor,
    Ks=(10, 20, 30, 40),
    eps=1e-12,
):
    feats = F.normalize(point_features, dim=1)
    prim_ids = prim_ids.view(-1)

    inertia_list = []
    label_list = []

    # ----------------------------------------------------
    # STEP 1: run k-means for all candidate Ks
    # ----------------------------------------------------
    for K in Ks:
        labels, centroids = fast_kmeans(feats, K=K, use_pca=False, use_cosine=True)

        labels_t = torch.from_numpy(labels).to(feats.device)
        centroids_t = torch.from_numpy(centroids).to(feats.device)

        # cosine distance cost
        dists = 1 - (feats * centroids_t[labels_t]).sum(dim=1)
        inertia = float(dists.mean().item())

        inertia_list.append(inertia)
        label_list.append(labels_t)

    inertia_list = torch.tensor(inertia_list)

    # ----------------------------------------------------
    # STEP 2: select K via elbow (inertia gain)
    # ----------------------------------------------------
    gains = (inertia_list[:-1] - inertia_list[1:]) / inertia_list[:-1]
    # if gains < threshold, stop increasing K
    threshold = 0.05
    elbow_idx = 0
    for i, g in enumerate(gains):
        if g < threshold:
            elbow_idx = i
            break
        elbow_idx = i + 1

    best_labels = label_list[elbow_idx]
    best_K = Ks[elbow_idx]

    # ----------------------------------------------------
    # STEP 3: compute NMI
    # ----------------------------------------------------
    C = best_K
    P = int(prim_ids.max().item()) + 1

    T = torch.zeros(P, C, device=prim_ids.device)
    T.index_put_((prim_ids, best_labels), torch.ones_like(prim_ids, dtype=torch.float32), accumulate=True)
    T = T / (T.sum() + eps)

    P_A = T.sum(dim=1, keepdim=True)
    P_C = T.sum(dim=0, keepdim=True)

    valid = T > 0
    MI = (T[valid] * torch.log((T[valid] + eps) / (P_A @ P_C + eps)[valid])).sum()

    H_A = -(P_A[P_A > 0] * torch.log(P_A[P_A > 0])).sum()
    H_C = -(P_C[P_C > 0] * torch.log(P_C[P_C > 0])).sum()

    NMI = (2 * MI) / (H_A + H_C + eps)

    return {
        "feat_ids": best_labels,
        "nmi": float(NMI.item()),
        "num_feat_clusters": int(best_K),
        "all_inertia": inertia_list.tolist(),
        "chosen_index": elbow_idx,
    }