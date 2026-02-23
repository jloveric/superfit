import numpy as np
import torch as th
import os
import random
import torch.nn.functional as F
from sysl.torch_compute.mat_combinators import sdf_geom_only_smooth_union, sdf_smooth_union
from .param_conversion import params_from_variables
from ..utils.config import AlgorithmConfig as AlgConf
from ..utils.constants import SEMANTIC_LOC
from ..utils.logger import logger

# if partfield is not installed, import the following. 
try:
    from partfield.config import default_argument_parser, setup
    from lightning.pytorch import seed_everything
    from partfield.model.PVCNN.encoder_pc import sample_triplane_feat
    from partfield.model_trainer_pvcnn_only_demo import Model
except ImportError:
    class Model:
        def __init__(self, cfg):
            pass
        def custom_predict(self, pts):
            return None, None, None
    pass
    

class SemanticLossHolder:
    def __init__(self):
        # Load the model.
        # GET GT Point features.

        parser = default_argument_parser()
        args = parser.parse_args([
            "-c", 
            os.path.join(SEMANTIC_LOC, "configs/final/demo.yaml"), 
            "--opts", 
            "continue_ckpt",
            os.path.join(SEMANTIC_LOC, "model/model_objaverse.ckpt")
            ])
        cfg = setup(args, freeze=False)
        seed_everything(cfg.seed)

        # Use config seed for consistency with rest of codebase
        self.model =  CustomModel(cfg, ckpt_path=cfg.continue_ckpt, device="cuda")
        self.use_point_mask = False

    def load_point_features_GT(self, points, subsample=False):
        n_points = points.shape[0]
        if subsample and n_points > 100_000:
            # subsample
            desired_ratio = 100_000 / n_points
            self.point_mask = th.rand(n_points) < desired_ratio
            self.use_point_mask = True
        else:
            self.point_mask = th.ones(n_points, dtype=th.bool)
        with th.no_grad():
            feat, center, scale = self.get_point_features(points[self.point_mask])
        self.point_features = feat

    def get_point_features(self, points):

        feat, center, scale = self.model.custom_predict(points)
        return feat, center, scale

    def instantiate_prim_features(self, points, sketcher,
                   compiled_func_relaxed, variable_list, tensor_list, has_temp, temperature):
        
        
        # Get transformed parameters
        transformed_params = params_from_variables(variable_list, tensor_list)
        # transformed_params, new_type_annotation = inject_temp_param_compiled(transformed_params, temperature, type_annotation)
        if has_temp:
            transformed_params.append(temperature)

        all_coords = sketcher.make_homogenous_coords(points)

        primitive_sdfs, output_sdf = compiled_func_relaxed(all_coords, transformed_params)
        
        
        feature_size = self.point_features.shape[1]
        n_primitives = primitive_sdfs.shape[0]
        primitive_features = 0.01*th.randn(n_primitives, feature_size, device=points.device)
        primitive_sdfs = primitive_sdfs[..., None]
        point_associations = self.point_association(primitive_sdfs, transformed_params)
        point_associations = point_associations[..., 1].squeeze(-1)
        for i in range(n_primitives):
            point_mask = point_associations == i
            mean_value = self.point_features[point_mask].mean(dim=0)
            if mean_value.isnan().any():
                if i > 0:
                    logger.warning(f"Mean value is nan for primitive {i}")
                    mean_value = self.point_features[:i-1].mean(dim=0)
                else:
                    mean_value = th.zeros_like(self.point_features[0])
            primitive_features[i] = mean_value
        
        self.primitive_features = th.autograd.Variable(primitive_features.detach().clone(), requires_grad=True)
    
    def update_param_groups(self, param_groups):
        param_groups.append({
            'params': self.primitive_features,
            'weight_decay': AlgConf.WEIGHT_DECAY,
            'lr': AlgConf.OPT_LR_RATE * 10.0,
        })
        return param_groups
        
        # Now, based on primitive_sdfs, and SU param -> directly associate points with primitives. 

    def point_association(self, primitive_sdfs, transformed_params):

        su_vals = transformed_params[1]
        K = primitive_sdfs.shape[0]

        prim_with_ids = []
        for i in range(K):
            ind_field = th.zeros_like(primitive_sdfs[i]) + i
            updated_prim = th.cat([primitive_sdfs[i], ind_field], dim=-1)
            prim_with_ids.append(updated_prim)

        out = prim_with_ids[0]
        for i in range(1, K):
            k_reshaped = su_vals[i-1].unsqueeze(-1)
            out = sdf_geom_only_smooth_union(out, prim_with_ids[i], k_reshaped)
        return out

    def feature_forward(self, primitive_sdfs, transformed_param):

        su_vals = transformed_param[1]
        K = primitive_sdfs.shape[0]
        primitive_sdfs_unsqueezed = primitive_sdfs.unsqueeze(-1)
        prim_with_ids = []
        for i in range(K):
            cur_prim_feature = self.primitive_features[i:i+1].clone()
            ind_field = cur_prim_feature.repeat(primitive_sdfs[i].shape[0], 1)
            updated_prim = th.cat([primitive_sdfs_unsqueezed[i], ind_field, ], dim=-1)
            prim_with_ids.append(updated_prim)

        out = prim_with_ids[0]
        for i in range(1, K):
            k_reshaped = su_vals[i-1].unsqueeze(-1)
            out = sdf_geom_only_smooth_union(out, prim_with_ids[i], k_reshaped)
            # out = sdf_smooth_union(out, prim_with_ids[i], k_reshaped)
        return out


    def get_semantic_loss(self, primitive_sdfs, output_sdf, transformed_params):
        if self.use_point_mask:
            primitive_sdfs = primitive_sdfs[:, self.point_mask]
            output_sdf = output_sdf[self.point_mask]
        # Combine them using 
        mask = (output_sdf <= AlgConf.LOSS_BAND)
        if not mask.sum() > 0:
            return th.tensor(0., device=output_sdf.device), th.tensor(0., device=output_sdf.device)



        feature_forward = self.feature_forward(primitive_sdfs, transformed_params)

        pred_features = feature_forward[..., 1:]
        # Ground truth features: shape (N, F)
        # Predicted features: shape (N, F)

        # Ground truth features: shape (N, F)
        target = self.point_features

        # Normalize along feature dimension
        # pred_norm = F.normalize(pred_features, p=2, dim=-1)
        # target_norm = F.normalize(target, p=2, dim=-1)
        pred_norm   = F.normalize(pred_features, p=2, dim=-1, eps=1e-8)
        target_norm = F.normalize(target,        p=2, dim=-1, eps=1e-8)

        # Cosine similarity: (N,)
        cos_sim = th.sum(pred_norm * target_norm, dim=-1)
        cos_sim = cos_sim[mask]

        if cos_sim.numel() == 0:
            loss = th.tensor(0., device=cos_sim.device)
        else:
            loss = 1.0 - cos_sim.mean()
        # Add a loss that is the similarlity between the primitive features
        primitive_features_loss = cosine_separation_loss(self.primitive_features.clone())

        return loss, primitive_features_loss


def cosine_separation_loss(feats: th.Tensor):
    N = feats.size(0)
    if N <= 1: 
        return th.tensor(0., device=feats.device)
    feats_norm = F.normalize(feats, dim=1, eps=1e-8)
    cos_sim = feats_norm @ feats_norm.T
    mask = ~th.eye(N, dtype=th.bool, device=feats.device)
    vals = cos_sim[mask]
    return th.tensor(0., device=feats.device) if vals.numel()==0 else vals.mean()

def _cosine_separation_loss(feats: th.Tensor):
    """
    feats: (N, D) tensor of feature vectors (e.g., 20 x 448)
    Returns a scalar loss that penalizes positive cosine similarity
    (encourages vectors to be orthogonal / diverse).
    """

    # Normalize each feature to unit length
    feats_norm = F.normalize(feats, dim=1, eps=1e-8)
    # feats_norm = F.normalize(feats, dim=1)      # (N, D)

    # Cosine similarity matrix: (N, N)
    cos_sim = feats_norm @ feats_norm.T

    # Remove diagonal entries (self-similarity = 1)
    N = cos_sim.size(0)
    mask = ~th.eye(N, dtype=th.bool, device=feats.device)
    loss = cos_sim[mask].mean()                 # average cosine similarity of all pairs

    return loss

class CustomModel(Model):   # Inherit from the lightning model
    def __init__(self, cfg, ckpt_path=None, device="cuda"):
        super().__init__(cfg)
        # self.device = device

        # Load weights
        if ckpt_path:
            ckpt = th.load(ckpt_path, map_location=device, weights_only=False)
            self.load_state_dict(ckpt["state_dict"], strict=False)

        self.eval()
        self.to(device)

    @th.no_grad()
    def custom_predict(self, pts: th.Tensor):
        """
        pts: torch.Tensor (N, 3) in original coordinates
        returns: torch.Tensor (N, C) predicted part features
        """

        assert pts.ndim == 2 and pts.shape[1] == 3, "Input must be (N,3)"

        # -----------------------------
        # 1. Preprocess like Demo_Dataset
        # -----------------------------
        bbmin = pts.min(0).values
        bbmax = pts.max(0).values
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * 0.9 / (bbmax - bbmin).max()

        pts_norm = (pts - center) * scale                # still (N,3)
        pts_norm = pts_norm.unsqueeze(0).to(self.device) # → (1,N,3)

        # -----------------------------
        # 2. EXACT inference path from predict_step()
        # -----------------------------
        
        pc_feat = self.pvcnn(pts_norm, pts_norm)             # (1, C_low, H, W)
        planes = self.triplane_transformer(pc_feat)          # (1, C_total, H, W)
        sdf_planes, part_planes = th.split(planes, [64, planes.shape[2] - 64], dim=2)

        feat = sample_triplane_feat(part_planes, pts_norm)   # (1, N, C)
        feat = feat.squeeze(0)                               # → (N, C)

        return feat, center, scale         
