import os
import re
import ot  # POT
import cubvh
import trimesh
import torch as th
import numpy as np
from glob import glob
import _pickle as cPickle
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import geolipi.symbolic as gls
from geolipi.torch_compute import recursive_evaluate, Sketcher
from sysl.utils import recursive_sm_to_smg, recursive_gls_to_sysl
from sysl.torch_compute.evaluate_mat_expr import recursive_evaluate_mat_expr

from ..symbolic.utils import n_prims_in_expr, gather_primitives
from ..optim.utils import sample_surface_proximal_points, get_sdf_and_gradients
from ..optim.measures import get_iou, get_iou_set, get_curvature_aware_iou, get_curvature_aware_iou_set
from ..utils.mesh_sdf import sdf_to_mesh, renorm_target_sdf, target_cleanup, get_target_cubvh
from ..optim.curvature import get_points_and_weights
from ..utils.config import AlgorithmConfig as AlgConf, reset_eval_seeds
from ..utils.stats import Stats
from ..utils.logger import logger
from ..symbolic.utils import n_prims_in_expr, fetch_singular_expr_eval
from ..optim.semantic_loss import SemanticLossHolder
from .kmeans import primitive_semantic_nmi_fast


RESOLUTIONS = [128,]
CD_RES = 2048
CD_MULTIPLIER = 100.0
# Dilated IOU for avoiding thin parts. 
VOX_IOU_DILATE_THRESHOLD = 0.05
MIX_IOU_RATIO = 0.15

class MeasurePack:

    def __init__(self, measure: str, target_mesh: trimesh.Trimesh, 
                    original_mesh: trimesh.Trimesh = None, target_sdf: th.Tensor = None, len_weight: float = -1e-5):
        self.measure = measure
        self.target_mesh = target_mesh
        self.original_mesh = original_mesh
        self.target_sdf = target_sdf
        self.len_weight = len_weight
        self.target_surface_pc = None
        self.pt_target_occ = None
        self.pt_target_sdf = None
        self.curvature_weights = None
    
    def reset(self):
        self.target_surface_pc = None
        self.pt_target_occ = None
        self.pt_target_sdf = None
        self.curvature_weights = None



def eval_shape(pred_program, measure_pack, semantic_loss_holder: SemanticLossHolder=None,
                inp_mode: str = "SuperFit"):
    # Reset seeds for evaluation to ensure consistent random sampling
    reset_eval_seeds()
    
    # Geometric
    # IOU at 128. 256. 512, IOU Surface proximal
    # # Code for resolution.
    input_mesh = measure_pack.target_mesh
    with Stats.scope("evaluation"):
        for resolution in RESOLUTIONS:
            sketcher_3d = Sketcher(resolution=resolution, n_dims=3)
            target = get_target_cubvh(input_mesh, sketcher_3d)
            target = target_cleanup(target, sketcher_3d)
            target = renorm_target_sdf(target, sketcher_3d)

            # target = target_cleanup(target, sketcher_3d)
            output = recursive_evaluate(pred_program.tensor(dtype=sketcher_3d.dtype), sketcher_3d)
            target_occ = (target <= 0.0).float()
            output_occ = (output <= 0.0).float()
            iou_val = get_iou(target_occ, output_occ).item()
            Stats.record(f"iou@{resolution}", iou_val)

        surface_proximal_samples = sample_surface_proximal_points(input_mesh, n_points=AlgConf.N_SURFACE_POINTS_EVAL, jitter_sigma=AlgConf.SURFACE_ADJ_PERTURBATION_SCALE)
        surface_proximal_samples = th.from_numpy(surface_proximal_samples).float().to(sketcher_3d.device)
        surface_proximal_sdf, surface_proximal_gradients = get_sdf_and_gradients(surface_proximal_samples, input_mesh)
        surface_proximal_occ = (surface_proximal_sdf <= 0.0).float()
        surface_eval = recursive_evaluate(pred_program.tensor(dtype=sketcher_3d.dtype), sketcher_3d, coords=surface_proximal_samples)
        surface_eval_occ = (surface_eval <= 0.0).float()
        iou_gt_surface = get_iou(surface_proximal_occ, surface_eval_occ).item()
        Stats.record("iou_gt_surface_proximal", iou_gt_surface)

        # Bi directonal Surface Iou
        output = recursive_evaluate(pred_program.tensor(dtype=sketcher_3d.dtype), sketcher_3d)
        pred_mesh = sdf_to_mesh(output, sketcher_3d)
        surface_proximal_samples = sample_surface_proximal_points(pred_mesh, n_points=AlgConf.N_SURFACE_POINTS_EVAL, jitter_sigma=AlgConf.SURFACE_ADJ_PERTURBATION_SCALE)
        surface_proximal_samples = th.from_numpy(surface_proximal_samples).float().to(sketcher_3d.device)
        # Eval the program: 
        surface_eval = recursive_evaluate(pred_program.tensor(dtype=sketcher_3d.dtype), sketcher_3d, coords=surface_proximal_samples)
        surface_eval_occ = (surface_eval <= 0.0).float()
        # get GT:
        surface_proximal_sdf, surface_proximal_gradients = get_sdf_and_gradients(surface_proximal_samples, input_mesh)
        surface_proximal_occ = (surface_proximal_sdf <= 0.0).float()
        iou_pred_surface = get_iou(surface_proximal_occ, surface_eval_occ).item()
        Stats.record("iou_pred_surface_proximal", iou_pred_surface)

        # Bi directonal Surface Iou
        bi_dir_iou = (iou_gt_surface + iou_pred_surface) / 2.0
        Stats.record("bi_directional_surface_iou", bi_dir_iou)
        
        surface_samples = sample_surface_proximal_points(input_mesh, n_points=AlgConf.N_SURFACE_POINTS_EVAL, jitter_sigma=0.0)
        surface_samples = th.from_numpy(surface_samples).float().to(sketcher_3d.device)
        surface_sdf, _  = get_sdf_and_gradients(surface_samples, pred_mesh)
        mean_delta = th.mean(th.abs(surface_sdf))
        Stats.record("mean_sdf_delta", mean_delta.item())

        surface_samples_cd = sample_surface_proximal_points(input_mesh, n_points=CD_RES, jitter_sigma=0.0)
        surface_samples_cd = th.from_numpy(surface_samples_cd).float().to(sketcher_3d.device)
        # How do I get points on my surface?
        # use 512 size program and onion... and farthest point sampling.
        pred_points = sample_surface_proximal_points(pred_mesh, n_points=CD_RES, jitter_sigma=0.0)
        pred_points = th.from_numpy(pred_points).float().to(sketcher_3d.device)

        # CD 
        cd = th.cdist(pred_points, surface_samples_cd, p=2) ** 2
        cd_1 = th.min(cd, dim=1)[0]
        cd_2 = th.min(cd, dim=0)[0] 
        cd_1_mean = th.mean(cd_1)
        cd_2_mean = th.mean(cd_2)
        cd_avg = (th.mean(cd_1) + th.mean(cd_2)) / 2.0
        cd_1_val = cd_1_mean.item() * CD_MULTIPLIER
        cd_2_val = cd_2_mean.item() * CD_MULTIPLIER
        cd_avg_val = cd_avg.item() * CD_MULTIPLIER
        Stats.record(f"CD_1_MEAN@{CD_RES}", cd_1_val)
        Stats.record(f"CD_2_MEAN@{CD_RES}", cd_2_val)
        Stats.record(f"CD_AVG@{CD_RES}", cd_avg_val)
        

        hd1 = th.quantile(cd_1, 0.95, dim=-1)
        hd2 = th.quantile(cd_2, 0.95, dim=-1)
        iqr_hd = (hd1 + hd2) / 2.0
        Stats.record("iqr_hausdorff", iqr_hd.item())

        emd = emd_sinkhorn(pred_points, surface_samples_cd)
        Stats.record("emd", emd.item())


        # Overall objective
        n_prims = n_prims_in_expr(pred_program)
        Stats.record("n_prims", n_prims)
        
        
        primitives = gather_primitives(pred_program)
        # number of parameters: 
        param_cost = get_param_cost(pred_program.sympy())
        if inp_mode == "PA":
            param_cost = param_cost - (6 * len(primitives) * 4)

            prim_type_cost = len(primitives) * 1 * 1
        else:
            prim_type_cost = 0
        total_param_cost = param_cost + prim_type_cost
        Stats.record('n_params', total_param_cost)

        # Compression based on obj
        orig_mesh = measure_pack.original_mesh
        orig_cost = orig_mesh.vertices.shape[0] * 3  * 4 + orig_mesh.faces.shape[0] * 3 * 4
        compression = total_param_cost / orig_cost
        Stats.record("compression_ratio", compression)

        objective = bi_dir_iou + measure_pack.len_weight * n_prims
        Stats.record("objective", objective)
        # ----------- Additional metrics -----------
        # Overlap Amount - fraction of shape which has some overlap. 
        prim_exec_sdf = [recursive_evaluate(x.tensor(dtype=sketcher_3d.dtype), sketcher_3d) for x in primitives]
        prim_exec_sdf = th.stack(prim_exec_sdf, dim=0)
        prim_execs_occ = (prim_exec_sdf <= 0.0).float()
        overlap_amount = th.sum(prim_execs_occ, dim=0)
        overlap_amount = (overlap_amount > 1.0).float().sum() / ((overlap_amount > 0.0).float().sum() + 1e-6)
        Stats.record("overlap_amount", overlap_amount.item())

        #
        output_sdf = recursive_evaluate(pred_program.tensor(dtype=sketcher_3d.dtype), sketcher_3d)
        output_occ = (output_sdf <= 0.0).float()
        prim_exec_union_sdf = th.min(prim_exec_sdf, dim=0)[0]
        delta_sdf = th.maximum(output_sdf, - prim_exec_union_sdf)
        delta_occ = (delta_sdf <= 0.0).float()
        unoverlap_amount = th.sum(delta_occ) / (th.sum(output_occ) + 1e-6)
        Stats.record("unoverlap_amount", unoverlap_amount.item())

        # partitioning. 
        sampled_expr = fetch_singular_expr_eval(pred_program.sympy(), relaxed_eval=False)
        new_expr = recursive_sm_to_smg(sampled_expr.sympy())
        mat_expr, _ = recursive_gls_to_sysl(new_expr, ind=0, version="v1")
        outputs = recursive_evaluate_mat_expr(mat_expr.tensor(dtype=sketcher_3d.dtype), sketcher_3d)
        output_sdf, prim_ids = outputs[..., 0], outputs[..., 1]
        prim_ids[output_sdf > 0] = -1

        # mask valid primitive voxels
        valid_mask = prim_ids >= 0
        active_ids = prim_ids[valid_mask].long()     # only primitives where sdf <= 0
        if active_ids.numel() == 0:
            raise ValueError("No active primitives found.")
        # ---- Count voxels per primitive efficiently ----
        # bincount works on GPU if active_ids is CUDA and dtype long
        counts = th.bincount(active_ids)      # shape: [num_primitives], counts[i] = volume of primitive i
        # total active volume
        total_active = counts.sum().float()

        vol_percentage = counts /  (total_active + 1e-12) # th.sum(prim_execs_occ, dim=-1) / (th.sum(output_occ) + 1e-6)
        avg_vol_percentage = th.mean(vol_percentage)
        Stats.record("avg_vol_percentage", avg_vol_percentage.item())

        max_vol_percentage = th.max(vol_percentage)
        Stats.record("max_vol_percentage", max_vol_percentage.item())

        median_vol_percentage = th.median(vol_percentage)
        Stats.record("median_vol_percentage", median_vol_percentage.item())
        

        # coverage ratios per primitive
        coverage = counts.float() / (total_active + 1e-12)
        # ---- Sort descending by coverage ----
        coverage_sorted, _ = th.sort(coverage, descending=True)
        # ---- Fractional coverage curve ----
        # cumulative fraction of volume explained as we add more primitives
        frac_coverage = th.cumsum(coverage_sorted, dim=0) / coverage_sorted.numel()
        Stats.record("frac_coverage", frac_coverage.sum().item())

        jumps = th.diff(frac_coverage)                   # [P-1]
        avg_jump = jumps.mean()
        Stats.record("avg_jump", avg_jump.item())

        # get primitive ids on surface points. 
        if semantic_loss_holder is not None:
            surface_samples_inp = sample_surface_proximal_points(input_mesh, n_points=AlgConf.N_SURFACE_POINTS_EVAL, jitter_sigma=0.0)
            surface_samples_inp = th.from_numpy(surface_samples_inp).float().to(sketcher_3d.device)
            semantic_loss_holder.load_point_features_GT(surface_samples_inp)
            point_features = semantic_loss_holder.point_features
            outputs = recursive_evaluate_mat_expr(mat_expr.tensor(dtype=sketcher_3d.dtype), sketcher_3d, coords=surface_samples_inp)
            surface_output_sdf, surface_prim_ids = outputs[..., 0], outputs[..., 1]

            kmeans_res = primitive_semantic_nmi_fast(point_features, surface_prim_ids.long())
            Stats.record('nmi', kmeans_res['nmi'])
            Stats.record('num_feat_clusters', kmeans_res['num_feat_clusters'])

            res = primitive_feature_stats(point_features, surface_prim_ids.long(), k=1)
            Stats.record('SEM_primitive_purity', res['intra_var_weighted'].mean().item())
            Stats.record('SEM_primitive_knn_sep', res['knn_sep'].item())
            Stats.record('SEM_inter_avg', res['inter_avg'].item())


        # SDF Error rate. 
        real_sdf = renorm_target_sdf(output_sdf, sketcher_3d) * 2.0
        sdf_error = th.abs(real_sdf - output_sdf)
        sdf_error_rate = sdf_error.mean()
        Stats.record("sdf_error", sdf_error_rate.item())

        # ALso only outside:
        outside = (real_sdf > 0.0)
        outside_sdf_error = sdf_error[outside]
        outside_sdf_error_rate = outside_sdf_error.mean()
        Stats.record("outside_sdf_error", outside_sdf_error_rate.item())
    
def get_param_cost(in_expr):
    current_cost = 0
    args = in_expr.sympy().get_args()
    for arg in args:
        if isinstance(arg, gls.GLFunction):
            current_cost += get_param_cost(arg)
        else:
            current_cost += len(arg) * 4
    return current_cost
    
def get_recon_measure(in_expr, sketcher, measure_pack: MeasurePack, convert_to_scalar=True):
    # Reset seeds for evaluation to ensure consistent random sampling
    reset_eval_seeds()
    in_expr = in_expr.tensor(dtype=sketcher.dtype)
    output_sdf = recursive_evaluate(in_expr, sketcher, relaxed_eval=False)
    target_mesh = measure_pack.target_mesh
    target_sdf = measure_pack.target_sdf
    measure = measure_pack.measure
    
    target_occ, target_pc = None, None
    if measure in ["iou", "tversky"]:
        target_occ = (target_sdf <= 0.0)
        target_occ = target_occ[None, ...]
    elif measure in ["cd", "iqr_hausdorff"]:
        target_pc = sample_surface_proximal_points(target_mesh, n_points=CD_RES, jitter_sigma=0.0)
        target_pc = th.from_numpy(target_pc).float().to(sketcher.device)
    elif measure in ["surface_iou", "surface_tversky"]:
        if measure_pack.target_surface_pc is None:
            BVH = cubvh.cuBVH(target_mesh.vertices, target_mesh.faces)
            surface_sampled_points = sample_surface_proximal_points(target_mesh, n_points=AlgConf.N_SURFACE_POINTS_EVAL, jitter_sigma=AlgConf.SURFACE_ADJ_PERTURBATION_SCALE)
            surface_sampled_points = th.from_numpy(surface_sampled_points).float().to(sketcher.device)
            # perturbations = (th.rand_like(surface_sampled_points)  - 0.5) * AlgConf.SURFACE_ADJ_PERTURBATION_SCALE
            surface_adj_points = surface_sampled_points.clone()#  + perturbations#[..., None] * sampled_normals
            measure_pack.target_surface_pc = surface_adj_points

            target_sdf, _, _ = BVH.signed_distance(surface_adj_points, return_uvw=False, mode="watertight")
            target_occ = (target_sdf <= 0.0)
            measure_pack.pt_target_occ = target_occ
            measure_pack.pt_target_sdf = target_sdf
        else:
            surface_adj_points = measure_pack.target_surface_pc
            target_occ = measure_pack.pt_target_occ
            target_sdf = measure_pack.pt_target_sdf
        output_sdf = recursive_evaluate(in_expr, sketcher, coords=surface_adj_points)
    elif measure == "surface_iou_wt_curvature":
        if measure_pack.target_surface_pc is None:
            BVH = cubvh.cuBVH(target_mesh.vertices, target_mesh.faces)
            surface_sampled_points, curvature_weights, _ = get_points_and_weights(target_mesh, sketcher, n_points=AlgConf.N_SURFACE_POINTS_EVAL)
            curvature_weights = AlgConf.CURVATURE_WEIGHTS_SCALE * curvature_weights

            perturbations = (th.rand_like(surface_sampled_points)  - 0.5) * AlgConf.SURFACE_ADJ_PERTURBATION_SCALE
            surface_adj_points = surface_sampled_points.clone() + perturbations#[..., None] * sampled_normals
            # surface_sampled_sdf = recompute_sdf_from_BVH(surface_adj_points, BVH, mode="watertight")
            measure_pack.target_surface_pc = surface_adj_points

            target_sdf, _, _ = BVH.signed_distance(surface_adj_points, return_uvw=False, mode="watertight")
            target_occ = (target_sdf <= 0.0)
            measure_pack.pt_target_occ = target_occ
            measure_pack.curvature_weights = curvature_weights
            measure_pack.pt_target_sdf = target_sdf
        else:
            surface_adj_points = measure_pack.target_surface_pc
            target_occ = measure_pack.pt_target_occ
            curvature_weights = measure_pack.curvature_weights
            target_sdf = measure_pack.pt_target_sdf

        output_sdf = recursive_evaluate(in_expr, sketcher, coords=surface_adj_points)

    elif measure == "surface_iou_wt_curvature_and_vox_iou":
        vox_target_occ = (target_sdf <= VOX_IOU_DILATE_THRESHOLD)
        vox_target_occ = vox_target_occ[None, ...]
        if measure_pack.target_surface_pc is None:
            BVH = cubvh.cuBVH(target_mesh.vertices, target_mesh.faces)
            surface_sampled_points, curvature_weights, _ = get_points_and_weights(target_mesh, sketcher, n_points=AlgConf.N_SURFACE_POINTS_EVAL)
            curvature_weights = AlgConf.CURVATURE_WEIGHTS_SCALE * curvature_weights

            perturbations = (th.rand_like(surface_sampled_points)  - 0.5) * AlgConf.SURFACE_ADJ_PERTURBATION_SCALE
            surface_adj_points = surface_sampled_points.clone() + perturbations#[..., None] * sampled_normals
            # surface_sampled_sdf = recompute_sdf_from_BVH(surface_adj_points, BVH, mode="watertight")
            measure_pack.target_surface_pc = surface_adj_points

            pt_target_sdf, _, _ = BVH.signed_distance(surface_adj_points, return_uvw=False, mode="watertight")
            pt_target_occ = (pt_target_sdf <= 0.0)
            measure_pack.pt_target_occ = pt_target_occ
            measure_pack.curvature_weights = curvature_weights
            measure_pack.pt_target_sdf = pt_target_sdf
        else:
            surface_adj_points = measure_pack.target_surface_pc
            pt_target_occ = measure_pack.pt_target_occ
            curvature_weights = measure_pack.curvature_weights
            pt_target_sdf = measure_pack.pt_target_sdf

        pt_output_sdf = recursive_evaluate(in_expr, sketcher, coords=surface_adj_points)
        
    if measure == "tversky":
        hard_output = (output_sdf <= 0.0)
        out_measure = get_tversky_iou(hard_output, target_occ)
    elif measure == "iou":
        hard_output = (output_sdf <= 0.0)
        out_measure = get_iou(hard_output, target_occ)
    elif measure  == "surface_iou":
        hard_output = (output_sdf <= 0.0)
        out_measure = get_iou(hard_output, target_occ)
    elif measure == "surface_tversky":
        hard_output = (output_sdf <= 0.0)
        out_measure = get_tversky_iou(hard_output, target_occ)
    elif measure == "surface_iou_wt_curvature":
        hard_output = (output_sdf <= 0.0)
        out_measure = get_curvature_aware_iou(hard_output, target_occ, curvature_weights=curvature_weights)
    elif measure == "surface_iou_wt_curvature_and_vox_iou":
        pt_hard_output = (pt_output_sdf <= 0.0)
        vox_hard_output = (output_sdf <= VOX_IOU_DILATE_THRESHOLD)
        out_measure_1 = get_curvature_aware_iou(pt_hard_output, pt_target_occ, curvature_weights=curvature_weights)
        out_measure_2 = get_iou(vox_hard_output, vox_target_occ)
        out_measure = out_measure_1 * (1-MIX_IOU_RATIO) + out_measure_2 * MIX_IOU_RATIO
    elif measure == "cd":
        try:
            out_measure = get_cd_measure(output_sdf, target_pc, sketcher)
        except Exception as e:
            logger.error(f"Error evaluating CD: {e}")
            out_measure = 0.0
    elif measure == "iqr_hausdorff":
        try:
            out_measure = get_iqr_hausdorff_measure(output_sdf, target_pc, sketcher)
        except Exception as e:
            logger.error(f"Error evaluating IQR Hausdorff: {e}")
            out_measure = 0.0
    if convert_to_scalar and isinstance(measure, th.Tensor):
        out_measure = out_measure.item()
    return out_measure

def get_recon_measure_packed(expr_set, sketcher, measure_pack: MeasurePack):
    # Reset seeds for evaluation to ensure consistent random sampling
    reset_eval_seeds()

    all_execs = []
    pt_all_execs = None
    pt_target_sdf = None
    target_mesh = measure_pack.target_mesh
    if measure_pack.measure in ["surface_iou", "surface_tversky"]:
        if measure_pack.target_surface_pc is None:
            BVH = cubvh.cuBVH(target_mesh.vertices, target_mesh.faces)
            surface_sampled_points = sample_surface_proximal_points(target_mesh, n_points=AlgConf.N_SURFACE_POINTS_EVAL, jitter_sigma=AlgConf.SURFACE_ADJ_PERTURBATION_SCALE)
            surface_sampled_points = th.from_numpy(surface_sampled_points).float().to(sketcher.device)
            # perturbations = (th.rand_like(surface_sampled_points)  - 0.5) * AlgConf.SURFACE_ADJ_PERTURBATION_SCALE
            surface_adj_points = surface_sampled_points.clone()#  + perturbations#[..., None] * sampled_normals
            measure_pack.target_surface_pc = surface_adj_points

            target_sdf, _, _ = BVH.signed_distance(surface_adj_points, return_uvw=False, mode="watertight")
            target_occ = (target_sdf <= 0.0)
            measure_pack.pt_target_occ = target_occ
            measure_pack.pt_target_sdf = target_sdf
        else:
            surface_adj_points = measure_pack.target_surface_pc
            target_occ = measure_pack.pt_target_occ
            target_sdf = measure_pack.pt_target_sdf
        for expr in expr_set:
            cur_exec = recursive_evaluate(expr.tensor(dtype=sketcher.dtype), sketcher, coords=surface_adj_points)
            all_execs.append(cur_exec)
    elif measure_pack.measure == "surface_iou_wt_curvature":
        if measure_pack.target_surface_pc is None:
            BVH = cubvh.cuBVH(target_mesh.vertices, target_mesh.faces)
            surface_sampled_points, curvature_weights, _ = get_points_and_weights(target_mesh, sketcher, n_points=AlgConf.N_SURFACE_POINTS_EVAL)
            curvature_weights = AlgConf.CURVATURE_WEIGHTS_SCALE * curvature_weights

            perturbations = (th.rand_like(surface_sampled_points)  - 0.5) * AlgConf.SURFACE_ADJ_PERTURBATION_SCALE
            surface_adj_points = surface_sampled_points.clone() + perturbations#[..., None] * sampled_normals
            # surface_sampled_sdf = recompute_sdf_from_BVH(surface_adj_points, BVH, mode="watertight")
            measure_pack.target_surface_pc = surface_adj_points

            target_sdf, _, _ = BVH.signed_distance(surface_adj_points, return_uvw=False, mode="watertight")
            target_occ = (target_sdf <= 0.0)
            measure_pack.pt_target_occ = target_occ
            measure_pack.curvature_weights = curvature_weights
            measure_pack.pt_target_sdf = target_sdf
        else:
            surface_adj_points = measure_pack.target_surface_pc
            target_occ = measure_pack.pt_target_occ
            curvature_weights = measure_pack.curvature_weights
            target_sdf = measure_pack.pt_target_sdf

        for expr in expr_set:
            cur_exec = recursive_evaluate(expr.tensor(dtype=sketcher.dtype), sketcher, coords=surface_adj_points)
            all_execs.append(cur_exec)
    elif measure_pack.measure == "surface_iou_wt_curvature_and_vox_iou":

        if measure_pack.target_surface_pc is None:
            BVH = cubvh.cuBVH(target_mesh.vertices, target_mesh.faces)
            surface_sampled_points, curvature_weights, _ = get_points_and_weights(target_mesh, sketcher, n_points=AlgConf.N_SURFACE_POINTS_EVAL)
            curvature_weights = AlgConf.CURVATURE_WEIGHTS_SCALE * curvature_weights

            perturbations = (th.rand_like(surface_sampled_points)  - 0.5) * AlgConf.SURFACE_ADJ_PERTURBATION_SCALE
            surface_adj_points = surface_sampled_points.clone() + perturbations#[..., None] * sampled_normals
            # surface_sampled_sdf = recompute_sdf_from_BVH(surface_adj_points, BVH, mode="watertight")
            measure_pack.target_surface_pc = surface_adj_points

            pt_target_sdf, _, _ = BVH.signed_distance(surface_adj_points, return_uvw=False, mode="watertight")
            target_occ = (pt_target_sdf <= 0.0)
            measure_pack.pt_target_occ = target_occ
            measure_pack.curvature_weights = curvature_weights
            measure_pack.pt_target_sdf = pt_target_sdf
        else:
            surface_adj_points = measure_pack.target_surface_pc
            target_occ = measure_pack.pt_target_occ
            curvature_weights = measure_pack.curvature_weights
            pt_target_sdf = measure_pack.pt_target_sdf

        pt_all_execs = []
        for expr in expr_set:
            cur_exec = recursive_evaluate(expr.tensor(dtype=sketcher.dtype), sketcher, coords=surface_adj_points)
            pt_all_execs.append(cur_exec)
        
        pt_all_execs = th.stack(pt_all_execs, dim=0)
        
        for expr in expr_set:
            cur_exec = recursive_evaluate(expr.tensor(dtype=sketcher.dtype), sketcher, relaxed_eval=False)
            all_execs.append(cur_exec)
        target_sdf = measure_pack.target_sdf
        
    else:
        for expr in expr_set:
            cur_exec = recursive_evaluate(expr.tensor(dtype=sketcher.dtype), sketcher, relaxed_eval=False)
            all_execs.append(cur_exec)
        target_sdf = measure_pack.target_sdf
    all_execs = th.stack(all_execs, dim=0)
    measure = measure_pack.measure
    return get_recon_measure_set(all_execs, sketcher, target_mesh,  target_sdf, curvature_weights=measure_pack.curvature_weights, measure=measure,
                                pt_output_sdfs=pt_all_execs, pt_target_sdf=pt_target_sdf)

def get_recon_measure_set(output_sdfs, sketcher, target_mesh, target_sdf, curvature_weights=None, measure="iou", pt_output_sdfs=None, pt_target_sdf=None):
    # Generate required targets
    target_occ, target_pc = None, None
    if measure in ["iou", "tversky"]:
        target_occ = (target_sdf <= 0.0)
        target_occ = target_occ[None, ...]
    elif measure in ["surface_iou", "surface_tversky"]:
        target_occ = (target_sdf <= 0.0)
        target_occ = target_occ[None, ...]
    elif measure == "surface_iou_wt_curvature":
        target_occ = (target_sdf <= 0.0)
        target_occ = target_occ[None, ...]
    elif measure in ["cd", "iqr_hausdorff"]:
        target_pc = sample_surface_proximal_points(target_mesh, n_points=CD_RES, jitter_sigma=0.0)
        target_pc = th.from_numpy(target_pc).float().to(sketcher.device)

    if measure == "tversky":
        hard_output = (output_sdfs.detach() <= 0.0)
        output = get_tversky_iou_set(hard_output, target_occ)
    elif measure == "iou":
        hard_output = (output_sdfs.detach() <= 0.0)
        output = get_iou_set(hard_output, target_occ)
    elif measure == "surface_iou":
        hard_output = (output_sdfs.detach() <= 0.0)
        output = get_iou_set(hard_output, target_occ)
    elif measure == "surface_tversky":
        hard_output = (output_sdfs.detach() <= 0.0)
        output = get_tversky_iou_set(hard_output, target_occ)
    elif measure == "surface_iou_wt_curvature":
        hard_output = (output_sdfs.detach() <= 0.0)
        output = get_curvature_aware_iou_set(hard_output, target_occ, curvature_weights=curvature_weights)
    elif measure == "surface_iou_wt_curvature_and_vox_iou":
        hard_output = (output_sdfs.detach() <= VOX_IOU_DILATE_THRESHOLD)
        target_occ = (target_sdf <= VOX_IOU_DILATE_THRESHOLD)
        pt_hard_output = (pt_output_sdfs.detach() <= 0.0)
        pt_target_occ = (pt_target_sdf <= 0.0)
        output_1 = get_curvature_aware_iou_set(pt_hard_output, pt_target_occ, curvature_weights=curvature_weights)
        output_2 = get_iou_set(hard_output, target_occ)
        output = output_1 * (1-MIX_IOU_RATIO) + output_2 * MIX_IOU_RATIO

    elif measure == "cd":
        output = get_cd_measure_set(output_sdfs, target_pc, sketcher)
    elif measure == "iqr_hausdorff":
        output = get_iqr_hausdorff_measure_set(output_sdfs, target_pc, sketcher)
    return output

def get_cd_measure_set(output_sdfs, target_pc, sketcher):

    output_pcs = []
    for output_sdf in output_sdfs:
        try:
            output_mesh = sdf_to_mesh(output_sdf.detach(), sketcher)
            output_pc = sample_surface_proximal_points(output_mesh, n_points=CD_RES, jitter_sigma=0.0)
            output_pc = np.asarray(output_pc)
            output_pc = th.from_numpy(output_pc).float().to(sketcher.device)
            output_pcs.append(output_pc)
        except Exception as e:
            logger.error(f"Error evaluating CD: {e}")
            output_pcs.append(target_pc * 0.0 + 100)
    output_pcs = th.stack(output_pcs, dim=0)   
    cd = th.cdist(output_pcs, target_pc[None, ...], p=2) ** 2
    cd_1 = th.min(cd, dim=-2)[0]
    cd_2 = th.min(cd, dim=-1)[0]
    cd_avg = (th.mean(cd_1, dim=-1) + th.mean(cd_2, dim=-1)) / 2.0
    output = 1 - (cd_avg * CD_MULTIPLIER)
    return output

def get_iqr_hausdorff_measure_set(output_sdfs, target_pc, sketcher):
    output_pcs = []
    for output_sdf in output_sdfs:
        try:
            output_mesh = sdf_to_mesh(output_sdf.detach(), sketcher)
            output_pc = sample_surface_proximal_points(output_mesh, n_points=CD_RES, jitter_sigma=0.0)
            output_pc = np.asarray(output_pc)
            output_pc = th.from_numpy(output_pc).float().to(sketcher.device)
            output_pcs.append(output_pc)
        except Exception as e:
            logger.error(f"Error evaluating IQR Hausdorff: {e}")
            output_pcs.append(target_pc * 0.0 + 100)
    output_pcs = th.stack(output_pcs, dim=0)
    cd = th.cdist(output_pcs, target_pc[None, ...], p=2)
    cd_1 = th.min(cd, dim=-2)[0]
    cd_2 = th.min(cd, dim=-1)[0]
    hd1 = th.quantile(cd_1, 0.95, dim=-1)
    hd2 = th.quantile(cd_2, 0.95, dim=-1)
    iqr_hd = (hd1 + hd2) / 2.0
    measure = 1 - (iqr_hd * 10.0)
    return measure

def get_tversky_iou(hard_output, target_occ, alpha=0.7, beta=0.3):
    TP = (hard_output & target_occ).sum()
    FP = (hard_output & ~target_occ).sum()
    FN = (~hard_output & target_occ).sum()
    tversky_iou = (TP) / (TP + alpha * FN + beta * FP + 1e-9)
    return tversky_iou


def get_tversky_iou_set(hard_output, target_occ, alpha=0.7, beta=0.3):
    TP = (hard_output & target_occ).sum(dim=-1)
    FP = (hard_output & ~target_occ).sum(dim=-1)
    FN = (~hard_output & target_occ).sum(dim=-1)
    tversky_iou = (TP) / (TP + alpha * FN + beta * FP + 1e-9)
    return tversky_iou


def get_iqr_hausdorff_measure(output_sdf, target_pc, sketcher):
    output_mesh = sdf_to_mesh(output_sdf.detach(), sketcher)
    output_pc = sample_surface_proximal_points(output_mesh, n_points=CD_RES, jitter_sigma=0.0)
    output_pc = np.asarray(output_pc)
    output_pc = th.from_numpy(output_pc).float().to(sketcher.device)
    cd = th.cdist(output_pc, target_pc, p=2)
    cd_1 = th.min(cd, dim=1)[0]
    cd_2 = th.min(cd, dim=0)[0]
    hd1 = th.quantile(cd_1, 0.95, dim=-1)
    hd2 = th.quantile(cd_2, 0.95, dim=-1)
    iqr_hd = (hd1 + hd2) / 2.0
    measure = 1 - (iqr_hd * 10.0)
    return measure

def get_cd_measure(output_sdf, target_pc, sketcher):
    output_mesh = sdf_to_mesh(output_sdf.detach(), sketcher)
    output_pc = sample_surface_proximal_points(output_mesh, n_points=CD_RES, jitter_sigma=0.0)
    output_pc = np.asarray(output_pc)
    output_pc = th.from_numpy(output_pc).float().to(sketcher.device)
    cd = th.cdist(output_pc, target_pc, p=2) ** 2
    cd_1 = th.min(cd, dim=1)[0]
    cd_2 = th.min(cd, dim=0)[0]
    cd_avg = (th.mean(cd_1) + th.mean(cd_2)) / 2.0
    measure = 1 - (cd_avg * CD_MULTIPLIER)
    return measure



def get_latest_prog_file_index(PROG_DIR, ind, max_stage=4):
    """
    Finds the file matching pattern:
        <PROG_DIR>/<ind>_shape_per_step_<num>.pkl
    and returns the one with the highest <num>.
    """
    pattern = os.path.join(PROG_DIR, f"{ind}_shape_per_step_*.pkl")
    files = glob(pattern)
    if not files:
        return None  # No match found

    # Extract numeric suffix and sort
    def extract_index(f):
        m = re.search(r'_shape_per_step_(\d+)\.pkl$', f)
        return int(m.group(1)) if m else -1

    files.sort(key=extract_index)

    if len(files) > max_stage:
        index = max_stage
    else:
        index = len(files) - 1
    return index

def measure_time_till_stage(PROG_DIR, ind, max_stage=4):
    """
    Measures the time taken to reach the given stage.
    """
    potential_files = [f"{ind}_shape_per_step_{i}.pkl" for i in range(max_stage + 1)]
    all_files = os.listdir(PROG_DIR)
    files = [x for x in potential_files if x in all_files]
    if not files:
        return None  # No match found
    total_time = 0.0
    for i in range((max_stage + 1)):
        if i < len(files):
            file = files[i]
            info = cPickle.load(open(os.path.join(PROG_DIR, file), "rb"))
            total_time += info["time_taken_overall"]
    return total_time

@th.no_grad()
def earth_mover_distance_exact(pc1: th.Tensor, pc2: th.Tensor) -> th.Tensor:
    """
    Exact Earth Mover's Distance (Wasserstein-1) between two point clouds
    using Hungarian (linear sum assignment).

    Inputs:
        pc1 : (N,3) float tensor, already on CUDA
        pc2 : (N,3) float tensor, already on CUDA

    Output:
        scalar float tensor (on CUDA) = mean matching distance
    """
    assert pc1.shape == pc2.shape
    assert pc1.ndim == 2 and pc1.shape[1] == 3

    N = pc1.shape[0]

    # pairwise cost matrix (Euclidean distances), computed on GPU
    # cost: [N, N]
    cost = th.cdist(pc1, pc2, p=2)        # CUDA tensor

    # linear_sum_assignment requires CPU numpy
    cost_cpu = cost.detach().cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(cost_cpu)

    # matched cost
    matched_cost = cost[row_ind, col_ind].mean()

    return matched_cost

@th.no_grad()
def emd_exact_pot(pc1, pc2):
    # pc1/pc2: (N,3) CUDA tensors
    N = pc1.shape[0]

    M = th.cdist(pc1, pc2)  # [N, N] on CUDA

    a = th.ones(N, device=pc1.device) / N
    b = th.ones(N, device=pc1.device) / N

    # POT will internally convert to CPU, run Hungarian
    M_np = M.detach().cpu().numpy()
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()

    G = ot.emd(a_np, b_np, M_np)     # optimal transport plan (Hungarian)
    emd = (G * M_np).sum()

    return th.tensor(emd, device=pc1.device, dtype=th.float32)

def emd_sinkhorn(pc1, pc2, eps=0.01):
    N = pc1.shape[0]
    M = th.cdist(pc1, pc2)
    a = th.ones(N, device=pc1.device) / N
    b = th.ones(N, device=pc2.device) / N
    G = ot.sinkhorn(a, b, M, reg=eps)  # stays on GPU
    emd = (G * M).sum()
    return emd

@th.no_grad()
def primitive_feature_stats(point_features: th.Tensor,
                            prim_ids: th.Tensor,
                            k: int = 1,
                            return_knn_margin: bool = False,
                            eps: float = 1e-12):
    """
    Compute semantic cluster statistics for primitive assignments.
    NOW with cluster-size weighting for final intra_var and knn_sep.
    """
    X = F.normalize(point_features, dim=1)    # (M, D)
    P = int(th.max(prim_ids).item()) + 1

    # ----------------------------------------------------
    # primitive point counts
    # ----------------------------------------------------
    counts = th.bincount(prim_ids, minlength=P).clamp(min=1)   # (P,)
    w = counts.float() / counts.sum()                          # weights (P,)

    # ----------------------------------------------------
    # primitive centroid features
    # ----------------------------------------------------
    sums = th.zeros(P, X.shape[1], device=X.device, dtype=X.dtype)
    sums.index_add_(0, prim_ids, X)
    mu = sums / counts.unsqueeze(1)
    mu = F.normalize(mu, dim=1)              # (P, D)

    # ----------------------------------------------------
    # intra-cluster cosine distance: 1 - dot(x, mu)
    # ----------------------------------------------------
    mu_per_point = mu[prim_ids]              # (M, D)
    cos_sim = (X * mu_per_point).sum(dim=1)
    intra_dist = 1.0 - cos_sim

    intra_sums = th.zeros(P, device=X.device, dtype=X.dtype)
    intra_sums.index_add_(0, prim_ids, intra_dist)
    intra_var = intra_sums / counts          # per-cluster (P,)

    # ✅ NEW: cluster-size–weighted intra variance (global scalar)
    intra_var_weighted = (intra_var * w).sum()

    # ----------------------------------------------------
    # inter-cluster centroid distances
    # ----------------------------------------------------
    dmat = 1.0 - (mu @ mu.T)                 # cosine distance (P,P)

    # mask diagonal
    mask = ~th.eye(P, dtype=th.bool, device=X.device)
    inter_avg = dmat[mask].mean()

    # ----------------------------------------------------
    # kNN centroid separation
    # ----------------------------------------------------
    dmat_masked = dmat + th.eye(P, device=X.device) * 1e9
    k = min(max(1, k), P - 1)
    knn_vals, _ = th.topk(dmat_masked, k=k, largest=False, dim=1)
    knn_per_cluster = knn_vals.mean(dim=1)   # (P,)

    # ✅ NEW: cluster-size–weighted kNN separation (global scalar)
    knn_sep_weighted = (knn_per_cluster * w).sum()

    result = {
        "mu": mu,
        "counts": counts,
        "intra_var": intra_var,                       # per cluster (unchanged)
        "intra_var_weighted": intra_var_weighted,     # ✅ new (scalar)
        "dmat": dmat,
        "inter_avg": inter_avg,
        "knn_sep": knn_sep_weighted,                  # ✅ now weighted
        "knn_per_cluster": knn_per_cluster,           # expose if needed
    }

    if return_knn_margin:
        knn_margin = knn_per_cluster / (intra_var + eps)
        result["knn_margin"] = (knn_margin * w).sum()

    return result
