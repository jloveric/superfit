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
from superfit.utils.config import AlgorithmConfig as AlgConf
OPT_EPSILON = AlgConf.OPT_EPSILON

def get_curvature_aware_iou(pred: th.Tensor,
                            target: th.Tensor,
                            curvature_weights: th.Tensor):
    """
    pred, target          : (...,) boolean tensors (0/1 or bool)
    curvature_weights     : (...,) float tensor with same shape
                            weighting value per point/voxel.

    Computes weighted IOU = sum(w * intersection) / sum(w * union)
    where w = (1 + curvature_weights).
    """
    # ensure boolean logic (in case pred/target are float)
    pred_bool = pred.bool()
    target_bool = target.bool()

    w = (1.0 + curvature_weights).float()  # (...,)

    intersection = (pred_bool & target_bool).float() * w
    union = (pred_bool | target_bool).float() * w

    iou = intersection.sum() / (union.sum() + OPT_EPSILON)
    return iou


def get_curvature_aware_iou_set(pred: th.Tensor,
                                target: th.Tensor,
                                curvature_weights: th.Tensor):
    """
    pred, target          : (B, ...) boolean (or {0,1}) tensors
    curvature_weights     : (...) float tensor
                             **same shape as each sample**, but NOT batched

    Returns:
        IOU: (B,) vector, weighted per sample using the same curvature weights.
    """
    # Ensure boolean logic
    pred_bool = pred.bool()
    target_bool = target.bool()

    # Shared weights, add 1 to incorporate curvature importance
    w = (1.0 + curvature_weights).float()  # shape (...)

    # Expand curvature weights to batch dimension (broadcast)
    # pred_bool / target_bool are (B, ...), w is (...,) → broadcasting works
    intersection = ((pred_bool & target_bool).float() * w).flatten(1).sum(-1)  # (B,)
    union        = ((pred_bool | target_bool).float() * w).flatten(1).sum(-1)  # (B,)

    return intersection / (union + OPT_EPSILON)
    
def get_iou(pred, target):
    intersection = th.logical_and(pred, target).float().sum()
    union = th.logical_or(pred, target).float().sum()
    iou = intersection / (union + OPT_EPSILON)
    return iou
    
def get_iou_set(pred, target):
    intersection = th.logical_and(pred, target).float().sum(dim=-1)
    union = th.logical_or(pred, target).float().sum(dim=-1)
    iou = intersection / (union + OPT_EPSILON)
    return iou

def get_local_iou(target, pred, mask):
    intersection = th.logical_and(pred, target).float()
    union = th.logical_or(pred, target).float()
    intersection = (intersection * mask).sum()
    union = (union * mask).sum()
    iou = intersection / (union + OPT_EPSILON)
    return iou