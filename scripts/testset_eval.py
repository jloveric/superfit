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

Evaluate primitive assemblies under an input directory.

For each folder: load best program from pkl, run sampling_based_pruning to pick the best
stochastic variant, run eval_shape, save *_eval.pkl in the folder and collect metrics.
At the end save a summary (means ± optional per-instance record) and print a markdown table.

Supports --start_ind / --end_ind for parallel runs; summary filename includes that range.
"""
import os
import re
import numpy as np
import argparse
import traceback
from typing import Optional, Dict, List, Any

import _pickle as cPickle
import torch as th
import geolipi.symbolic as gls

from superfit.algos.eval_tools import MeasurePack, eval_shape
from superfit.algos.prune import sampling_based_pruning
from superfit.utils.config import AlgorithmConfig as AlgConf
from superfit.utils.mesh_preprocess import extract_mesh, cd_based_process_mesh_to_sdf, normalize_to_unit_cube
from superfit.utils.mesh_sdf import get_target_cubvh, renorm_target_sdf
from geolipi.torch_compute import Sketcher
from superfit.utils.stats import Stats
from superfit.utils.logger import logger
from superfit.utils.io import get_best_expr
import superfit.utils.config as config_options
from superfit.optim.semantic_loss import SemanticLossHolder


def get_best_pkl_in_folder(folder_path: str) -> Optional[str]:
    """Path to pkl to use: primitive_assembly.pkl if present, else first non-*_textured.pkl."""
    p = os.path.join(folder_path, "primitive_assembly.pkl")
    if os.path.isfile(p):
        return p
    for name in sorted(os.listdir(folder_path)):
        if name.endswith(".pkl") and not name.endswith("_textured.pkl"):
            return os.path.join(folder_path, name)
    return None

def _numeric_stats(flat: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric evaluation metrics from flat stats (strip iter_X. prefix)."""
    out = {}
    for k, v in flat.items():
        if isinstance(v, dict) and v.get("GLFunction"):
            continue
        if isinstance(v, (int, float)):
            out[re.sub(r"^iter_\d+\.", "", k)] = float(v)
    return out


def eval_one_folder(pkl_path: str, eval_last_only: bool, save_per_instance: bool, semantic_loss_holder: Optional[SemanticLossHolder] = None) -> Optional[Dict[str, float]]:
    """
    Load pkl, build mesh + measure_pack. For the chosen iter(s): load best program,
    run sampling_based_pruning, then eval_shape.
    If save_per_instance is True, save *_eval.pkl in the folder.
    Returns numeric metrics dict (last iter's metrics).
    """
    folder = os.path.dirname(pkl_path)
    basename = os.path.basename(pkl_path)

    with open(pkl_path, "rb") as f:
        info = cPickle.load(f)

    mesh_file = info.get("input_mesh_file")
    n_iters = info.get("n_iters", 0)
    if not mesh_file or n_iters == 0:
        raise ValueError("pkl missing input_mesh_file or n_iters")

    config_options.main_setting()
    sketcher = Sketcher(resolution=AlgConf.DATA_RESOLUTION, n_dims=3)
    prune_sketcher = Sketcher(resolution=AlgConf.PRUNE_RESOLUTION, dtype=th.float16, n_dims=3)

    mesh, target_sdf, _ = cd_based_process_mesh_to_sdf(mesh_file, sketcher)
    target_sdf = get_target_cubvh(mesh, prune_sketcher, mode="watertight")
    target_sdf = renorm_target_sdf(target_sdf, prune_sketcher)
    input_mesh = extract_mesh(mesh_file)
    input_mesh = normalize_to_unit_cube(input_mesh)

    measure_pack = MeasurePack(
        measure=AlgConf.PRUNE_METRIC,
        target_mesh=mesh,
        original_mesh=input_mesh,
        target_sdf=target_sdf,
        len_weight=AlgConf.MPS_LEN_WEIGHT,
    )

    iter_indices = [n_iters - 1] if eval_last_only else list(range(n_iters))
    Stats.reset()
    for i in iter_indices:
        with Stats.scope(f"iter_{i}"):
            in_expr = get_best_expr(info, i)
            best_program, _, _, _ = sampling_based_pruning(in_expr, sketcher, measure_pack)
            eval_shape(best_program, measure_pack, semantic_loss_holder)

    flat = Stats.get_dict()
    metrics = _numeric_stats(flat)

    if save_per_instance:
        out_pkl = os.path.join(folder, basename.replace(".pkl", "_eval.pkl") if basename.endswith(".pkl") else basename + "_eval.pkl")
        cPickle.dump(flat, open(out_pkl, "wb"))
        logger.info("Saved per-instance eval to %s", out_pkl)

    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate primitive assemblies under input_path")
    p.add_argument("--input_path", type=str, required=True, help="Root with one subdir per shape (each has .pkl)")
    p.add_argument("--eval", type=str, default="last", choices=["last", "all_iters"], help="Eval last iter or all iters")
    p.add_argument("--save_per_instance_metrics", action="store_true", help="Save *_eval.pkl inside each folder")
    p.add_argument("--start_ind", type=int, default=None, help="Start index (inclusive) for subdirs")
    p.add_argument("--end_ind", type=int, default=None, help="End index (exclusive) for subdirs")
    p.add_argument("--include_semantic", action="store_true", help="Include semantics in the evaluation")
    args = p.parse_args()
    if not os.path.isdir(args.input_path):
        raise FileNotFoundError(f"Not a directory: {args.input_path}")
    return args


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input_path)

    subdirs = sorted(
        d for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d)) and not d.startswith(".")
    )
    if not subdirs:
        logger.warning("No subdirs in %s", input_path)
        return

    start = args.start_ind if args.start_ind is not None else 0
    end = args.end_ind if args.end_ind is not None else len(subdirs)
    subdirs = subdirs[start:end]
    eval_last_only = args.eval == "last"

    if args.include_semantic:
        semantic_loss_holder = SemanticLossHolder()
    else:
        semantic_loss_holder = None

    all_metrics = []
    failed = []
    for name in subdirs:
        folder = os.path.join(input_path, name)
        pkl_path = get_best_pkl_in_folder(folder)
        if not pkl_path:
            logger.warning("Skipping %s: no pkl", name)
            continue
        try:
            metrics = eval_one_folder(pkl_path, eval_last_only, args.save_per_instance_metrics, semantic_loss_holder)
            if metrics is not None:
                all_metrics.append(metrics)
        except Exception as e:
            logger.error("Error %s: %s", name, e)
            failed.append(name)
            traceback.print_exc()

    if failed:
        logger.warning("Failed: %s", failed)
    if not all_metrics:
        logger.warning("No successful evals; no summary.")
        return

    # record: metric_name -> list of values
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())
    record = {}
    for k in sorted(all_keys):
        vals = [m[k] for m in all_metrics if k in m and m[k] is not None]
        if vals:
            record[k] = vals
            
    means = {k: np.nanmean(v) for k, v in record.items()}

    summary = {"means": means, "record": record}
    out_name = f"eval_summary_start{start}_end{end}.pkl"
    out_path = os.path.join(input_path, out_name)
    cPickle.dump(summary, open(out_path, "wb"))
    logger.info("Saved %s", out_path)

    print("\n| Metric | Mean |")
    print("|--------|------|")
    for k in sorted(means.keys()):
        print(f"| {k} | {means[k]:.6f} |")
    print()

    md_path = os.path.join(input_path, out_name.replace(".pkl", ".md"))
    with open(md_path, "w") as f:
        f.write("| Metric | Mean |\n|--------|------|\n")
        for k in sorted(means.keys()):
            f.write(f"| {k} | {means[k]:.6f} |\n")
    logger.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
