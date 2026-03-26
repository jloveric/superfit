"""
Run texture fitting on all shapes under an input directory at once.

Expects <input-dir>/<folder-name>/ with at least one .pkl file per folder
(uses primitive_assembly.pkl if present, otherwise the best/first .pkl).
Runs the same texture optimization as fit_texture.py for each folder.
"""
import os
import argparse
import traceback
from typing import Optional

import torch as th
import _pickle as cPickle
from geolipi.torch_compute import Sketcher, recursive_evaluate

from superfit.utils.logger import logger
from superfit.utils.mesh_preprocess import extract_mesh, normalize_to_unit_cube
from superfit.utils.mesh_sdf import sdf_to_mesh
from superfit.utils.stats import Stats
from superfit.utils.io import get_best_expr
from superfit.mat_opt.optim import optimize_color
from superfit.mat_opt.utils import get_material_expr, save_html_mat_expr
from superfit.utils.editing import save_edit_mode_html
from superfit.utils.mesh_preprocess import cd_based_process_mesh_to_sdf
from superfit.algos.eval_tools import MeasurePack
from superfit.algos.prune import sampling_based_pruning
import superfit.utils.config as config_options
from superfit.utils.config import AlgorithmConfig as AlgConf


def get_best_pkl_in_folder(folder_path: str) -> Optional[str]:
    """Return path to the pkl file to use in this folder.
    Prefers primitive_assembly.pkl; otherwise returns the first .pkl found.
    """
    preferred = os.path.join(folder_path, "primitive_assembly.pkl")
    if os.path.isfile(preferred):
        return preferred
    for name in sorted(os.listdir(folder_path)):
        if name.endswith(".pkl") and not name.endswith("_textured.pkl"):
            return os.path.join(folder_path, name)
    return None


def _fit_texture_one(input_file: str, save_html: bool, save_edit_html: bool, ablation: int) -> None:
    """Run texture optimization for a single pkl file (same logic as fit_texture.py)."""
    config_options.main_setting()
    config_options.set_config_ablation(ablation, fastmode=True)
    sketcher_3d = Sketcher(resolution=AlgConf.DATA_RESOLUTION, n_dims=3)

    logger.info("Loading pkl file: %s", input_file)
    info_dict = cPickle.load(open(input_file, "rb"))

    n_iters = info_dict.get("n_iters", 0)
    if n_iters == 0:
        raise ValueError("n_iters not found or is 0 in pkl file")
    iter_idx = n_iters - 1
    base_geometric_expr = get_best_expr(info_dict, iter_idx)

    # Do a pruning pass here. 
    input_mesh_file = info_dict['input_mesh_file']
    target_mesh, target_sdf_prune, cd_avg = cd_based_process_mesh_to_sdf(input_mesh_file, sketcher_3d)
    measure_pack = MeasurePack(
        measure=AlgConf.PRUNE_METRIC,
        target_mesh=target_mesh,
        original_mesh=target_mesh,
        target_sdf=target_sdf_prune,
        len_weight=AlgConf.MPS_LEN_WEIGHT
    )
    base_geometric_expr, best_recon_measure, best_n_prim, best_obj = sampling_based_pruning(base_geometric_expr, sketcher_3d, measure_pack)

    input_mesh_file = info_dict.get("input_mesh_file", None)
    if input_mesh_file is None:
        raise ValueError("input_mesh_file not found in pkl file")

    logger.info("Loading and processing mesh...")
    global_mesh = extract_mesh(input_mesh_file)
    global_mesh = normalize_to_unit_cube(global_mesh)

    output_sdf = recursive_evaluate(base_geometric_expr.tensor(), sketcher_3d)
    sample_mesh = sdf_to_mesh(output_sdf, sketcher_3d)
    material_expr = get_material_expr(base_geometric_expr)

    logger.info("Input mesh file: %s, using iteration: %s", input_mesh_file, iter_idx)
    Stats.reset()
    logger.info("Starting texture optimization...")
    logger.info("Sample mesh vertices: %s, global mesh vertices: %s", len(sample_mesh.vertices), len(global_mesh.vertices))

    with Stats.scope("texture_optimization"):
        optimized_program, optimized_obj = optimize_color(
            global_mesh, sample_mesh, material_expr, sketcher_3d, verbose=True
        )

    optimized_obj_val = optimized_obj.item() if isinstance(optimized_obj, th.Tensor) else optimized_obj
    Stats.record("optimized_obj", optimized_obj_val)
    logger.info("Optimization complete. Final objective: %.6f", optimized_obj_val)

    pkl_dir = os.path.dirname(input_file)
    pkl_basename = os.path.basename(input_file)
    output_basename = f"{pkl_basename}_textured.pkl"
    output_file = os.path.join(pkl_dir, output_basename)

    Stats.record("input_mesh_file", input_mesh_file)
    Stats.record("iter_idx", iter_idx)
    Stats.record("base_geometric_expr", base_geometric_expr.sympy().state(), log=False)
    Stats.record("material_expr", optimized_program.state(), log=False)

    logger.info("Saving texture optimization results to: %s", output_file)
    cPickle.dump(Stats.get_dict(), open(output_file, "wb"))

    if save_html:
        save_file_name = os.path.join(pkl_dir, "best_program_textured.html")
        save_html_mat_expr(material_expr, sketcher_3d, save_file_name)
        logger.info("Saved HTML to %s", save_file_name)
    if save_edit_html:
        save_file_name = os.path.join(pkl_dir, "best_edit_mode_textured.html")
        save_edit_mode_html(material_expr, sketcher_3d, save_file_name, is_textured=True)
        logger.info("Saved HTML to %s", save_file_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit texture on all shapes under input_dir (one subdir per shape, each with .pkl)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Base directory containing one subdir per shape, each with e.g. primitive_assembly.pkl",
    )
    parser.add_argument("--save_html", action="store_true", help="Save HTML per shape")
    parser.add_argument("--save_edit_html", action="store_true", help="Save edit-mode HTML per shape")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run texture fit even if *_textured.pkl already exists",
    )
    parser.add_argument(
        "--start_ind",
        type=int,
        default=None,
        help="Start index (inclusive) for subdirs to process; use with --end_ind for parallel runs",
    )
    parser.add_argument(
        "--end_ind",
        type=int,
        default=None,
        help="End index (exclusive) for subdirs to process; use with --start_ind for parallel runs",
    )
    parser.add_argument("--ablation", type=int, default=0, help="Ablation number for config selection.")
    args = parser.parse_args()
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    return args


def main(args: argparse.Namespace) -> None:
    input_dir = os.path.abspath(args.input_dir)
    subdirs = sorted([
        d
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith(".")
    ])
    if not subdirs:
        logger.warning("No subdirectories found under %s", input_dir)
        return

    if args.start_ind is not None or args.end_ind is not None:
        start = args.start_ind if args.start_ind is not None else 0
        end = args.end_ind if args.end_ind is not None else len(subdirs)
        subdirs = subdirs[start:end]
        logger.info("Processing subdirs from index %s to %s (%d folders)", start, end, len(subdirs))

    failed = []
    for folder_name in subdirs:
        folder_path = os.path.join(input_dir, folder_name)
        pkl_path = get_best_pkl_in_folder(folder_path)
        if not pkl_path:
            logger.warning("Skipping %s: no .pkl found", folder_name)
            continue

        textured_name = os.path.basename(pkl_path).replace(".pkl", "_textured.pkl")
        textured_path = os.path.join(folder_path, textured_name)
        if os.path.isfile(textured_path) and not args.overwrite:
            logger.info("Skipping %s: already textured (%s)", folder_name, textured_path)
            continue

        logger.info("Fitting texture for: %s (from %s)", folder_name, pkl_path)
        try:
            _fit_texture_one(pkl_path, args.save_html, args.save_edit_html, args.ablation)
            logger.info("Done: %s", folder_name)
        except Exception as e:
            logger.error("Error fitting texture for %s: %s", folder_name, e)
            failed.append(folder_name)
            traceback.print_exc()
        logger.info("=" * 50)

    if failed:
        logger.warning("Failed folders: %s", failed)
    else:
        logger.info("Texture fitting finished for all %d folders.", len(subdirs))


if __name__ == "__main__":
    args = parse_args()
    main(args)
