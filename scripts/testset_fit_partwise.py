"""
Partwise fitting: for each mesh, load instance ids, split into parts, normalize each part
to unit cube (saving the transform), close open meshes, run resfit per part, save each part separately.
Requires partobjaverse-style instance id files (per-face .npy).
"""
import os
import argparse
import traceback
import torch as th
import numpy as np
import trimesh
import _pickle as cPickle
import distinctipy
from geolipi.torch_compute import Sketcher
from superfit.algos.resfit import resfit
from superfit.utils.mesh_preprocess import (
    extract_mesh,
    normalize_to_unit_cube,
    normalize_to_unit_cube_with_transform,
    mesh_with_segments_to_part_targets,
    open_mesh_to_closed_mesh,
    renorm_target_sdf,
    target_cleanup_v2,
    sdf_to_mesh,
    get_target_cubvh,
)
from superfit.utils.config import AlgorithmConfig as AlgConf
import superfit.utils.config as config_options
from superfit.utils.stats import Stats
from superfit.utils.logger import logger
from superfit.utils.constants import AOT_ARTIFACT_DIR, SAVE_DIR_BASE, PARTOBJAVERSE_INSTANCE_DIR
from superfit.utils.io import load_partobjaverse_mesh_paths, to_cpu_recursive


th.set_float32_matmul_precision("medium")
th.backends.cudnn.benchmark = True
th._dynamo.config.cache_size_limit = 32
th.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_ind", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end_ind", type=int, default=100, help="End index (exclusive)")
    parser.add_argument("--ablation", type=int, default=0, help="Ablation number")
    parser.add_argument("--fastmode", action="store_true", default=False, help="Enable fastmode")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing save files")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR_BASE, help="Save directory")
    parser.add_argument("--aot_postfix", type=str, default="aott", help="AOT postfix")
    return parser.parse_args()


def load_mesh_and_instance_ids(input_mesh_file):
    """Load mesh (no normalization) and per-face instance ids from partobjaverse paths."""
    # mesh = extract_mesh(input_mesh_file)
    mesh = trimesh.load(input_mesh_file, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh = normalize_to_unit_cube(mesh)
    mesh_name = os.path.splitext(os.path.basename(input_mesh_file))[0]
    instance_id_path = os.path.join(PARTOBJAVERSE_INSTANCE_DIR, f"{mesh_name}.npy")
    if not os.path.exists(instance_id_path):
        raise FileNotFoundError(f"Instance ids not found: {instance_id_path}")
    instance_ids = np.load(instance_id_path)
    if len(instance_ids) != len(mesh.faces):
        raise ValueError(
            f"Instance id count ({len(instance_ids)}) != face count ({len(mesh.faces)}) for {input_mesh_file}"
        )
    return mesh, instance_ids


def part_wise_resfit_for_mesh(
    input_mesh_file,
    save_dir_base,
    fastmode,
    ablation,
    overwrite,
    aot_postfix,
):
    """
    Load one mesh, split by instance ids into parts, then for each part:
    normalize to unit cube (save transform in Stats), close mesh, run resfit, save to save_dir_base/part_<i>/.
    """

    config_options.main_setting()
    config_options.set_config_ablation(ablation, fastmode=fastmode)
    AlgConf.AOT_ARTIFACT_FILE = os.path.join(AOT_ARTIFACT_DIR, f"aot_artifact_{aot_postfix}_{ablation}.pt")

    mesh, instance_ids = load_mesh_and_instance_ids(input_mesh_file)

    n_colors = np.max(instance_ids) + 1
    colors = distinctipy.get_colors(n_colors)
    # distinctipy returns floats in [0,1]; build (n_faces, 4) RGBA uint8
    colors_uint8 = (np.array(colors) * 255).astype(np.uint8)

    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    face_colors[:, 3] = 255
    for i in range(n_colors):
        face_colors[instance_ids == i, :3] = colors_uint8[i]
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=face_colors)
    part_meshes = mesh_with_segments_to_part_targets(mesh, instance_ids)
    n_parts = len(part_meshes)
    logger.info(f"Split into {n_parts} parts")

    sketcher_3d = Sketcher(resolution=AlgConf.DATA_RESOLUTION, n_dims=3)

    for part_idx in range(n_parts):
        part_mesh = part_meshes[part_idx]
        save_dir = os.path.join(save_dir_base, f"part_{part_idx:03d}")
        save_file = os.path.join(save_dir, "primitive_assembly.pkl")

        if os.path.exists(save_file) and not overwrite:
            logger.info(f"Skipping part {part_idx}: already exists {save_file}")
            continue

        os.makedirs(save_dir, exist_ok=True)
        save_config_file = os.path.join(save_dir, "config.json")
        AlgConf.save_to_file(save_config_file)

        Stats.reset()
        Stats.record("input_mesh_file", input_mesh_file)
        Stats.record("part_index", part_idx)
        Stats.record("n_parts_total", n_parts)

        try:
            # Normalize part to unit cube and save transform (translation first, then scale)

            # Close open mesh via SDF/winding
            closed_mesh = open_mesh_to_closed_mesh(part_mesh, sketcher_3d)
            closed_mesh_norm = closed_mesh
            # closed_mesh_norm, translation, scale = normalize_to_unit_cube_with_transform(closed_mesh.copy(), margin=0.9)
            # Stats.record("part_normalize_translation", np.asarray(translation, dtype=np.float64))
            # Stats.record("part_normalize_scale", scale)
            
            target_sdf = get_target_cubvh(closed_mesh_norm, sketcher_3d, mode="raystab")
            target_sdf = renorm_target_sdf(target_sdf, sketcher_3d)
            target_sdf = target_cleanup_v2(target_sdf, sketcher_3d)
            target_sdf = renorm_target_sdf(target_sdf, sketcher_3d)
            mesh = sdf_to_mesh(target_sdf, sketcher_3d)
            
            if not mesh.is_watertight:
                logger.warning(f"Part {part_idx} mesh not watertight after closing; continuing anyway")

            with Stats.timer("resfit_total"):
                resfit(mesh, original_mesh=None, original_annotations=None)

            cPickle.dump(to_cpu_recursive(Stats.get_dict()), open(save_file, "wb"))
            logger.info(f"Saved part {part_idx} to {save_file}")
        except Exception as e:
            logger.error(f"Error processing part {part_idx}: {e}")
            traceback.print_exc()
            err_file = save_file.replace(".pkl", "_error.pkl")
            cPickle.dump(to_cpu_recursive(Stats.get_dict()), open(err_file, "wb"))
            logger.info(f"Saved error stats to {err_file}")


def main(args):
    mesh_paths = load_partobjaverse_mesh_paths()
    indices = np.arange(args.start_ind, args.end_ind)
    failed_indices = []

    for idx in indices:
        if idx >= len(mesh_paths):
            logger.warning(f"Index {idx} out of range (max {len(mesh_paths) - 1})")
            continue

        input_mesh_file = mesh_paths[idx]
        folder_name = os.path.splitext(os.path.basename(input_mesh_file))[0]
        save_dir_base = os.path.join(args.save_dir, "partobjaverse_partwise", f"ablation_{args.ablation}_v4", folder_name)

        # Check if any part already exists (and skip whole shape if no overwrite)
        part_0_file = os.path.join(save_dir_base, "part_000", "primitive_assembly.pkl")
        if os.path.exists(part_0_file) and not args.overwrite:
            logger.info(f"Skipping index {idx}: {folder_name} (part outputs exist)")
            continue

        logger.info(f"Processing index {idx}: {folder_name}")
        logger.info(f"  Input: {input_mesh_file}")
        logger.info(f"  Output: {save_dir_base}")

        try:
            part_wise_resfit_for_mesh(
                input_mesh_file,
                save_dir_base,
                args.fastmode,
                args.ablation,
                args.overwrite,
                args.aot_postfix,
            )
            logger.info("=" * 50)
            logger.info(f"Successfully processed index {idx}: {folder_name}")
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"Error processing index {idx}: {e}")
            failed_indices.append(idx)
            traceback.print_exc()

    logger.info(f"\nPartwise fitting complete. Failed indices: {failed_indices}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
