import os
import argparse
import traceback
import torch as th
import numpy as np
import _pickle as cPickle
from geolipi.torch_compute import recursive_evaluate, Sketcher
from superfit.algos.resfit import resfit
from superfit.utils.mesh_preprocess import process_mesh_to_sdf, cd_based_process_mesh_to_sdf
from superfit.utils.config import AlgorithmConfig as AlgConf
from superfit.utils.stats import Stats
from superfit.utils.logger import logger
from superfit.utils.constants import AOT_ARTIFACT_DIR, SAVE_DIR_BASE
from superfit.utils.io import load_toy4k_mesh_paths, load_partobjaverse_mesh_paths
import superfit.utils.config as config_options
from superfit.utils.io import to_cpu_recursive


th.set_float32_matmul_precision("medium")
th.backends.cudnn.benchmark = True
th._dynamo.config.cache_size_limit = 32
th.autograd.set_detect_anomaly(True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toys4k", choices=["toys4k", "partobjaverse"], help="Dataset to use: 'toys4k' or 'partobjaverse'")
    parser.add_argument("--start_ind", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end_ind", type=int, default=100, help="End index (exclusive)")
    parser.add_argument("--ablation", type=int, default=0, help="Ablation number")
    parser.add_argument("--fastmode", action="store_true", required=False, default=False, help="Enable fastmode")
    parser.add_argument("--overwrite", action="store_true", required=False, default=False, help="Overwrite existing save files")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR_BASE, help="Save directory")
    parser.add_argument("--aot_postfix", type=str, default="aott", help="AOT postfix")
    return parser.parse_args()


def shape_wise_resfit(input_mesh_file, save_dir, fastmode, ablation):
    """Process a single mesh file with resfit algorithm."""
    config_options.main_setting()
    config_options.set_config_ablation(ablation, fastmode=fastmode)
    
    AlgConf.AOT_ARTIFACT_FILE = os.path.join(AOT_ARTIFACT_DIR, f"aot_artifact_{args.aot_postfix}_{ablation}.pt")    
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # also save entire shape. 
    save_config_file = os.path.join(save_dir, "config.json")
    AlgConf.save_to_file(save_config_file)

    save_file = os.path.join(save_dir, "primitive_assembly.pkl")
    sketcher_3d = Sketcher(resolution=AlgConf.DATA_RESOLUTION, n_dims=3)
    
    mesh, target_sdf, cd_avg = cd_based_process_mesh_to_sdf(input_mesh_file, sketcher_3d)

    if not mesh.is_watertight:
        raise ValueError(f"------- Non Watertight Mesh -------")
    logger.info(f"CD_AVG: {cd_avg}")
    Stats.reset()
    Stats.record("input_mesh_file", input_mesh_file)
    # inner_save_file = os.path.join(save_dir, "stepwise.pkl")
    with Stats.timer("resfit_total"):
        resfit(mesh)
    cPickle.dump(to_cpu_recursive(Stats.get_dict()), open(save_file, "wb"))
    logger.info(f"Saved to {save_file}")


def main(args):
    """Main function that processes all meshes based on the selected dataset."""
    # Load mesh paths based on dataset
    if args.dataset == "toys4k":
        mesh_paths = load_toy4k_mesh_paths()
    elif args.dataset == "partobjaverse":
        mesh_paths = load_partobjaverse_mesh_paths()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    failed_indices = []
    indices = np.arange(args.start_ind, args.end_ind)
    all_cds = []
    # Process each index in the list
    for idx in indices:
        if idx >= len(mesh_paths):
            logger.warning(f"Index {idx} is out of range (max: {len(mesh_paths) - 1})")
            continue
        
        try:
        # if True:
            input_mesh_file = mesh_paths[idx]
            
            # Extract folder name from path for save directory
            # For toys4k: .../truck/truck_028/mesh.obj -> truck_028
            # For partobjaverse: .../file.glb -> file (without extension)
            if args.dataset == "toys4k":
                mesh_dir = os.path.dirname(input_mesh_file)
                folder_name = os.path.basename(mesh_dir)
            else:  # partobjaverse
                folder_name = os.path.splitext(os.path.basename(input_mesh_file))[0]
            
            # Set save directory
            save_dir = os.path.join(args.save_dir, args.dataset, f"ablation_{args.ablation}_v4", folder_name)
            
            # Check if save file exists and skip if overwrite is False
            save_file = os.path.join(save_dir, "primitive_assembly.pkl")
            if os.path.exists(save_file):
                if not args.overwrite:
                    logger.info(f"Skipping index {idx}: {folder_name} (file already exists: {save_file})")
                    continue
                else:
                    logger.info(f"Overwriting index {idx}: {folder_name} (file already exists: {save_file})")
                    # Remove everything in the save directory
                    for file in os.listdir(save_dir):
                        os.remove(os.path.join(save_dir, file))
            
            logger.info(f"Processing index {idx}: {folder_name}")
            logger.info(f"  Input: {input_mesh_file}")
            logger.info(f"  Output: {save_dir}")
            
            # Process the mesh
            shape_wise_resfit(input_mesh_file, save_dir, args.fastmode, args.ablation)
            logger.info("="*50)
            logger.info("="*50)
            logger.info(f"Successfully processed index {idx}: {folder_name}\n")
            logger.info("="*50)
            logger.info("="*50)
        except Exception as e:
            logger.error(f"Error processing index {idx}: {str(e)}")
            failed_indices.append(idx)
            traceback.print_exc()
            cPickle.dump(to_cpu_recursive(Stats.get_dict()), open(save_file.replace(".pkl", "_error.pkl"), "wb"))
            logger.info(f"Saved error stats to {save_file.replace('.pkl', '_error.pkl')}")
            continue
    
    logger.info(f"\nProcessing complete. Failed indices: {failed_indices}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
