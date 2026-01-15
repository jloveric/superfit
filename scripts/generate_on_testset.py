import os
import csv
import time
import argparse
import traceback
import torch as th
import numpy as np
import _pickle as cPickle
from dataclasses import dataclass
from geolipi.torch_compute import recursive_evaluate, Sketcher
from superfit.algos.resfit import resfit
from superfit.utils.mesh_preprocess import process_mesh_to_sdf
from superfit.utils.config import AlgorithmConfig as AlgConf
from superfit.utils.stats import Stats
from superfit.utils.logger import logger
import superfit.utils.config as config_options
th.set_float32_matmul_precision("medium")
th.backends.cudnn.benchmark = True

# Path configuration
OLD_PATH_PREFIX = "/sensei-fs-3/users/aganeshan/data/toy4k/"
NEW_PATH_PREFIX = "/users/aganesh8/data/aganesh8/data/toys4k_obj_files"
SAVE_DIR_BASE = "/users/aganesh8/data/aganesh8/projects/project_neo/outputs"
DEFAULT_CSV_FILE = "test_set_1.csv"

# Hardcoded index list - modify as needed
INDICES = np.arange(0, 100)  # Example indices, update with your desired indices

@dataclass
class ProcessArgs:
    input_mesh_file: str
    save_dir: str
    fastmode: bool
    ablation: int

def main_shape_wise(args):
    #### Set Mode.  
    input_mesh_file = args.input_mesh_file
    save_dir = args.save_dir
    config_options.main_setting()
    if args.fastmode:
        AlgConf.FastMode = True
        AlgConf.TorchCompile = True
    else:
        AlgConf.FastMode = False
        AlgConf.TorchCompile = False
    config_options.low_cost_mode()
    if args.ablation == 0:
        pass
    elif args.ablation == 1:
        AlgConf.N_SURFACE_POINTS = 150_000
        AlgConf.OPT_RESOLUTION = 32
    elif args.ablation == 2:
        AlgConf.SCALE_FACTOR_START = 13.0
        AlgConf.N_ITERS = 250
        AlgConf.SAT_PATIENCE = 100
        AlgConf.MAX_ITER = 1000
        AlgConf.OPT_LR_RATE = 0.02
    elif args.ablation == 3:
        pass
    elif args.ablation == 4:
        AlgConf.N_SURFACE_POINTS = 75_000
        AlgConf.OPT_RESOLUTION = 32
    elif args.ablation == 5:
        AlgConf.N_ITERS = 400
        AlgConf.SAT_PATIENCE = 100
        AlgConf.MAX_ITER = 1000
        AlgConf.OPT_LR_RATE = 0.01
    elif args.ablation == 6:
        config_options.low_cost_mode_v2()
    elif args.ablation == 7:
        config_options.low_cost_mode_v2()
        AlgConf.N_SURFACE_POINTS = 50_000
    
    AlgConf.AOT_ARTIFACT_FILE = os.path.join(AlgConf.AOT_ARTIFACT_DIR, f"aot_artifact_{args.ablation}.pt")    

    # TEST
    failed_indices = []
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # also save entire shape. 
    save_config_file = os.path.join(save_dir, "config.json")
    AlgConf.save_to_file(save_config_file)

    save_file = os.path.join(save_dir, "primitive_assembly.pkl")
    # if os.path.exists(save_file):
    #     continue

    sketcher_3d = Sketcher(resolution=AlgConf.DATA_RESOLUTION, n_dims=3)
    save_file_temp = os.path.join(save_dir, "resfit_prog.pkl")
    
    mesh, target_sdf = process_mesh_to_sdf(input_mesh_file, sketcher_3d)

    if not mesh.is_watertight:
        raise ValueError(f"------- Non Watertight Mesh -------")
    min_sdf = target_sdf.min().item()
    max_sdf = target_sdf.max().item()
    if min_sdf <-1.0 or max_sdf > 1.0:
        logger.error(f"Invalid SDF range: min={min_sdf:.6f}, max={max_sdf:.6f}")
        raise ValueError(f"----------- INVALID SDF RANGE -----------")
        # continue
    Stats.reset()
    with Stats.timer("resfit_total"):
        resfit(mesh, save_file=save_file_temp)
    cPickle.dump(Stats.get_dict(), open(save_file, "wb"))
    logger.info(f"Saved to {save_file}")
    if failed_indices:
        logger.warning(f"Failed indices: {failed_indices}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default=DEFAULT_CSV_FILE, help="Path to CSV file with mesh paths")
    parser.add_argument("--fastmode", action="store_true", required=False, default=False, help="Enable fastmode")
    parser.add_argument("--ablation", type=int, default=0, help="Ablation number")
    
    args = parser.parse_args()
    
    # Parse CSV file
    csv_path = args.csv_file
    if not os.path.isabs(csv_path):
        # If relative path, assume it's in the project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        csv_path = os.path.join(project_root, csv_path)
    
    mesh_cat_and_file_list = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            mesh_cat_and_file_list.append((row[0], row[1]))
    # Skip header row
    mesh_cat_and_file_list = mesh_cat_and_file_list[1:]
    
    failed_indices = []
    
    # Process each index in the list
    for idx in INDICES:
        if idx >= len(mesh_cat_and_file_list):
            logger.warning(f"Index {idx} is out of range (max: {len(mesh_cat_and_file_list) - 1})")
            continue
        
        try:
            category, mesh_file = mesh_cat_and_file_list[idx]
            
            # Replace path prefix
            input_mesh_file = mesh_file.replace(OLD_PATH_PREFIX, NEW_PATH_PREFIX)
            
            # Extract last folder name from path (e.g., truck_028 from .../truck/truck_028/mesh.obj)
            mesh_dir = os.path.dirname(input_mesh_file)
            folder_name = os.path.basename(mesh_dir)
            
            # Set save directory
            save_dir = os.path.join(SAVE_DIR_BASE, "ablation", f"ablation_{args.ablation}_v3", folder_name)
            
            logger.info(f"Processing index {idx}: {folder_name}")
            logger.info(f"  Input: {input_mesh_file}")
            logger.info(f"  Output: {save_dir}")
            args.input_mesh_file = input_mesh_file
            # Process the mesh
            process_args = ProcessArgs(input_mesh_file=input_mesh_file, save_dir=save_dir, fastmode=args.fastmode, ablation=args.ablation)
            main_shape_wise(process_args)
            logger.info(f"Successfully processed index {idx}: {folder_name}\n")
            
        except Exception as e:
            logger.error(f"Error processing index {idx}: {str(e)}")
            failed_indices.append(idx)
            traceback.print_exc()
            continue
    
    logger.info(f"\nProcessing complete. Failed indices: {failed_indices}")
