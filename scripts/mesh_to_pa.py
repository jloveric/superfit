import os
import time
import argparse
import torch as th
import _pickle as cPickle
from geolipi.torch_compute import recursive_evaluate, Sketcher
from superfit.utils.stats import Stats
from superfit.utils.logger import logger
from superfit.algos.resfit import resfit
import superfit.utils.config as config_options
from superfit.utils.mesh_preprocess import process_mesh_to_sdf
from superfit.utils.config import AlgorithmConfig as AlgConf, initialize_seeds
# Over parameterize - even more - see what happens - all on the base version. 

th.set_float32_matmul_precision("medium")
th.backends.cudnn.benchmark = True


def main_shape_wise(args):
        
    input_mesh_file = args.input_mesh_file
    save_dir = args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Configuration setup. 
    config_options.main_setting()
    config_options.low_cost_mode()
    config_options.low_cost_mode_v2()
    if args.fastmode:
        AlgConf.FastMode = True
        AlgConf.TorchCompile = True
    else:
        AlgConf.FastMode = False
        AlgConf.TorchCompile = False
    config_options.check_config()
    save_config_file = os.path.join(save_dir, "config.json")
    AlgConf.save_to_file(save_config_file)
    
    # Initialize seeds after config setup
    initialize_seeds()
    
    save_file = os.path.join(save_dir, f"final.pkl")
    save_file_temp = os.path.join(save_dir, f"resfit_stepwise.pkl")

    sketcher_3d = Sketcher(resolution=AlgConf.DATA_RESOLUTION, n_dims=3)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mesh_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--profile_path", type=str, required=False, default=None)
    parser.add_argument("--fastmode", action="store_true", required=False, default=False)

    args = parser.parse_args()
    if args.profile_path is not None:
        import cProfile
        import pstats
        def main():
            main_shape_wise(args)
        cProfile.run("main()", args.profile_path)
        pstats.Stats(args.profile_path).strip_dirs().sort_stats("cumtime").print_stats(50)
    else:
        main_shape_wise(args)
# python scripts/mesh_to_pa.py --input_mesh_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/airplane/airplane_007/mesh.obj --save_dir ../outputs/basic