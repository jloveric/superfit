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
from superfit.utils.mesh_preprocess import cd_based_process_mesh_to_sdf
from superfit.utils.io import to_cpu_recursive
from superfit.utils.constants import AOT_ARTIFACT_DIR
from superfit.utils.io import save_html
from superfit.utils.config import AlgorithmConfig as AlgConf, initialize_seeds
from superfit.utils.editing import save_edit_mode_html
# Over parameterize - even more - see what happens - all on the base version. 

th.set_float32_matmul_precision("medium")
th.backends.cudnn.benchmark = True


def main_shape_wise(args):
        
    input_file = args.input_file
    save_dir = args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Configuration setup. 
    config_options.main_setting()
    if args.fastmode:
        AlgConf.FastMode = True
        AlgConf.TorchCompile = True
    else:
        AlgConf.FastMode = False
        AlgConf.TorchCompile = False
    AlgConf.PRIM_TYPE = "VarAxisSF"
    AlgConf.OPT_POST_PRUNE = True
    AlgConf.BIDIR = True
    save_config_file = os.path.join(save_dir, "config.json")
    AlgConf.save_to_file(save_config_file)
    # Assuming we are running the baseline version.
    AlgConf.AOT_ARTIFACT_FILE = os.path.join(AOT_ARTIFACT_DIR, f"aot_artifact_{0}.pt")    
    
    
    # Initialize seeds after config setup
    initialize_seeds()
    
    save_file = os.path.join(save_dir, f"final_content.pkl")
    save_file_temp = os.path.join(save_dir, f"stepwise.pkl")

    sketcher_3d = Sketcher(resolution=AlgConf.DATA_RESOLUTION, n_dims=3)
    mesh, target_sdf = cd_based_process_mesh_to_sdf(input_file, sketcher_3d)

    if not mesh.is_watertight:
        raise ValueError(f"------- Non Watertight Mesh -------")
    min_sdf = target_sdf.min().item()
    max_sdf = target_sdf.max().item()
    if min_sdf <-1.0 or max_sdf > 1.0:
        logger.error(f"Invalid SDF range: min={min_sdf:.6f}, max={max_sdf:.6f}")
        raise ValueError(f"----------- INVALID SDF RANGE -----------")
        # continue
        
    Stats.reset()
    Stats.record("input_file", input_file)
    with Stats.timer("resfit_total"):
        best_program, running_program = resfit(mesh, save_file=save_file_temp)
    cPickle.dump(to_cpu_recursive(Stats.get_dict()), open(save_file, "wb"))
    logger.info(f"Saved to {save_file}")

    # If save html
    if args.save_html:
        save_file_name = os.path.join(save_dir, "best_program.html")
        html_code = save_html(best_program, save_file_name)
        logger.info(f"Saved HTML to {save_file_name}")
    if args.save_edit_html:
        save_file_name = os.path.join(save_dir, "best_edit_mode.html")
        html_code = save_edit_mode_html(best_program, sketcher_3d, save_file_name)
        logger.info(f"Saved HTML to {save_file_name}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--profile_path", type=str, required=False, default=None)
    parser.add_argument("--fastmode", action="store_true", required=False, default=False)
    parser.add_argument("--save_html", action="store_true", required=False, default=False)
    parser.add_argument("--save_edit_html", action="store_true", required=False, default=False)

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
# python scripts/mesh_to_pa.py --input_file /media/aditya/OS/data/toys_4k/toys4k_obj_files/airplane/airplane_007/mesh.obj --save_dir ../outputs/basic