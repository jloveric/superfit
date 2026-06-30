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
import os
import argparse
import _pickle as cPickle
import superfit.utils.config as config_options
import torch as th
from geolipi.torch_compute import Sketcher, recursive_evaluate
from superfit.utils.stats import Stats
from superfit.utils.logger import logger
from superfit.utils.mesh_sdf import sdf_to_mesh
from superfit.utils.editing import save_edit_mode_html
from superfit.utils.io import to_cpu_recursive, save_html
from superfit.utils.mesh_preprocess import cd_based_process_mesh_to_sdf
from superfit.utils.config import AlgorithmConfig as AlgConf, initialize_seeds
from superfit.utils.constants import AOT_ARTIFACT_DIR
from superfit.algos.resfit import resfit
from superfit.algos.eval_tools import MeasurePack
from superfit.algos.prune import sampling_based_pruning
from superfit.symbolic.utils import gather_primitives, fetch_singular_expr_eval
# Over parameterize - even more - see what happens - all on the base version. 

th.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--profile_path", type=str, required=False, default=None)
    parser.add_argument("--fastmode", action="store_true", required=False, default=False)
    parser.add_argument("--ablation", type=int, default=0, help="Ablation number.")
    parser.add_argument("--aot_postfix", type=str, default="aott", help="AOT artifact postfix.")
    parser.add_argument("--save_html", action="store_true", required=False, default=False)
    parser.add_argument("--save_edit_html", action="store_true", required=False, default=False)
    parser.add_argument("--save_mesh", action="store_true", required=False, default=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for optimization.")
    return parser.parse_args()


def main_shape_wise(args):
        
    input_path = args.input_path
    save_dir = args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Configuration setup. 
    config_options.main_setting()
    config_options.set_config_ablation(args.ablation, fastmode=args.fastmode)
    
    save_config_file = os.path.join(save_dir, "config.json")
    AlgConf.save_to_file(save_config_file)
    AlgConf.AOT_ARTIFACT_FILE = os.path.join(
        AOT_ARTIFACT_DIR, f"aot_artifact_{args.aot_postfix}_{args.ablation}.pt"
    )
    
    
    # Initialize seeds after config setup.
    initialize_seeds(seed=args.seed)
    th.backends.cudnn.benchmark = True
    
    save_file = os.path.join(save_dir, f"final_content.pkl")
    save_file_temp = os.path.join(save_dir, f"stepwise.pkl")

    sketcher_3d = Sketcher(resolution=AlgConf.DATA_RESOLUTION, n_dims=3)
    mesh, target_sdf, _ = cd_based_process_mesh_to_sdf(input_path, sketcher_3d)

    if not mesh.is_watertight:
        raise ValueError(f"------- Non Watertight Mesh -------")
    min_sdf = target_sdf.min().item()
    max_sdf = target_sdf.max().item()
    if min_sdf <-1.0 or max_sdf > 1.0:
        logger.error(f"Invalid SDF range: min={min_sdf:.6f}, max={max_sdf:.6f}")
        raise ValueError(f"----------- INVALID SDF RANGE -----------")
        # continue
        
    Stats.reset()
    Stats.record("input_path", input_path)
    with Stats.timer("resfit_total"):
        best_program, running_program = resfit(mesh, save_file=save_file_temp)
    cPickle.dump(to_cpu_recursive(Stats.get_dict()), open(save_file, "wb"))
    logger.info(f"Saved to {save_file}")

    # If save html
    # convert to singular best expressions: 
    measure_pack = MeasurePack(
        measure=AlgConf.PRUNE_METRIC,
        target_mesh=mesh,
        original_mesh=mesh,
        target_sdf=target_sdf,
        len_weight=AlgConf.MPS_LEN_WEIGHT
    )
    singular_best_program,_, _, _ = sampling_based_pruning(best_program, sketcher_3d, measure_pack)
    if args.save_html:
        singular_best_program_resolved = fetch_singular_expr_eval(singular_best_program.tensor(), temperature=0.1, 
                                                                  relaxed_eval=False).sympy()
        save_file_name = os.path.join(save_dir, "best_program.html")
        html_code = save_html(singular_best_program_resolved, save_file_name)
        logger.info(f"Saved HTML to {save_file_name}")
    if args.save_edit_html:
        singular_best_program_resolved = fetch_singular_expr_eval(singular_best_program.tensor(), temperature=0.1, 
                                                                relaxed_eval=False, remove_marker=False, 
                                                                use_euler_angle=True).sympy()
        save_file_name = os.path.join(save_dir, "best_edit_mode.html")
        html_code = save_edit_mode_html(singular_best_program_resolved, sketcher_3d, save_file_name)
        logger.info(f"Saved HTML to {save_file_name}")
    if args.save_mesh:
        save_file_name = os.path.join(save_dir, "full_mesh.obj")
        full_sdf = recursive_evaluate(singular_best_program.tensor(), sketcher_3d)
        full_mesh = sdf_to_mesh(full_sdf, sketcher_3d)
        full_mesh.export(save_file_name)
        logger.info(f"Saved Mesh to {save_file_name}")
        # partwise: 
        # NOTE: Under Smooth Union; Union(Primitive) != FullAssembly.
        primitives = gather_primitives(singular_best_program)
        for i, primitive in enumerate(primitives):
            primitive_sdf = recursive_evaluate(primitive.tensor(), sketcher_3d)
            primitive_mesh = sdf_to_mesh(primitive_sdf, sketcher_3d)
            primitive_mesh.export(os.path.join(save_dir, f"primitive_{i}.obj"))
            logger.info(f"Saved Primitive {i} Mesh to {os.path.join(save_dir, f'primitive_{i}.obj')}")

        
    

def cli_main():
    args = parse_args()
    if args.profile_path is not None:
        import cProfile
        import pstats
        def main():
            main_shape_wise(args)
        cProfile.run("main()", args.profile_path)
        pstats.Stats(args.profile_path).strip_dirs().sort_stats("cumtime").print_stats(50)
    else:
        main_shape_wise(args)


if __name__ == "__main__":
    cli_main()
# python scripts/mesh_to_pa.py --input_path /media/aditya/OS/data/toys_4k/toys4k_obj_files/airplane/airplane_007/mesh.obj --save_dir ../outputs/basic
