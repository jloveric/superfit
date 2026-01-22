import os
import argparse
import _pickle as cPickle
import torch as th
import geolipi.symbolic as gls
from superfit.algos.eval_tools import MeasurePack, eval_shape
from superfit.utils.config import AlgorithmConfig as AlgConf
from superfit.utils.mesh_preprocess import process_mesh_to_sdf, extract_mesh
from superfit.utils.mesh_sdf import get_target_cubvh, renorm_target_sdf
from geolipi.torch_compute import Sketcher
from superfit.symbolic.utils import fetch_singular_expr_eval
from superfit.utils.stats import Stats
from superfit.utils.logger import logger
import superfit.utils.config as config_options


def main():
    parser = argparse.ArgumentParser(description="Evaluate primitive assembly from pkl file")
    parser.add_argument("--input_file", type=str, help="Path to primitive_assembly.pkl file")
    parser.add_argument("--eval", type=str, default="last", choices=["last", "all_iters"], 
                       help="Evaluate last iteration only or all iterations")
    
    args = parser.parse_args()
    
    # Load the pkl file
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"File not found: {args.input_file}")
    
    logger.info(f"Loading pkl file: {args.input_file}")
    info_dict = cPickle.load(open(args.input_file, "rb"))
    
    # Get input_mesh_file from stats
    input_mesh_file = info_dict.get("input_mesh_file", None)
    if input_mesh_file is None:
        raise ValueError("input_mesh_file not found in pkl file")
    
    # Get number of iterations
    n_iters = info_dict.get("n_iters", 0)
    if n_iters == 0:
        raise ValueError("n_iters not found or is 0 in pkl file")
    
    logger.info(f"Input mesh file: {input_mesh_file}")
    logger.info(f"Number of iterations: {n_iters}")
    
    # Setup config
    config_options.main_setting()
    
    # Setup sketchers
    sketcher_3d = Sketcher(resolution=AlgConf.DATA_RESOLUTION, n_dims=3)
    prune_sketcher = Sketcher(resolution=AlgConf.PRUNE_RESOLUTION, dtype=th.float16, n_dims=3)
    
    # Load and process mesh
    input_mesh = extract_mesh(input_mesh_file)
    mesh, target_sdf = process_mesh_to_sdf(input_mesh_file, sketcher_3d)
    target_mesh = mesh
    
    # Create target SDF for pruning
    target_sdf_prune = get_target_cubvh(target_mesh, prune_sketcher, mode="watertight")
    target_sdf_prune = renorm_target_sdf(target_sdf_prune, prune_sketcher)
    
    # Create measure pack
    measure_pack = MeasurePack(
        measure=AlgConf.PRUNE_METRIC,
        target_mesh=target_mesh,
        original_mesh=input_mesh,
        target_sdf=target_sdf_prune,
        len_weight=AlgConf.MPS_LEN_WEIGHT
    )
    
    # Reset stats
    Stats.reset()
    
    # Determine which iterations to evaluate
    if args.eval == "all_iters":
        iter_indices = range(n_iters)
        logger.info(f"Evaluating all {n_iters} iterations")
    else:
        iter_indices = [n_iters - 1]
        logger.info(f"Evaluating last iteration (iter {n_iters - 1})")
    
    # Evaluate each iteration
    for iter_idx in iter_indices:
        prog_type = "best_program"
        iter_key = f"iter_{iter_idx}.{prog_type}"
        
        # Check if this iteration exists
        expr_key = f"{iter_key}.expr_str"
        if expr_key not in info_dict:
            logger.warning(f"Iteration {iter_idx} not found in pkl file, skipping")
            continue
        
        logger.info(f"Evaluating iteration {iter_idx}")
        
        # Use scoping to avoid stats overwriting when evaluating multiple iterations
        with Stats.scope(f"iter_{iter_idx}"):
            # Load the program
            out_expr = {
                "expr_str": info_dict[expr_key],
                "symbol_tensor_map": info_dict.get(f"{iter_key}.symbol_tensor_map", {})
            }
            
            init_expr = gls.GLFunction.from_state(out_expr).sympy()
            expr_in = fetch_singular_expr_eval(
                init_expr.tensor(), 
                temperature=100.0, 
                relaxed_eval=True, 
                remove_marker=False
            ).sympy()
            
            # Run evaluation
            eval_shape(expr_in, measure_pack, None)
    
    # Get the stats dictionary
    eval_stats = Stats.get_dict()
    
    # Determine output file path
    pkl_dir = os.path.dirname(args.input_file)
    pkl_basename = os.path.basename(args.input_file)
    # Replace extension or add _eval suffix
    if pkl_basename.endswith(".pkl"):
        output_basename = pkl_basename.replace(".pkl", "_eval.pkl")
    else:
        output_basename = f"{pkl_basename}_eval.pkl"
    
    output_file = os.path.join(pkl_dir, output_basename)
    
    # Save evaluation results
    logger.info(f"Saving evaluation results to: {output_file}")
    cPickle.dump(eval_stats, open(output_file, "wb"))
    logger.info(f"Evaluation complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
