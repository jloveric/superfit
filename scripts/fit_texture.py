"""
Script to optimize material textures on primitives after geometric fitting is complete.
"""
import os
import argparse
import torch as th
import _pickle as cPickle
from geolipi.torch_compute import Sketcher, recursive_evaluate
from superfit.utils.mesh_preprocess import extract_mesh, normalize_to_unit_cube
from superfit.utils.mesh_sdf import sdf_to_mesh
from superfit.utils.stats import Stats
from superfit.utils.logger import logger
from superfit.mat_opt.optim import optimize_color   
from superfit.mat_opt.utils import get_material_expr
from superfit.utils.io import get_best_expr
from superfit.mat_opt.utils import save_html_mat_expr
import superfit.utils.config as config_options
from superfit.utils.config import AlgorithmConfig as AlgConf
from superfit.utils.editing import save_edit_mode_html


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimize material textures on primitives after fitting")
    parser.add_argument("--input_file", type=str, required=True, help="Path to primitive_assembly.pkl file")
    parser.add_argument("--save_html", action="store_true", required=False, default=False)
    parser.add_argument("--save_edit_html", action="store_true", required=False, default=False)
    args = parser.parse_args()
    
    # Validate pkl file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"File not found: {args.input_file}")
    
    return args


def main(args: argparse.Namespace):
    
    # Setup config
    config_options.main_setting()

    # Setup sketchers
    sketcher_3d = Sketcher(resolution=AlgConf.DATA_RESOLUTION, n_dims=3)
    
    # Load the pkl file
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"File not found: {args.input_file}")
    
    logger.info(f"Loading pkl file: {args.input_file}")
    info_dict = cPickle.load(open(args.input_file, "rb"))
    
    # Get input_mesh_file from info_dict
    n_iters = info_dict.get("n_iters", 0)
    if n_iters == 0:
        raise ValueError("n_iters not found or is 0 in pkl file")
    iter_idx = n_iters - 1
    base_geometric_expr = get_best_expr(info_dict, iter_idx)

    input_mesh_file = info_dict.get("input_mesh_file", None)
    if input_mesh_file is None:
        raise ValueError("input_mesh_file not found in pkl file")
    
    # Load and process mesh
    logger.info("Loading and processing mesh...")
    global_mesh = extract_mesh(input_mesh_file)
    global_mesh = normalize_to_unit_cube(global_mesh)
    # target_mesh, target_sdf = process_mesh_to_sdf(input_mesh_file, sketcher_3d)
    
    output_sdf = recursive_evaluate(base_geometric_expr.tensor(), sketcher_3d)
    sample_mesh = sdf_to_mesh(output_sdf, sketcher_3d)

    # Material Expression:
    material_expr = get_material_expr(base_geometric_expr)

    logger.info(f"Input mesh file: {input_mesh_file}")
    logger.info(f"Using iteration: {iter_idx}")

    
    # Reset stats
    Stats.reset()
    
    # Use scoping for this texture optimization
    logger.info("==================== Starting texture optimization ====================")
    
    # Get global mesh (ground truth with textures)
    
    logger.info(f"Sample mesh vertices: {len(sample_mesh.vertices)}")
    logger.info(f"Global mesh vertices: {len(global_mesh.vertices)}")
    
    # Run texture optimization
    logger.info("Running texture optimization...")
    with Stats.scope(f"texture_optimization"):
        optimized_program, optimized_obj = optimize_color(
            global_mesh, 
            sample_mesh, 
            material_expr, 
            sketcher_3d,
            verbose=True
        )
        
    # Record optimization results
    if isinstance(optimized_obj, th.Tensor):
        optimized_obj_val = optimized_obj.item()
    else:
        optimized_obj_val = optimized_obj
    
    Stats.record("optimized_obj", optimized_obj_val)
    
    logger.info(f"Optimization complete. Final objective: {optimized_obj_val:.6f}")
    
    # Save results
    pkl_dir = os.path.dirname(args.input_file)
    pkl_basename = os.path.basename(args.input_file)
    
    output_basename = f"{pkl_basename}_textured.pkl"
    
    output_file = os.path.join(pkl_dir, output_basename)
    
    # Get stats and add optimized program
    Stats.record("input_mesh_file", input_mesh_file)
    Stats.record("iter_idx", iter_idx)
    Stats.record("base_geometric_expr", base_geometric_expr.sympy().state(), log=False)
    Stats.record("material_expr", optimized_program.state(), log=False)
    
    logger.info(f"Saving texture optimization results to: {output_file}")
    cPickle.dump(Stats.get_dict(), open(output_file, "wb"))
    
    logger.info("==================== Texture optimization complete ====================")
    logger.info(f"Results saved to: {output_file}")
    
    if args.save_html:
        save_file_name = os.path.join(pkl_dir, "best_program_textured.html")
        html_code = save_html_mat_expr(material_expr, sketcher_3d, save_file_name)
        logger.info(f"Saved HTML to {save_file_name}")
    if args.save_edit_html:
        save_file_name = os.path.join(pkl_dir, "best_edit_mode_textured.html")
        html_code = save_edit_mode_html(material_expr, sketcher_3d, save_file_name, is_textured=True)
        logger.info(f"Saved HTML to {save_file_name}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
