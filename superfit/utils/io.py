import os
import csv
import distinctipy
import numpy as np
import geolipi.symbolic as gls
import superfit.symbolic as sps
from sysl.shader import evaluate_to_shader
from sysl.shader_runtime import create_multibuffer_shader_html
from sysl.utils import recursive_sm_to_smg, recursive_gls_to_sysl
from ..symbolic.utils import fetch_singular_expr_eval, n_prims_in_expr
from .constants import TOY4K_PATH_PREFIX, TOY4K_CSV_FILE, PARTOBJAVERSE_MESH_DIR
# from mpspy.symbolic.utils import recursive_ntco_to_ntco_ss, recursive_axisangle_to_euler

def save_html(expression, save_file_name="resfit_best_program.html"):
    expr_in = fetch_singular_expr_eval(expression.tensor(), temperature=100.0, relaxed_eval=True, remove_marker=True).sympy()
    expr_smg = recursive_sm_to_smg(expr_in)
    mat_expr, _ = recursive_gls_to_sysl(expr_smg, version="v4", ind=2)
    shader_info = evaluate_to_shader(mat_expr.sympy(), mode="multipass", post_process_shader=["part_outline_nobg"])
    html_code = create_multibuffer_shader_html(shader_info)
    with open(save_file_name, "w") as f:
        f.write(html_code)
    return html_code


def load_toy4k_mesh_paths(csv_file=TOY4K_CSV_FILE, toy4k_path_prefix=TOY4K_PATH_PREFIX):
    """
    Load mesh paths from toy4k dataset CSV file.
    Expected format: CSV with single column of relative paths (no header).
    
    Args:
        csv_file: Path to CSV file. If None, uses default from constants.
                 If relative path, assumes it's in the project root.
    
    Returns:
        List of absolute mesh file paths (strings).
    """
    
    if not os.path.isabs(csv_file):
        # If relative path, assume it's in the project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        csv_path = os.path.join(project_root, csv_file)
    else:
        csv_path = csv_file
    
    mesh_paths = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 1 and row[0].strip():
                mesh_paths.append(row[0].strip())
    
    # Convert relative paths to absolute paths
    processed_paths = []
    for relative_path in mesh_paths:
        # Join with the new path prefix to get absolute path
        absolute_path = os.path.join(toy4k_path_prefix, relative_path)
        processed_paths.append(absolute_path)
    
    return processed_paths

def load_partobjaverse_mesh_paths(location=PARTOBJAVERSE_MESH_DIR):
    """
    Load mesh paths from PartObjaverse dataset.
    
    Returns:
        List of mesh file paths (strings) for all .glb files in the dataset directory.
    """
    files = os.listdir(location)
    files = [os.path.join(location, f) for f in files]
    files = [f for f in files if f.endswith(".glb")]
    return files


def get_best_expr(info_dict, iter_idx, prog_type = "best_program"):
    iter_key = f"iter_{iter_idx}.{prog_type}"
    out_expr = info_dict[iter_key]
    init_expr = gls.GLFunction.from_state(out_expr).sympy()
    return init_expr
