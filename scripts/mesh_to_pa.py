import os
import _pickle as cPickle
import time
from geolipi.torch_compute import recursive_evaluate, Sketcher
import argparse
from superfit.algos.resfit import resfit
from superfit.utils.mesh_preprocess import normalize_to_unit_cube, extract_mesh, target_cleanup_v2
from superfit.utils.mesh_sdf import get_target_cubvh, renorm_target_sdf, sdf_to_mesh
from superfit.utils.config import AlgorithmConfig as AlgConf
import superfit.utils.config as config_options
# Over parameterize - even more - see what happens - all on the base version. 
import torch

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

def main_shape_wise(args):
    #### Set Mode.  
    input_mesh_file = args.input_mesh_file
    save_dir = args.save_dir
    config_options.main_setting()

    failed_indices = []
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # also save entire shape. 
    save_config_file = os.path.join(save_dir, "config.json")
    AlgConf.save_to_file(save_config_file)


    save_file = os.path.join(save_dir, f"primitive_assembly.pkl")
    # if os.path.exists(save_file):
    #     continue

    sketcher_3d = Sketcher(resolution=AlgConf.DATA_RESOLUTION, n_dims=3)
    save_file_temp = os.path.join(save_dir, f"resfit_prog.pkl")
    
    input_mesh = extract_mesh(input_mesh_file)
    input_mesh = normalize_to_unit_cube(input_mesh)
    target_sdf = get_target_cubvh(input_mesh, sketcher_3d, mode="raystab")
    target_sdf = target_cleanup_v2(target_sdf, sketcher_3d)
    target_sdf = renorm_target_sdf(target_sdf, sketcher_3d)
    mesh = sdf_to_mesh(target_sdf, sketcher_3d)

    if not mesh.is_watertight:
        raise ValueError(f"------- Non Watertight Mesh -------")
    min_sdf = target_sdf.min().item()
    max_sdf = target_sdf.max().item()
    if min_sdf <-1.0 or max_sdf > 1.0:
        print(min_sdf, max_sdf)
        raise ValueError(f"----------- INVALID SDF RANGE -----------")
        # continue
    start_time = time.time()
    out_expr, opt_program, stats = resfit(mesh, save_file=save_file_temp)
    end_time = time.time()
    part_info = {
        "out_expr": out_expr.sympy().state(),
        "opt_program": opt_program.sympy().state(),
        "time_taken": end_time - start_time,
    }
    part_info.update(stats)
    cPickle.dump(part_info, open(save_file, "wb"))
    print(f"Saved to {save_file}")
    print(f"Failed indices: {failed_indices}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mesh_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    main_shape_wise(args)
    