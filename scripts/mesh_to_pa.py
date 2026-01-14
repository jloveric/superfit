import os
import _pickle as cPickle
import time
from geolipi.torch_compute import recursive_evaluate, Sketcher
import argparse
from superfit.algos.resfit import resfit
from superfit.utils.mesh_preprocess import process_mesh_to_sdf
from superfit.utils.config import AlgorithmConfig as AlgConf
import superfit.utils.config as config_options
# Over parameterize - even more - see what happens - all on the base version. 
import torch as th

th.set_float32_matmul_precision("medium")
th.backends.cudnn.benchmark = True

# th.backends.fp32_precision = "tf32"                 # global default
# th.backends.cuda.matmul.fp32_precision = "tf32"     # matmul/bmm
# th.backends.cudnn.fp32_precision = "tf32"           # cuDNN backend default

# Fast matmuls (TF32 on Ampere+)
# th.backends.cuda.matmul.fp32_precision = "tf32"   # or "ieee" for full precision  [oai_citation:1‡PyTorch Documentation](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html?utm_source=chatgpt.com)
# Fast cuDNN convs (if you do convs; safe to set even if you don’t)
# th.backends.cudnn.conv.fp32_precision = "tf32"    # 2.9+ recommended API  [oai_citation:2‡GitHub](https://github.com/pytorch/pytorch/issues/166286?utm_source=chatgpt.com)
# cuDNN algorithm search (only helps if conv input sizes are mostly constant)
# th.backends.cudnn.benchmark = True

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
    
    mesh, target_sdf = process_mesh_to_sdf(input_mesh_file, sketcher_3d)

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