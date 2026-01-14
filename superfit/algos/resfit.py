import time
import torch as th
import trimesh
import numpy as np
import _pickle as cPickle
from geolipi.torch_compute import Sketcher, recursive_evaluate
from ..utils.mesh_sdf import get_target_cubvh, renorm_target_sdf
from ..symbolic.utils import gather_primitives
from ..utils.mesh_sdf import get_masked
from ..optim.entry import optimize_primitive_assembly
from ..optim.measures import get_iou
from ..utils.mesh_sdf import target_cleanup, CLEAN_UP_DELTA
from .prim_initialize import get_init_prim_program, simple_cleanup_volumetric
from .prim_initialize import generate_prim_initializations
import superfit.symbolic as sps
from .decompose_msd import msd_new
from ..utils.config import AlgorithmConfig as AlgConf
from .prune import main_pruning_pipeline


def resfit(target_mesh, 
    save_file=None,
    early_stop=True,
    record_param=False,
    ):

    prune_sketcher = Sketcher(resolution=AlgConf.PRUNE_RESOLUTION, dtype=th.float16, n_dims=3)
    decompose_sketcher = Sketcher(resolution=AlgConf.DECOMPOSE_RESOLUTION, dtype=th.float16, n_dims=3)
    optim_sketcher = Sketcher(resolution=AlgConf.OPT_RESOLUTION, n_dims=3, dtype=AlgConf.OPT_DTYPE)
    # target = get_target_mesh2sdf(mesh)
    opt_program = None
    best_program = None
    best_obj = 0.0
    best_recon_measure = 0.0
    best_iter = 0
    cur_iter = 0
    stochastic_dropout = AlgConf.STOCHASTIC_DROPOUT

    proc_start_time = time.time()
    decompose_target_sdf = get_target_cubvh(target_mesh, decompose_sketcher, mode="watertight")
    decompose_target_sdf = renorm_target_sdf(decompose_target_sdf, decompose_sketcher)
    masked_target_sdf = decompose_target_sdf.clone()
    
    target_sdf_opt = get_target_cubvh(target_mesh, optim_sketcher, mode="watertight")
    target_sdf_opt = renorm_target_sdf(target_sdf_opt, optim_sketcher)
    
    target_sdf_prune = get_target_cubvh(target_mesh, prune_sketcher, mode="watertight")
    target_sdf_prune = renorm_target_sdf(target_sdf_prune, prune_sketcher)
    proc_end_time = time.time()
    print(f"Time taken for processing target SDFs: {proc_end_time - proc_start_time}")
    
    primitives = []
    init_start_time = time.time()
    last_run = False
    n_prims_prev = 0
    while cur_iter < AlgConf.MPS_MAX_ITER:
        # try:
        with th.no_grad():
            cur_mps_iter_stats = {}
            cur_iter_start_time = time.time()
            # pruned_parts, all_indices = generate_partitions_sdf_v3(masked_target_sdf, sketcher, max_iter=max_iter_msd, min_part_size_global=min_volume_requirement, min_eroded_part_size_ratio=min_eroded_part_size_ratio)
            if AlgConf.RUN_LAST_OPT and last_run:
                pass
            else:
                pruned_parts, _ = msd_new(masked_target_sdf, decompose_sketcher, **AlgConf.DECOMPOSE_CONFIG)
                if len(pruned_parts) == 0:
                    print("No parts found - stopping")
                    break

                pruned_parts, _ = simple_cleanup_volumetric(pruned_parts, None, size_limit=AlgConf.DECOMPOSE_SIZE_LIMIT)
                print("found", len(pruned_parts), "parts")
                n_prims = len(pruned_parts) + len(primitives)
                inverse_rate = n_prims
                desired_prob = 0.9 ** (1 / inverse_rate)
                desired_prob = min(desired_prob, 0.975) # Should we?
                delta = np.log(desired_prob/ (1 - desired_prob))
                primitive_fits = generate_prim_initializations(pruned_parts, decompose_sketcher)
                # primitive_fits = generate_neo_prim_initializations_v3_rigid(pruned_parts, decompose_sketcher)

                opt_program = get_init_prim_program(primitive_fits, decompose_sketcher, opt_program,
                                                        stochastic_dropout=stochastic_dropout,
                                                        logits_keep_drop=(delta/2, -delta/2),
                                                        version=getattr(sps, AlgConf.PRIM_TYPE))
                
                end_time = time.time()

            # Now optimize the program.
            cur_mps_iter_stats["init_program"] = opt_program.sympy().state()
            cur_mps_iter_stats["time_taken_initialization"] = end_time - cur_iter_start_time
            # opt_program = convert_to_chained_expr(opt_program.tensor())
            # opt_program = batch_primitives(opt_program)
        start_time = time.time()
        new_opt_program, stats = optimize_primitive_assembly(opt_program.tensor(dtype=AlgConf.OPT_DTYPE), target_mesh, target_sdf_opt, optim_sketcher, n_prims_prev=n_prims_prev)
        if stats['best_obj'] > stats['cur_obj']:
            opt_program = new_opt_program
        cur_mps_iter_stats["opt_program"] = opt_program.sympy().state()

        end_time = time.time()
        print(f"Time taken for optimization: {end_time - start_time}")
        cur_mps_iter_stats["time_taken_optimization"] = end_time - start_time
        cur_mps_iter_stats["optim_stats"] = stats
        if AlgConf.DO_PRUNE:
            start_time = time.time()
            with th.no_grad():
                opt_program = opt_program.tensor(dtype=prune_sketcher.dtype)
                cur_best_program, prune_stats = main_pruning_pipeline(opt_program, target_mesh, target_sdf_prune, prune_sketcher, measure=AlgConf.PRUNE_METRIC)
                # if AlgConf.TVERSKY_MODE:
                opt_program = cur_best_program
                # opt_program, _ = main_pruning_pipeline(opt_program, target_mesh, target_sdf_prune, prune_sketcher, measure="surface_tversky")
                # else:
                # opt_program = cur_best_program
            end_time = time.time()
            print(f"Time taken for pruning: {end_time - start_time}")
            cur_mps_iter_stats["time_taken_pruning"] = end_time - start_time
            cur_mps_iter_stats["prune_stats"] = prune_stats
            cur_mps_iter_stats["pruned_program"] = cur_best_program.sympy().state()
        else:
            opt_program = opt_program.tensor()
            cur_best_program = opt_program
            prune_stats = stats
        cur_obj = prune_stats["best_obj"]
        cur_recon = prune_stats["best_recon_measure"]
        if cur_obj > best_obj + AlgConf.MPS_MIN_IMPROVEMENT:
            best_iter = cur_iter
        if cur_obj >= best_obj:
            print("==================== New best obj found ====================")
            print("Previous best iter: ", best_iter, "New best iter: ", cur_iter)
            print("Previous best obj: ", best_obj, "New best obj: ", cur_obj)
            print("Previous best recon measure: ", best_recon_measure, "New best recon measure: ", cur_recon)
            best_obj = cur_obj
            best_recon_measure = cur_recon
            best_program = cur_best_program.sympy()
        cur_mps_iter_stats["best_obj"] = best_obj
        cur_mps_iter_stats["best_recon_measure"] = best_recon_measure
        cur_mps_iter_stats["best_program"] = best_program.sympy().state()
        cur_mps_iter_stats["best_iter"] = best_iter
        # Better way to handle failure comes here. 
        # Mask target the occupied regions. 
        masked_target_sdf = get_masked(opt_program, decompose_target_sdf.clone(), decompose_sketcher)
        if AlgConf.DECOMPOSE_MODE == "COACD":
            # Other wise coacd fails to get good primitives. 
            masked_target_sdf = masked_target_sdf - CLEAN_UP_DELTA
            masked_target_sdf = target_cleanup(masked_target_sdf, decompose_sketcher, AlgConf.MIN_VOLUME_LIMIT_FOR_REINIT)
        masked_target_sdf = renorm_target_sdf(masked_target_sdf, decompose_sketcher)
        primitives = gather_primitives(opt_program)
        n_prims_main = len(primitives)
        print(f"Cur Iter: {cur_iter}, Best Recon Measure: {best_recon_measure:.3f}, Best Obj: {best_obj:.3f}, program_size: {len(primitives)}, best_iter: {best_iter}")

        if save_file is not None:
            stats = {
                "n_iters": cur_iter,
                "best_obj": best_obj,
                "best_recon_measure": best_recon_measure,
                "success": True,
            }
            if record_param:
                raise ValueError("Record param not supported")
                # stats["record_param_list"] = record_param_list

            end_time = time.time()
            part_info = {
                "out_expr": cur_best_program.sympy().state(),
                "opt_target": opt_program.sympy().state(),
                "time_taken_overall": end_time - cur_iter_start_time,
            }
            part_info.update(stats)
            part_info.update(cur_mps_iter_stats)
            cur_save_file = save_file.replace(".pkl", f"_{cur_iter}.pkl")
            cPickle.dump(part_info, open(cur_save_file, "wb"))
            print(f"Saved to {cur_save_file}")
        # except Exception as e:
        #     print(f"Error in loop {cur_iter}: {e}")
        #     continue
        
        if masked_target_sdf.min() > 0.0:
            print("Target SDF is fully occupied - stopping")

            if AlgConf.RUN_LAST_OPT:
                if not last_run:
                    print("-------------------------------- Running last optimization --------------------------------")
                    last_run = True
                    cur_iter += 1
                    continue
                else:
                    print("==================== Finished running last optimization ====================")
                    break
            else:
                break

        if early_stop and cur_iter - best_iter >= AlgConf.EARLY_STOP_ITER:
            print(f"Early stopping at iteration {cur_iter} as no improvement")
            print("Previous best iter: ", best_iter)
            
            if AlgConf.RUN_LAST_OPT:
                if not last_run:
                    print("-------------------------------- Running last optimization --------------------------------")
                    last_run = True
                    cur_iter += 1
                    continue
                else:
                    print("==================== Finished running last optimization ====================")
                    break
            else:
                break
        else:
            print("==================== Did not reach early stop condition ====================")
        if AlgConf.RUN_LAST_OPT and last_run:
            print("==================== Finished running last optimization ====================")
            break
        n_prims_prev = n_prims_main
        cur_iter += 1
    if best_program is None:
        stats = {'success': False, 'n_iters': cur_iter, 'best_obj': 0.0, 'best_iou': 0}
        best_program, opt_program = None, None
    else:
        best_out = recursive_evaluate(best_program.tensor(), decompose_sketcher)
        hard_out = (best_out <= 0).bool()
        hard_target = (decompose_target_sdf <= 0).bool()
        best_iou = get_iou(hard_out, hard_target)
        stats = {
            "n_iters": cur_iter,
            "best_obj": best_obj,
            "best_iou": best_iou,
            "best_recon_measure": best_recon_measure,
            "success": True,
            "total_time": time.time() - init_start_time,
            "last_step": cur_iter,
            "last_step_stats": part_info,
            "stats": cur_mps_iter_stats,
        }
    print(f"===========MPS: Time taken: {time.time() - init_start_time} ===========")
    stats['processing_time'] = proc_end_time - proc_start_time
    return best_program, opt_program, stats

