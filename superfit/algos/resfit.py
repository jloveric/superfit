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
from .eval import get_recon_measure, MeasurePack
from .decompose_msd import msd_new
from ..utils.config import AlgorithmConfig as AlgConf
from ..utils.stats import Stats
from ..utils.logger import logger
from .prune import main_pruning_pipeline

def get_delta(n_prims):
    inverse_rate = n_prims
    desired_prob = 0.9 ** (1 / inverse_rate)
    desired_prob = min(desired_prob, 0.975) # Should we?
    delta = np.log(desired_prob/ (1 - desired_prob))
    return delta

def resfit(target_mesh, 
    save_file=None,
    early_stop=True,
    record_param=False,
    ):

    prune_sketcher = Sketcher(resolution=AlgConf.PRUNE_RESOLUTION, dtype=th.float16, n_dims=3)
    decompose_sketcher = Sketcher(resolution=AlgConf.DECOMPOSE_RESOLUTION, dtype=th.float16, n_dims=3)
    optim_sketcher = Sketcher(resolution=AlgConf.OPT_RESOLUTION, n_dims=3, dtype=AlgConf.OPT_DTYPE)
    # target = get_target_mesh2sdf(mesh)
    running_program, best_program = None, None
    best_obj, cur_best_obj = 0.0, 0.0
    best_recon_measure, cur_best_recon_measure = 0.0, 0.0
    cur_iter, best_iter = 0, 0

    with Stats.timer("processing_target_sdfs"):
        decompose_target_sdf = get_target_cubvh(target_mesh, decompose_sketcher, mode="watertight")
        decompose_target_sdf = renorm_target_sdf(decompose_target_sdf, decompose_sketcher)
        masked_target_sdf = decompose_target_sdf.clone()
        
        target_sdf_opt = get_target_cubvh(target_mesh, optim_sketcher, mode="watertight")
        target_sdf_opt = renorm_target_sdf(target_sdf_opt, optim_sketcher)
        
        target_sdf_prune = get_target_cubvh(target_mesh, prune_sketcher, mode="watertight")
        target_sdf_prune = renorm_target_sdf(target_sdf_prune, prune_sketcher)
    processing_time = Stats.get("time_processing_target_sdfs", root=True) or 0
    logger.info(f"Time taken for processing target SDFs: {processing_time:.3f}s")
    
    measure_pack = MeasurePack(
        measure=AlgConf.PRUNE_METRIC,
        target_mesh=target_mesh,
        target_sdf=target_sdf_prune,
        len_weight=AlgConf.MPS_LEN_WEIGHT
    )

    primitives = []
    init_start_time = time.time()
    n_prims_prev = 0
    last_run = False
    while cur_iter < AlgConf.MPS_MAX_ITER:
        with Stats.scope(f"iter_{cur_iter}"):
            # try:
            if running_program:
                cur_recon_measure = get_recon_measure(running_program, prune_sketcher, measure_pack)
                cur_n_prim = len(gather_primitives(running_program))
                cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
                Stats.record("pre_init_recon_measure", cur_recon_measure)
                Stats.record("pre_init_n_prim", cur_n_prim)
                Stats.record("pre_init_obj", cur_obj)
                Stats.record("pre_init_program", running_program.sympy().state(), log=False)
                cur_best_program = running_program

            with Stats.timer("initialization"):
                with th.no_grad():
                    pruned_parts, _ = msd_new(masked_target_sdf, decompose_sketcher, **AlgConf.DECOMPOSE_CONFIG)
                    if len(pruned_parts) == 0:
                        logger.info("===No parts found - Reached Stopping Criteria===")
                        break

                    pruned_parts, _ = simple_cleanup_volumetric(pruned_parts, None, size_limit=AlgConf.DECOMPOSE_SIZE_LIMIT)
                    logger.info(f"Found {len(pruned_parts)} parts")
                    n_prims = len(pruned_parts) + len(primitives)
                    primitive_fits = generate_prim_initializations(pruned_parts, decompose_sketcher)
                    delta = get_delta(n_prims)
                    running_program = get_init_prim_program(primitive_fits, decompose_sketcher, running_program,
                                                            logits_keep_drop=(delta/2, -delta/2))
            Stats.record("init_program", running_program.sympy().state(), log=False)
            # Always assume this modification is accepted. 
            cur_recon_measure = get_recon_measure(running_program, prune_sketcher, measure_pack)
            cur_n_prim = len(gather_primitives(running_program))
            cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
            Stats.record("init_recon_measure", cur_recon_measure)
            Stats.record("init_n_prim", cur_n_prim)
            Stats.record("init_obj", cur_obj)
            if cur_obj > cur_best_obj:
                cur_best_program = running_program
                cur_best_obj = cur_obj
            
            with Stats.timer("optimization"):
                measure_pack.target_sdf = target_sdf_opt
                measure_pack.reset()
                running_program = optimize_primitive_assembly(running_program.tensor(dtype=AlgConf.OPT_DTYPE), target_mesh, target_sdf_opt, optim_sketcher, measure_pack)
            Stats.record("opt_program", running_program.sympy().state(), log=False)
            Stats.record("opt_recon_measure", Stats.get("optimization.end_recon_measure"))
            Stats.record("opt_n_prim", Stats.get("optimization.end_n_prim"))
            Stats.record("opt_obj", Stats.get("optimization.end_obj"))
            if Stats.get("opt_obj") > cur_best_obj:
                cur_best_program = running_program
                cur_best_obj = Stats.get("opt_obj")
            
            if AlgConf.DO_PRUNE:
                with th.no_grad():
                    with Stats.timer("pruning"):
                        running_program = running_program.tensor(dtype=prune_sketcher.dtype)
                        measure_pack.target_sdf = target_sdf_prune
                        measure_pack.reset()
                        running_program = main_pruning_pipeline(running_program, prune_sketcher, measure_pack)
                Stats.record("pruned_program", running_program.sympy().state(), log=False)
                Stats.record("pruned_recon_measure", Stats.get("pruning.best_recon_measure"))
                Stats.record("pruned_n_prim", Stats.get("pruning.best_n_prim"))
                Stats.record("pruned_obj", Stats.get("pruning.best_obj"))
                if Stats.get("pruned_obj") > cur_best_obj:
                    cur_best_program = running_program
                    cur_best_obj = Stats.get("pruned_obj")
            else:
                Stats.record("pruned_recon_measure", Stats.get("opt_recon_measure"))
                Stats.record("pruned_n_prim", Stats.get("opt_n_prim"))
                Stats.record("pruned_obj", Stats.get("opt_obj"))
            
            cur_best_recon_measure = max(Stats.get("init_recon_measure"), Stats.get("opt_recon_measure"), Stats.get("pruned_recon_measure"))
            if cur_best_obj > best_obj + AlgConf.MPS_MIN_IMPROVEMENT:
                best_iter = cur_iter
            if cur_best_obj >= best_obj:
                logger.info("==================== New best obj found ====================")
                logger.info(f"Previous best iter: {best_iter}, New best iter: {cur_iter}")
                logger.info(f"Previous best obj: {best_obj:.6f}, New best obj: {cur_best_obj:.6f}")
                logger.info(f"Previous best recon measure: {best_recon_measure:.6f}, New best recon measure: {cur_best_recon_measure:.6f}")
                best_obj = cur_best_obj
                best_recon_measure = cur_best_recon_measure
                best_program = cur_best_program.sympy()
            Stats.record("best_obj", best_obj)
            Stats.record("best_recon_measure", best_recon_measure)
            Stats.record("best_program", best_program.sympy().state(), log=False)
            Stats.record("running_program", running_program.sympy().state(), log=False)
            Stats.record("best_iter", best_iter)
            
            if save_file is not None:
                iter_stats = Stats.get_dict()  # Get current scope (iter_{cur_iter})
                if record_param:
                    raise ValueError("Record param not supported")
                cur_save_file = save_file.replace(".pkl", f"_{cur_iter}.pkl")
                cPickle.dump(iter_stats, open(cur_save_file, "wb"))
                logger.info(f"Saved to {cur_save_file}")
        # Better way to handle failure comes here. 
        # Mask target the occupied regions. 
        masked_target_sdf = get_masked(running_program, decompose_target_sdf.clone(), decompose_sketcher)
        if AlgConf.DECOMPOSE_MODE == "COACD":
            # Other wise coacd fails to get good primitives. 
            masked_target_sdf = masked_target_sdf - CLEAN_UP_DELTA
            masked_target_sdf = target_cleanup(masked_target_sdf, decompose_sketcher, AlgConf.MIN_VOLUME_LIMIT_FOR_REINIT)
        masked_target_sdf = renorm_target_sdf(masked_target_sdf, decompose_sketcher)
        primitives = gather_primitives(running_program)
        n_prims_main = len(primitives)
        logger.info(f"Cur Iter: {cur_iter}, Best Recon Measure: {best_recon_measure:.3f}, Best Obj: {best_obj:.3f}, program_size: {len(primitives)}, best_iter: {best_iter}")
        
        if masked_target_sdf.min() > 0.0:
            logger.info("Target SDF is fully occupied - stopping")

            if AlgConf.RUN_LAST_OPT:
                if not last_run:
                    logger.info("-------------------------------- Running last optimization --------------------------------")
                    last_run = True
                    cur_iter += 1
                    continue
                else:
                    logger.info("==================== Finished running last optimization ====================")
                    break
            else:
                break

        if early_stop and cur_iter - best_iter >= AlgConf.EARLY_STOP_ITER:
            logger.info(f"Early stopping at iteration {cur_iter} as no improvement")
            logger.info(f"Previous best iter: {best_iter}")
            
            if AlgConf.RUN_LAST_OPT:
                if not last_run:
                    logger.info("-------------------------------- Running last optimization --------------------------------")
                    last_run = True
                    cur_iter += 1
                    continue
                else:
                    logger.info("==================== Finished running last optimization ====================")
                    break
            else:
                break
        else:
            logger.debug("Did not reach early stop condition")
        if AlgConf.RUN_LAST_OPT and last_run:
            logger.info("==================== Finished running last optimization ====================")
            break
        n_prims_prev = n_prims_main
        cur_iter += 1
    total_time = time.time() - init_start_time
    Stats.record("total_time", total_time)
    Stats.record("n_iters", cur_iter)
    
    if best_program is None:
        Stats.record("success", False)
        Stats.record("best_obj", 0.0)
        Stats.record("best_iou", 0)
        best_program, running_program = None, None
    else:
        best_out = recursive_evaluate(best_program.tensor(), decompose_sketcher)
        hard_out = (best_out <= 0).bool()
        hard_target = (decompose_target_sdf <= 0).bool()
        best_iou = get_iou(hard_out, hard_target)
        Stats.record("success", True)
        Stats.record("best_obj", best_obj)
        Stats.record("best_iou", best_iou)
        Stats.record("best_recon_measure", best_recon_measure)
    
    logger.info(f"===========MPS: Time taken: {total_time:.3f}s ===========")

    return best_program, running_program

