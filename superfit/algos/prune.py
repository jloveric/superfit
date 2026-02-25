from dataclasses import dataclass
import torch as th
import numpy as np
import trimesh
import random
import itertools
import geolipi.symbolic as gls
from geolipi.torch_compute.evaluate_expression import recursive_evaluate

from ..symbolic.utils import inject_temp_param, gather_primitives, fetch_singular_expr_eval
from .eval_tools import get_recon_measure, get_recon_measure_packed, MeasurePack
from ..symbolic.utils import gather_instance_dropout_alternatives, gather_smooth_union_ops, generate_from_sm_ops_and_primitives
from ..utils.config import AlgorithmConfig as AlgConf, initialize_seeds
from ..utils.stats import Stats
from ..utils.logger import logger

MAX_SAMPLING_TRIES = 100
SAMPLING_PATIENCE = 10
TEMP_SCALE_FACTOR = 2.0
MAX_INNER_LOOP_LEAVE_ONE_OUT = 100
MIN_VOL_LIMIT = 1e-4

def main_pruning_pipeline(in_expr, sketcher, measure_pack, post_prune=False):
    # Initialize seeds for pruning (set once at start)
    if post_prune: 
        scope_name = "pp_pruning"
    else:
        scope_name = "pruning"
    with Stats.scope(scope_name):
        cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
        cur_n_prim = len(gather_primitives(in_expr))
        cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
        Stats.record("init_recon_measure", cur_recon_measure)
        Stats.record("init_n_prim", cur_n_prim)
        Stats.record("init_obj", cur_obj)

        sel_opt_program, best_recon_measure, best_n_prim, best_obj = sampling_based_pruning(in_expr, sketcher, measure_pack)
        Stats.record("sampling_based_recon_measure", best_recon_measure)
        Stats.record("sampling_based_n_prim", best_n_prim)
        Stats.record("sampling_based_obj", best_obj)
        
        sel_opt_program, best_recon_measure, best_n_prim, best_obj = top_down_pruning(sel_opt_program, sketcher, measure_pack)
        Stats.record("top_down_recon_measure", best_recon_measure)
        Stats.record("top_down_n_prim", best_n_prim)
        Stats.record("top_down_obj", best_obj)
        
        sel_opt_program, best_recon_measure, best_n_prim, best_obj = leave_one_out_greedy_pruning(sel_opt_program, sketcher, measure_pack)
        Stats.record("leave_one_out_greedy_recon_measure", best_recon_measure)
        Stats.record("leave_one_out_greedy_n_prim", best_n_prim)
        Stats.record("leave_one_out_greedy_obj", best_obj)
        
        sel_opt_program, best_recon_measure, best_n_prim, best_obj = prune_tiny_parts(sel_opt_program, sketcher, measure_pack)
        Stats.record("prune_tiny_parts_recon_measure", best_recon_measure)
        Stats.record("prune_tiny_parts_n_prim", best_n_prim)
        Stats.record("prune_tiny_parts_obj", best_obj)
        
        sel_opt_program_v2, best_recon_measure_v2, best_n_prim_v2, best_obj_v2 = prune_tiny_parts_v2(sel_opt_program, sketcher, measure_pack)
        Stats.record("prune_tiny_parts_v2_recon_measure", best_recon_measure_v2)
        Stats.record("prune_tiny_parts_v2_n_prim", best_n_prim_v2)
        Stats.record("prune_tiny_parts_v2_obj", best_obj_v2)

        Stats.record("running_recon_measure", best_recon_measure_v2)
        Stats.record("running_n_prim", best_n_prim_v2)
        Stats.record("running_obj", best_obj_v2)

        real_measure = measure_pack.measure
        measure_pack.measure = "iou"
        cur_iou_v2 = get_recon_measure(sel_opt_program_v2, sketcher, measure_pack)
        cur_iou = get_recon_measure(sel_opt_program, sketcher, measure_pack)
        measure_pack.measure = real_measure
        Stats.record("running_iou", cur_iou_v2)

        if best_obj_v2 > best_obj:
            best_program = sel_opt_program_v2
            best_recon_measure = best_recon_measure_v2
            best_n_prim = best_n_prim_v2
            best_obj = best_obj
            best_iou = cur_iou_v2
        else:
            best_program = sel_opt_program
            best_iou = cur_iou
        Stats.record("best_iou", best_iou)
        Stats.record("best_recon_measure", best_recon_measure)
        Stats.record("best_n_prim", best_n_prim)
        Stats.record("best_obj", best_obj)


        logger.info(f"Final {measure_pack.measure}: {best_recon_measure:.6f}")
        logger.info(f"Final IOU: {best_iou:.6f}")
        logger.info(f"Final OBJ: {best_obj:.6f}")
        logger.info(f"Final program size: {best_n_prim}")
        # real_measure = measure_pack.measure
        # measure_pack.measure = "iou"

    
    return best_program, sel_opt_program_v2
    

def sampling_based_pruning(in_expr, sketcher, measure_pack, n_samples=100):
    
    logger.info("==== Sampling based pruning ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    logger.info(f"Initial {measure_pack.measure}: {cur_recon_measure:.6f}")
    logger.info(f"Initial OBJ: {cur_obj:.6f}")
    logger.info(f"Initial program size: {cur_n_prim}")

    n_tries, patience, temp_scale, prev_n_exprs = 0, 0, 1.0, 0
    stochastic_expr = inject_temp_param(in_expr, (temp_scale,))
    expr_set = set()
    while (prev_n_exprs < n_samples):
        new_expr = fetch_singular_expr_eval(stochastic_expr, relaxed_eval=True, remove_marker=False)
        if not new_expr is None:
            expr_set.add(new_expr.sympy())
        n_exprs = len(expr_set)
        if patience > SAMPLING_PATIENCE:
            temp_scale *= TEMP_SCALE_FACTOR
            stochastic_expr = inject_temp_param(in_expr, (temp_scale,))
            patience = 0
        if n_exprs == prev_n_exprs:
            patience += 1
        n_tries += 1
        prev_n_exprs = n_exprs
        if n_tries > MAX_SAMPLING_TRIES:
            break
    expr_set = list(expr_set)
    logger.info(f"Found {len(expr_set)} unique expressions")
    # Batch eval:
    recon_measures, n_prims, objs = batch_eval(expr_set, measure_pack, sketcher)
    best_ind = th.argmax(objs).item()
    best_expr = expr_set[best_ind]
    logger.info(f"Best {measure_pack.measure}: {recon_measures[best_ind]:.6f}")
    logger.info(f"Best OBJ: {objs[best_ind]:.6f}")
    logger.info(f"Best program size: {n_prims[best_ind]}")
    best_recon_measure = recon_measures[best_ind]
    best_obj = objs[best_ind]
    best_n_prim = n_prims[best_ind]
    return best_expr, best_recon_measure, best_n_prim, best_obj

def top_down_pruning(in_expr, sketcher, measure_pack):

    logger.info("==== Top down pruning ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    logger.info(f"Initial {measure_pack.measure}: {cur_recon_measure:.6f}")
    logger.info(f"Initial OBJ: {cur_obj:.6f}")
    logger.info(f"Initial program size: {cur_n_prim}")

    top_down_alternatives = gather_top_down_alterns(in_expr)
    top_down_alternatives.append(in_expr)
    recon_measures, n_prims, objs = batch_eval(top_down_alternatives, measure_pack, sketcher)
    best_ind = th.argmax(objs)
    best_expr = top_down_alternatives[best_ind].tensor(dtype=sketcher.dtype)
    logger.info(f"Best {measure_pack.measure}: {recon_measures[best_ind]:.6f}")
    logger.info(f"Best OBJ: {objs[best_ind]:.6f}")
    logger.info(f"Best program size: {n_prims[best_ind]}")
    best_recon_measure = recon_measures[best_ind]
    best_obj = objs[best_ind]
    best_n_prim = n_prims[best_ind]
    return best_expr, best_recon_measure, best_n_prim, best_obj

def top_down_pruning_recursive(in_expr, sketcher, measure_pack):
    logger.info("==== Top down pruning recursive ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    logger.info(f"Initial {measure_pack.measure}: {cur_recon_measure:.6f}")
    logger.info(f"Initial OBJ: {cur_obj:.6f}")
    logger.info(f"Initial program size: {cur_n_prim}")

    best_expr, best_recon_measure, best_n_prim, best_obj = __top_down_pruning(in_expr, sketcher, measure_pack)
    logger.info(f"Best {measure_pack.measure}: {best_recon_measure:.6f}")
    logger.info(f"Best OBJ: {best_obj:.6f}")
    logger.info(f"Best program size: {best_n_prim}")
    return best_expr, best_recon_measure, best_n_prim, best_obj

def leave_one_out_pruning(in_expr, sketcher, measure_pack):
    logger.info("==== Leave one out pruning ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    logger.info(f"Initial {measure_pack.measure}: {cur_recon_measure:.6f}")
    logger.info(f"Initial OBJ: {cur_obj:.6f}")
    logger.info(f"Initial program size: {cur_n_prim}")

    # Check which singular ones can be left out.
    # Try all NCN combinations of leaving them out. 
    primitives = gather_primitives(in_expr)
    sm_ops = gather_smooth_union_ops(in_expr)
    if len(primitives) <= 1:
        logger.info("==== Only one primitive found ====")
        return in_expr, cur_recon_measure, cur_n_prim, cur_obj

    instance_dropout_alternatives = gather_instance_dropout_alternatives(in_expr)


    recon_measures, n_prims, objs = batch_eval(instance_dropout_alternatives, measure_pack, sketcher)
    n_alterns = len(instance_dropout_alternatives)
    loo_valids = []
    for i in range(n_alterns):
        if objs[i] >= cur_obj:
            loo_valids.append(i)
    if 0 in loo_valids:
        loo_valids.remove(0)
    if len(loo_valids) == 0:
        # Early exit!
        return in_expr, cur_recon_measure, cur_n_prim, cur_obj
    logger.info(f"==== Found {len(loo_valids)} drop options ====")

    n_selects = len(loo_valids)
    best_obj = cur_obj
    best_recon_measure = cur_recon_measure
    best_n_prim = cur_n_prim
    best_expr = in_expr
    while n_selects > 0:
        logger.info(f"=== Selecting {n_selects} out of {len(loo_valids)} ====")
        removal_options = list(itertools.combinations(loo_valids, n_selects))
        if len(removal_options) > MAX_INNER_LOOP_LEAVE_ONE_OUT:
            logger.info(f"==== Sampling {MAX_INNER_LOOP_LEAVE_ONE_OUT} out of {len(removal_options)} ====")
            removal_options = random.sample(removal_options, MAX_INNER_LOOP_LEAVE_ONE_OUT)
        all_exprs = []
        for removal_option in removal_options:
            temp_ops = [x for ind, x in enumerate(sm_ops) if ind +1 not in removal_option]
            temp_primitives = [x for ind, x in enumerate(primitives) if ind not in removal_option]
            temp_expr = generate_from_sm_ops_and_primitives(temp_ops, temp_primitives)
            if not temp_expr is None:
                all_exprs.append(temp_expr)
        recon_measures, n_prims, objs = batch_eval(all_exprs, measure_pack, sketcher)
        recon_measures = np.stack(recon_measures)
        cur_best_ind = np.argmax(recon_measures)
        cur_best_recon_measure = recon_measures[cur_best_ind]   
        cur_best_n_prim = n_prims[cur_best_ind]
        cur_best_obj = objs[cur_best_ind]
        cur_best_expr = all_exprs[cur_best_ind]
        if cur_best_obj >= best_obj:
            best_expr = cur_best_expr
            best_recon_measure = cur_best_recon_measure
            best_n_prim = cur_best_n_prim
            best_obj = cur_best_obj
            break
        else:
            n_selects -=1
    logger.info(f"Best {measure_pack.measure}: {best_recon_measure:.6f}")
    logger.info(f"Best OBJ: {best_obj:.6f}")
    logger.info(f"Best program size: {best_n_prim}")
    return best_expr, best_recon_measure, best_n_prim, best_obj

def leave_one_out_greedy_pruning(in_expr, sketcher, measure_pack):
    logger.info("==== Leave one out greedy pruning ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    logger.info(f"Initial {measure_pack.measure}: {cur_recon_measure:.6f}")
    logger.info(f"Initial OBJ: {cur_obj:.6f}")
    logger.info(f"Initial program size: {cur_n_prim}")

    # Check which singular ones can be left out.
    # Try all NCN combinations of leaving them out. 
    primitives = gather_primitives(in_expr)
    sm_ops = gather_smooth_union_ops(in_expr)
    if len(primitives) <= 1:
        logger.info("==== Only one primitive found ====")
        return in_expr, cur_recon_measure, cur_n_prim, cur_obj

    instance_dropout_alternatives = gather_instance_dropout_alternatives(in_expr)

    if len(instance_dropout_alternatives) == 0:
        logger.info("==== No drop options found ====")
        return in_expr, cur_recon_measure, cur_n_prim, cur_obj
    else:
        logger.info(f"==== Found {len(instance_dropout_alternatives)} drop options ====")

    recon_measures, n_prims, objs = batch_eval(instance_dropout_alternatives, measure_pack, sketcher)
    n_alterns = len(instance_dropout_alternatives)
    loo_valids = []
    for i in range(n_alterns):
        if objs[i] >= cur_obj:
            loo_valids.append(i)
    if len(loo_valids) == 0:
        # Early exit!
        return in_expr, cur_recon_measure, cur_n_prim, cur_obj
    logger.info(f"==== Found {len(loo_valids)} drop options ====")
    try_0_removal = False
    if 0 in loo_valids:
        # Remove it separately?
        loo_valids.remove(0)
        try_0_removal = True
    candidate_removals = [x for x in loo_valids]
    best_obj = cur_obj
    best_recon_measure = cur_recon_measure
    best_n_prim = cur_n_prim
    best_expr = in_expr
    additional_removes = []

    while len(candidate_removals) > 0:
        all_exprs = []
        for removal_option in candidate_removals:
            cur_removes = [removal_option] + additional_removes
            temp_ops = [x for ind, x in enumerate(sm_ops) if ind + 1 not in cur_removes]
            temp_primitives = [x for ind, x in enumerate(primitives) if ind not in cur_removes]
            temp_expr = generate_from_sm_ops_and_primitives(temp_ops, temp_primitives)
            if not temp_expr is None:
                all_exprs.append(temp_expr)
        if not all_exprs:
            logger.info("==== No valid expressions found ====")
            break
        recon_measures, n_prims, objs = batch_eval(all_exprs, measure_pack, sketcher)
        recon_measures = np.stack(recon_measures)
        cur_best_ind = np.argmax(recon_measures)
        cur_best_recon_measure = recon_measures[cur_best_ind]   
        cur_best_n_prim = n_prims[cur_best_ind]
        cur_best_obj = objs[cur_best_ind]
        cur_best_expr = all_exprs[cur_best_ind]
        best_option = candidate_removals[cur_best_ind]
        if cur_best_obj >= best_obj:
            logger.info(f"Best option {best_option} improved from {best_obj:.6f} to {cur_best_obj:.6f}")
            best_expr = cur_best_expr
            best_recon_measure = cur_best_recon_measure
            best_n_prim = cur_best_n_prim
            best_obj = cur_best_obj
            candidate_removals = [x for x in candidate_removals if x != best_option]
            # And also those who lowered the score.
            # for ind in range(len(candidate_removals)):
            #     if objs[ind] < best_obj:
            #         candidate_removals.remove(candidate_removals[ind])
            additional_removes.append(best_option)
        else:
            logger.info(f"Best option {best_option} did not improve from {best_obj:.6f} to {cur_best_obj:.6f}")
            break
    if try_0_removal:
        logger.info("==== Trying 0 removal ====")
        primitives = gather_primitives(best_expr)
        sm_ops = gather_smooth_union_ops(best_expr)
        temp_ops = [x for ind, x in enumerate(sm_ops) if ind != 0]
        temp_primitives = [x for ind, x in enumerate(primitives) if ind != 0]
        
        temp_expr = generate_from_sm_ops_and_primitives(temp_ops, temp_primitives)

        cur_recon_measure = get_recon_measure(temp_expr, sketcher, measure_pack)
        cur_n_prim = len(gather_primitives(temp_expr))
        cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
        if cur_obj >= best_obj:
            logger.info(f"0 removal improved from {best_obj:.6f} to {cur_obj:.6f}")
            best_expr = temp_expr
            best_recon_measure = cur_recon_measure
            best_n_prim = cur_n_prim
            best_obj = cur_obj
        
    logger.info(f"Best {measure_pack.measure}: {best_recon_measure:.6f}")
    logger.info(f"Best OBJ: {best_obj:.6f}")
    logger.info(f"Best program size: {best_n_prim}")
    return best_expr, best_recon_measure, best_n_prim, best_obj

# Alternative - For tiny parts, if Exec(P-A) > 0, or if I(A, T) = 0, then remove A
def prune_tiny_parts(in_expr, sketcher, measure_pack):

    logger.info("==== Prune Tiny Parts greedy pruning ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    logger.info(f"Initial {measure_pack.measure}: {cur_recon_measure:.6f}")
    logger.info(f"Initial OBJ: {cur_obj:.6f}")
    logger.info(f"Initial program size: {cur_n_prim}")

    # Check which singular ones can be left out.
    # Try all NCN combinations of leaving them out. 
    primitives = gather_primitives(in_expr)
    sm_ops = gather_smooth_union_ops(in_expr)
    if len(primitives) <= 1:
        logger.info("==== Only one primitive found ====")
        return in_expr, cur_recon_measure, cur_n_prim, cur_obj

    n_prims = []
    all_execs = []
    for expr in primitives:
        cur_exec = recursive_evaluate(expr.tensor(dtype=sketcher.dtype), sketcher, relaxed_eval=False)
        all_execs.append(cur_exec)
        n_prims.append(len(gather_primitives(expr)))
    all_execs = th.stack(all_execs, dim=0)

    vol = (all_execs <= 0).float().sum(dim=1) / (sketcher.resolution ** 3)
    removal_options = [ind for ind, x in enumerate(vol) if x < MIN_VOL_LIMIT]

    if 0 in removal_options:
        removal_options.remove(0)
    if len(removal_options) == 0:
        logger.info("==== No tiny parts found ====")
        # Early exit!
        return in_expr, cur_recon_measure, cur_n_prim, cur_obj
        
    logger.info(f"==== Found {len(removal_options)} drop options ====")
    candidate_removals = [x for x in removal_options]
    best_obj = cur_obj
    best_recon_measure = cur_recon_measure
    best_n_prim = cur_n_prim
    best_expr = in_expr
    additional_removes = []
    while len(candidate_removals) > 0:
        all_exprs = []
        for removal_option in candidate_removals:
            cur_removes = [removal_option] + additional_removes
            temp_ops = [x for ind, x in enumerate(sm_ops) if ind +1 not in cur_removes]
            temp_primitives = [x for ind, x in enumerate(primitives) if ind not in cur_removes]
            temp_expr = generate_from_sm_ops_and_primitives(temp_ops, temp_primitives)
            if not temp_expr is None:
                all_exprs.append(temp_expr)
        if not all_exprs:
            logger.info("==== No valid expressions found ====")
            break
        recon_measures, n_prims, objs = batch_eval(all_exprs, measure_pack, sketcher)
        recon_measures = np.stack(recon_measures)
        cur_best_ind = np.argmax(recon_measures)
        cur_best_recon_measure = recon_measures[cur_best_ind]   
        cur_best_n_prim = n_prims[cur_best_ind]
        cur_best_obj = objs[cur_best_ind]
        cur_best_expr = all_exprs[cur_best_ind]
        best_option = candidate_removals[cur_best_ind]
        if cur_best_obj >= best_obj:
            logger.info(f"Best option {best_option} improved from {best_obj:.6f} to {cur_best_obj:.6f}")
            best_expr = cur_best_expr
            best_recon_measure = cur_best_recon_measure
            best_n_prim = cur_best_n_prim
            best_obj = cur_best_obj
            candidate_removals = [x for x in candidate_removals if x != best_option]
            additional_removes.append(best_option)
            # for ind in range(len(candidate_removals)):
            #     if objs[ind] < best_obj:
            #         candidate_removals.remove(candidate_removals[ind])
        else:
            logger.info(f"Best option {best_option} did not improve from {best_obj:.6f} to {cur_best_obj:.6f}")
            break
    logger.info(f"Best {measure_pack.measure}: {best_recon_measure:.6f}")
    logger.info(f"Best OBJ: {best_obj:.6f}")
    logger.info(f"Best program size: {best_n_prim}")
    return best_expr, best_recon_measure, best_n_prim, best_obj


# Alternative - For tiny parts, if I(A, T) = 0, then remove A
def prune_tiny_parts_v2(in_expr, sketcher, measure_pack):

    logger.info("==== Prune Tiny Parts V2 greedy pruning ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    logger.info(f"Initial {measure_pack.measure}: {cur_recon_measure:.6f}")
    logger.info(f"Initial OBJ: {cur_obj:.6f}")
    logger.info(f"Initial program size: {cur_n_prim}")

    # Check which singular ones can be left out.
    # Try all NCN combinations of leaving them out. 
    primitives = gather_primitives(in_expr)
    sm_ops = gather_smooth_union_ops(in_expr)
    if len(primitives) <= 1:
        logger.info("==== Only one primitive found ====")
        return in_expr, cur_recon_measure, cur_n_prim, cur_obj

    n_prims = []
    all_execs = []
    for expr in primitives:
        cur_exec = recursive_evaluate(expr.tensor(dtype=sketcher.dtype), sketcher, relaxed_eval=False)
        all_execs.append(cur_exec)
        n_prims.append(len(gather_primitives(expr)))
    all_execs = th.stack(all_execs, dim=0)
    
    all_occs = all_execs <= 0
    vol = (all_occs).float().sum(dim=1) # / (sketcher.resolution ** 3)
    # Second condition - I(A, T) = 0
    intersection_measures = th.logical_and(all_occs, measure_pack.target_sdf[None, :] <= 0).float().sum(dim=1)
    # intersection_measures = intersection_measures / (sketcher.resolution ** 3)
    intersection_measures = intersection_measures / vol
    # vol_removal_options = [ind for ind, x in enumerate(vol) if x < MIN_VOL_LIMIT]
    intersection_removal_options = [ind for ind, x in enumerate(intersection_measures) if x < 0.6]
    removal_options = intersection_removal_options# set(vol_removal_options).intersection(set(intersection_removal_options))
    try_0_removal = False
    if 0 in removal_options:
        try_0_removal = True
        removal_options.remove(0)
    
    if len(removal_options) == 0:
        logger.info("==== No tiny parts found ====")
        # Early exit!
        return in_expr, cur_recon_measure, cur_n_prim, cur_obj

    logger.info(f"==== Found {len(removal_options)} prims to drop ====")
    temp_ops = [x for ind, x in enumerate(sm_ops) if ind +1 not in removal_options]
    temp_primitives = [x for ind, x in enumerate(primitives) if ind not in removal_options]
    if try_0_removal and len(temp_ops) > 1:
        # Is this correct?
        temp_ops = temp_ops[1:]
        temp_primitives = temp_primitives[1:]
    best_expr = generate_from_sm_ops_and_primitives(temp_ops, temp_primitives)

    best_recon_measure = get_recon_measure(best_expr, sketcher, measure_pack)
    best_n_prim = len(gather_primitives(best_expr))
    best_obj = best_recon_measure + measure_pack.len_weight * best_n_prim
    logger.info(f"Best {measure_pack.measure}: {best_recon_measure:.6f}")
    logger.info(f"Best OBJ: {best_obj:.6f}")
    logger.info(f"Best program size: {best_n_prim}")
    return best_expr, best_recon_measure, best_n_prim, best_obj



def __top_down_pruning(in_expr, sketcher, measure_pack, best_expr=None, best_recon_measure=0.0, best_n_prim=0, best_obj=0.0):
    if best_expr is None:
        best_expr = in_expr
    # At each instance try the child OP. 
    # if it is lower -> mark as best program and go down. 
    # else just return
    if isinstance(in_expr, gls.SmoothUnion):
        check_child_1, check_child_2 = False, False
        test_expr_1 = in_expr.args[0]
        cur_recon_measure = get_recon_measure(test_expr_1, sketcher, measure_pack)
        cur_n_prim = len(gather_primitives(test_expr_1))
        cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
        if cur_obj >= best_obj:
            best_expr = test_expr_1
            best_recon_measure = cur_recon_measure
            best_n_prim = cur_n_prim
            best_obj = cur_obj
            check_child_1 = True
        test_expr_2 = in_expr.args[1]
        cur_recon_measure = get_recon_measure(test_expr_2, sketcher, measure_pack)
        cur_n_prim = len(gather_primitives(test_expr_2))
        cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
        if cur_obj >= best_obj:
            best_expr = test_expr_2
            best_recon_measure = cur_recon_measure
            best_n_prim = cur_n_prim
            best_obj = cur_obj   
            check_child_2 = True 
        if check_child_1:
            best_expr, best_recon_measure, best_n_prim, best_obj = __top_down_pruning(test_expr_1, sketcher, measure_pack, best_expr, best_recon_measure, best_n_prim, best_obj)
        if check_child_2:
            best_expr, best_recon_measure, best_n_prim, best_obj = __top_down_pruning(test_expr_2, sketcher, measure_pack, best_expr, best_recon_measure, best_n_prim, best_obj)
    elif isinstance(in_expr, gls.GLFunction):
        # just check eval and return. 
        new_args = []
        for arg in in_expr.args:
            if isinstance(arg, gls.GLFunction):
                best_expr, best_recon_measure, best_n_prim, best_obj = __top_down_pruning(arg, sketcher, measure_pack, best_expr, best_recon_measure, best_n_prim, best_obj)
    return best_expr, best_recon_measure, best_n_prim, best_obj


def gather_top_down_alterns(program, prim_list=None):
    if prim_list is None:
        prim_list = []

    if isinstance(program, gls.SmoothUnion):
        prim_list.append(program.args[0])
        new_prim_list = gather_top_down_alterns(program.args[0])
        prim_list.extend(new_prim_list)
        prim_list.append(program.args[1])
        new_prim_list = gather_top_down_alterns(program.args[1])
        prim_list.extend(new_prim_list)
        return prim_list
    else:
        if isinstance(program, gls.GLFunction):
            for arg in program.args:
                new_prim_list = gather_top_down_alterns(arg)
                prim_list.extend(new_prim_list)
        return prim_list


def batch_eval(expr_set, measure_pack, sketcher):

    n_prims = []
    for expr in expr_set:
        n_prims.append(len(gather_primitives(expr)))
    recon_measures = get_recon_measure_packed(expr_set, sketcher,measure_pack)
    if isinstance(recon_measures, th.Tensor):
        recon_measures = [x.item() for x in recon_measures]
    target_sdf = measure_pack.target_sdf
    objs = th.tensor(recon_measures).to(target_sdf.device) + measure_pack.len_weight * th.tensor(n_prims).to(target_sdf.device)

    obj = [x.item() for x in objs]
    best_ind = th.argmax(objs)
    # Log best and worst metrics
    logger.info(f"Lowest {measure_pack.measure}: {min(recon_measures):.6f}")
    logger.info(f"Highest {measure_pack.measure}: {max(recon_measures):.6f}")
    # length of the program
    logger.info(f"Lowest program size: {min(n_prims)}")
    logger.info(f"Highest program size: {max(n_prims)}")
    # obj
    logger.info(f"Lowest OBJ: {min(obj):.6f}")
    logger.info(f"Highest OBJ: {max(obj):.6f}")
    return recon_measures, n_prims, objs