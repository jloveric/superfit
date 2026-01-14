from dataclasses import dataclass
import torch as th
import numpy as np
import trimesh
import random
import itertools
import geolipi.symbolic as gls
from geolipi.torch_compute.evaluate_expression import recursive_evaluate

from ..symbolic.utils import inject_temp_param, fetch_singular_expr, gather_primitives
from .eval import get_recon_measure, get_recon_measure_packed, MeasurePack
from ..symbolic.utils import gather_instance_dropout_alternatives, gather_smooth_union_ops, generate_from_sm_ops_and_primitives
from ..utils.config import AlgorithmConfig as AlgConf

MAX_SAMPLING_TRIES = 100
SAMPLING_PATIENCE = 10
TEMP_SCALE_FACTOR = 1.5
MAX_INNER_LOOP_LEAVE_ONE_OUT = 100
MIN_VOL_LIMIT = 1e-4

def main_pruning_pipeline(in_expr, mesh, target_sdf, sketcher, measure=None):

    measure_pack = MeasurePack(
        measure=AlgConf.PRUNE_METRIC,
        target_mesh=mesh,
        target_sdf=target_sdf,
        len_weight=AlgConf.MPS_LEN_WEIGHT
    )
    stats = {}
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    print(f"Initial {measure_pack.measure}: {cur_recon_measure}")
    print(f"Initial OBJ: {cur_obj}")
    print(f"Initial program size: {cur_n_prim}")
    stats = {
        "pruning_initial_recon_measure": cur_recon_measure,
        "pruning_initial_n_prim": cur_n_prim,
        "pruning_initial_obj": cur_obj,
    }

    sel_opt_program, best_recon_measure, best_n_prim, best_obj = sampling_based_pruning(in_expr, sketcher, measure_pack)
    stats["pruning_sampling_based_recon_measure"] = best_recon_measure
    stats["pruning_sampling_based_n_prim"] = best_n_prim
    stats["pruning_sampling_based_obj"] = best_obj
    sel_opt_program, best_recon_measure, best_n_prim, best_obj = top_down_pruning(sel_opt_program, sketcher, measure_pack)
    stats["pruning_top_down_recon_measure"] = best_recon_measure
    stats["pruning_top_down_n_prim"] = best_n_prim
    stats["pruning_top_down_obj"] = best_obj
    # sel_opt_program, best_recon_measure, best_n_prim, best_obj = leave_one_out_pruning(opt_program, sketcher_3d, measure_pack)
    sel_opt_program, best_recon_measure, best_n_prim, best_obj = leave_one_out_greedy_pruning(sel_opt_program, sketcher, measure_pack)
    stats["pruning_leave_one_out_greedy_recon_measure"] = best_recon_measure
    stats["pruning_leave_one_out_greedy_n_prim"] = best_n_prim
    stats["pruning_leave_one_out_greedy_obj"] = best_obj
    sel_opt_program, best_recon_measure, best_n_prim, best_obj = prune_tiny_parts(sel_opt_program, sketcher, measure_pack)
    stats["pruning_prune_tiny_parts_recon_measure"] = best_recon_measure
    stats["pruning_prune_tiny_parts_n_prim"] = best_n_prim
    stats["pruning_prune_tiny_parts_obj"] = best_obj
    sel_opt_program, best_recon_measure, best_n_prim, best_obj = prune_tiny_parts_v2(sel_opt_program, sketcher, measure_pack)
    stats["pruning_prune_tiny_parts_recon_measure"] = best_recon_measure
    stats["pruning_prune_tiny_parts_n_prim"] = best_n_prim
    stats["pruning_prune_tiny_parts_obj"] = best_obj
    # sel_opt_program, best_recon_measure, best_n_prim, best_obj = top_down_pruning_recursive(opt_program, sketcher_3d, measure_pack)
    # Alternative subtractions:
    cur_recon_measure = get_recon_measure(sel_opt_program, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(sel_opt_program))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    print(f"Final {measure_pack.measure}: {cur_recon_measure}")
    print(f"Final OBJ: {cur_obj}")
    print(f"Final program size: {cur_n_prim}")
    real_measure = measure_pack.measure
    measure_pack.measure = "iou"
    cur_iou = get_recon_measure(sel_opt_program, sketcher, measure_pack)
    best_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    measure_pack.measure = real_measure
    stats["best_iou"] = cur_iou
    stats["best_recon_measure"] = cur_recon_measure
    stats["best_n_prim"] = cur_n_prim
    stats["best_obj"] = best_obj
    return sel_opt_program, stats
    

def sampling_based_pruning(in_expr, sketcher, measure_pack, n_samples=100):
    
    print("==== Sampling based pruning ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    print(f"Initial {measure_pack.measure}: {cur_recon_measure}")
    print(f"Initial OBJ: {cur_obj}")
    print(f"Initial program size: {cur_n_prim}")

    n_tries, patience, temp_scale, prev_n_exprs = 0, 0, 1.0, 0
    stochastic_expr = inject_temp_param(in_expr, (temp_scale,))
    expr_set = set()
    while (prev_n_exprs < n_samples):
        new_expr = fetch_singular_expr(stochastic_expr, relaxed_eval=True, remove_marker=False)
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
    print(f"found {len(expr_set)} unique expressions")
    # Batch eval:
    recon_measures, n_prims, objs = batch_eval(expr_set, measure_pack, sketcher)
    best_ind = th.argmax(objs)
    best_expr = expr_set[best_ind].tensor()
    print(f"Best {measure_pack.measure}: {recon_measures[best_ind]}")
    print(f"Best OBJ: {objs[best_ind]}")
    print(f"Best program size: {n_prims[best_ind]}")
    best_recon_measure = recon_measures[best_ind]
    best_obj = objs[best_ind]
    best_n_prim = n_prims[best_ind]
    return best_expr, best_recon_measure, best_n_prim, best_obj

def top_down_pruning(in_expr, sketcher, measure_pack):

    print("==== Top down pruning ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    print(f"Initial {measure_pack.measure}: {cur_recon_measure}")
    print(f"Initial OBJ: {cur_obj}")
    print(f"Initial program size: {cur_n_prim}")

    top_down_alternatives = gather_top_down_alterns(in_expr)
    top_down_alternatives.append(in_expr)
    recon_measures, n_prims, objs = batch_eval(top_down_alternatives, measure_pack, sketcher)
    best_ind = th.argmax(objs)
    best_expr = top_down_alternatives[best_ind].tensor()
    print(f"Best {measure_pack.measure}: {recon_measures[best_ind]}")
    print(f"Best OBJ: {objs[best_ind]}")
    print(f"Best program size: {n_prims[best_ind]}")
    best_recon_measure = recon_measures[best_ind]
    best_obj = objs[best_ind]
    best_n_prim = n_prims[best_ind]
    return best_expr, best_recon_measure, best_n_prim, best_obj

def top_down_pruning_recursive(in_expr, sketcher, measure_pack):
    print("==== Top down pruning recursive ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    print(f"Initial {measure_pack.measure}: {cur_recon_measure}")
    print(f"Initial OBJ: {cur_obj}")
    print(f"Initial program size: {cur_n_prim}")

    best_expr, best_recon_measure, best_n_prim, best_obj = __top_down_pruning(in_expr, sketcher, measure_pack)
    print(f"Best {measure_pack.measure}: {best_recon_measure}")
    print(f"Best OBJ: {best_obj}")
    print(f"Best program size: {best_n_prim}")
    return best_expr, best_recon_measure, best_n_prim, best_obj

def leave_one_out_pruning(in_expr, sketcher, measure_pack):
    print("==== Leave one out pruning ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    print(f"Initial {measure_pack.measure}: {cur_recon_measure}")
    print(f"Initial OBJ: {cur_obj}")
    print(f"Initial program size: {cur_n_prim}")

    # Check which singular ones can be left out.
    # Try all NCN combinations of leaving them out. 
    primitives = gather_primitives(in_expr)
    sm_ops = gather_smooth_union_ops(in_expr)

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
    print(f"==== found {len(loo_valids)} drop options ====")

    n_selects = len(loo_valids)
    best_obj = cur_obj
    best_recon_measure = cur_recon_measure
    best_n_prim = cur_n_prim
    best_expr = in_expr
    while n_selects > 0:
        print(f"=== Selecting {n_selects} out of {len(loo_valids)} ====")
        removal_options = list(itertools.combinations(loo_valids, n_selects))
        if len(removal_options) > MAX_INNER_LOOP_LEAVE_ONE_OUT:
            print(f"==== Sampling {MAX_INNER_LOOP_LEAVE_ONE_OUT} out of {len(removal_options)} ====")
            removal_options = random.sample(removal_options, MAX_INNER_LOOP_LEAVE_ONE_OUT)
        all_exprs = []
        for removal_option in removal_options:
            temp_ops = [x for ind, x in enumerate(sm_ops) if ind +1 not in removal_option]
            temp_primitives = [x for ind, x in enumerate(primitives) if ind not in removal_option]
            temp_expr = generate_from_sm_ops_and_primitives(temp_ops, temp_primitives)
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
    print(f"Best {measure_pack.measure}: {best_recon_measure}")
    print(f"Best OBJ: {best_obj}")
    print(f"Best program size: {best_n_prim}")
    return best_expr, best_recon_measure, best_n_prim, best_obj

def leave_one_out_greedy_pruning(in_expr, sketcher, measure_pack):
    print("==== Leave one out greedy pruning ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    print(f"Initial {measure_pack.measure}: {cur_recon_measure}")
    print(f"Initial OBJ: {cur_obj}")
    print(f"Initial program size: {cur_n_prim}")

    # Check which singular ones can be left out.
    # Try all NCN combinations of leaving them out. 
    primitives = gather_primitives(in_expr)
    sm_ops = gather_smooth_union_ops(in_expr)

    instance_dropout_alternatives = gather_instance_dropout_alternatives(in_expr)

    if len(instance_dropout_alternatives) == 0:
        print("==== No drop options found ====")
        return in_expr, cur_recon_measure, cur_n_prim, cur_obj
    else:
        print(f"==== found {len(instance_dropout_alternatives)} drop options ====")

    recon_measures, n_prims, objs = batch_eval(instance_dropout_alternatives, measure_pack, sketcher)
    n_alterns = len(instance_dropout_alternatives)
    loo_valids = []
    for i in range(n_alterns):
        if objs[i] >= cur_obj:
            loo_valids.append(i)
    if len(loo_valids) == 0:
        # Early exit!
        return in_expr, cur_recon_measure, cur_n_prim, cur_obj
    print(f"==== found {len(loo_valids)} drop options ====")
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
            all_exprs.append(temp_expr)
        recon_measures, n_prims, objs = batch_eval(all_exprs, measure_pack, sketcher)
        recon_measures = np.stack(recon_measures)
        cur_best_ind = np.argmax(recon_measures)
        cur_best_recon_measure = recon_measures[cur_best_ind]   
        cur_best_n_prim = n_prims[cur_best_ind]
        cur_best_obj = objs[cur_best_ind]
        cur_best_expr = all_exprs[cur_best_ind]
        best_option = candidate_removals[cur_best_ind]
        if cur_best_obj >= best_obj:
            print(f"Best option {best_option} improved from {best_obj} to {cur_best_obj}")
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
            print(f"Best option {best_option} did not improve from {best_obj} to {cur_best_obj}")
            break
    if try_0_removal:
        print("==== Trying 0 removal ====")
        primitives = gather_primitives(best_expr)
        sm_ops = gather_smooth_union_ops(best_expr)
        temp_ops = [x for ind, x in enumerate(sm_ops) if ind != 0]
        temp_primitives = [x for ind, x in enumerate(primitives) if ind != 0]
        
        temp_expr = generate_from_sm_ops_and_primitives(temp_ops, temp_primitives)

        cur_recon_measure = get_recon_measure(temp_expr, sketcher, measure_pack)
        cur_n_prim = len(gather_primitives(temp_expr))
        cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
        if best_obj >= best_obj:
            print(f"0 removal improved from {best_obj} to {cur_obj}")
            best_expr = temp_expr
            best_recon_measure = cur_recon_measure
            best_n_prim = cur_n_prim
            best_obj = cur_obj
        
    print(f"Best {measure_pack.measure}: {best_recon_measure}")
    print(f"Best OBJ: {best_obj}")
    print(f"Best program size: {best_n_prim}")
    return best_expr, best_recon_measure, best_n_prim, best_obj

# Alternative - For tiny parts, if Exec(P-A) > 0, or if I(A, T) = 0, then remove A
def prune_tiny_parts(in_expr, sketcher, measure_pack):

    print("==== Prune Tiny Parts greedy pruning ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    print(f"Initial {measure_pack.measure}: {cur_recon_measure}")
    print(f"Initial OBJ: {cur_obj}")
    print(f"Initial program size: {cur_n_prim}")

    # Check which singular ones can be left out.
    # Try all NCN combinations of leaving them out. 
    primitives = gather_primitives(in_expr)
    sm_ops = gather_smooth_union_ops(in_expr)

    n_prims = []
    all_execs = []
    for expr in primitives:
        cur_exec = recursive_evaluate(expr.tensor(), sketcher, relaxed_eval=False)
        all_execs.append(cur_exec)
        n_prims.append(len(gather_primitives(expr)))
    all_execs = th.stack(all_execs, dim=0)

    vol = (all_execs <= 0).float().sum(dim=1) / (sketcher.resolution ** 3)
    removal_options = [ind for ind, x in enumerate(vol) if x < MIN_VOL_LIMIT]

    if 0 in removal_options:
        removal_options.remove(0)
    if len(removal_options) == 0:
        print("==== No tiny parts found ====")
        # Early exit!
        return in_expr, cur_recon_measure, cur_n_prim, cur_obj
        
    print(f"==== found {len(removal_options)} drop options ====")
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
            all_exprs.append(temp_expr)
        recon_measures, n_prims, objs = batch_eval(all_exprs, measure_pack, sketcher)
        recon_measures = np.stack(recon_measures)
        cur_best_ind = np.argmax(recon_measures)
        cur_best_recon_measure = recon_measures[cur_best_ind]   
        cur_best_n_prim = n_prims[cur_best_ind]
        cur_best_obj = objs[cur_best_ind]
        cur_best_expr = all_exprs[cur_best_ind]
        best_option = candidate_removals[cur_best_ind]
        if cur_best_obj >= best_obj:
            print(f"Best option {best_option} improved from {best_obj} to {cur_best_obj}")
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
            print(f"Best option {best_option} did not improve from {best_obj} to {cur_best_obj}")
            break
    print(f"Best {measure_pack.measure}: {best_recon_measure}")
    print(f"Best OBJ: {best_obj}")
    print(f"Best program size: {best_n_prim}")
    return best_expr, best_recon_measure, best_n_prim, best_obj


# Alternative - For tiny parts, if I(A, T) = 0, then remove A
def prune_tiny_parts_v2(in_expr, sketcher, measure_pack):

    print("==== Prune Tiny Parts V2 greedy pruning ====")
    cur_recon_measure = get_recon_measure(in_expr, sketcher, measure_pack)
    cur_n_prim = len(gather_primitives(in_expr))
    cur_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    print(f"Initial {measure_pack.measure}: {cur_recon_measure}")
    print(f"Initial OBJ: {cur_obj}")
    print(f"Initial program size: {cur_n_prim}")

    # Check which singular ones can be left out.
    # Try all NCN combinations of leaving them out. 
    primitives = gather_primitives(in_expr)
    sm_ops = gather_smooth_union_ops(in_expr)

    n_prims = []
    all_execs = []
    for expr in primitives:
        cur_exec = recursive_evaluate(expr.tensor(), sketcher, relaxed_eval=False)
        all_execs.append(cur_exec)
        n_prims.append(len(gather_primitives(expr)))
    all_execs = th.stack(all_execs, dim=0)
    
    all_occs = all_execs <= 0
    vol = (all_occs).float().sum(dim=1) / (sketcher.resolution ** 3)
    # Second condition - I(A, T) = 0
    intersection_measures = th.logical_and(all_occs, measure_pack.target_sdf[None, :] <= 0).float().sum(dim=1)

    vol_removal_options = [ind for ind, x in enumerate(vol) if x < MIN_VOL_LIMIT]
    intersection_removal_options = [ind for ind, x in enumerate(intersection_measures) if x == 0]
    removal_options = set(vol_removal_options).intersection(set(intersection_removal_options))
    try_0_removal = False
    if 0 in removal_options:
        try_0_removal = True
        removal_options.remove(0)
    
    if len(removal_options) == 0:
        print("==== No tiny parts found ====")
        # Early exit!
        return in_expr, cur_recon_measure, cur_n_prim, cur_obj

    print(f"==== found {len(removal_options)} prims to drop ====")
    temp_ops = [x for ind, x in enumerate(sm_ops) if ind +1 not in removal_options]
    temp_primitives = [x for ind, x in enumerate(primitives) if ind not in removal_options]
    if try_0_removal and len(temp_ops) > 1:
        # Is this correct?
        temp_ops = temp_ops[1:]
        temp_primitives = temp_primitives[1:]
    best_expr = generate_from_sm_ops_and_primitives(temp_ops, temp_primitives)

    best_recon_measure = get_recon_measure(best_expr, sketcher, measure_pack)
    best_n_prim = len(gather_primitives(in_expr))
    best_obj = cur_recon_measure + measure_pack.len_weight * cur_n_prim
    print(f"Best {measure_pack.measure}: {best_recon_measure}")
    print(f"Best OBJ: {best_obj}")
    print(f"Best program size: {best_n_prim}")
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
    # print best and worst IOU
    print(f"Lowest {measure_pack.measure}: {min(recon_measures)}")
    print(f"Highest {measure_pack.measure}: {max(recon_measures)}")
    # length of the program
    print(f"Lowest program size: {min(n_prims)}")
    print(f"Highest program size: {max(n_prims)}")
    # obj
    print(f"Lowest OBJ: {min(obj)}")
    print(f"Highest OBJ: {max(obj)}")
    return recon_measures, n_prims, objs