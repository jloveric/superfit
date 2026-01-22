
import os
import ast
import time
import torch as th
import superfit.symbolic as sps
from ..symbolic.utils import (inject_temp_param, remove_temp_param,)

from ..utils.config import AlgorithmConfig as AlgConf
from ..utils.stats import Stats
from ..utils.logger import logger
from .primitive_registry import HANDLER_REGISTRY
from ..symbolic.utils import gather_primitives
from .expr_conversion import convert_to_packed, convert_to_unpacked, convert_to_batched, convert_to_unbatched
from .param_conversion import transform_to_tunable
from .main_opt import run_optimization_loop
from .fast_opt import run_optimization_loop_fast
from ..algos.eval_tools import get_recon_measure, MeasurePack
from .compile_function import compile_program_jit, compile_cached_with_dummy_opt

def optimize_primitive_assembly(in_program, target_mesh, target_sdf, sketcher,
                       measure_pack):
    """
    Optimized version of opt_till_saturation with configurable optimizer setup.
    
    Args:
        optimizer_config: Optional dict with optimizer configuration overrides
    """
    ### Other option -> for some link fixed execution and stop opt.
    version = getattr(sps, AlgConf.PRIM_TYPE)
    handler = HANDLER_REGISTRY[version]
    assert handler is not None, f"No handler found for {in_program.base_class}"



    with Stats.scope("optimization"):
        n_prims = len(gather_primitives(in_program))
        cur_recon_measure = get_recon_measure(in_program, sketcher, measure_pack)
        if isinstance(cur_recon_measure, th.Tensor):
            cur_recon_measure = cur_recon_measure.item()
        cur_obj = cur_recon_measure + AlgConf.MPS_LEN_WEIGHT * n_prims
        Stats.record("init_obj", cur_obj)
        Stats.record("init_recon_measure", cur_recon_measure)

        opt_program = in_program
        opt_program = convert_to_packed(opt_program, handler)
        opt_program = convert_to_batched(opt_program, handler)
        
        opt_program = remove_temp_param(opt_program)
        tensor_list = opt_program.gather_tensor_list(type_annotate=True, index_annotate=True)
        # _, variable_list = transform_to_tunable(tensor_list)
        variable_list = transform_to_tunable(tensor_list, handler)
        opt_var_list = [x for ind, x in enumerate(variable_list)]
        type_annotation = [tuple(x[1:]) for x in tensor_list]
        
        # Group parameters by type for different learning rates
        special_params = []
        regular_params = []
        
        for i, type_annot in enumerate(type_annotation):
            if issubclass(type_annot[0], (sps.StochasticPrimitive)):
                special_params.append(opt_var_list[i])
            elif issubclass(type_annot[0], (sps.SuperFrustumPackedBatchedStochastic, 
                                            sps.SolidSFPackedBatchedStochasticSU,
                                            )) and type_annot[2] == 2:
                special_params.append(opt_var_list[i])
            else:
                regular_params.append(opt_var_list[i])
        
        temperature = 1.0
        gmbled_opt_program = inject_temp_param(opt_program.tensor(dtype=AlgConf.OPT_DTYPE), temperature)
        
        if AlgConf.FastMode:
            compiled_func_relaxed = compile_cached_with_dummy_opt(gmbled_opt_program, sketcher, handler, torch_compile=AlgConf.TorchCompile)
            # compiled_func_relaxed = compile_program_jit_cached(gmbled_opt_program, sketcher, torch_compile=torch_compile)
        else:
            compiled_func_relaxed = compile_program_jit(gmbled_opt_program, sketcher, torch_compile=AlgConf.TorchCompile)
        
        # Create parameter groups with different learning rates
        param_groups = []
        if special_params:
            param_groups.append({'params': special_params, 'lr': AlgConf.OPT_LR_RATE * 5.0})
        if regular_params:
            param_groups.append({'params': regular_params, 'lr': AlgConf.OPT_LR_RATE})
        
        if AlgConf.LOWER_SP:
            ratio = 1 - (cur_recon_measure)**2
            AlgConf.STOCHASTIC_PRECONDITION_INIT_VAL_LOWER = AlgConf.STOCHASTIC_PRECONDITION_INIT_VAL * ratio
            logger.info(f"Lowering stochastic precondition init val to {AlgConf.STOCHASTIC_PRECONDITION_INIT_VAL_LOWER}")
        if AlgConf.FastMode:
            out_program = run_optimization_loop_fast(
                init_opt_program=opt_program,
                target_mesh=target_mesh,
                target=target_sdf,
                sketcher=sketcher,
                #
                variable_list=variable_list,
                tensor_list=tensor_list,
                param_groups=param_groups,
                compiled_ops=compiled_func_relaxed,
                render_mode=AlgConf.RENDER_MODE,
                render_iter=AlgConf.RENDER_ITER,
            )
        else:
            out_program = run_optimization_loop(
                init_opt_program=opt_program,
                target_mesh=target_mesh,
                target=target_sdf,
                sketcher=sketcher,
                #
                variable_list=variable_list,
                tensor_list=tensor_list,
                type_annotation=type_annotation,
                param_groups=param_groups,
                compiled_func_relaxed=compiled_func_relaxed,
            )
        
        out_program = convert_to_unbatched(out_program, handler)
        out_program = convert_to_unpacked(out_program, handler)
            
        n_prims = len(gather_primitives(out_program))
        end_recon_measure = get_recon_measure(out_program, sketcher, measure_pack)
        if isinstance(end_recon_measure, th.Tensor):
            end_recon_measure = end_recon_measure.item()
        end_obj = end_recon_measure + AlgConf.MPS_LEN_WEIGHT * n_prims
        Stats.record("end_recon_measure", end_recon_measure)
        Stats.record("end_obj", end_obj)
        logger.info("==================== Optimization stopped ====================")

    return out_program

