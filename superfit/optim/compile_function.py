import ast
import time
import os
import torch as th
import torch._dynamo as dynamo
import geolipi.symbolic as gls
from dataclasses import dataclass
from typing import Callable, Any, List
from geolipi.torch_compute.unroll_expression import unroll_expression
from ..torch_compute.compile_friendly import batched_sf_packed_stochastic_eval, _sdf_smooth_union_pair, batched_sf_packed_stochastic_su_eval
from .losses import compute_total_loss
from ..utils.config import AlgorithmConfig as AlgConf
from ..utils.logger import logger
from .param_conversion import params_from_variables
from .primitive_registry import PrimitiveHandler

N_OPT_ITERS = 5

def compile_program_jit(in_program, sketcher, handler: PrimitiveHandler,
            isolated_vars=True,
            torch_compile=False):
    # This is the old pass. 
    # In the new version we literally give the compiled function directly. 
    opt_program, _ = in_program.tensor(dtype=AlgConf.OPT_DTYPE).get_varnamed_expr()
    compiled_func_relaxed, func_def, _ = unroll_expression(opt_program, sketcher, 
        isolated_vars=isolated_vars, param_mode="varlist", relaxed_eval=True)
    logger.debug(ast.unparse(func_def))
    logger.debug(opt_program.sympy().pretty_print())
    if torch_compile:
        # JIT compilation is too slow as it has to be done many times!
        compiled_func_relaxed = th.compile(compiled_func_relaxed, mode="reduce-overhead", fullgraph=True)
    return compiled_func_relaxed

@dataclass(slots=True)
class CompiledOps:
    compiled_assembly_execution: Callable[..., Any] = batched_sf_packed_stochastic_eval
    compiled_loss_function: Callable[..., Any] = compute_total_loss
    param_from_variables: Callable[..., Any] = params_from_variables


# Run a dummy Opt function. 
def compile_cached_with_dummy_opt(in_program, sketcher,
            handler: PrimitiveHandler, torch_compile=False, *args, **kwargs):

    forward_function = handler.batched_eval_function

    if AlgConf.COMPILED_FUNCTIONS is not None:
        compiled_ops = AlgConf.COMPILED_FUNCTIONS
        return compiled_ops

    if not torch_compile:
        def execute(coords, all_params):
            output = forward_function(coords, *all_params)
            return output
        compiled_ops = CompiledOps(
            compiled_assembly_execution=execute,
            compiled_loss_function=compute_total_loss,
            param_from_variables=params_from_variables,
        )
        return compiled_ops


    comp_func = th.compile(forward_function, 
        backend="inductor",
        # mode="max-autotune",
        mode="default",
        fullgraph=True,
        dynamic=True,
    )
    # comp_func = forward_function
    compiled_su_func = th.compile(_sdf_smooth_union_pair, 
        backend="inductor",
        # mode="max-autotune",
        mode="default",
        fullgraph=True,
        dynamic=True,
    )
    # compiled_su_func = _sdf_smooth_union_pair

    def compiled_assembly_execution(coords, all_params):
        params, su_vals, logits, temperature = all_params 
        output = comp_func(coords, params, logits, temperature)
        K = output.shape[0]

        out = output[0]
        for i in range(1, K):
            k_reshaped = su_vals[i-1].unsqueeze(-1)
            out = compiled_su_func(out, output[i], k_reshaped)
        
        return (output, out)

    compiled_loss_function = th.compile(compute_total_loss, 
        backend="inductor",
        # mode="max-autotune",
        mode="default",
        fullgraph=True,
        dynamic=True,
    )
    # compiled_loss_function = compute_total_loss

    compiled_param_from_variables = th.compile(handler.param_from_variables_fast, 
        backend="inductor",
        # mode="max-autotune",
        mode="default",
        fullgraph=True,
        dynamic=True,
    )
    # compiled_param_from_variables = handler.param_from_variables_fast
    # Primitive Instantiation
    # Loop: 
    artifact_file = AlgConf.AOT_ARTIFACT_FILE
    # This way or from AlgConf?
    arg_0 = in_program.get_arg(0)
    dtype = arg_0.dtype
    device = arg_0.device
    # Just a rough size estimate.
    PC_SIZE = 2_00_000
    scale_factor = 1.0
    SURF_SIZE = AlgConf.N_SURFACE_POINTS
    BATCH_SIZE = arg_0.shape[0]
    _coords = th.randn(1, PC_SIZE, 3, dtype=dtype, device=device).clone().detach().requires_grad_(False)
    _surface_coords = th.randn(1, SURF_SIZE, 3, dtype=dtype, device=device).clone().detach().requires_grad_(False)
    _surface_adj_coords = th.randn(1, SURF_SIZE, 3, dtype=dtype, device=device).clone().detach().requires_grad_(False)

    _params = th.randn(BATCH_SIZE, handler.batched_param_size, dtype=dtype, device=device).clone().detach().requires_grad_(True)
    _su_vals = th.randn(BATCH_SIZE-1, 1, dtype=dtype, device=device).clone().detach().requires_grad_(True)
    _logits = th.randn(BATCH_SIZE, 2, dtype=dtype, device=device).clone().detach().requires_grad_(True)
    _temperature = th.randn([1,], dtype=dtype, device=device).clone().detach().requires_grad_(False)

    # Just create the tensors for outputs: 
    output_shape_occ = th.randn(PC_SIZE, dtype=dtype, device=device).clone().detach().requires_grad_(True)
    hard_target_fl = th.randn(PC_SIZE, dtype=dtype, device=device).clone().detach().requires_grad_(False)
    mask_shape = th.randn(PC_SIZE, dtype=dtype, device=device).clone().detach().requires_grad_(False)

    output_surface_adj_occ = th.randn(SURF_SIZE, dtype=dtype, device=device).clone().detach().requires_grad_(True)
    hard_target_surface_adj_fl = th.randn(SURF_SIZE, dtype=dtype, device=device).clone().detach().requires_grad_(False)
    mask_surface_adj = th.randn(SURF_SIZE, dtype=dtype, device=device).clone().detach().requires_grad_(False)
    _curvature_weights = th.randn(SURF_SIZE, dtype=dtype, device=device).clone().detach().requires_grad_(False)
    
    output_surface_sdf = th.randn(SURF_SIZE, dtype=dtype, device=device).clone().detach().requires_grad_(True)
    mask_surface = th.randn(SURF_SIZE, dtype=dtype, device=device).clone().detach().requires_grad_(False)

    primitive_sdfs = th.randn(BATCH_SIZE, PC_SIZE + 2 * SURF_SIZE, dtype=dtype, device=device).clone().detach().requires_grad_(True)
    output_sdf = th.randn(PC_SIZE + 2 * SURF_SIZE, dtype=dtype, device=device).clone().detach().requires_grad_(True)

    dynamo.mark_dynamic(_coords, 1)
    dynamo.mark_dynamic(_params, 0, min=2, max=100)
    dynamo.mark_dynamic(_logits, 0, min=2, max=100)
    dynamo.mark_dynamic(_su_vals, 0, min=1, max=99)

    dynamo.mark_dynamic(output_shape_occ, 0)
    dynamo.mark_dynamic(hard_target_fl, 0)
    dynamo.mark_dynamic(mask_shape, 0)

    dynamo.mark_dynamic(primitive_sdfs, 0, min=2, max=100)
    dynamo.mark_dynamic(primitive_sdfs, 1)
    dynamo.mark_dynamic(output_sdf, 0)


    if os.path.exists(artifact_file):
        logger.info(f"Loading artifacts from {artifact_file}")
        artifact_bytes = th.load(artifact_file)
        th.compiler.load_cache_artifacts(artifact_bytes)
    
    cur_vars = [_params, _su_vals, _logits]
    optim = th.optim.Adam(cur_vars, lr=0.00001)


    for i in range(N_OPT_ITERS):

        th.compiler.cudagraph_mark_step_begin()
        optim.zero_grad()
        start_time = time.time()
        transformed_params = compiled_param_from_variables(cur_vars)
        transformed_params.append(_temperature)
        # transformed_params = [x for x in cur_vars] + [_temperature]
        # Concatenate all coordinates
        all_coords = th.cat([_coords, _surface_coords, _surface_adj_coords], dim=1)
        ## MAIN FORWARD
        primitive_sdfs, output_sdf = compiled_assembly_execution(all_coords, transformed_params)
        loss_1 = primitive_sdfs.sum() + output_sdf.sum()
        loss_2 = compiled_loss_function(output_shape_occ, hard_target_fl, 
                                        output_surface_adj_occ, hard_target_surface_adj_fl, 
                                        output_surface_sdf,
                                        primitive_sdfs, output_sdf, 
                                        mask_shape, mask_surface, mask_surface_adj,
                                        transformed_params, 
                                        scale_factor, _curvature_weights)
        total_loss = loss_1 + loss_2
        total_loss.backward()
        optim.step()
        end_time = time.time()
        logger.info(f"Time taken for iteration: {end_time - start_time:.3f}s")
    # assign the compiled functions to an object.
    compiled_ops = CompiledOps(
        compiled_assembly_execution=compiled_assembly_execution,
        compiled_loss_function=compiled_loss_function,
        param_from_variables=compiled_param_from_variables,
    )
    logger.info("Finished compiling with dummy opt")
    if AlgConf.SAVE_JIT_CACHE:
        if not os.path.exists(artifact_file) or AlgConf.OVERWRITE_JIT_CACHE:
            parent_dir = os.path.dirname(artifact_file)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            logger.info(f"Saving artifacts to {artifact_file}")
            artifacts = th.compiler.save_cache_artifacts()
            assert artifacts is not None
            artifact_bytes, cache_info = artifacts
            th.save(artifact_bytes, artifact_file)
    AlgConf.COMPILED_FUNCTIONS = compiled_ops
    return compiled_ops
    