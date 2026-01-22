
import geolipi.symbolic as gls
import torch as th
import torch.nn.functional as F
from geolipi.torch_compute.sketcher import Sketcher
from typing import Optional
from geolipi.symbolic.registry import register_symbol
from geolipi.torch_compute.evaluate_expression import rec_eval, _parse_param_from_expr
from geolipi.torch_compute.unroll_expression import rec_unroll, LocalContext, _process_params
from geolipi.torch_compute.compile_expression import CompiledLocalContext, rec_compiled_unroll

from sysl.shader.global_shader_context import GlobalShaderContext
from sysl.shader.evaluate_singlepass import rec_shader_eval, _inline_parse_param_from_expr
from sysl.shader.shader_mod_ext import FixedArityShaderModule
from sysl.shader.shader_module import SMMap
from string import Template
import superfit.symbolic as sps
from superfit.symbolic.utils import sample_gumbel
DROP_CONST = 1.0

@rec_eval.register
def rec_eval_primitive_marker(expression: sps.PrimitiveMarker, sketcher: Sketcher,
             secondary_sketcher: Optional[Sketcher] = None, coords: Optional[th.Tensor] = None,
             relaxed_eval: bool = True,
             *args, **kwargs) -> th.Tensor:
    sub_expr = expression.args[0]
    output = rec_eval(sub_expr, sketcher, coords=coords, relaxed_eval=relaxed_eval, *args, **kwargs)
    return output

    
@rec_unroll.register
def rec_unroll_prim_marker(expression:sps.PrimitiveMarker, local_context: LocalContext, sketcher: Sketcher,
               secondary_sketcher: Optional[Sketcher] = None, isolated_vars: bool = False,
               relaxed_eval: bool = True,
               *args, **kwargs) -> LocalContext:
    prim_expr = expression.args[0]
    local_context = rec_unroll(prim_expr, local_context, sketcher, secondary_sketcher, isolated_vars,  relaxed_eval=relaxed_eval, *args, **kwargs)
    return local_context



@rec_shader_eval.register
def eval_primitive_marker(expression: sps.PrimitiveMarker, global_sc: GlobalShaderContext):
    sub_expr = expression.args[0]
    global_sc = rec_shader_eval(sub_expr, global_sc)
    return global_sc

# ------------------------------
# Stochastic Primitive
# ------------------------------


def mix_stochastic_relaxed(primitive_evals: th.Tensor,  # (..., 2)
                           logits: th.Tensor,           # (..., 2) or (2,)
                           temperature: float | th.Tensor = 0.1
                           ) -> th.Tensor:              # (...,) matching pv's leading dims (excl. last)
    """
    Soft (relaxed) Gumbel-Softmax mix of two primitives.
    Works with pv=(B,N,2), lg=(B,2)  → (B,N)
             pv=(N,2),    lg=(2,)    → (N,)
    """
    pv = primitive_evals                         # (..., 2)
    lg = logits                                  # (..., 2) or (2,)
    g  = sample_gumbel(lg.shape, device=lg.device)
    w  = th.softmax((lg + g) / temperature, dim=-1)   # (..., 2)
    w = w.unsqueeze(-2)
    # Broadcasting handles the middle N dim automatically.
    out = (pv * w).sum(dim=-1)                   # (...)
    return out


def mix_stochastic_st(primitive_evals: th.Tensor,       # (..., 2)
                      logits: th.Tensor,                # (..., 2) or (2,)
                      temperature: float | th.Tensor = 0.1
                      ) -> th.Tensor:                   # (...,)
    """
    Straight-through (hard in fwd, soft in bwd) Gumbel-Softmax mix.
    Compatible with the two target shape regimes; no repeats, no loops.
    """
    pv = primitive_evals
    lg = logits
    g   = sample_gumbel(lg.shape, device=lg.device)
    soft= th.softmax((lg + g) / temperature, dim=-1)   # (..., 2)

    idx = th.argmax(soft, dim=-1)                      # (...,)
    hard= F.one_hot(idx, num_classes=2).to(soft.dtype) # (..., 2)
    w   = (hard - soft).detach() + soft                # straight-through
    w = w.unsqueeze(-2)
    out = (pv * w).sum(dim=-1)                         # (...)
    return out


def mix_stochastic_non_relaxed(primitive_evals: th.Tensor,  # (..., 2)
                               logits: th.Tensor,           # (..., 2) or (2,)
                               temperature: float | th.Tensor = None
                               ) -> th.Tensor:              # (...,)
    """
    Hard argmax over logits (no Gumbel/temperature).
    Uses gather with an index expanded to pv's leading dims (zero-copy expand).
    """
    pv = primitive_evals
    lg = logits
    idx = th.argmax(lg, dim=-1)                      # (...,)
    hard= F.one_hot(idx, num_classes=2).to(lg.dtype) # (..., 2)ough
    hard = hard.unsqueeze(-2)

    out = (pv * hard).sum(dim=-1)                         # (...)
    return out
# ------------------------------
# Evaluate (numeric)
# ------------------------------

@rec_eval.register
def eval_stochastic_prim(
    expression: sps.StochasticPrimitive,
    sketcher: Sketcher,
    secondary_sketcher: Optional[Sketcher] = None,
    coords: Optional[th.Tensor] = None,
    relaxed_eval: bool = True,
    straight_through: bool = False,
    *args, **kwargs
) -> th.Tensor:
    # child value s(x): (K,) or batched (..., K)
    sub_expr = expression.args[0]
    s = rec_eval(sub_expr, sketcher, coords=coords)

    # params: logits (2,), optional temperature
    params = _parse_param_from_expr(expression, expression.args[1:], sketcher)
    if len(params) == 1:
        logits, = params
        temperature = 0.1
    else:
        logits, temperature = params

    # build [keep, drop] = [s, 1]
    keep_drop = th.stack([s, th.ones_like(s) * DROP_CONST], dim=-1)

    # dispatch with no branching inside the mixers
    if relaxed_eval and straight_through:
        return mix_stochastic_st(keep_drop, logits, float(temperature))
    elif relaxed_eval:
        return mix_stochastic_relaxed(keep_drop, logits, float(temperature))
    else:
        return mix_stochastic_non_relaxed(keep_drop, logits)

# ------------------------------
# Unroll (source -> Python code)
# ------------------------------

@rec_unroll.register
def unroll_stochastic_prim(
    expression: sps.StochasticPrimitive,
    local_context: LocalContext,
    sketcher: Sketcher,
    secondary_sketcher: Optional[Sketcher] = None,
    isolated_vars: bool = False,
    relaxed_eval: bool = True,
    straight_through: bool = False,
    *args, **kwargs
) -> LocalContext:
    cur_coords = local_context.coords_stack.pop()
    cur_transform = local_context.transform_stack.pop()

    # single child
    child = expression.args[0]
    local_context.transform_stack.append(cur_transform)
    local_context.coords_stack.append(cur_coords)
    if not isolated_vars:
        local_context.transform_count += 1
        local_context.coords_count += 1
        nt = f"transform_{local_context.transform_count}"
        nc = f"coords_{local_context.coords_count}"
        local_context.add_codeline(f"{nt} = {cur_transform}.clone()")
        local_context.add_codeline(f"{nc} = {cur_coords}.clone()")
    local_context.coords_stack.append(cur_coords)
    local_context = rec_unroll(child, local_context, sketcher, secondary_sketcher, isolated_vars, *args, **kwargs)

    # child result
    child_res = local_context.res_stack.pop()

    # build [s, ones] -> (..., 2, 1)
    local_context.res_count += 1
    ones_res = f"res_{local_context.res_count}"
    local_context.add_codeline(f"{ones_res} = th.ones_like({child_res}) * {DROP_CONST}")

    local_context.res_count += 1
    new_res = f"res_{local_context.res_count}"
    local_context.add_codeline(f"{new_res} = th.stack([{child_res}, {ones_res}], dim=-1)")

    # params -> logits, (temperature?)
    param_list = _process_params(expression, expression.args[1:], local_context, sketcher)

    # choose the concrete mixer at unroll-time so generated code has no runtime branch
    if relaxed_eval and straight_through:
        call = f"{new_res} = mix_stochastic_st({new_res}, {param_list})"
        local_context.add_dependency("mix_stochastic_st", mix_stochastic_st)
        local_context.add_dependency("sample_gumbel", sample_gumbel)
    elif relaxed_eval:
        call = f"{new_res} = mix_stochastic_relaxed({new_res}, {param_list})"
        local_context.add_dependency("mix_stochastic_relaxed", mix_stochastic_relaxed)
        local_context.add_dependency("sample_gumbel", sample_gumbel)
    else:
        call = f"{new_res} = mix_stochastic_non_relaxed({new_res}, {param_list})"
        local_context.add_dependency("mix_stochastic_non_relaxed", mix_stochastic_non_relaxed)

    local_context.add_codeline(call)
    local_context.res_stack.append(new_res)
    return local_context

# ------------------------------
# Compiled unroll
# ------------------------------

@rec_compiled_unroll.register
def rec_compiled_unroll_stochastic_prim(
    expression: sps.StochasticPrimitive,
    local_context: CompiledLocalContext,
    sketcher: Sketcher,
    secondary_sketcher: Optional[Sketcher] = None,
    isolated_vars: bool = False,
    relaxed_eval: bool = True,
    straight_through: bool = False,
    *args, **kwargs
) -> CompiledLocalContext:
    raise NotImplementedError("Not implemented")

