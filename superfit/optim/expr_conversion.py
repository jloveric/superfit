"""
ADOBE

Copyright 2026 Adobe

All Rights Reserved.

NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.
"""
import torch as th
import geolipi.symbolic as gls
import superfit.symbolic as sps
from .primitive_registry import PrimitiveHandler
from ..symbolic.utils import gather_primitives
from ..symbolic.utils import gather_smooth_union_ops, generate_from_sm_ops_and_primitives
from ..utils.config import AlgorithmConfig as AlgConf
from ..symbolic.symbolic_types import VALID_PACKED_CLASSES, VALID_BATCHED_SU_CLASSES, VALID_BATCHED_STOCHASTIC_SU_CLASSES, VARAXIS_EXECUTED_CLASSES

def convert_to_packed(expr, handler: PrimitiveHandler):
    if isinstance(expr, sps.PrimitiveMarker):
        if isinstance(expr.args[0], sps.StochasticPrimitive):
            has_stochastic = True
            stochastic_expr = expr.args[0]
            stochastic_arg = stochastic_expr.get_arg(1)
            next_expr = stochastic_expr
        else:
            has_stochastic = False
            next_expr = expr

        translate_expr = next_expr.args[0]
        rotate_expr = translate_expr.args[0]
        prim_expr = rotate_expr.args[0]

        translate_arg = translate_expr.get_arg(1)
        rotate_arg = rotate_expr.get_arg(1)
        n_args = len(prim_expr.args)
        sp_tapered_arg = [prim_expr.get_arg(i) for i in range(n_args)]
        if isinstance(prim_expr, VARAXIS_EXECUTED_CLASSES):
            log_reinit_param = handler.reinit_params(prim_expr, sp_tapered_arg[0])
            sp_tapered_arg.extend(log_reinit_param)
        new_param = [translate_arg, *sp_tapered_arg, rotate_arg]
        new_param = th.cat(new_param, dim=-1)
        packed_class = handler.packed_class
        new_expr = packed_class(new_param)
        if has_stochastic:
            new_expr = sps.StochasticPrimitive(new_expr, stochastic_arg)
        new_expr = sps.PrimitiveMarker(new_expr)
        return new_expr
    elif isinstance(expr, gls.GLFunction):
        new_args = []
        for arg in expr.args:
            if arg in expr.lookup_table:
                arg = expr.lookup_table[arg]
            new_args.append(convert_to_packed(arg, handler))
        return expr.func(*new_args)
    else:
        return expr

def convert_to_unpacked(expr, handler: PrimitiveHandler):
    if isinstance(expr, sps.PrimitiveMarker):
        if isinstance(expr.args[0], sps.StochasticPrimitive):
            has_stochastic = True
            stochastic_expr = expr.args[0]
            stochastic_arg = stochastic_expr.get_arg(1)
            next_expr = stochastic_expr
        else:
            has_stochastic = False
            next_expr = expr

        packed_expr = next_expr.args[0]
        packed_arg = packed_expr.get_arg(0)
        translate_arg = packed_arg[..., :3]
        rotate_arg = packed_arg[..., -3:]
        unpacked_vars = handler.unpack_params(packed_arg[..., 3:-3])
        base_class = handler.base_class
        new_primitive = base_class(*unpacked_vars)

        new_primitive = gls.AxisAngleRotate3D(new_primitive, rotate_arg)
        new_primitive = gls.Translate3D(new_primitive, translate_arg)
        if has_stochastic:
            new_primitive = sps.StochasticPrimitive(new_primitive, stochastic_arg)
        new_expr = sps.PrimitiveMarker(new_primitive)
        return new_expr

    elif isinstance(expr, gls.GLFunction):
        new_args = []
        for arg in expr.args:
            if arg in expr.lookup_table:
                arg = expr.lookup_table[arg]
            new_args.append(convert_to_unpacked(arg, handler))
        return expr.func(*new_args)
    else:
        return expr



def convert_to_batched(program, handler: PrimitiveHandler):
    primitives = gather_primitives(program)
    sm_ops = gather_smooth_union_ops(program)
    if sm_ops:
        sm_ops = th.stack(sm_ops, dim=0)
    else:
        sm_ops = None

    if AlgConf.STOCHASTIC_DROPOUT: 
        logits = []
        prim_params = []
        prim_class = None
        for sub_expr in primitives:
            sub_expr = sub_expr.args[0]
            if isinstance(sub_expr, sps.StochasticPrimitive):
                stochastic_expr = sub_expr
                log_param = stochastic_expr.get_arg(1)
                logits.append(log_param)
                prim_expr = stochastic_expr.get_arg(0)
                prim_param = prim_expr.get_arg(0)
                prim_params.append(prim_param)
            elif isinstance(sub_expr, VALID_PACKED_CLASSES):
                # This means its second time optimization.
                prim_expr = sub_expr
                prim_param = prim_expr.get_arg(0)
                prim_params.append(prim_param)
                logit_fake = th.Tensor(AlgConf.DEFAULT_LOGITS_RESTART_VALUES).to(prim_param.device)
                logits.append(logit_fake)
        logits = th.stack(logits, dim=0)
        prim_params = th.stack(prim_params, dim=0)
        batched_class = handler.packed_batched_stochastic_su_class
        if sm_ops is not None:
            out_program = batched_class(prim_params, sm_ops, logits)
        else:
            # TODO: THIS IS MESSY AND WRONG.
            raise ValueError("No smooth union ops found for batched conversion.")
    else:
        prim_params = []
        prim_class = None
        for sub_expr in primitives:
            prim_expr = sub_expr.args[0]
            prim_param = prim_expr.get_arg(0)
            if prim_param in prim_expr.lookup_table:
                prim_param = prim_expr.lookup_table[prim_param]
            prim_params.append(prim_param)
            prim_class = prim_expr.__class__
        prim_params = th.stack(prim_params, dim=0)
        batched_class = handler.packed_batched_su_class
        if sm_ops is not None:
            out_program = batched_class(prim_params, sm_ops)
        else:
            out_program = batched_class(prim_params)
    return out_program


def convert_to_unbatched(program, handler: PrimitiveHandler):
    if isinstance(program, VALID_BATCHED_STOCHASTIC_SU_CLASSES):
        prim_params = program.get_arg(0)
        sm_ops = program.get_arg(1)
        logits = program.get_arg(2)
        n_prims = prim_params.shape[0]
        prim_class = handler.packed_class
        new_prims = []
        for i in range(n_prims):
            cur_prim = prim_params[i]
            new_prim = sps.PrimitiveMarker(sps.StochasticPrimitive(prim_class(cur_prim), logits[i]))
            new_prims.append(new_prim)
        out_program = generate_from_sm_ops_and_primitives(sm_ops, new_prims)
    elif isinstance(program, VALID_BATCHED_SU_CLASSES):
        prim_params = program.get_arg(0)
        sm_ops = program.get_arg(1)
        new_prims = []
        n_prims = prim_params.shape[0]
        prim_class = handler.packed_class
        for i in range(n_prims):
            cur_prim = prim_params[i]
            new_prim = sps.PrimitiveMarker(prim_class(cur_prim))
            new_prims.append(new_prim)
        out_program = generate_from_sm_ops_and_primitives(sm_ops, new_prims)
    else:
        raise ValueError(f"Unsupported command symbol: {program}")
    return out_program

