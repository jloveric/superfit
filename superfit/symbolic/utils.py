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
import numpy as np
import torch as th
import sympy as sp
import sysl.symbolic as sls
import geolipi.symbolic as gls
from geolipi.symbolic.symbol_types import PRIM_TYPE
from geolipi.torch_compute.transforms import axis_angle_to_rotation_matrix
import superfit.symbolic as sps
from . import rotation_functions as rotf
from .symbolic_types import VALID_BATCHED_STOCHASTIC_SU_CLASSES, VARAXIS_CLASSES, SFSP_CLASSES, SFSP_UNRAVEL_CLASSES
from .base import StochasticPrimitive, PrimitiveMarker

VARAXIS_MAP = {
    sps.VarAxisSF: {
        0: sps.SuperFrustumY,
        1: sps.SuperFrustumZ,
        2: sps.SuperFrustumX,
    },
    sps.VarAxisSQ: {
        0: sps.SuperQuadricY,
        1: sps.SuperQuadricZ,
        2: sps.SuperQuadricX,
    },
    sps.VarAxisSG: {
        0: sps.SuperGeonY,
        1: sps.SuperGeonZ,
        2: sps.SuperGeonX,
    },
    sps.VarAxisSPP: {
        0: sps.SPProtoY,
        1: sps.SPProtoZ,
        2: sps.SPProtoX,
    },
}

_AXIS_VARIANT_TO_BASE = {
    sps.SuperFrustumX: sps.SuperFrustum,
    sps.SuperFrustumY: sps.SuperFrustum,
    sps.SuperFrustumZ: sps.SuperFrustum,
    sps.SuperQuadricX: sps.SuperQuadric,
    sps.SuperQuadricY: sps.SuperQuadric,
    sps.SuperQuadricZ: sps.SuperQuadric,
    sps.SPProtoX: sps.SPProto,
    sps.SPProtoY: sps.SPProto,
    sps.SPProtoZ: sps.SPProto,
    sps.SuperGeonX: sps.SuperGeon,
    sps.SuperGeonY: sps.SuperGeon,
    sps.SuperGeonZ: sps.SuperGeon,
}

_AXIS_VARIANT_PERM = {
    # Y is identity (base orientation)
    sps.SuperFrustumY: (0, 1, 2),
    sps.SuperQuadricY: (0, 1, 2),
    sps.SPProtoY: (0, 1, 2),
    sps.SuperGeonY: (0, 1, 2),
    # Z variants use [x, y, z] -> [y, z, x]
    sps.SuperFrustumZ: (1, 2, 0),
    sps.SuperQuadricZ: (1, 2, 0),
    sps.SPProtoZ: (1, 2, 0),
    sps.SuperGeonZ: (1, 2, 0),
    # X variants use [x, y, z] -> [z, x, y]
    sps.SuperFrustumX: (2, 0, 1),
    sps.SuperQuadricX: (2, 0, 1),
    sps.SPProtoX: (2, 0, 1),
    sps.SuperGeonX: (2, 0, 1),
}


SFSP_CONVERSION_MAP = {
    sps.SFSPX: sps.SuperFrustumX,
    sps.SFSPY: sps.SuperFrustumY,
    sps.SFSPZ: sps.SuperFrustumZ,
    sps.SuperFrustumX: sps.SFSPX,
    sps.SuperFrustumY: sps.SFSPY,
    sps.SuperFrustumZ: sps.SFSPZ,
    sps.SuperFrustum: sps.SFSP,
    sps.SFSP: sps.SuperFrustum,
    sps.SPPSPX: sps.SPProtoX,
    sps.SPPSPY: sps.SPProtoY,
    sps.SPPSPZ: sps.SPProtoZ,
    sps.SPPSP: sps.SPProto,
    sps.SGSPX: sps.SuperGeonX,
    sps.SGSPY: sps.SuperGeonY,
    sps.SGSPZ: sps.SuperGeonZ,
    sps.SGSP: sps.SuperGeon,
    sps.SPProtoX: sps.SPPSPX,
    sps.SPProtoY: sps.SPPSPY,
    sps.SPProtoZ: sps.SPPSPZ,
    sps.SPProto: sps.SPPSP,
    sps.SuperGeonX: sps.SGSPX,
    sps.SuperGeonY: sps.SGSPY,
    sps.SuperGeonZ: sps.SGSPZ,
    sps.SuperGeon: sps.SGSP,
}

INVERSE_PRIM_MAP = {
    "VarAxisSF": "SuperFrustum",
    "VarAxisSQ": "SuperQuadric",
    "VarAxisSPP": "SPProto",
    "VarAxisSG": "SuperGeon",
    "Cuboid": "Cuboid",
    "SuperFrustum": "SuperFrustum",
}

def sample_gumbel(shape, eps=1e-10, device=None, dtype=th.float32):
    U = th.rand(shape, device=device, dtype=dtype)
    return -th.log(-th.log(U + eps) + eps)

# Also do the effect of stochastic dropout using exp
def inject_stochastic_prim(
    expression: gls.GLBase,
    logits_keep_drop=(0.0, -0.0),  # default prior: prefer keep
    temperature: float = 0.1
) -> gls.GLBase:
    """
    Wrap leaf primitives marked by PrimitiveMarker with StochasticPrimitive.
    Uses direct logits (2-vector): (logit_keep, logit_drop) to match Gumbel usage.
    """
    if isinstance(expression, (PrimitiveMarker,)):
        sub_expr = expression.args[0]
        new_expr = StochasticPrimitive(sub_expr, logits_keep_drop, (temperature,))
        return expression.__class__(new_expr)
    elif isinstance(expression, gls.GLFunction):
        new_args = []
        for arg in expression.args:
            if isinstance(arg, gls.GLFunction):
                new_args.append(inject_stochastic_prim(arg, logits_keep_drop, temperature))
            elif arg in expression.lookup_table:
                new_args.append(expression.lookup_table[arg])
            else:
                new_args.append(arg)
        return expression.__class__(*new_args)
    else:
        return expression



def recursive_packed_to_prim(gls_expr):
    # SPPSP + SPPSPX|Y|Z (all subclass SPPSP) -> SPProto family
    if isinstance(gls_expr, sps.SPPSP):
        cur_expr = gls_expr.sympy()
        a = cur_expr.get_args()
        size, rnd, doe = a[0], a[1], a[2]
        rnd4 = (rnd[0], rnd[1], rnd[2], rnd[3])
        ctor = SFSP_CONVERSION_MAP[gls_expr.__class__]
        return ctor(size, rnd4, (doe[0],), (doe[1],), (doe[2], doe[3]))

    # SGSP (incl. SGSPX/Y/Z) -> SuperGeon family
    if isinstance(gls_expr, sps.SGSP):
        cur_expr = gls_expr.sympy()
        a = cur_expr.get_args()
        size, p, p2 = a[0], a[1], a[2]
        ctor = SFSP_CONVERSION_MAP[gls_expr.__class__]
        return ctor(
            size,
            (p[0],), (p[1],), (p[2],), (p[3],),
            (p2[0],), (p2[1],), (p2[2],), (p2[3],),
        )

    if isinstance(gls_expr, sps.SFSP):
        cur_expr = gls_expr.sympy()
        ntc_args = cur_expr.get_args()
        new_expr = SFSP_CONVERSION_MAP[gls_expr.__class__](ntc_args[0], 
                    (ntc_args[1][0],), 
                    (ntc_args[1][1],), 
                    (ntc_args[1][2],), 
                    (ntc_args[1][3],), 
                    ntc_args[2])
        return new_expr
    if isinstance(gls_expr, gls.GLFunction):
        old_args = gls_expr.get_args()
        new_args = []
        for arg in old_args:
            if isinstance(arg, gls.GLBase):
                new_args.append(recursive_packed_to_prim(arg))
            else:
                new_args.append(arg)
        return gls_expr.__class__(*new_args)
    return gls_expr


def recursive_prim_to_packed(gls_expr):
    if isinstance(gls_expr, sps.SuperFrustum):
        new_args = []
        cur_expr = gls_expr.sympy()
        ntc_args = cur_expr.get_args()
        arg_0 = ntc_args[0]
        arg_1 = (ntc_args[1][0], ntc_args[2][0], ntc_args[3][0], ntc_args[4][0],)
        arg_2 = ntc_args[5]
        new_expr = SFSP_CONVERSION_MAP[gls_expr.__class__](arg_0, arg_1, arg_2)
        return new_expr
    if isinstance(gls_expr, sps.SPProto):
        a = gls_expr.sympy().get_args()
        doe = (a[2][0], a[3][0], a[4][0], a[4][1])
        return SFSP_CONVERSION_MAP[gls_expr.__class__](a[0], a[1], doe)
    if isinstance(gls_expr, sps.SuperGeon):
        cur_expr = gls_expr.sympy()
        a = cur_expr.get_args()
        params = (a[1][0], a[2][0], a[3][0], a[4][0])
        params2 = (a[5][0], a[6][0], a[7][0], a[8][0])
        return SFSP_CONVERSION_MAP[gls_expr.__class__](a[0], params, params2)
    else:
        if isinstance(gls_expr, gls.GLFunction):
            new_args = []
            for arg in gls_expr.args:
                if isinstance(arg, gls.GLBase):
                    out_expr = recursive_prim_to_packed(arg)
                    new_args.append(out_expr)
                else:
                    new_args.append(arg)
            return gls_expr.__class__(*new_args)
        else:
            return gls_expr

def n_prims_in_expr(expr, n_prims=0):
    if isinstance(expr, PRIM_TYPE):
        return n_prims + 1
    elif isinstance(expr, gls.GLFunction):
        overall_prims = n_prims
        for arg in expr.args:
            new_prims = n_prims_in_expr(arg, 0)
            overall_prims += new_prims
        return overall_prims
    else:
        return 0


def fetch_singular_expr_stochastic(expr, relaxed_eval=True):
    expr = expr.tensor()
    logits = expr.args[1]
    logits = expr.lookup_table[logits]
    if len(expr.args) == 2:
        temperature = 0.01
    else:
        temperature = expr.lookup_table[expr.args[2]]
    if relaxed_eval:
        N = logits.shape[0]
        device = logits.device
        gumbel_noise = sample_gumbel((N,), device=device)
        sel_ind = (logits / temperature + gumbel_noise).argmax(dim=-1).item()
        # gumbel_soft = th.nn.functional.softmax((logits + gumbel_noise) / temperature, dim=0) 
        # sel_ind = th.multinomial(gumbel_soft, 1).item()  # int
    else:
        sel_ind = th.argmax(logits).item()  # int
    if sel_ind == 0:
        selected_expr = expr.args[0]
    else:
        selected_expr = gls.NullExpression3D()
    return selected_expr


def reduce_varaxis(expr, temperature: float = 0.1, relaxed_eval: bool = True):
    params = expr.get_args()

    # Expect last argument to be logits over 3 axis variants
    logits = params[-1]          # shape (3,) (or (K,))
    remaining_args = params[:-1]

    if relaxed_eval:
        # Gumbel-max sampling: argmax((logits + g)/tau) gives a categorical sample.
        g = sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
        sel_ind = (logits / temperature + g).argmax(dim=-1).item()
    else:
        sel_ind = logits.argmax(dim=-1).item()

    new_expr = VARAXIS_MAP[expr.__class__][int(sel_ind)](*remaining_args)
    return new_expr

def fetch_singular_expr_eval(expr, temperature=0.1, relaxed_eval=True, remove_marker=True, fix_axis_variant=False, use_euler_angle=False):
    if isinstance(expr, StochasticPrimitive):
        new_expr = fetch_singular_expr_stochastic(expr, relaxed_eval=relaxed_eval)
        out_expr = fetch_singular_expr_eval(new_expr, temperature=temperature, relaxed_eval=relaxed_eval, remove_marker=remove_marker, fix_axis_variant=fix_axis_variant, use_euler_angle=use_euler_angle)
        return out_expr
    elif isinstance(expr, VARAXIS_CLASSES):
        new_expr = reduce_varaxis(expr, temperature=temperature, relaxed_eval=relaxed_eval)
        return new_expr
    elif isinstance(expr, PrimitiveMarker):
        in_expr = fetch_singular_expr_eval(expr.args[0], temperature=temperature, relaxed_eval=relaxed_eval, remove_marker=remove_marker, fix_axis_variant=fix_axis_variant, use_euler_angle=use_euler_angle)
        if fix_axis_variant:
            in_expr = convert_marked_axis_variant_to_single_rotation(in_expr, use_euler_angle=use_euler_angle)
        if use_euler_angle:
            in_expr = rotf.recursive_axisangle_to_eulerangle(in_expr)
        else:
            in_expr = rotf.recursive_eulerangle_to_axisangle(in_expr)
        if remove_marker:
            return in_expr
        else:
            if isinstance(in_expr, gls.NullExpression3D):
                return in_expr
            else:
                return expr.__class__(in_expr)
    elif isinstance(expr, gls.GLFunction):
        new_args = []
        has_null = False
        for arg in expr.args:
            if isinstance(arg, gls.GLFunction):
                new_expr = fetch_singular_expr_eval(arg, temperature=temperature, relaxed_eval=relaxed_eval, remove_marker=remove_marker, fix_axis_variant=fix_axis_variant, use_euler_angle=use_euler_angle)
                if not isinstance(new_expr, gls.NullExpression3D):
                    new_args.append(new_expr)
                else:
                    has_null = True
            elif arg in expr.lookup_table:
                new_args.append(expr.lookup_table[arg])
            else:
                new_args.append(arg)
        if isinstance(expr, gls.SmoothUnion):
            if len(new_args) == 1:
                return gls.NullExpression3D()
            elif len(new_args) == 2:
                return new_args[0]
            else: 
                return expr.__class__(*new_args)
        elif isinstance(expr, gls.Union):
            if len(new_args) == 0:
                return gls.NullExpression3D()
            elif len(new_args) == 1:
                return new_args[0]
            else: 
                return expr.__class__(*new_args)
        else:
            if has_null:
                return gls.NullExpression3D()
            else:
                return expr.__class__(*new_args)
    else:
        return expr

########################################################
# TEMP PARAMETERS
########################################################

def inject_temp_param(expression: gls.GLBase, temperature: float | th.Tensor) -> gls.GLBase:
    if isinstance(expression, StochasticPrimitive):
        if len(expression.args) == 2:
            prim_expr, dropout_prob = expression.args
            if isinstance(dropout_prob, sp.Symbol):
                dropout_prob = expression.lookup_table[dropout_prob]
            return StochasticPrimitive(prim_expr, dropout_prob, temperature)
        elif len(expression.args) == 3:
            prim_expr, dropout_prob, old_temperature = expression.args
            if isinstance(dropout_prob, sp.Symbol):
                dropout_prob = expression.lookup_table[dropout_prob]
            return StochasticPrimitive(prim_expr, dropout_prob, temperature)
        else:
            raise ValueError(f"Invalid number of arguments for StochasticPrimitive: {len(expression.args)}")
    elif isinstance(expression, VALID_BATCHED_STOCHASTIC_SU_CLASSES):
        proc_args = []
        for arg in expression.args:
            if arg in expression.lookup_table:
                arg = expression.lookup_table[arg]
            proc_args.append(arg)
        if len(proc_args) == 3:
            prim_expr, sm_ops, logits = proc_args
            if prim_expr in expression.lookup_table:
                prim_expr = expression.lookup_table[prim_expr]
            return expression.__class__(prim_expr, sm_ops, logits, temperature)
        elif len(proc_args) == 4:
            prim_expr, sm_ops, logits, old_temperature = proc_args
            if prim_expr in expression.lookup_table:
                prim_expr = expression.lookup_table[prim_expr]
            return expression.__class__(prim_expr, sm_ops, logits, temperature)
        else:
            raise ValueError(f"Invalid number of arguments for {expression.__class__.__name__}: {len(expression.args)}")
    else:
        if isinstance(expression, gls.GLFunction):
            new_args = []
            for arg in expression.args:
                if isinstance(arg, gls.GLBase):
                    new_args.append(inject_temp_param(arg, temperature))
                else:
                    if arg in expression.lookup_table:
                        new_args.append(expression.lookup_table[arg])
                    else:
                        new_args.append(arg)
            return expression.__class__(*new_args)
        else:
            return expression

def remove_temp_param(expression: gls.GLBase) -> gls.GLBase:
    if isinstance(expression, StochasticPrimitive):
        if len(expression.args) == 3:
            prim_expr, dropout_prob, temperature = expression.args
            if dropout_prob in expression.lookup_table:
                dropout_prob = expression.lookup_table[dropout_prob]
            return StochasticPrimitive(prim_expr, dropout_prob)
        elif len(expression.args) == 2:
            return expression
        else:
            raise ValueError(f"Invalid number of arguments for StochasticPrimitive: {len(expression.args)}")
    elif isinstance(expression, VALID_BATCHED_STOCHASTIC_SU_CLASSES):
        if len(expression.args) == 4:
            prim_params, sm_ops, logits, temperature = expression.args
            if prim_params in expression.lookup_table:
                prim_params = expression.lookup_table[prim_params]
            if sm_ops in expression.lookup_table:
                sm_ops = expression.lookup_table[sm_ops]
            if logits in expression.lookup_table:
                logits = expression.lookup_table[logits]
            return expression.__class__(prim_params, sm_ops, logits)
        elif len(expression.args) == 3:
            return expression
        else:
            raise ValueError(f"Invalid number of arguments for {expression.__class__.__name__}: {len(expression.args)}")
    else:
        if isinstance(expression, gls.GLFunction):
            new_args = []
            for arg in expression.args:
                if isinstance(arg, gls.GLBase):
                    new_args.append(remove_temp_param(arg))
                else:
                    if arg in expression.lookup_table:
                        new_args.append(expression.lookup_table[arg])
                    else:
                        new_args.append(arg)
            return expression.__class__(*new_args)
        else:
            return expression

def gather_smooth_union_ops(program, prim_list=None):
    if prim_list is None:
        prim_list = []

    if isinstance(program, gls.GLFunction):
        for arg in program.args:
            new_prim_list = gather_smooth_union_ops(arg)
            prim_list.extend(new_prim_list)
    if isinstance(program, gls.SmoothUnion):
        param = program.args[2]
        if param in program.lookup_table:
            param = program.lookup_table[param]
        prim_list.append(param)
    return prim_list


def generate_from_sm_ops_and_primitives(sm_ops, primitives):
    if len(primitives) == 0:
        return None
    if len(sm_ops) == 0:
        return primitives[0]
    expr = None
    for i in range(len(sm_ops)):
        if expr is None:
            expr = gls.SmoothUnion(primitives[i], primitives[i+1], sm_ops[i])
        else:
            expr = gls.SmoothUnion(expr, primitives[i+1], sm_ops[i])
    return expr

def gather_primitives(program, prim_list=None):
    if prim_list is None:
        prim_list = []

    if isinstance(program, PrimitiveMarker):
        prim_list.append(program)
        return prim_list
    else:
        if isinstance(program, gls.GLFunction):
            for arg in program.args:
                new_prim_list = gather_primitives(arg)
                prim_list.extend(new_prim_list)
        return prim_list

def gather_instance_dropout_alternatives(program):
    # consider the primitive to be SM with the thing below. 
    
    primitives = gather_primitives(program)
    sm_ops = gather_smooth_union_ops(program)

    n_ops = len(sm_ops)
    alternative_exprs = []
    for i in range(n_ops):
        temp_ops = [x for ind, x in enumerate(sm_ops) if ind != i]
        temp_primitives = [x for ind, x in enumerate(primitives) if ind != i+1]
        temp_expr = generate_from_sm_ops_and_primitives(temp_ops, temp_primitives)
        alternative_exprs.append(temp_expr)

    temp_ops = [x for x in sm_ops[1:]]
    temp_primitives = [x for x in primitives[1:]]
    temp_expr = generate_from_sm_ops_and_primitives(temp_ops, temp_primitives)
    alternative_exprs.insert(0, temp_expr)
    alternative_exprs = [x for x in alternative_exprs if x is not None]
    return alternative_exprs


def extract_primitive_bundles(expression):
    # Convert to varmapped. 
    # gather prim and sm ops. 
    # bundle them together. 
    varnamed_expr, _, var_map_base = expression._get_varnamed_expr(exclude_class_set=(gls.UniformVariable, sls.MaterialV4))
    
    primitives = gather_primitives(varnamed_expr)
    sm_ops = gather_smooth_union_ops(varnamed_expr)
    
    primitive_param_bundles = {}
    n_ops = len(sm_ops)
    for op_ind in range(n_ops):
        prim_ind = op_ind + 1
        cur_op = sm_ops[op_ind]
        cur_prim = primitives[prim_ind]
        prim_params, _ = cur_prim.get_params(index_annotate=False)
        prim_params.append(cur_op)
        primitive_param_bundles[prim_ind] = prim_params

    cur_prim = primitives[0]
    prim_params, _ = cur_prim.get_params(index_annotate=False)
    primitive_param_bundles[0] = prim_params
    return primitive_param_bundles


def inject_temperature_solid_ntco(program, temperature=None):
    if temperature is None:
        temperature = gls.UniformFloat((0.001,), (1.0,), (2.0,), "tem")
        
    if isinstance(program, sps.SolidSF):
        cur_args = program.get_args()
        args = list(cur_args) + [temperature]
        return sps.SolidSF(*args)
    elif isinstance(program, gls.GLFunction):
        new_args = []
        for arg in program.args:
            arg = inject_temperature_solid_ntco(arg, temperature)
            new_args.append(arg)
        return program.__class__(*new_args)
    return program


def _perm_to_matrix(perm: tuple[int, int, int], ref: th.Tensor) -> th.Tensor:
    # Build permutation matrix P such that:
    #   (P @ v)[i] = v[perm[i]]
    mat = th.zeros((3, 3), dtype=ref.dtype, device=ref.device)
    for i, j in enumerate(perm):
        mat[i, j] = 1.0
    return mat


def _permute_vec3_expr(vec3_expr: gls.GLBase, perm: tuple[int, int, int]) -> gls.GLBase:
    """
    Construct a new Vector[3] expression with components permuted like:
      new_vec[i] = vec[perm[i]]
    """
    return gls.Vec3(
        gls.VarSplitter(vec3_expr, perm[0]),
        gls.VarSplitter(vec3_expr, perm[1]),
        gls.VarSplitter(vec3_expr, perm[2]),
    )


def _rotation_matrix_to_euler_xyz(R: th.Tensor) -> th.Tensor:
    return rotf._rotation_matrix_to_euler_xyz(R)


def _euler_xyz_to_rotation_matrix(euler: th.Tensor) -> th.Tensor:
    return rotf._euler_xyz_to_rotation_matrix(euler)


def rotation_matrix_to_axis_angle(R: th.Tensor, variant: str = "default", eps: float = 1e-6) -> th.Tensor:
    return rotf.rotation_matrix_to_axis_angle(R, variant=variant, eps=eps)


def _rotation_matrix_to_axis_angle(R: th.Tensor, eps_theta: float = 1e-6) -> th.Tensor:
    # Backward-compatible internal helper.
    return rotation_matrix_to_axis_angle(R, variant="default", eps=eps_theta)


def convert_marked_axis_variant_to_single_rotation(translate_expr: gls.GLBase, use_euler_angle=True) -> gls.GLBase:
    """
    Convert one guaranteed `PrimitiveMarker` of the form:
      Translate3D( AxisAngleRotate3D(VariantPrim, aa), t ) 
    into:
      Translate3D( EulerRotate3D(BasePrim(permute size), euler), t ) 
    """

    if not isinstance(translate_expr, gls.Translate3D):
        raise TypeError(
            "Expected PrimitiveMarker.args[0] to be Translate3D("
            f"AxisAngleRotate3D(<VariantPrim>, aa), t) but got {type(translate_expr)}"
        )

    axis_angle_expr = translate_expr.args[0]
    if not isinstance(axis_angle_expr, gls.AxisAngleRotate3D):
        raise TypeError(
            "Expected Translate3D.expr to be AxisAngleRotate3D(<VariantPrim>, aa), "
            f"but got {type(axis_angle_expr)}"
        )

    variant_prim = axis_angle_expr.args[0]
    if not isinstance(variant_prim, tuple(_AXIS_VARIANT_TO_BASE.keys())):
        # Nothing to do; return marker as-is.
        return translate_expr

    # Extract fixed remap and fold:
    #   Translate3D( AxisAngleRotate3D(VariantPrim, aa), t )
    # into:
    #   Translate3D( EulerRotate3D(BasePrim(permute size), euler), t )
    variant_cls = variant_prim.__class__
    base_cls = _AXIS_VARIANT_TO_BASE[variant_cls]
    perm = _AXIS_VARIANT_PERM[variant_cls]

    # In the Torch evaluator wrappers, axis variants only permute `coords` and `size`.
    # So we permute only the `size` vector (arg0) and keep the remaining params unchanged.
    prim_args = list(variant_prim.get_args())
    prim_args[0] = prim_args[0][list(perm)]
    base_prim = base_cls(*prim_args)

    rot_aa = axis_angle_expr.tensor().get_arg(1)
    aa_b = rot_aa.reshape(-1, 3)
    R1 = axis_angle_to_rotation_matrix(aa_b)

    # Variant wrapper permutes coords AFTER the incoming axis-angle rotation:
    #   v_new = P @ (R1 @ v)
    R2 = _perm_to_matrix(perm, rot_aa).to(R1.dtype).to(R1.device)
    R2 = R2.expand(R1.shape[0], 3, 3)
    R = R2 @ R1
    if use_euler_angle:
        euler = _rotation_matrix_to_euler_xyz(R).reshape(*rot_aa.shape[:-1], 3)
        new_rotate = gls.EulerRotate3D(base_prim, euler).sympy()
    else:
        # Axis-angle fallback: this is only valid for a single rotation.
        # `R` is built from `rot_aa.reshape(-1, 3)` so the batch dim is the first dim.
        if R.shape[0] != 1:
            raise ValueError(
                "convert_marked_axis_variant_to_single_rotation(axis-angle fallback) "
                f"expects a single rotation, but got {R.shape[0]} rotations."
            )

        R_single = R[0]
        composed_axis_angle = rotation_matrix_to_axis_angle(R_single)
        new_rotate = gls.AxisAngleRotate3D(base_prim, composed_axis_angle).sympy()

    translate_arg = translate_expr.get_arg(1)
    new_translate = gls.Translate3D(new_rotate, translate_arg)
    return new_translate


