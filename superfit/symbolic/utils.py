import numpy as np
import torch as th
import sympy as sp
import sysl.symbolic as sls
import geolipi.symbolic as gls
from geolipi.symbolic.symbol_types import PRIM_TYPE
from geolipi.torch_compute.transforms import axis_angle_to_rotation_matrix
import superfit.symbolic as sps
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

SFSP_CONVERSION_MAP = {
    sps.SFSPX: sps.SuperFrustumX,
    sps.SFSPY: sps.SuperFrustumY,
    sps.SFSPZ: sps.SuperFrustumZ,
    sps.SuperFrustumX: sps.SFSPX,
    sps.SuperFrustumY: sps.SFSPY,
    sps.SuperFrustumZ: sps.SFSPZ,
    sps.SuperFrustum: sps.SFSP,
    sps.SFSP: sps.SuperFrustum,
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



def recursive_sfsp_to_sf(gls_expr):
    if isinstance(gls_expr, SFSP_CLASSES):
        new_args = []
        cur_expr = gls_expr.sympy()
        ntc_args = cur_expr.get_args()
        new_expr = SFSP_CONVERSION_MAP[gls_expr.__class__](ntc_args[0], 
                    (ntc_args[1][0],), 
                    (ntc_args[1][1],), 
                    (ntc_args[1][2],), 
                    (ntc_args[1][3],), 
                    ntc_args[2])
        return new_expr
    else:
        if isinstance(gls_expr, gls.GLFunction):
            old_args = gls_expr.get_args()
            new_args = []
            for arg in old_args:
                if isinstance(arg, gls.GLBase):
                    out_expr = recursive_sfsp_to_sf(arg)
                    new_args.append(out_expr)
                else:
                    new_args.append(arg)
            return gls_expr.__class__(*new_args)
        else:
            return gls_expr

def recursive_sf_to_sfsp(gls_expr):
    if isinstance(gls_expr, SFSP_UNRAVEL_CLASSES):
        new_args = []
        cur_expr = gls_expr.sympy()
        ntc_args = cur_expr.get_args()
        arg_0 = ntc_args[0]
        arg_1 = (ntc_args[1][0], ntc_args[2][0], ntc_args[3][0], ntc_args[4][0],)
        arg_2 = ntc_args[5]
        new_expr = SFSP_CONVERSION_MAP[gls_expr.__class__](arg_0, arg_1, arg_2)
        return new_expr
    else:
        if isinstance(gls_expr, gls.GLFunction):
            new_args = []
            for arg in gls_expr.args:
                if isinstance(arg, gls.GLBase):
                    out_expr = recursive_sf_to_sfsp(arg)
                    new_args.append(out_expr)
                else:
                    new_args.append(arg)
            return gls_expr.__class__(*new_args)
        else:
            return gls_expr

def convert_axis_angle_to_euler(axis_angle: th.Tensor) -> th.Tensor:
    """
    Convert a 3D axis-angle vector (..., 3) into Euler angles (..., 3)
    following the rotation order defined by Settings.ROT_ORDER
    (e.g., 'XYZ', 'ZXY', etc.).

    Returns a tensor of Euler angles (same batch shape as axis_angle).
    """

    # 1) Convert AA → rotation matrix (...,3,3)
    R = axis_angle_to_rotation_matrix(axis_angle)  # (..., 3, 3)


    # Prepare output
    # Result shape is (..., 3)
    euler = []

    # R = R_x * R_y * R_z
    sy = -R[..., 2, 0]
    cy = th.sqrt(1 - sy * sy)
    x = th.atan2(R[..., 2, 1], R[..., 2, 2])
    y = th.atan2(sy, cy)
    z = th.atan2(R[..., 1, 0], R[..., 0, 0])
    euler = th.stack([x, z, -y], dim=-1)

    return euler

def recursive_axisangle_to_euler(gls_expr):
    if isinstance(gls_expr, gls.AxisAngleRotate3D):
        new_args = []
        arg_0 = gls_expr.get_arg(0)
        arg_1 = gls_expr.tensor().get_arg(1)
        new_arg_0 = recursive_axisangle_to_euler(arg_0)
        new_arg_1 = convert_axis_angle_to_euler(arg_1)
        new_expr = gls.EulerRotate3D(new_arg_0, new_arg_1).sympy()
        return new_expr
    else:
        if isinstance(gls_expr, gls.GLFunction):
            in_args = gls_expr.get_args()
            new_args = []
            for arg in in_args:
                if isinstance(arg, gls.GLBase):
                    out_expr = recursive_axisangle_to_euler(arg)
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

def fetch_singular_expr_eval(expr, temperature=0.1, relaxed_eval=True, remove_marker=True):
    if isinstance(expr, StochasticPrimitive):
        new_expr = fetch_singular_expr_stochastic(expr, relaxed_eval=relaxed_eval)
        out_expr = fetch_singular_expr_eval(new_expr, temperature=temperature, relaxed_eval=relaxed_eval, remove_marker=remove_marker)
        return out_expr
    elif isinstance(expr, VARAXIS_CLASSES):
        new_expr = reduce_varaxis(expr, temperature=temperature, relaxed_eval=relaxed_eval)
        return new_expr
    elif isinstance(expr, PrimitiveMarker):
        in_expr = fetch_singular_expr_eval(expr.args[0], temperature=temperature, relaxed_eval=relaxed_eval, remove_marker=remove_marker)
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
                new_expr = fetch_singular_expr_eval(arg, temperature=temperature, relaxed_eval=relaxed_eval, remove_marker=remove_marker)
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

