import geolipi.symbolic as gls
import sysl.symbolic as csls
import numpy as np
from geolipi.symbolic.symbol_types import PRIM_TYPE
from .base import StochasticPrimitive, PrimitiveMarker
import superfit.symbolic as sps
import torch as th
import sympy as sp
from geolipi.torch_compute.transforms import axis_angle_to_rotation_matrix
import sysl.symbolic as sls
from .base import sample_gumbel




def convert_to_packed(expr):
    if isinstance(expr, PrimitiveMarker):
        # Expr = PrimitiveMarker(StochasticPrimitive(Translate(AARotate(NeoTapered))))
        if isinstance(expr.args[0], StochasticPrimitive):
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
        
        new_param = [translate_arg, rotate_arg, *sp_tapered_arg]
        new_param = th.cat(new_param, dim=-1)
        if isinstance(prim_expr, sps.SuperFrustum):
            new_expr = sps.SuperFrustumPacked(new_param)
        elif isinstance(prim_expr, sps.SolidSF):
            new_expr = sps.SolidSFPacked(new_param)
        if has_stochastic:
            new_expr = StochasticPrimitive(new_expr, stochastic_arg)
        new_expr = PrimitiveMarker(new_expr)
        return new_expr
    elif isinstance(expr, gls.GLFunction):
        new_args = []
        for arg in expr.args:
            if arg in expr.lookup_table:
                arg = expr.lookup_table[arg]
            new_args.append(convert_to_packed(arg))
        return expr.func(*new_args)
    else:
        return expr

def convert_to_unpacked(expr):
    if isinstance(expr, PrimitiveMarker):
        # Expr = PrimitiveMarker(StochasticPrimitive(Translate(AARotate(NeoTapered))))
        if isinstance(expr.args[0], StochasticPrimitive):
            has_stochastic = True
            stochastic_expr = expr.args[0]
            stochastic_arg = stochastic_expr.get_arg(1)
            next_expr = stochastic_expr
        else:
            has_stochastic = False
            next_expr = expr

        tapered_packed_expr = next_expr.args[0]
        tapered_packed_arg = tapered_packed_expr.get_arg(0)
        translate_arg = tapered_packed_arg[..., :3]
        rotate_arg = tapered_packed_arg[..., 3:6]
        sp_arg_0 = tapered_packed_arg[..., 6:9]
        
        if isinstance(tapered_packed_expr, sps.SuperFrustumPacked):
            sp_arg_1 = tapered_packed_arg[..., 9:10]
            sp_arg_2 = tapered_packed_arg[..., 10:11]
            sp_arg_3 = tapered_packed_arg[..., 11:12]
            sp_arg_4 = tapered_packed_arg[..., 12:13]
            sp_arg_5 = tapered_packed_arg[..., 13:14]
            new_primitive = sps.SuperFrustum(sp_arg_0, sp_arg_1, sp_arg_2, sp_arg_3, sp_arg_4, sp_arg_5)
        elif isinstance(tapered_packed_expr, sps.SolidSFPacked):
            sp_arg_1 = tapered_packed_arg[..., 9:10]
            sp_arg_2 = tapered_packed_arg[..., 10:11]
            sp_arg_3 = tapered_packed_arg[..., 11:12]
            sp_arg_4 = tapered_packed_arg[..., 12:13]
            sp_arg_5 = tapered_packed_arg[..., 13:14]
            sp_arg_6 = tapered_packed_arg[..., 14:18]
            new_primitive = sps.SolidSF(sp_arg_0, sp_arg_1, sp_arg_2, sp_arg_3, sp_arg_4, sp_arg_5, sp_arg_6)
        else:
            raise ValueError(f"Invalid tapered packed expression: {tapered_packed_expr}")
        new_primitive = gls.AxisAngleRotate3D(new_primitive, rotate_arg)
        new_primitive = gls.Translate3D(new_primitive, translate_arg)
        if has_stochastic:
            new_primitive = StochasticPrimitive(new_primitive, stochastic_arg)
        new_expr = PrimitiveMarker(new_primitive)
        return new_expr

    elif isinstance(expr, gls.GLFunction):
        new_args = []
        for arg in expr.args:
            if arg in expr.lookup_table:
                arg = expr.lookup_table[arg]
            new_args.append(convert_to_unpacked(arg))
        return expr.func(*new_args)
    else:
        return expr

def split_tapered_packed_param(param):
    param_0 = param[..., :3]
    param_1 = param[..., 3:6]
    param_2 = param[..., 6:9]
    param_3 = param[..., 9:10]
    param_4 = param[..., 10:11]
    param_5 = param[..., 11:12]
    return param_0, param_1, param_2, param_3, param_4, param_5


def split_superfrustum_packed_param(param):
    param_0 = param[..., :3]
    param_1 = param[..., 3:6]
    param_2 = param[..., 6:9]
    param_3 = param[..., 9:10]
    param_4 = param[..., 10:11]
    param_5 = param[..., 11:12]
    param_6 = param[..., 12:13]
    param_7 = param[..., 13:14]
    return param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7

def split_solid_sf_packed_param(param):
    param_0 = param[..., :3]
    param_1 = param[..., 3:6]
    param_2 = param[..., 6:9]
    param_3 = param[..., 9:10]
    param_4 = param[..., 10:11]
    param_5 = param[..., 11:12]
    param_6 = param[..., 12:13]
    param_7 = param[..., 13:14]
    param_8 = param[..., 14:18]
    return param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8

    
def recursive_ntc_to_ntc_ss(gls_expr):
    if isinstance(gls_expr, sps.NTC):
        new_args = []
        cur_expr = gls_expr.sympy()
        ntc_args = [cur_expr.get_arg(i) for i in range(len(gls_expr.args))]
        arg_0 = ntc_args[0]
        arg_1 = (ntc_args[1][0], ntc_args[2][0], ntc_args[3][0], ntc_args[4][0],)
        new_expr = sps.NTC_SS(arg_0, arg_1)
        return new_expr
    else:
        if isinstance(gls_expr, gls.GLFunction):
            new_args = []
            for arg in gls_expr.args:
                if isinstance(arg, gls.GLBase):
                    out_expr = recursive_ntc_to_ntc_ss(arg)
                    new_args.append(out_expr)
                else:
                    new_args.append(arg)
            return gls_expr.__class__(*new_args)
        else:
            return gls_expr

def recursive_ntco_to_ntco_ss(gls_expr):
    if isinstance(gls_expr, sps.NTCO):
        new_args = []
        cur_expr = gls_expr.sympy()
        ntc_args = [cur_expr.get_arg(i) for i in range(len(gls_expr.args))]
        arg_0 = ntc_args[0]
        arg_1 = (ntc_args[1][0], ntc_args[2][0], ntc_args[3][0], ntc_args[4][0],)
        arg_2 = ntc_args[5]
        new_expr = sps.NTCO_SS(arg_0, arg_1, arg_2)
        return new_expr
    else:
        if isinstance(gls_expr, gls.GLFunction):
            new_args = []
            for arg in gls_expr.args:
                if isinstance(arg, gls.GLBase):
                    out_expr = recursive_ntco_to_ntco_ss(arg)
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


            
def fetch_singular_gmbl(expr, relaxed_eval=True):
    # Sample a y
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
        gumbel_soft = th.nn.functional.softmax((logits + gumbel_noise) / temperature, dim=0) 
        sel_ind = th.argmax(gumbel_soft).item()  # int
    else:
        sel_ind = th.argmax(logits).item()  # int
    selected_expr = expr.args[0].args[sel_ind]
    return selected_expr

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
        gumbel_soft = th.nn.functional.softmax((logits + gumbel_noise) / temperature, dim=0) 
        sel_ind = th.argmax(gumbel_soft).item()  # int
    else:
        sel_ind = th.argmax(logits).item()  # int
    if sel_ind == 0:
        selected_expr = expr.args[0]
    else:
        selected_expr = gls.NullExpression3D()
    return selected_expr

def fetch_singular_expr(expr, relaxed_eval=True, remove_marker=True):
    if isinstance(expr, StochasticPrimitive):
        return fetch_singular_expr_stochastic(expr, relaxed_eval=relaxed_eval)
    elif isinstance(expr, PrimitiveMarker):
        in_expr = fetch_singular_expr(expr.args[0], relaxed_eval=relaxed_eval, remove_marker=remove_marker)
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
                new_expr = fetch_singular_expr(arg, relaxed_eval=relaxed_eval, remove_marker=remove_marker)
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
    elif isinstance(expression, (sps.SuperFrustumPackedBatchedStochasticSU, 
                                sps.SolidSFPackedBatchedStochasticSU)):
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
            raise ValueError(f"Invalid number of arguments for NeoTaperedPackedBatchedSU: {len(expression.args)}")
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
    elif isinstance(expression, (sps.SuperFrustumPackedBatchedStochasticSU, 
                                sps.SolidSFPackedBatchedStochasticSU)):
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
            raise ValueError(f"Invalid number of arguments for NeoTaperedPackedBatchedStochasticSU: {len(expression.args)}")
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
    varnamed_expr, _, var_map_base = expression._get_varnamed_expr(exclude_class_list=(gls.UniformVariable, sls.MaterialV4))
    
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

