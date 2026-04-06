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

Custom evaulation to speed up SuperGeon.
Did not really speed up things :(.
uncomment the @rec_shader_eval.register decorator to use this.
"""
import geolipi.symbolic as gls
import superfit.symbolic as sps
from sysl.shader.global_shader_context import GlobalShaderContext
from sysl.shader.evaluate_shader_trace import _inline_parse_param_from_expr
from sysl.shader.evaluate_singlepass import rec_shader_eval
from sysl.shader.evaluate_shader_trace import rec_sdf_shader_eval

# @rec_shader_eval.register
def eval_prim_sdf(expression: sps.SuperGeon, global_sc: GlobalShaderContext, *args, **kwargs) -> GlobalShaderContext:
    # basic version
    params = expression.args
    func_name = expression.__class__.__name__
    shader_params = _inline_parse_param_from_expr(expression, params, global_sc)
    # global_sc = PRIMITIVE_MAP[type(expression)](global_sc, *shader_params)
    primitive_param = ",".join(shader_params)
    global_sc.local_sc.add_dependency(func_name)
    global_sc.add_shader_module(func_name, primitive_param=primitive_param)
    updated_params = global_sc.shader_modules[func_name].get_updated_param(primitive_param)
    cur_pos = global_sc.local_sc.pos_stack.pop()
    sdf_name = f"sdf_{global_sc.local_sc.res_sdf_count}"
    global_sc.local_sc.res_sdf_count += 1
    # GLSL code for sphere (sphere_param[0] is the vec4 sphere parameters)
    
    code_line = f"float {sdf_name} = {func_name}({cur_pos}, {updated_params});"
    global_sc.local_sc.add_codeline(code_line)
    global_sc.local_sc.res_sdf_stack.append(("float", sdf_name))
    return global_sc


# @rec_sdf_shader_eval.register
def eval_prim_sdf(expression: gls.Primitive3D, global_sc) -> GlobalShaderContext:
    
    params = expression.args
    func_name = expression.__class__.__name__
    shader_params = _inline_parse_param_from_expr(expression, params, global_sc)
    primitive_param = ",".join(shader_params)

    global_sc.local_sc.add_dependency(func_name)
    global_sc.add_shader_module(func_name, primitive_param=primitive_param)
    updated_params = global_sc.shader_modules[func_name].get_updated_param(primitive_param)

    cur_pos = global_sc.local_sc.pos_stack.pop()
    sdf_name = f"sdf_{global_sc.local_sc.res_sdf_count}"
    global_sc.local_sc.res_sdf_count += 1
    code_line = f"float {sdf_name} = {func_name}({cur_pos}, {updated_params});"
    global_sc.local_sc.add_codeline(code_line)
    prim_id = global_sc.prim_count  # Store BEFORE increment
    prim_name = f"prim_{prim_id}"
    global_sc.prim_count += 1
    code_line = f"vec2 {prim_name} = vec2({sdf_name}, {prim_id});"  # Use stored value
    global_sc.local_sc.add_codeline(code_line)
    global_sc.local_sc.res_sdf_stack.append(("vec2", prim_name))

    return global_sc
