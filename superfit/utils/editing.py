import numpy as np
import distinctipy
import geolipi.symbolic as gls
import superfit.symbolic as sps
from sysl.shader.global_shader_context import GlobalShaderContext
from sysl.shader.evaluate_multipass import rec_sdf_shader_eval
from ..symbolic.utils import extract_primitive_bundles, n_prims_in_expr, recursive_sf_to_sfsp
from sysl.shader_runtime import create_multibuffer_shader_html
from sysl.utils import recursive_sm_to_smg, recursive_gls_to_sysl
from ..symbolic.utils import fetch_singular_expr_eval
from ..mat_opt.utils import reconf_sp_rgb, convert_to_atlased_encoded
from sysl.shader import evaluate_to_shader
import sysl.symbolic as sls

DEFAULT_EDITING_SETTINGS = {
    "render_mode": "v4",
    "variables": {
        "_AA": 1,
        "_ADD_FLOOR_PLANE": False,
        "_RAYCAST_MAX_STEPS": 150,
    },
    "extract_vars": True,
    "use_define_vars": True,
    "set_to_ubo": False,
    "set_param_to_texture": False,
}
def get_sfsp_editing_expr():

    primitive_expr = gls.SmoothUnion(
        gls.Translate3D(
            gls.AxisAngleRotate3D(
                sps.SFSP(
                    gls.UniformVec3(
                        (0, 0, 0),
                        (0.5, 0, 0),
                        (2, 2, 2),
                        "size"
                    ),
                    gls.UniformVec4(
                        (0, 0, 0, 0),
                        (0.5, 0, 0, 0),
                        (1, 1.0, 2, 2),
                        "round_dilate_taper_bend"
                    ),
                    gls.UniformFloat(
                        (0.0, ),
                        (0.5, ),
                        (1.0,),
                        "Onion_Ratio"
                    ),
                ),
                gls.UniformVec3(
                    (-np.pi, -np.pi, -np.pi),
                    (0, 0, 0),
                    (np.pi, np.pi, np.pi),
                    "axis_angle"
                )
            ), 
            gls.UniformVec3(
                (-1, -1, -1),
                (0.5, 0, 0),
                (1, 1, 1),
                "translate"
            )
        ),
        gls.Sphere3D((0.5,)),
        gls.UniformFloat(
            (0.0, ),
            (0.5, ),
            (1.0,),
            "SmoothUnion_Amount"
        )
    )
    return primitive_expr


def create_auxiliary_sf(new_expr, primitive_expr=None):
    if primitive_expr is None:
        primitive_expr = get_sfsp_editing_expr()
    varnamed_expr, _, var_map_base = new_expr._get_varnamed_expr(exclude_class_set=(gls.UniformVariable, sls.MaterialV4))
    var_map_base = {x:list([float(t) for t in y]) for x, y in var_map_base.items()}
    # varnamed_expr_no_mat = recursive_sysl_to_gls(varnamed_expr)
    primitive_parameter_bundles = extract_primitive_bundles(varnamed_expr)
    for key, value in primitive_parameter_bundles.items():
        primitive_parameter_bundles[key] = [str(x) for x in value]
    gc_prim = GlobalShaderContext()
    gc_prim = rec_sdf_shader_eval(primitive_expr, global_sc=gc_prim)
    uniforms = gc_prim.uniforms
    new_uniforms = {
        'size': uniforms['size'],
        'round_dilate_taper_bend': uniforms['round_dilate_taper_bend'],
        'axis_angle': uniforms['axis_angle'],
        'translate': uniforms['translate'],
        'SmoothUnion_Amount': uniforms['SmoothUnion_Amount'],
        'Onion_Ratio': uniforms['Onion_Ratio'],
    }
    uniform_map = {
        0: 'size',
        1: 'round_dilate_taper_bend',
        2: 'Onion_Ratio',
        3: None,
        4: None,
        5: 'axis_angle',
        6: 'translate',
        7: 'SmoothUnion_Amount',
    }
    auxiliary = {
        "primitive_map": primitive_parameter_bundles,
        "uniforms": new_uniforms,
        "var_map": var_map_base,
        "uniform_map": uniform_map,
    }
    return auxiliary


def create_auxiliary_sf_textured(new_expr, primitive_expr=None):
    if primitive_expr is None:
        primitive_expr = get_sfsp_editing_expr()
    varnamed_expr, _, var_map_base = new_expr._get_varnamed_expr(exclude_class_set=(gls.UniformVariable, sls.MaterialV4))
    var_map_base = {x:list([float(t) for t in y]) for x, y in var_map_base.items()}
    # varnamed_expr_no_mat = recursive_sysl_to_gls(varnamed_expr)
    primitive_parameter_bundles = extract_primitive_bundles(varnamed_expr)
    for key, value in primitive_parameter_bundles.items():
        primitive_parameter_bundles[key] = [str(x) for x in value]
    gc_prim = GlobalShaderContext()
    gc_prim = rec_sdf_shader_eval(primitive_expr, global_sc=gc_prim)
    uniforms = gc_prim.uniforms
    new_uniforms = {
        'size': uniforms['size'],
        'round_dilate_taper_bend': uniforms['round_dilate_taper_bend'],
        'axis_angle': uniforms['axis_angle'],
        'translate': uniforms['translate'],
        'SmoothUnion_Amount': uniforms['SmoothUnion_Amount'],
        'Onion_Ratio': uniforms['Onion_Ratio'],
    }
    uniform_map = {
        0: 'size',
        1: 'round_dilate_taper_bend',
        2: 'Onion_Ratio',
        3: None,
        4: None,
        5: None,
        6: None,
        7: None,
        8: None,
        9: None,
        10: None,
        11: 'axis_angle',
        12: 'translate',
        13: 'SmoothUnion_Amount',
    }
    auxiliary = {
        "primitive_map": primitive_parameter_bundles,
        "uniforms": new_uniforms,
        "var_map": var_map_base,
        "uniform_map": uniform_map,
    }
    return auxiliary


def save_edit_mode_html(expression, sketcher_3d, save_file_name="edit_mode.html", is_textured=False):

    if is_textured:
        mat_expr_reconf = reconf_sp_rgb(expression.tensor())
        expression = convert_to_atlased_encoded(mat_expr_reconf, sketcher_3d)

    new_expr = fetch_singular_expr_eval(expression.sympy(), relaxed_eval=False, remove_marker=False)
    new_expr = recursive_sm_to_smg(new_expr.sympy())
    ntc_ss_expr = recursive_sf_to_sfsp(new_expr.sympy())
    n_prims = n_prims_in_expr(ntc_ss_expr)
    print(f"n_prims: {n_prims}")

    if not is_textured:
        colors = distinctipy.get_colors(n_prims+2)
        ntc_ss_expr, _ = recursive_gls_to_sysl(ntc_ss_expr.sympy(), 1, 
                                version="v4", mode="simple", colors=colors)

    if is_textured:
        auxiliary = create_auxiliary_sf_textured(ntc_ss_expr, primitive_expr)  
    else:
        auxiliary = create_auxiliary_sf(ntc_ss_expr, primitive_expr)  
    new_expr = fetch_singular_expr_eval(ntc_ss_expr.sympy(), relaxed_eval=False, remove_marker=True)
    
    primitive_expr = get_sfsp_editing_expr()

    shader_bundles = evaluate_to_shader(new_expr.sympy(), settings=DEFAULT_EDITING_SETTINGS, 
                                            mode="multipass",
                                            post_process_shader=["selection_highlight"],
                                            primitive_editing_mode=True,
                                            prim_expr=primitive_expr
                                            )
    html_code = create_multibuffer_shader_html(shader_bundles, show_controls=True, 
                    layout_horizontal=True,
                    primitive_editing_mode=True,
                    auxiliary=auxiliary,
                    show_primitive_tracking=False)

    with open(save_file_name, "w") as f:
        f.write(html_code)