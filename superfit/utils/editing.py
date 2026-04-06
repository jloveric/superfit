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
import json
import numpy as np
import distinctipy
import sympy as sp
import torch as th
import geolipi.symbolic as gls
import superfit.symbolic as sps
from sympy import Tuple as SympyTuple, Float as SympyFloat, Integer as SympyInteger
from sysl.shader.global_shader_context import GlobalShaderContext
from sysl.shader.evaluate_multipass import rec_sdf_shader_eval
from ..symbolic.utils import (
    extract_primitive_bundles,
    n_prims_in_expr,
    recursive_prim_to_packed,
)
from ..symbolic.rotation_functions import recursive_axisangle_to_eulerangle

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

# Editing shader pipeline: replace AxisAngleRotate3D with EulerRotate3D before evaluate_to_shader.
# Set to False (or comment out the call to apply_editing_aa_to_euler) to keep axis-angle.
EDITING_CONVERT_AXISANGLE_TO_EULER = True


def apply_editing_aa_to_euler(expr):
    """Apply ``recursive_axisangle_to_eulerangle`` when EDITING_CONVERT_AXISANGLE_TO_EULER is True."""
    if not EDITING_CONVERT_AXISANGLE_TO_EULER:
        return expr
    return recursive_axisangle_to_eulerangle(expr)

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
                        (0, 0, 0, -2),
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


def get_sppsp_editing_expr():
    """Editing primitive wrapper for packed SPPSP primitives."""
    primitive_expr = gls.SmoothUnion(
        gls.Translate3D(
            gls.AxisAngleRotate3D(
                sps.SPPSP(
                    gls.UniformVec3(
                        (0, 0, 0),
                        (0.5, 0, 0),
                        (2, 2, 2),
                        "size",
                    ),
                    gls.UniformVec4(
                        (0, 0, 0, 0),
                        (0.5, 0, 0, 0),
                        (1, 1.0, 1.0, 1.0),
                        "roundness",
                    ),
                    gls.UniformVec4(
                        (0, 0, 0, 0),
                        (0.5, 0, 0, 0),
                        (1, 1.0, 1, 1),
                        "doe",
                    ),
                ),
                gls.UniformVec3(
                    (-np.pi, -np.pi, -np.pi),
                    (0, 0, 0),
                    (np.pi, np.pi, np.pi),
                    "axis_angle",
                ),
            ),
            gls.UniformVec3(
                (-1, -1, -1),
                (0.5, 0, 0),
                (1, 1, 1),
                "translate",
            ),
        ),
        gls.Sphere3D((0.5,)),
        gls.UniformFloat(
            (0.0,),
            (0.5,),
            (1.0,),
            "SmoothUnion_Amount",
        ),
    )
    return primitive_expr


def get_sgsp_editing_expr():
    """Editing primitive wrapper for packed SGSP primitives."""
    primitive_expr = gls.SmoothUnion(
        gls.Translate3D(
            gls.AxisAngleRotate3D(
                sps.SGSP(
                    gls.UniformVec3(
                        (0, 0, 0),
                        (0.5, 0, 0),
                        (2, 2, 2),
                        "size",
                    ),
                    gls.UniformVec4(
                        (0, 0, 0, -2),
                        (0.5, 0, 0, 0),
                        (1, 1.0, 2, 2),
                        "roundness_dilate_taper_bulge",
                    ),
                    gls.UniformVec4(
                        (0, -1, -1, -np.pi),
                        (0.5, 0, 0, 0),
                        (1, 1.0, 1, np.pi),
                        "onion_ratio_trapeze_taper_bulge_rot2d",
                    ),
                ),
                gls.UniformVec3(
                    (-np.pi, -np.pi, -np.pi),
                    (0, 0, 0),
                    (np.pi, np.pi, np.pi),
                    "axis_angle",
                ),
            ),
            gls.UniformVec3(
                (-1, -1, -1),
                (0.5, 0, 0),
                (1, 1, 1),
                "translate",
            ),
        ),
        gls.Sphere3D((0.5,)),
        gls.UniformFloat(
            (0.0,),
            (0.5,),
            (1.0,),
            "SmoothUnion_Amount",
        ),
    )
    return primitive_expr


def get_cuboid_editing_expr():
    """Editing primitive wrapper for Cuboid (single vec3 size; already packed, no VarAxis)."""
    primitive_expr = gls.SmoothUnion(
        gls.Translate3D(
            gls.AxisAngleRotate3D(
                sps.Cuboid(
                    gls.UniformVec3(
                        (0, 0, 0),
                        (0.5, 0, 0),
                        (2, 2, 2),
                        "size",
                    ),
                ),
                gls.UniformVec3(
                    (-np.pi, -np.pi, -np.pi),
                    (0, 0, 0),
                    (np.pi, np.pi, np.pi),
                    "axis_angle",
                ),
            ),
            gls.UniformVec3(
                (-1, -1, -1),
                (0.5, 0, 0),
                (1, 1, 1),
                "translate",
            ),
        ),
        gls.Sphere3D((0.5,)),
        gls.UniformFloat(
            (0.0,),
            (0.5,),
            (1.0,),
            "SmoothUnion_Amount",
        ),
    )
    return primitive_expr


def _build_packed_class_to_base_map():
    """Map leaf / axis-variant classes to the packed base used for editing (SFSP/SPPSP/SGSP/Cuboid)."""
    mapping = {}

    def add(base_cls, *variant_names):
        for name in variant_names:
            cls = getattr(sps, name, None)
            if cls is not None:
                mapping[cls] = base_cls

    add(sps.SFSP, "SFSP", "SFSPX", "SFSPY", "SFSPZ")
    add(sps.SPPSP, "SPPSP", "SPPSPX", "SPPSPY", "SPPSPZ")
    add(sps.SGSP, "SGSP", "SGSPX", "SGSPY", "SGSPZ")
    mapping[sps.Cuboid] = sps.Cuboid
    return mapping


PACKED_CLASS_TO_BASE = _build_packed_class_to_base_map()


def detect_packed_prim_type(expr):
    """Detect which packed primitive family exists inside `expr`."""
    if expr is None:
        return None
    mapped = PACKED_CLASS_TO_BASE.get(expr.__class__)
    if mapped is not None:
        return mapped
    if isinstance(expr, gls.GLFunction):
        for arg in getattr(expr, "args", []):
            found = detect_packed_prim_type(arg)
            if found is not None:
                return found
    return None


PRIM_EDITING_CONFIG = {
    sps.SFSP: {
        "get_editing_expr": get_sfsp_editing_expr,
        "uniform_names": [
            "size",
            "round_dilate_taper_bend",
            "axis_angle",
            "translate",
            "SmoothUnion_Amount",
            "Onion_Ratio",
        ],
        "uniform_map": {
            0: "size",
            1: "round_dilate_taper_bend",
            2: "Onion_Ratio",
            3: None,
            4: None,
            5: "axis_angle",
            6: "translate",
            7: "SmoothUnion_Amount",
        },
        "uniform_map_textured": {
            0: "size",
            1: "round_dilate_taper_bend",
            2: "Onion_Ratio",
            3: None,
            4: None,
            5: None,
            6: None,
            7: None,
            8: None,
            9: None,
            10: None,
            11: "axis_angle",
            12: "translate",
            13: "SmoothUnion_Amount",
        },
    },
    sps.SPPSP: {
        "get_editing_expr": get_sppsp_editing_expr,
        "uniform_names": [
            "size",
            "roundness",
            "doe",
            "axis_angle",
            "translate",
            "SmoothUnion_Amount",
        ],
        "uniform_map": {
            0: "size",
            1: "roundness",
            2: "doe",
            3: None,
            4: None,
            5: "axis_angle",
            6: "translate",
            7: "SmoothUnion_Amount",
        },
        "uniform_map_textured": {
            0: "size",
            1: "roundness",
            2: "doe",
            3: None,
            4: None,
            5: None,
            6: None,
            7: None,
            8: None,
            9: None,
            10: None,
            11: "axis_angle",
            12: "translate",
            13: "SmoothUnion_Amount",
        },
    },
    sps.SGSP: {
        "get_editing_expr": get_sgsp_editing_expr,
        "uniform_names": [
            "size",
            "roundness_dilate_taper_bulge",
            "onion_ratio_trapeze_taper_bulge_rot2d",
            "axis_angle",
            "translate",
            "SmoothUnion_Amount",
        ],
        "uniform_map": {
            0: "size",
            1: "roundness_dilate_taper_bulge",
            2: "onion_ratio_trapeze_taper_bulge_rot2d",
            3: None,
            4: None,
            5: "axis_angle",
            6: "translate",
            7: "SmoothUnion_Amount",
        },
        "uniform_map_textured": {
            0: "size",
            1: "roundness_dilate_taper_bulge",
            2: "onion_ratio_trapeze_taper_bulge_rot2d",
            3: None,
            4: None,
            5: None,
            6: None,
            7: None,
            8: None,
            9: None,
            10: None,
            11: "axis_angle",
            12: "translate",
            13: "SmoothUnion_Amount",
        },
    },
    sps.Cuboid: {
        "get_editing_expr": get_cuboid_editing_expr,
        "uniform_names": [
            "size",
            "axis_angle",
            "translate",
            "SmoothUnion_Amount",
        ],
        "uniform_map": {
            0: "size",
            1: None,
            2: None,
            3: None,
            4: None,
            5: "axis_angle",
            6: "translate",
            7: "SmoothUnion_Amount",
        },
        "uniform_map_textured": {
            0: "size",
            1: None,
            2: None,
            3: None,
            4: None,
            5: None,
            6: None,
            7: None,
            8: None,
            9: None,
            10: None,
            11: "axis_angle",
            12: "translate",
            13: "SmoothUnion_Amount",
        },
    },
}


def _flatten_to_floats(x):
    """Coerce nested vec values (e.g. tuples of tuples) into a flat float list."""
    if isinstance(x, (list, tuple, sp.Tuple)):
        out = []
        for item in x:
            out.extend(_flatten_to_floats(item))
        return out
    return [float(x)]


def _create_auxiliary_prim(new_expr, *, primitive_expr=None, textured: bool):
    prim_type = detect_packed_prim_type(new_expr) or sps.SFSP
    config = PRIM_EDITING_CONFIG[prim_type]
    if primitive_expr is None:
        primitive_expr = config["get_editing_expr"]()

    varnamed_expr, _, var_map_base = new_expr._get_varnamed_expr(
        exclude_class_set=(gls.UniformVariable, sls.MaterialV4)
    )
    # Some primitives produce nested tuple shapes; flatten them into a single vec.
    var_map_base = {k: _flatten_to_floats(v) for k, v in var_map_base.items()}

    primitive_parameter_bundles = extract_primitive_bundles(varnamed_expr)
    for key, value in primitive_parameter_bundles.items():
        primitive_parameter_bundles[key] = [str(x) for x in value]

    gc_prim = GlobalShaderContext()
    gc_prim = rec_sdf_shader_eval(primitive_expr, global_sc=gc_prim)
    uniforms = gc_prim.uniforms

    new_uniforms = {name: uniforms[name] for name in config["uniform_names"]}
    uniform_map = config["uniform_map_textured"] if textured else config["uniform_map"]

    return {
        "primitive_map": primitive_parameter_bundles,
        "uniforms": new_uniforms,
        "var_map": var_map_base,
        "uniform_map": uniform_map,
    }


def create_auxiliary_prim(new_expr, primitive_expr=None):
    return _create_auxiliary_prim(new_expr, primitive_expr=primitive_expr, textured=False)


def create_auxiliary_prim_textured(new_expr, primitive_expr=None):
    return _create_auxiliary_prim(new_expr, primitive_expr=primitive_expr, textured=True)


def expression_from_varmap(varnamed_expr_str, varmap):
    """
    Build a GeoLIPI expression from the varnamed expression string and edited variable values.

    Uses the same state format as geolipi: a dict with "expr_str" and "symbol_tensor_map".
    The varmap (from the JS editor or var_map_edited.json) provides the values for var_0,
    var_1, ... which are injected via gls.GLFunction.from_state.

    Args:
        varnamed_expr_str: String representation of the expression with var_0, var_1, ...
            (from the varnamed expression used to generate the shader / editor).
        varmap: Dict of {"var_0": [0.5, 0.5, 0.5], "var_1": [0, 0, 0, 0], ...}
                (edited values from the frontend or loaded JSON).

    Returns:
        A gls.GLFunction expression with the varmap values applied.
    """
    symbol_tensor_map = {
        sp.Symbol(k): th.tensor(v, dtype=th.float32) if isinstance(v, (list, tuple)) else th.tensor([float(v)], dtype=th.float32)
        for k, v in varmap.items()
    }
    state = {
        "expr_str": varnamed_expr_str,
        "symbol_tensor_map": symbol_tensor_map,
        "GLFunction": True,
    }
    return gls.GLFunction.from_state(state)


def load_varmap_json(filepath):
    """Load a varmap JSON file exported from the editor."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_edit_mode_html(expression, sketcher_3d, save_file_name="edit_mode.html", is_textured=False):

    expression = apply_editing_aa_to_euler(expression)
    if is_textured:
        mat_expr_reconf = reconf_sp_rgb(expression.tensor())
        expression = convert_to_atlased_encoded(mat_expr_reconf, sketcher_3d)

    new_expr = fetch_singular_expr_eval(expression.sympy(), relaxed_eval=False, remove_marker=False)
    new_expr = recursive_sm_to_smg(new_expr.sympy())
    ntc_ss_expr = recursive_prim_to_packed(new_expr.sympy())
    n_prims = n_prims_in_expr(ntc_ss_expr)
    print(f"n_prims: {n_prims}")

    if not is_textured:
        colors = distinctipy.get_colors(n_prims+2)
        ntc_ss_expr, _ = recursive_gls_to_sysl(ntc_ss_expr.sympy(), 1, 
                                version="v4", mode="simple", colors=colors)

    prim_type = detect_packed_prim_type(ntc_ss_expr) or sps.SFSP
    primitive_expr = PRIM_EDITING_CONFIG[prim_type]["get_editing_expr"]()
    
    if is_textured:
        auxiliary = create_auxiliary_prim_textured(ntc_ss_expr, primitive_expr)  
    else:
        auxiliary = create_auxiliary_prim(ntc_ss_expr, primitive_expr)  
    new_expr = fetch_singular_expr_eval(ntc_ss_expr.sympy(), relaxed_eval=False, remove_marker=True)

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