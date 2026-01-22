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
DROP_CONST = 1.0

@register_symbol
class PrimitiveMarker(gls.GLFunction):
    ...

@register_symbol
class StochasticPrimitive(gls.GLFunction):
    ...
