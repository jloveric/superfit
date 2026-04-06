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
from sysl.torch_compute.evaluate_mat_expr import rec_eval_mat_expr
import superfit.symbolic as sps

# Register PrimitiveMarker handler for material expression evaluation
@rec_eval_mat_expr.register
def prim_marker_eval(expr: sps.PrimitiveMarker, *args, **kwargs):
    subexpr = expr.get_arg(0)
    return rec_eval_mat_expr(subexpr, *args, **kwargs)