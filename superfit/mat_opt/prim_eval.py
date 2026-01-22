from sysl.torch_compute.evaluate_mat_expr import rec_eval_mat_expr
import superfit.symbolic as sps

# Register PrimitiveMarker handler for material expression evaluation
@rec_eval_mat_expr.register
def prim_marker_eval(expr: sps.PrimitiveMarker, *args, **kwargs):
    subexpr = expr.get_arg(0)
    return rec_eval_mat_expr(subexpr, *args, **kwargs)