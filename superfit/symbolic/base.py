import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol
DROP_CONST = 1.0

@register_symbol
class PrimitiveMarker(gls.GLFunction):
    ...

@register_symbol
class StochasticPrimitive(gls.GLFunction):
    ...
