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
import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol
DROP_CONST = 1.0

@register_symbol
class PrimitiveMarker(gls.GLFunction):
    ...

@register_symbol
class StochasticPrimitive(gls.GLFunction):
    ...
