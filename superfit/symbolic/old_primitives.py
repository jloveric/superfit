import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol

@register_symbol
class SPTaperedOnion(gls.Primitive3D):
    """
    Similar to sdUberprim by paniq
    source: https://www.shadertoy.com/view/MsVGWG
    """
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "float"},
            "dilate_3d": {"type": "float"},
            "scale": {"type": "float"},
            "onion_ratio": {"type": "float"},
        }

# TBD: Rewire the shaders for these primitives. 
@register_symbol
class SPTaperedWrongV1(gls.Primitive3D):
    ...

@register_symbol
class SPTaperedWrongV2(gls.Primitive3D):
    ...

@register_symbol
class SPTaperedCorrectV1(gls.Primitive3D):
    ...

@register_symbol
class SPTaperedCorrectV2(gls.Primitive3D):
    ...

@register_symbol
class SPTaperedApproxV1(gls.Primitive3D):
    ...

@register_symbol
class SPTaperedApproxV2(gls.Primitive3D):
    ...

@register_symbol
class SPChamfered(gls.Primitive3D):
    ...
