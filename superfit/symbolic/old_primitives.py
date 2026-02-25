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

@register_symbol
class SPTaperedWrongV1(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "float"},
            "dilate_3d": {"type": "float"},
            "scale": {"type": "float"},
        }

@register_symbol
class SPTaperedWrongV2(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "Vector[2]"},
            "onion_ratio": {"type": "float"},
            "dilate_3d": {"type": "float"},
            "scale": {"type": "Vector[2]"},
        }

@register_symbol
class SPTaperedNewtonV1(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "float"},
            "dilate_3d": {"type": "float"},
            "scale": {"type": "float"},
        }

@register_symbol
class SPTaperedApproxV1(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "float"},
            "dilate_3d": {"type": "float"},
            "scale_opp": {"type": "float"},
        }

@register_symbol
class SPTaperedApproxV2(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "Vector[2]"},
            "dilate_3d": {"type": "float"},
            "scale_opp": {"type": "Vector[2]"},
        }

@register_symbol
class SPChamferedV1(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "ch": {"type": "float"},
            "dilate_3d": {"type": "float"},
        }

@register_symbol
class SPChamferedV2(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "Vector[2]"},
            "dilate_3d": {"type": "float"},
            "scale_opp": {"type": "Vector[2]"},
        }

@register_symbol
class SPTaperedQuarticV1(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "float"},
            "dilate_3d": {"type": "float"},
            "scale_opp": {"type": "float"},
        }
