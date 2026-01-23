import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol

@register_symbol
class SuperFrustum(gls.Primitive3D):
    """
    Primitive described in https://arxiv.org/abs/2512.09201.
    """
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "float"},
            "dilate_3d": {"type": "float"},
            "taper": {"type": "float"},
            "bulge": {"type": "float"},
            "onion": {"type": "float"},
        }

@register_symbol
class SFSP(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "round_dilate_taper_bend": {"type": "Vector[4]"},
            "Onion_Ratio": {"type": "float"},
            # Nove to two packed variant.
        }
            
@register_symbol
class SolidSF(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "float"},
            "dilate_3d": {"type": "float"},
            "taper": {"type": "float"},
            "bulge": {"type": "float"},
            "onion": {"type": "float"},
            "logits": {"type": "Vector[4]"},
        }

@register_symbol
class VarAxisSF(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "float"},
            "dilate_3d": {"type": "float"},
            "taper": {"type": "float"},
            "bulge": {"type": "float"},
            "onion": {"type": "float"},
            "logits": {"type": "Vector[3]"},
        }

@register_symbol
class SuperFrustumX(SuperFrustum):
    """
    About the X Axis - same as SuperFrustum
    """

@register_symbol
class SuperFrustumY(SuperFrustum):
    """
    About the Y Axis - same as SuperFrustum
    """

@register_symbol
class SuperFrustumZ(SuperFrustum):
    """
    About the Z Axis - same as SuperFrustum
    """


# Syntactic sugar for existing Primitives
class Cuboid(gls.Cuboid3D):
    ...

class SuperQuadric(gls.InexactSuperQuadric3D):
    ...

@register_symbol
class SPNeo(gls.Primitive3D):

    """
    Close to Primitive used in Project Neo.
    Coverage: Cuboid, Cylinder, Sphere, Uniform Capsule, Torus, Cable, OnionedRoundedRectExtrusion
    """
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "Vector[4]"},
            "dilate_3d": {"type": "float"},
            "onion": {"type": "float"},
        }

# TBD: Rewire the shaders for these primitives. 


@register_symbol
class SPBase(gls.Primitive3D):

    """
    Basic Rounded Rectangle Extrusion.
    Coverage: Cuboid, Cylinder, Sphere, Uniform Capsule
    """
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "float"},
            "dilate_3d": {"type": "float"},
        }

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
