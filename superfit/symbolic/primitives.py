import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol

# Syntactic sugar for existing Primitives
@register_symbol
class Cuboid(gls.Cuboid3D):
    """
    Cuboid Primitive.
    """

@register_symbol
class SuperQuadric(gls.InexactSuperQuadric3D):
    """
    SuperQuadric Primitive.
    """


@register_symbol
class SPProto(gls.Primitive3D):
    """
    Based on sdSuperprim by paniq
    source: https://www.shadertoy.com/view/Xdy3Rm
    Coverage: Cuboid, Cylinder, Sphere, Uniform Capsule, Torus, Cable, OnionedRoundedRectExtrusion
    """
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "Vector[4]"},
            "dilate_3d": {"type": "float"},
            "onion": {"type": "float"},
            "extrussion": {"type": "Vector[2]"},
        }

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
class SuperGeon(gls.Primitive3D):
    """
    SuperGeon -> Make Triangle/Prism Possible - via trapezoid + gourd like scaling.
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
            "trapeze": {"type": "float"},
            "taper_bulge": {"type": "float"},
            "rot2d": {"type": "float"},
        }


@register_symbol
class SolidSF(gls.Primitive3D):
    """
    Solid SuperFrustum - convergent to one of Cuboid, Cylinder, Sphere, Cone (ref: https://arxiv.org/abs/2512.09201)
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
            "logits": {"type": "Vector[4]"},
        }

# Other Variants. 
@register_symbol
class SFSP(gls.Primitive3D):
    """
    SuperFrustum packed variable version for shader code generation.
    """
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "round_dilate_taper_bend": {"type": "Vector[4]"},
            "Onion_Ratio": {"type": "float"},
            # Nove to two packed variant.
        }
            
@register_symbol
class SGSP(gls.Primitive3D):
    """
    SuperGeon packed variable version for shader code generation.
    """
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness_dilate_taper_bulge": {"type": "Vector[4]"},
            "onion_ratio_trapeze_taper_bulge_rot2d": {"type": "Vector[4]"},
        }