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
from .primitives import (SuperFrustum, SuperQuadric, SPProto, SuperGeon, SFSP, SPPSP, SGSP)

@register_symbol
class VarAxisSF(gls.Primitive3D):
    """
    Variable Axis SuperFrustum - models SuperFrustum over 3 axes - X, Y, Z.
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
            "axis_logits": {"type": "Vector[3]"},
        }

@register_symbol
class SuperFrustumX(SuperFrustum):
    """
    SuperFrustum over the X Axis.
    """

@register_symbol
class SuperFrustumY(SuperFrustum):
    """
    SuperFrustum over the Y Axis.
    """

@register_symbol
class SuperFrustumZ(SuperFrustum):
    """
    SuperFrustum over the Z Axis.
    """

@register_symbol
class SFSPX(SFSP):
    """
    SFSP over the X Axis.
    """

@register_symbol
class SFSPY(SFSP):
    """
    SFSP over the Y Axis.
    """

@register_symbol
class SFSPZ(SFSP):
    """
    SFSP over the Z Axis.
    """

@register_symbol
class VarAxisSQ(gls.Primitive3D):
    """
    Variable Axis SuperQuadric - models SuperQuadric over 3 axes - X, Y, Z.
    """
    @classmethod
    def default_specs(cls):
        return {"skew_vec": {"type": "Vector[3]"}, 
            "epsilon_1": {"type": "float", "min": 0.0}, 
            "epsilon_2": {"type": "float", "min": 0.0},
            "axis_logits": {"type": "Vector[3]"}
        }

@register_symbol
class SuperQuadricX(SuperQuadric):
    """
    SuperQuadric over the X Axis.
    """

@register_symbol
class SuperQuadricY(SuperQuadric):
    """
    SuperQuadric over the Y Axis.
    """
@register_symbol
class SuperQuadricZ(SuperQuadric):
    """
    SuperQuadric over the Z Axis.
    """

@register_symbol
class VarAxisSPP(gls.Primitive3D):
    """
    Variable Axis SPProto - models SPProto over 3 axes - X, Y, Z.
    """
    @classmethod
    def default_specs(cls):
        return {
            "size": {"type": "Vector[3]"},
            "roundness": {"type": "Vector[4]"},
            "dilate_3d": {"type": "float"},
            "onion": {"type": "float"},
            "extrussion": {"type": "Vector[2]"},
            "axis_logits": {"type": "Vector[3]"}
        }

@register_symbol
class SPProtoX(SPProto):
    """
    SPProto over the X Axis.
    """

@register_symbol
class SPProtoY(SPProto):
    """
    SPProto over the Y Axis.
    """
@register_symbol
class SPProtoZ(SPProto):
    """
    SPProto over the Z Axis.
    """
@register_symbol
class SPPSPX(SPPSP):
    """
    SPPSP over the X Axis.
    """

@register_symbol
class SPPSPY(SPPSP):
    """
    SPPSP over the Y Axis.
    """

@register_symbol
class SPPSPZ(SPPSP):
    """
    SPPSP over the Z Axis.
    """

@register_symbol
class VarAxisSG(gls.Primitive3D):
    """
    Variable Axis SuperGeon - models SuperGeon over 3 axes - X, Y, Z.
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
            "axis_logits": {"type": "Vector[3]"}
        }

@register_symbol
class SuperGeonX(SuperGeon):
    """
    SuperGeon over the X Axis.
    """

@register_symbol
class SuperGeonY(SuperGeon):
    """
    SuperGeon over the Y Axis.
    """

@register_symbol
class SuperGeonZ(SuperGeon):
    """
    SuperGeon over the Z Axis.
    """

@register_symbol
class SGSPX(SGSP):
    """
    SGSP over the X Axis.
    """
@register_symbol
class SGSPY(SGSP):
    """
    SGSP over the Y Axis.
    """
@register_symbol
class SGSPZ(SGSP):
    """
    SGSP over the Z Axis.
    """