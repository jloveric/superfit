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
