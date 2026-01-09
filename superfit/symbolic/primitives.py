import geolipi.symbolic as gls
import torch as th
import torch.nn.functional as F
from geolipi.torch_compute.sketcher import Sketcher
from typing import Optional, List, Tuple
import sympy as sp
from geolipi.symbolic.registry import register_symbol


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

class SolidSF(gls.GLFunction):
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

# Packed versions - used for speeding up batched evals.
class SuperFrustumPacked(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Vector[14]"},
        }

class SolidSFPacked(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Vector[18]"},
        }

#### OTHERS> 
# Wrong V1 v2
# Exact V1 V2
# Approx V1 V2
# CHamfered. 