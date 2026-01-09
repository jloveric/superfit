import geolipi.symbolic as gls
import torch as th
import torch.nn.functional as F
from geolipi.torch_compute.sketcher import Sketcher
from typing import Optional, List, Tuple
import sympy as sp
from geolipi.symbolic.registry import register_symbol


# Packed versions - used for speeding up batched evals.
class SuperFrustumPackedBatched(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 14]"},
        }

class SolidSFPackedBatched(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 18]"},
        }

class SuperFrustumPackedBatchedStochastic(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 14]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

class SolidSFPackedBatchedStochastic(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 18]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

class SuperFrustumPackedBatchedSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 14]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
        }

class SolidSFPackedBatchedSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 18]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
        }

class SuperFrustumPackedBatchedStochasticSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 14]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

class SolidSFPackedBatchedStochasticSU(gls.Primitive3D):    
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 18]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }