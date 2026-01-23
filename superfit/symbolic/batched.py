import geolipi.symbolic as gls

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

class VarAxisSFPacked(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Vector[17]"},
        }

class CuboidPacked(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Vector[9]"},
        }

class SQPacked(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Vector[11]"},
        }
    
# Batched Variants

# SuperFrustum batched variants
class SuperFrustumPackedBatched(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 14]"},
        }

class SuperFrustumPackedBatchedStochastic(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 14]"},
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

class SuperFrustumPackedBatchedStochasticSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 14]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

# SolidSF batched variants
class SolidSFPackedBatched(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 18]"},
        }

class SolidSFPackedBatchedStochastic(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 18]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

class SolidSFPackedBatchedSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 18]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
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

# VarAxisSF batched variants
class VarAxisSFPackedBatched(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 17]"},
        }

class VarAxisSFPackedBatchedStochastic(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 17]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

class VarAxisSFPackedBatchedSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 17]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
        }

class VarAxisSFPackedBatchedStochasticSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 17]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

# Cuboid batched variants
class CuboidPackedBatched(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 9]"},
        }

class CuboidPackedBatchedStochastic(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 9]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

class CuboidPackedBatchedSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 9]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
        }

class CuboidPackedBatchedStochasticSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 9]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

# SQ batched variants
class SQPackedBatched(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 11]"},
        }

class SQPackedBatchedStochastic(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 11]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

class SQPackedBatchedSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 11]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
        }

class SQPackedBatchedStochasticSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 11]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }
