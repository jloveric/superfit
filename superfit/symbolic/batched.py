import geolipi.symbolic as gls

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

class SPProtoPacked(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Vector[17]"},
        }

class SuperFrustumPacked(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Vector[14]"},
        }

class SuperGeonPacked(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Vector[17]"},
        }

class SolidSFPacked(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Vector[18]"},
        }


class VarAxisSQPacked(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Vector[14]"},
        }

class VarAxisSPPPacked(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Vector[20]"},
        }

class VarAxisSFPacked(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Vector[17]"},
        }

class VarAxisSGPacked(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Vector[20]"},
        }
    
# Batched Variants

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

# SPProto batched variants
class SPProtoPackedBatched(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 17]"},
        }

class SPProtoPackedBatchedStochastic(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 17]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

class SPProtoPackedBatchedSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 17]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
        }

class SPProtoPackedBatchedStochasticSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 17]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

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

# SuperGeon
class SGPackedBatched(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 17]"},
        }

class SGPackedBatchedStochastic(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 17]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

class SGPackedBatchedSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 17]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
        }

class SGPackedBatchedStochasticSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 17]"},
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

# VarAxis

# VarAxisSQ batched variants
class VarAxisSQPackedBatched(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 14]"},
        }

class VarAxisSQPackedBatchedStochastic(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 14]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

class VarAxisSQPackedBatchedSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 14]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
        }

class VarAxisSQPackedBatchedStochasticSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 14]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

# SP Proto
class VarAxisSPPPackedBatched(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 20]"},
        }

class VarAxisSPPPackedBatchedStochastic(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 20]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

class VarAxisSPPPackedBatchedSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 20]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
        }

class VarAxisSPPPackedBatchedStochasticSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 20]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }


# SF
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

# VarAxisSG
class VarAxisSGPackedBatched(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 20]"},
        }

class VarAxisSGPackedBatchedStochastic(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 20]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }

class VarAxisSGPackedBatchedSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 20]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
        }

class VarAxisSGPackedBatchedStochasticSU(gls.Primitive3D):
    @classmethod
    def default_specs(cls):
        return {
            "params": {"type": "Matrix[B, 20]"},
            "su_vals": {"type": "Vector[B -1, 1]"},
            "logits": {"type": "Vector[B, 2]"},
            "temperature": {"type": "float", "optional": True},
        }
