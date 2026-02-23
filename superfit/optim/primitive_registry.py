"""
Registry system for primitive type handlers.

This module provides a registry pattern to manage different primitive types
and their associated conversion, loss, and initialization functions.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Type, Optional
import torch as th


@dataclass
class PrimitiveHandler:
    """
    Handler for a primitive type that encapsulates all type-specific operations.
    """
    # Base packed class (e.g., SuperFrustum)
    base_class: Type
    packed_class: Type
    packed_batched_class: Type
    packed_batched_stochastic_class: Type
    packed_batched_su_class: Type
    packed_batched_stochastic_su_class: Type
    
    convert_to_batched: Callable[[th.Tensor], th.Tensor]
    
    convert_to_unbatched: Callable[[th.Tensor], th.Tensor]
    
    # Conversion functions: param <-> variable
    unpack_params: Callable[[th.Tensor], th.Tensor]
    param_to_var: Callable[[th.Tensor, int], th.Tensor]
    var_to_param: Callable[[th.Tensor, int], th.Tensor]
    # For the batched Version.
    params_from_variables: Callable[[th.Tensor], th.Tensor]
    param_from_variables_fast: Callable[[th.Tensor], th.Tensor]

    
    batched_eval_function: Callable[[th.Tensor], th.Tensor]
    
    # loss function
    get_param_loss: Callable[[th.Tensor], th.Tensor]

    point2prim_hard: Callable[[th.Tensor], th.Tensor]
    point2prim_soft: Callable[[th.Tensor], th.Tensor]

    # Only for VarAxis / SolidSF
    reinit_params: Callable[[th.Tensor, th.Tensor], th.Tensor]

    PARAM_IND_TO_NAME: Dict[int, str]

    
HANDLER_REGISTRY = {}


def register_handler(handler: PrimitiveHandler):
    HANDLER_REGISTRY[handler.base_class] = handler
    return handler

def get_handler(base_class: Type) -> Optional[PrimitiveHandler]:
    return HANDLER_REGISTRY.get(base_class)

def has_handler(base_class: Type) -> bool:
    return base_class in HANDLER_REGISTRY