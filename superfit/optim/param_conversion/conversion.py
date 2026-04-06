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
import torch as th
import sysl.symbolic as sls
import geolipi.symbolic as gls
import superfit.symbolic as sps
from ..primitive_registry import PrimitiveHandler
from .utils import su_param_to_var, su_var_to_param, process_var_to_param, process_param_to_var, RANGE_DICT


def transform_to_tunable(variable_list, handler: PrimitiveHandler):

    params = []
    parsed_variables = []
    for cur_var in variable_list:
        param, inverted_variable = invert_variable(cur_var, handler)
        params.append(param)
        parsed_variables.append(inverted_variable)

    return parsed_variables


def params_from_variables(variable_list, tensor_list, handler: PrimitiveHandler ):
    params = []
    for ind, inverted_variable in enumerate(variable_list):
        info = tensor_list[ind]
        cur_var = (inverted_variable, info[1], info[2], info[3])
        param = revert_variable(cur_var, handler)
        params.append(param)

    return params


## SUPPORT - Translate, Rotate, Smooth Union, StochasticPrimitive
def invert_variable(variable_info_set, handler: PrimitiveHandler):
    param, command_symbol, var_type, local_ind = variable_info_set
    if issubclass(command_symbol, gls.SmoothUnion):
        variable = su_param_to_var(param)
    elif issubclass(command_symbol, gls.Translate3D):
        pmin, pmax = RANGE_DICT[command_symbol]
        mul  = 0.5 * (pmax - pmin)
        extra = 0.5 * (pmax + pmin)
        param = th.clip(param, pmin, pmax)
        variable = th.atanh((param - extra) / mul)
        variable = th.autograd.Variable(variable, requires_grad=True)
    elif issubclass(command_symbol, (sps.StochasticPrimitive, 
                                    sls.SphericalRGBGrid3D, 
                                    gls.AxisAngleRotate3D)):
        variable = th.autograd.Variable(param, requires_grad=True)
    elif issubclass(command_symbol, (handler.packed_batched_class, handler.packed_batched_stochastic_su_class)):
        variable = handler.param_to_var(param, local_ind)
    elif issubclass(command_symbol, handler.base_class):
        sym_list = [handler.PARAM_IND_TO_NAME[local_ind],]
        v_parts  = [param,]
        var_list = process_param_to_var(sym_list, v_parts)
        variable = var_list[0]
    else:
        raise ValueError(f"Unsupported command symbol: {command_symbol}")
    return param, variable

def revert_variable(variable_info_set, handler: PrimitiveHandler):
    variable, command_symbol, var_type, local_ind = variable_info_set
    if issubclass(command_symbol, gls.SmoothUnion):
        param = su_var_to_param(variable)
    elif issubclass(command_symbol, gls.Translate3D):
        pmin, pmax = RANGE_DICT[command_symbol]
        mul  = 0.5 * (pmax - pmin)
        extra = 0.5 * (pmax + pmin)
        param = th.tanh(variable) * mul + extra
    elif issubclass(command_symbol, (sps.StochasticPrimitive, 
                                    sls.SphericalRGBGrid3D, 
                                    gls.AxisAngleRotate3D)):
        param = variable
    elif issubclass(command_symbol, (handler.packed_batched_class, handler.packed_batched_stochastic_su_class)):
        param = handler.var_to_param(variable, local_ind)
    elif issubclass(command_symbol, handler.base_class):
        sym_list = [handler.PARAM_IND_TO_NAME[local_ind],]
        v_parts  = [variable,]
        p_list = process_var_to_param(sym_list, v_parts)
        param = p_list[0]
    else:
        raise ValueError(f"Unsupported command symbol: {command_symbol}")
    return param
