from __future__ import annotations

import sys
import importlib
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import superfit.symbolic as sps
import superfit.utils.config as config_options
from superfit.optim.entry import (
    _convert_custom_vasf_to_varaxis,
    _convert_varaxis_to_custom_vasf,
)
from superfit.optim.param_conversion import CustomVASFHandler, VarAxisSFHandler
from superfit.optim.primitive_registry import HANDLER_REGISTRY
from superfit.symbolic.symbolic_types import VALID_BATCHED_STOCHASTIC_SU_CLASSES
from superfit.symbolic.utils import inject_temp_param, remove_temp_param
from superfit.utils.config import AlgorithmConfig as AlgConf

vasf_module = importlib.import_module("superfit.custom_ops.varaxis_sf_cuda")


def _make_args():
    params = torch.arange(3 * 17, dtype=torch.float32).reshape(3, 17) / 100.0
    su_vals = torch.tensor([[0.11], [0.23]], dtype=torch.float32)
    logits = torch.tensor([[1.0, -1.0], [0.2, 0.3], [-0.4, 0.7]], dtype=torch.float32)
    return params, su_vals, logits


def _assert_args_match(lhs, rhs):
    assert len(lhs.args) == len(rhs.args)
    for i in range(len(lhs.args)):
        left = lhs.get_arg(i)
        right = rhs.get_arg(i)
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right)
        else:
            assert left == right


def test_custom_vasf_round_trip_preserves_args():
    params, su_vals, logits = _make_args()
    program = sps.VarAxisSFPackedBatchedStochasticSU(params, su_vals, logits)

    custom = _convert_varaxis_to_custom_vasf(program)
    assert isinstance(custom, sps.CustomVASF)
    _assert_args_match(custom, program)

    restored = _convert_custom_vasf_to_varaxis(custom)
    assert isinstance(restored, sps.VarAxisSFPackedBatchedStochasticSU)
    _assert_args_match(restored, program)


def test_custom_vasf_temp_injection_and_removal():
    params, su_vals, logits = _make_args()
    program = sps.CustomVASF(params, su_vals, logits)

    with_temp = inject_temp_param(program, 0.5)
    assert isinstance(with_temp, sps.CustomVASF)
    assert len(with_temp.args) == 4
    assert with_temp.get_arg(3) == 0.5

    stripped = remove_temp_param(with_temp)
    assert isinstance(stripped, sps.CustomVASF)
    assert len(stripped.args) == 3
    _assert_args_match(stripped, program)


def test_custom_vasf_handler_is_optim_only():
    assert sps.CustomVASF in VALID_BATCHED_STOCHASTIC_SU_CLASSES
    assert CustomVASFHandler.packed_batched_stochastic_su_class is sps.CustomVASF
    assert CustomVASFHandler.param_from_variables_fast is VarAxisSFHandler.param_from_variables_fast
    assert HANDLER_REGISTRY[sps.VarAxisSF] is VarAxisSFHandler


def test_custom_ops_import_without_built_extension_when_disabled():
    fields = {"USE_CUSTOM_OP": AlgConf.USE_CUSTOM_OP}
    try:
        AlgConf.USE_CUSTOM_OP = False
        assert callable(vasf_module.varaxis_sf_cuda)
    finally:
        for name, value in fields.items():
            setattr(AlgConf, name, value)


def test_missing_custom_op_extension_has_build_instructions(monkeypatch):
    old_ext = vasf_module._EXT

    def fake_import_module(name, package=None):
        if name == ".varaxis_sf_cuda_ext" and package == "superfit.custom_ops":
            raise ImportError("fake missing extension")
        return original_import_module(name, package)

    original_import_module = vasf_module.importlib.import_module
    monkeypatch.setattr(vasf_module.importlib, "import_module", fake_import_module)
    vasf_module._EXT = None
    try:
        with pytest.raises(
            vasf_module.CustomOpNotBuiltError,
            match="python -m superfit.custom_ops.build",
        ):
            vasf_module._load_ext()
    finally:
        vasf_module._EXT = old_ext


def test_ablation_8_enables_custom_varaxis_op():
    fields = {
        "PRIM_TYPE": AlgConf.PRIM_TYPE,
        "USE_CUSTOM_OP": AlgConf.USE_CUSTOM_OP,
        "TORCH_COMPILE": AlgConf.TORCH_COMPILE,
    }
    try:
        AlgConf.USE_CUSTOM_OP = False
        AlgConf.TORCH_COMPILE = True
        config_options.set_config_ablation(8, fastmode=True)
        assert AlgConf.PRIM_TYPE == "VarAxisSF"
        assert AlgConf.USE_CUSTOM_OP is True
        assert AlgConf.TORCH_COMPILE is False
    finally:
        for name, value in fields.items():
            setattr(AlgConf, name, value)
