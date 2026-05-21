# SPDX-FileCopyrightText: 2026 Aditya Ganeshan
# SPDX-License-Identifier: MIT

"""Optional CUDA extension wrapper for fused VarAxisSF evaluation."""

from __future__ import annotations

import importlib

import torch


_EXT = None
BUILD_COMMAND = "python -m superfit.custom_ops.build"


class CustomOpNotBuiltError(RuntimeError):
    """Raised when the optional CustomVASF CUDA extension is not built."""


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    try:
        _EXT = importlib.import_module(".varaxis_sf_cuda_ext", __package__)
    except ImportError as exc:
        raise CustomOpNotBuiltError(
            "The optional CustomVASF CUDA extension is not built. "
            f"Build it from the superfit project root with `{BUILD_COMMAND}`, "
            "or set AlgorithmConfig.USE_CUSTOM_OP = False to use the PyTorch path."
        ) from exc
    return _EXT


def _raise_with_arch_hint(exc: Exception) -> None:
    message = str(exc)
    if (
        "no kernel image is available for execution on the device" in message
        or "cudaErrorNoKernelImageForDevice" in message
    ):
        raise RuntimeError(
            "The CustomVASF CUDA extension was built without a kernel image for "
            "this GPU. Rebuild it on the target machine, or build a portable "
            f"wheel from the superfit project root with `{BUILD_COMMAND}`. "
            "Leave TORCH_CUDA_ARCH_LIST unset to use the builder's filtered "
            "broad architecture list, or set it explicitly for a narrower build."
        ) from exc
    raise exc


def _check_inputs(
    coords: torch.Tensor,
    params: torch.Tensor,
    temperature: float | torch.Tensor,
    gumbel: torch.Tensor,
    grad_mode: str,
) -> float:
    if not coords.is_cuda or not params.is_cuda or not gumbel.is_cuda:
        raise ValueError("coords, params, and gumbel must be CUDA tensors")
    if coords.dtype != torch.float32 or params.dtype != torch.float32 or gumbel.dtype != torch.float32:
        raise ValueError("coords, params, and gumbel must be float32 tensors")
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (B, M, 3), got {tuple(coords.shape)}")
    if params.ndim != 2 or params.shape[-1] != 17:
        raise ValueError(f"params must have shape (B, 17), got {tuple(params.shape)}")
    if gumbel.ndim != 2 or gumbel.shape[-1] != 3:
        raise ValueError(f"gumbel must have shape (B, 3), got {tuple(gumbel.shape)}")
    if coords.shape[0] not in {1, params.shape[0]}:
        raise ValueError("coords B dimension must be 1 or match params B")
    if gumbel.shape[0] != params.shape[0]:
        raise ValueError("gumbel B dimension must match params B")
    if not coords.is_contiguous() or not params.is_contiguous() or not gumbel.is_contiguous():
        raise ValueError("coords, params, and gumbel must be contiguous")
    if grad_mode not in {"auto", "full", "params"}:
        raise ValueError("grad_mode must be one of 'auto', 'full', or 'params'")

    if isinstance(temperature, torch.Tensor):
        if temperature.numel() != 1:
            raise ValueError("temperature tensor must contain exactly one value")
        temperature_value = float(temperature.detach().item())
    else:
        temperature_value = float(temperature)
    if temperature_value <= 0.0:
        raise ValueError("temperature must be positive")
    return temperature_value


def _check_assembly_inputs(
    coords: torch.Tensor,
    params: torch.Tensor,
    su_vals: torch.Tensor,
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
    inner_gumbel: torch.Tensor,
    outer_gumbel: torch.Tensor,
) -> float:
    temperature_value = _check_inputs(coords, params, temperature, inner_gumbel, "params")
    if not su_vals.is_cuda or not logits.is_cuda or not outer_gumbel.is_cuda:
        raise ValueError("su_vals, logits, and outer_gumbel must be CUDA tensors")
    if su_vals.dtype != torch.float32 or logits.dtype != torch.float32 or outer_gumbel.dtype != torch.float32:
        raise ValueError("su_vals, logits, and outer_gumbel must be float32 tensors")
    if su_vals.ndim != 2 or su_vals.shape != (params.shape[0] - 1, 1):
        raise ValueError(
            f"su_vals must have shape (params.shape[0] - 1, 1), got {tuple(su_vals.shape)}"
        )
    if logits.ndim != 2 or logits.shape != (params.shape[0], 2):
        raise ValueError(f"logits must have shape (params.shape[0], 2), got {tuple(logits.shape)}")
    if outer_gumbel.ndim != 2 or outer_gumbel.shape != (params.shape[0], 2):
        raise ValueError(
            f"outer_gumbel must have shape (params.shape[0], 2), got {tuple(outer_gumbel.shape)}"
        )
    if not su_vals.is_contiguous() or not logits.is_contiguous() or not outer_gumbel.is_contiguous():
        raise ValueError("su_vals, logits, and outer_gumbel must be contiguous")
    if params.shape[0] > 128:
        raise ValueError("varaxis_sf_assembly_cuda currently supports at most 128 primitives")
    return temperature_value


class _VarAxisSFCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, params, temperature, gumbel, grad_mode):
        temperature_value = _check_inputs(coords, params, temperature, gumbel, grad_mode)
        if coords.shape[0] != params.shape[0] and coords.requires_grad and grad_mode != "params":
            raise ValueError(
                "broadcast coords with coord gradients are unsupported; use grad_mode='params' "
                "or pass coords with the same B dimension as params"
            )
        try:
            out = _load_ext().forward(coords, params, temperature_value, gumbel)
        except Exception as exc:
            _raise_with_arch_hint(exc)
        ctx.save_for_backward(coords, params, gumbel)
        ctx.temperature = temperature_value
        ctx.grad_mode = grad_mode
        return out

    @staticmethod
    def backward(ctx, grad_out):
        coords, params, gumbel = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        need_coords, need_params = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        mode = ctx.grad_mode
        if mode == "auto":
            mode = "full" if need_coords else "params"
        if coords.shape[0] != params.shape[0] and need_coords and mode != "params":
            raise RuntimeError(
                "broadcast coords with coord gradients are unsupported; use grad_mode='params' "
                "or pass coords with the same B dimension as params"
            )
        if coords.shape[0] != params.shape[0] and mode == "full" and not need_coords:
            mode = "params"

        grad_coords = None
        grad_params = None
        ext = _load_ext()
        if mode == "params":
            if need_params:
                try:
                    grad_params = ext.backward_params(
                        grad_out,
                        coords,
                        params,
                        ctx.temperature,
                        gumbel,
                    )
                except Exception as exc:
                    _raise_with_arch_hint(exc)
        else:
            try:
                grad_coords_raw, grad_params_raw = ext.backward(
                    grad_out,
                    coords,
                    params,
                    ctx.temperature,
                    gumbel,
                )
            except Exception as exc:
                _raise_with_arch_hint(exc)
            if need_coords:
                grad_coords = grad_coords_raw
            if need_params:
                grad_params = grad_params_raw
        return grad_coords, grad_params, None, None, None


def varaxis_sf_cuda(
    coords: torch.Tensor,
    params: torch.Tensor,
    temperature: float | torch.Tensor,
    gumbel: torch.Tensor,
    grad_mode: str = "auto",
) -> torch.Tensor:
    """Evaluate VarAxisSF with one fused forward kernel and custom backward.

    Args:
        coords: Contiguous CUDA float32 tensor of shape ``(B, M, 3)``. ``B``
            may be ``1`` to broadcast the same coordinates across all params.
        params: Contiguous CUDA float32 tensor of shape ``(B, 17)`` using the
            current VarAxisSF packed layout.
        temperature: Positive scalar softmax temperature.
        gumbel: Contiguous CUDA float32 tensor of shape ``(B, 3)``.  Passing the
            noise in keeps tests deterministic and avoids kernel-side RNG.
        grad_mode: ``"auto"``, ``"full"``, or ``"params"``.  ``"auto"`` uses the
            params-only backward when ``coords`` does not require gradients.

    Returns:
        CUDA float32 tensor of shape ``(params.shape[0], M)``.

    Raises:
        CustomOpNotBuiltError: If the optional CUDA extension has not been built.
    """
    return _VarAxisSFCUDA.apply(coords, params, temperature, gumbel, grad_mode)


class _VarAxisSFAssemblyCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, params, su_vals, logits, temperature, inner_gumbel, outer_gumbel):
        temperature_value = _check_assembly_inputs(
            coords,
            params,
            su_vals,
            logits,
            temperature,
            inner_gumbel,
            outer_gumbel,
        )
        if coords.requires_grad:
            raise ValueError(
                "varaxis_sf_assembly_cuda is optimizer-oriented and supports params-only "
                "gradients; pass coords without requires_grad"
            )
        try:
            primitive_sdfs, out = _load_ext().assembly_forward(
                coords,
                params,
                su_vals,
                logits,
                temperature_value,
                inner_gumbel,
                outer_gumbel,
            )
        except Exception as exc:
            _raise_with_arch_hint(exc)
        ctx.save_for_backward(coords, params, su_vals, logits, inner_gumbel, outer_gumbel, primitive_sdfs)
        ctx.temperature = temperature_value
        return primitive_sdfs, out.unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_primitive, grad_out):
        coords, params, su_vals, logits, inner_gumbel, outer_gumbel, primitive_sdfs = ctx.saved_tensors
        if grad_primitive is None:
            grad_primitive = torch.zeros_like(primitive_sdfs)
        else:
            grad_primitive = grad_primitive.contiguous()
        if grad_out is None:
            grad_out = torch.zeros((coords.shape[1],), device=coords.device, dtype=coords.dtype)
        else:
            grad_out = grad_out.reshape(-1).contiguous()

        need_params = ctx.needs_input_grad[1]
        need_su = ctx.needs_input_grad[2]
        need_logits = ctx.needs_input_grad[3]
        grad_params = grad_su = grad_logits = None
        if need_params or need_su or need_logits:
            try:
                grad_params_raw, grad_su_raw, grad_logits_raw = _load_ext().assembly_backward_params(
                    grad_primitive,
                    grad_out,
                    primitive_sdfs,
                    coords,
                    params,
                    su_vals,
                    logits,
                    ctx.temperature,
                    inner_gumbel,
                    outer_gumbel,
                )
            except Exception as exc:
                _raise_with_arch_hint(exc)
            if need_params:
                grad_params = grad_params_raw
            if need_su:
                grad_su = grad_su_raw
            if need_logits:
                grad_logits = grad_logits_raw
        return None, grad_params, grad_su, grad_logits, None, None, None


def varaxis_sf_assembly_cuda(
    coords: torch.Tensor,
    params: torch.Tensor,
    su_vals: torch.Tensor,
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
    inner_gumbel: torch.Tensor,
    outer_gumbel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate optimizer-style CustomVASF primitive SDFs and smooth union.

    This is a params-only training helper for the optimization loop shape:
    broadcast coords, trainable primitive params / smooth-union values / outer
    logits, fixed caller-provided Gumbel tensors.  It returns the stochastic
    primitive SDFs ``(K, M)`` and the final smooth-union SDF ``(1, M)``.
    """
    return _VarAxisSFAssemblyCUDA.apply(
        coords,
        params,
        su_vals,
        logits,
        temperature,
        inner_gumbel,
        outer_gumbel,
    )
