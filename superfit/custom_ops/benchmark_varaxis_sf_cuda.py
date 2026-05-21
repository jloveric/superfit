# SPDX-FileCopyrightText: 2026 Aditya Ganeshan
# SPDX-License-Identifier: MIT

"""Benchmark CustomVASF against the optimizer's torch.compile baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch

from superfit.custom_ops.varaxis_sf_cuda import _load_ext, varaxis_sf_assembly_cuda, varaxis_sf_cuda
from superfit.torch_compute.batched_sf import (
    batched_sf_packed_eval_part_2,
    common_transform_coords,
    unpacked_params_varsf,
)
from superfit.torch_compute.compile_friendly import _sdf_smooth_union_pair


@dataclass(frozen=True)
class Case:
    batch: int
    points: int


@dataclass(frozen=True)
class Timing:
    ms: float
    peak_mb: float


def make_inputs(batch: int, points: int, coord_batch: int, seed: int):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(seed)

    coords = torch.empty(coord_batch, points, 3, device=device).uniform_(
        -0.8, 0.8, generator=gen
    )
    translate = torch.empty(batch, 3, device=device).uniform_(-0.2, 0.2, generator=gen)
    size = torch.empty(batch, 3, device=device).uniform_(0.35, 1.1, generator=gen)
    size[:, 1] = size[:, 1] + 0.071
    size[:, 2] = size[:, 2] + 0.113
    roundness = torch.empty(batch, 1, device=device).uniform_(0.05, 0.45, generator=gen)
    dilate = torch.empty(batch, 1, device=device).uniform_(-0.04, 0.04, generator=gen)
    scale = torch.empty(batch, 1, device=device).uniform_(0.45, 1.45, generator=gen)

    bulge_mag = torch.empty(batch, 1, device=device).uniform_(0.12, 0.75, generator=gen)
    bulge_sign = torch.where(
        torch.arange(batch, device=device).view(batch, 1) % 2 == 0,
        torch.ones(batch, 1, device=device),
        -torch.ones(batch, 1, device=device),
    )
    bulge = bulge_mag * bulge_sign
    onion = torch.empty(batch, 1, device=device).uniform_(0.08, 0.75, generator=gen)

    base_logits = torch.tensor(
        [[3.0, -1.0, -2.0], [-1.0, 3.0, -2.0], [-2.0, -1.0, 3.0]],
        device=device,
    )
    axis_logits = base_logits[torch.arange(batch, device=device) % 3].clone()
    axis_logits = axis_logits + torch.empty(batch, 3, device=device).uniform_(
        -0.25, 0.25, generator=gen
    )
    rotate = torch.empty(batch, 3, device=device).uniform_(-0.75, 0.75, generator=gen)
    rotate = rotate + torch.tensor([0.17, -0.11, 0.23], device=device)

    params = torch.cat(
        [translate, size, roundness, dilate, scale, bulge, onion, axis_logits, rotate],
        dim=-1,
    ).contiguous()
    su_vals = torch.empty(batch - 1, 1, device=device).uniform_(0.01, 0.08, generator=gen)
    outer_logits = torch.empty(batch, 2, device=device).normal_(0.0, 1.0, generator=gen)
    inner_gumbel = torch.empty(batch, 3, device=device).uniform_(-0.6, 0.6, generator=gen)
    outer_gumbel = torch.empty(batch, 2, device=device).uniform_(-0.6, 0.6, generator=gen)
    primitive_grad = torch.empty(batch, points, device=device).uniform_(-0.7, 0.7, generator=gen)
    assembly_grad = torch.empty(points, device=device).uniform_(-0.7, 0.7, generator=gen)
    return (
        coords.contiguous(),
        params.contiguous(),
        su_vals.contiguous(),
        outer_logits.contiguous(),
        0.73,
        inner_gumbel.contiguous(),
        outer_gumbel.contiguous(),
        primitive_grad.contiguous(),
        assembly_grad.contiguous(),
    )


def reference_varaxis_sf(coords, params, temperature, gumbel):
    translate, size, roundness, dilate_3d, scale, bulge, onion, logits, rotate = (
        unpacked_params_varsf(params)
    )
    transformed = common_transform_coords(coords, translate, rotate)
    sdf_y = batched_sf_packed_eval_part_2(
        transformed, size, roundness, dilate_3d, scale, bulge, onion
    )
    sdf_z = batched_sf_packed_eval_part_2(
        transformed[:, :, [1, 2, 0]],
        size[:, [1, 2, 0]],
        roundness,
        dilate_3d,
        scale,
        bulge,
        onion,
    )
    sdf_x = batched_sf_packed_eval_part_2(
        transformed[:, :, [2, 0, 1]],
        size[:, [2, 0, 1]],
        roundness,
        dilate_3d,
        scale,
        bulge,
        onion,
    )
    w = torch.softmax((logits + gumbel) / float(temperature), dim=-1)
    return w[:, 0:1] * sdf_y + w[:, 1:2] * sdf_z + w[:, 2:3] * sdf_x


def reference_stochastic_eval(coords, params, logits, temperature, inner_gumbel, outer_gumbel):
    outputs = reference_varaxis_sf(coords, params, temperature, inner_gumbel)
    w = torch.softmax((logits + outer_gumbel) / float(temperature), dim=-1)
    return outputs * w[:, 0:1] + w[:, 1:2]


def custom_stochastic_eval(coords, params, logits, temperature, inner_gumbel, outer_gumbel):
    outputs = varaxis_sf_cuda(coords, params, temperature, inner_gumbel, grad_mode="auto")
    w = torch.softmax((logits + outer_gumbel) / float(temperature), dim=-1)
    return outputs * w[:, 0:1] + w[:, 1:2]


def assembly_from_primitive_sdfs(primitive_sdfs, su_vals):
    out = primitive_sdfs[0]
    for i in range(1, primitive_sdfs.shape[0]):
        out = _sdf_smooth_union_pair(out, primitive_sdfs[i], su_vals[i - 1].unsqueeze(-1))
    return out


def make_compiled_assembly(eval_func):
    comp_func = torch.compile(
        eval_func,
        backend="inductor",
        mode="default",
        fullgraph=True,
        dynamic=True,
    )
    compiled_su_func = torch.compile(
        _sdf_smooth_union_pair,
        backend="inductor",
        mode="default",
        fullgraph=True,
        dynamic=True,
    )

    def compiled_assembly(coords, all_params, inner_gumbel, outer_gumbel):
        params, su_vals, logits, temperature = all_params
        output = comp_func(coords, params, logits, temperature, inner_gumbel, outer_gumbel)
        out = output[0]
        for i in range(1, output.shape[0]):
            out = compiled_su_func(out, output[i], su_vals[i - 1].unsqueeze(-1))
        return output, out

    return compiled_assembly


def custom_fused_assembly(coords, all_params, inner_gumbel, outer_gumbel):
    params, su_vals, logits, temperature = all_params
    return varaxis_sf_assembly_cuda(
        coords,
        params,
        su_vals,
        logits,
        temperature,
        inner_gumbel,
        outer_gumbel,
    )


def clear_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def time_cuda(fn: Callable[[], None], warmup: int, iters: int) -> Timing:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return Timing(start.elapsed_time(end) / float(iters), peak_mb)


def diff_stats(name: str, a: torch.Tensor, b: torch.Tensor) -> str:
    diff = (a - b).detach().abs()
    return (
        f"{name}:max={diff.max().item():.6g},mean={diff.mean().item():.6g},"
        f"gt1e-4={int((diff > 1e-4).sum().item())},"
        f"gt1e-3={int((diff > 1e-3).sum().item())}"
    )


def parity_report(case: Case, inputs) -> None:
    coords, params, su_vals, logits, temperature, inner_gumbel, outer_gumbel, prim_grad, sdf_grad = inputs
    params_ref = params.detach().clone().requires_grad_(True)
    su_ref = su_vals.detach().clone().requires_grad_(True)
    logits_ref = logits.detach().clone().requires_grad_(True)
    ref_prim = reference_stochastic_eval(
        coords.detach(), params_ref, logits_ref, temperature, inner_gumbel, outer_gumbel
    )
    ref_sdf = assembly_from_primitive_sdfs(ref_prim, su_ref)
    (ref_prim * prim_grad).sum().add((ref_sdf * sdf_grad).sum()).backward()

    params_cuda = params.detach().clone().requires_grad_(True)
    su_cuda = su_vals.detach().clone().requires_grad_(True)
    logits_cuda = logits.detach().clone().requires_grad_(True)
    custom_prim, custom_sdf = varaxis_sf_assembly_cuda(
        coords.detach(),
        params_cuda,
        su_cuda,
        logits_cuda,
        temperature,
        inner_gumbel,
        outer_gumbel,
    )
    (custom_prim * prim_grad).sum().add((custom_sdf * sdf_grad).sum()).backward()

    print(
        f"parity,B={case.batch},M={case.points},"
        + ",".join(
            [
                diff_stats("prim", custom_prim, ref_prim),
                diff_stats("sdf", custom_sdf, ref_sdf),
                diff_stats("grad_params", params_cuda.grad, params_ref.grad),
                diff_stats("grad_su", su_cuda.grad, su_ref.grad),
                diff_stats("grad_logits", logits_cuda.grad, logits_ref.grad),
            ]
        )
    )


def bench_forward(eval_ref, eval_custom, inputs, warmup: int, iters: int):
    coords, params, _, logits, temperature, inner_gumbel, outer_gumbel, _, _ = inputs
    with torch.no_grad():
        ref_timing = time_cuda(
            lambda: eval_ref(coords, params, logits, temperature, inner_gumbel, outer_gumbel),
            warmup,
            iters,
        )
        custom_timing = time_cuda(
            lambda: eval_custom(coords, params, logits, temperature, inner_gumbel, outer_gumbel),
            warmup,
            iters,
        )
    return ref_timing, custom_timing


def bench_assembly_forward(assembly_ref, assembly_custom, inputs, warmup: int, iters: int):
    coords, params, su_vals, logits, temperature, inner_gumbel, outer_gumbel, _, _ = inputs
    all_params = (params, su_vals, logits, temperature)
    with torch.no_grad():
        ref_timing = time_cuda(
            lambda: assembly_ref(coords, all_params, inner_gumbel, outer_gumbel),
            warmup,
            iters,
        )
        custom_timing = time_cuda(
            lambda: assembly_custom(coords, all_params, inner_gumbel, outer_gumbel),
            warmup,
            iters,
        )
    return ref_timing, custom_timing


def bench_assembly_backward(assembly_ref, assembly_custom, inputs, warmup: int, iters: int):
    coords, params, su_vals, logits, temperature, inner_gumbel, outer_gumbel, prim_grad, sdf_grad = inputs
    coords = coords.detach()
    params_ref = params.detach().clone().requires_grad_(True)
    su_ref = su_vals.detach().clone().requires_grad_(True)
    logits_ref = logits.detach().clone().requires_grad_(True)
    params_cuda = params.detach().clone().requires_grad_(True)
    su_cuda = su_vals.detach().clone().requires_grad_(True)
    logits_cuda = logits.detach().clone().requires_grad_(True)

    def ref_step():
        clear_grads(params_ref, su_ref, logits_ref)
        prim, sdf = assembly_ref(
            coords,
            (params_ref, su_ref, logits_ref, temperature),
            inner_gumbel,
            outer_gumbel,
        )
        (prim * prim_grad).sum().add((sdf * sdf_grad).sum()).backward()

    def custom_step():
        clear_grads(params_cuda, su_cuda, logits_cuda)
        prim, sdf = assembly_custom(
            coords,
            (params_cuda, su_cuda, logits_cuda, temperature),
            inner_gumbel,
            outer_gumbel,
        )
        (prim * prim_grad).sum().add((sdf * sdf_grad).sum()).backward()

    return time_cuda(ref_step, warmup, iters), time_cuda(custom_step, warmup, iters)


def bench_custom_backward_components(inputs, warmup: int, iters: int) -> dict[str, Timing]:
    (
        coords,
        params,
        su_vals,
        logits,
        temperature,
        inner_gumbel,
        outer_gumbel,
        prim_grad,
        sdf_grad,
    ) = inputs
    ext = _load_ext()
    coords = coords.detach()
    params = params.detach()
    su_vals = su_vals.detach()
    logits = logits.detach()

    with torch.no_grad():
        primitive_sdfs, _ = varaxis_sf_assembly_cuda(
            coords,
            params,
            su_vals,
            logits,
            temperature,
            inner_gumbel,
            outer_gumbel,
        )
        grad_raw, _, _ = ext.assembly_backward_prep(
            prim_grad,
            sdf_grad,
            primitive_sdfs,
            params,
            su_vals,
            logits,
            temperature,
            outer_gumbel,
        )
        partials = ext.backward_params_partials(
            grad_raw,
            coords,
            params,
            temperature,
            inner_gumbel,
        )

    return {
        "custom_assembly_backward_prep": time_cuda(
            lambda: ext.assembly_backward_prep(
                prim_grad,
                sdf_grad,
                primitive_sdfs,
                params,
                su_vals,
                logits,
                temperature,
                outer_gumbel,
            ),
            warmup,
            iters,
        ),
        "custom_param_partials_vjp": time_cuda(
            lambda: ext.backward_params_partials(
                grad_raw,
                coords,
                params,
                temperature,
                inner_gumbel,
            ),
            warmup,
            iters,
        ),
        "custom_param_reduction": time_cuda(
            lambda: ext.reduce_param_partials(partials),
            warmup,
            iters,
        ),
    }


def print_timing(case: Case, mode: str, ref: Timing, custom: Timing) -> None:
    print(
        f"{case.batch},{case.points},{mode},"
        f"{ref.ms:.4f},{custom.ms:.4f},{ref.ms / custom.ms:.3f},"
        f"{ref.peak_mb:.1f},{custom.peak_mb:.1f}"
    )


def print_custom_component_timing(case: Case, mode: str, custom: Timing) -> None:
    print(
        f"diagnostic,{case.batch},{case.points},{mode},"
        f"{custom.ms:.4f},{custom.peak_mb:.1f}"
    )


def parse_case(text: str) -> Case:
    batch_text, points_text = text.lower().split("x", maxsplit=1)
    return Case(batch=int(batch_text), points=int(points_text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        default=None,
        help="Benchmark case as BxM, e.g. 8x8192. May be repeated.",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--no-parity", action="store_true")
    parser.add_argument("--include-primitive", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run this benchmark on a GPU node.")

    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    print(
        f"device,name={torch.cuda.get_device_name(device)},"
        f"capability=sm_{major}{minor},torch={torch.__version__},cuda={torch.version.cuda}"
    )

    cases = args.case or [
        Case(3, 200_000),
        Case(8, 200_000),
        Case(16, 200_000),
        Case(30, 200_000),
    ]

    compiled_ref_eval = torch.compile(
        reference_stochastic_eval,
        backend="inductor",
        mode="default",
        fullgraph=True,
        dynamic=True,
    )
    custom_eval = custom_stochastic_eval
    compiled_ref_assembly = make_compiled_assembly(reference_stochastic_eval)
    custom_assembly = custom_fused_assembly

    print(
        "B,M,mode,torch_compile_ms,custom_cuda_ms,speedup,"
        "torch_peak_mb,custom_peak_mb"
    )
    if args.diagnostics:
        print("diagnostic,B,M,component,custom_cuda_ms,custom_peak_mb")
    for idx, case in enumerate(cases):
        inputs = make_inputs(case.batch, case.points, coord_batch=1, seed=1000 + idx)
        if not args.no_parity:
            parity_report(case, inputs)

        if args.include_primitive:
            ref_timing, custom_timing = bench_forward(
                compiled_ref_eval,
                custom_eval,
                inputs,
                args.warmup,
                args.iters,
            )
            print_timing(case, "primitive_forward", ref_timing, custom_timing)

        ref_timing, custom_timing = bench_assembly_forward(
            compiled_ref_assembly,
            custom_assembly,
            inputs,
            args.warmup,
            args.iters,
        )
        print_timing(case, "fastopt_forward", ref_timing, custom_timing)

        ref_timing, custom_timing = bench_assembly_backward(
            compiled_ref_assembly,
            custom_assembly,
            inputs,
            args.warmup,
            args.iters,
        )
        print_timing(case, "fastopt_params_fwd_bwd", ref_timing, custom_timing)

        if args.diagnostics:
            timings = bench_custom_backward_components(inputs, args.warmup, args.iters)
            for mode, timing in timings.items():
                print_custom_component_timing(case, mode, timing)


if __name__ == "__main__":
    main()
