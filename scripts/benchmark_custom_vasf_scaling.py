# SPDX-FileCopyrightText: 2026 Aditya Ganeshan
# SPDX-License-Identifier: MIT

"""Benchmark CustomVASF fast-opt scaling and write plots.

Run from the repository's ``superfit/`` root:

    python scripts/benchmark_custom_vasf_scaling.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "superfit_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch._dynamo as dynamo

torch.set_float32_matmul_precision("high")

from superfit.custom_ops.benchmark_varaxis_sf_cuda import (
    assembly_from_primitive_sdfs,
    custom_fused_assembly,
    make_inputs,
    reference_stochastic_eval,
)


COLORS = {
    "custom": "black",
    "compiled": "#d62728",
    "uncompiled": "#f2c300",
}
LABELS = {
    "custom": "Custom CUDA",
    "compiled": "torch.compile",
    "uncompiled": "Eager PyTorch",
}
TITLES = {
    "forward_ms": "Forward",
    "backward_ms": "Backward",
    "together_ms": "Forward + Backward",
}


def make_compiled_dynamic_assembly():
    comp_func = torch.compile(
        reference_stochastic_eval,
        backend="inductor",
        mode="default",
        fullgraph=True,
        dynamic=True,
    )

    from superfit.torch_compute.compile_friendly import _sdf_smooth_union_pair

    compiled_su_func = torch.compile(
        _sdf_smooth_union_pair,
        backend="inductor",
        mode="default",
        fullgraph=True,
        dynamic=True,
    )

    def assembly(coords, all_params, inner_gumbel, outer_gumbel):
        params, su_vals, logits, temperature = all_params
        primitive_sdfs = comp_func(
            coords,
            params,
            logits,
            temperature,
            inner_gumbel,
            outer_gumbel,
        )
        output_sdf = primitive_sdfs[0]
        for i in range(1, primitive_sdfs.shape[0]):
            output_sdf = compiled_su_func(
                output_sdf,
                primitive_sdfs[i],
                su_vals[i - 1].unsqueeze(-1),
            )
        return primitive_sdfs, output_sdf

    return assembly


def make_uncompiled_assembly():
    def assembly(coords, all_params, inner_gumbel, outer_gumbel):
        params, su_vals, logits, temperature = all_params
        primitive_sdfs = reference_stochastic_eval(
            coords,
            params,
            logits,
            temperature,
            inner_gumbel,
            outer_gumbel,
        )
        return primitive_sdfs, assembly_from_primitive_sdfs(primitive_sdfs, su_vals)

    return assembly


def mark_dynamic_dim(
    tensor: torch.Tensor,
    dim: int,
    *,
    min: int | None = None,
    max: int | None = None,
) -> None:
    """Mark a dimension dynamic without rejecting valid size-1 benchmark cases."""
    if tensor.size(dim) == 1:
        maybe_mark_dynamic = getattr(dynamo, "maybe_mark_dynamic", None)
        if maybe_mark_dynamic is not None:
            maybe_mark_dynamic(tensor, dim)
        return

    kwargs = {}
    if min is not None:
        kwargs["min"] = min
    if max is not None:
        kwargs["max"] = max
    dynamo.mark_dynamic(tensor, dim, **kwargs)


def mark_fastopt_dynamic(
    coords: torch.Tensor,
    params: torch.Tensor,
    su_vals: torch.Tensor,
    logits: torch.Tensor,
    inner_gumbel: torch.Tensor,
    outer_gumbel: torch.Tensor,
) -> None:
    """Mirror the dynamic-shape assumptions used by optim/compile_function.py."""
    mark_dynamic_dim(coords, 1)
    mark_dynamic_dim(params, 0, min=2, max=128)
    mark_dynamic_dim(su_vals, 0, min=1, max=127)
    mark_dynamic_dim(logits, 0, min=2, max=128)
    mark_dynamic_dim(inner_gumbel, 0, min=2, max=128)
    mark_dynamic_dim(outer_gumbel, 0, min=2, max=128)


def clear_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def make_trainable_inputs(inputs):
    coords, params, su_vals, logits, temperature, inner_gumbel, outer_gumbel, prim_grad, sdf_grad = inputs
    train_coords = coords.detach()
    train_params = params.detach().clone().requires_grad_(True)
    train_su_vals = su_vals.detach().clone().requires_grad_(True)
    train_logits = logits.detach().clone().requires_grad_(True)
    train_inner_gumbel = inner_gumbel.detach()
    train_outer_gumbel = outer_gumbel.detach()
    mark_fastopt_dynamic(
        train_coords,
        train_params,
        train_su_vals,
        train_logits,
        train_inner_gumbel,
        train_outer_gumbel,
    )
    return (
        train_coords,
        train_params,
        train_su_vals,
        train_logits,
        temperature,
        train_inner_gumbel,
        train_outer_gumbel,
        prim_grad,
        sdf_grad,
    )


def time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / float(iters)


def time_backward_cuda(assembly, inputs, warmup: int, iters: int) -> float:
    coords, params, su_vals, logits, temperature, inner_gumbel, outer_gumbel, prim_grad, sdf_grad = (
        make_trainable_inputs(inputs)
    )
    all_params = (params, su_vals, logits, temperature)

    for _ in range(warmup):
        clear_grads(params, su_vals, logits)
        primitive_sdfs, output_sdf = assembly(coords, all_params, inner_gumbel, outer_gumbel)
        torch.autograd.backward(
            (primitive_sdfs, output_sdf),
            (prim_grad, sdf_grad.reshape_as(output_sdf)),
        )
    torch.cuda.synchronize()

    elapsed = 0.0
    for _ in range(iters):
        clear_grads(params, su_vals, logits)
        primitive_sdfs, output_sdf = assembly(coords, all_params, inner_gumbel, outer_gumbel)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.autograd.backward(
            (primitive_sdfs, output_sdf),
            (prim_grad, sdf_grad.reshape_as(output_sdf)),
        )
        end.record()
        torch.cuda.synchronize()
        elapsed += start.elapsed_time(end)
    return elapsed / float(iters)


def benchmark_method(assembly, inputs, warmup: int, iters: int) -> dict[str, float]:
    coords, params, su_vals, logits, temperature, inner_gumbel, outer_gumbel, prim_grad, sdf_grad = (
        make_trainable_inputs(inputs)
    )
    all_params = (params, su_vals, logits, temperature)

    def forward_step():
        assembly(coords, all_params, inner_gumbel, outer_gumbel)

    def together_step():
        clear_grads(params, su_vals, logits)
        primitive_sdfs, output_sdf = assembly(coords, all_params, inner_gumbel, outer_gumbel)
        torch.autograd.backward(
            (primitive_sdfs, output_sdf),
            (prim_grad, sdf_grad.reshape_as(output_sdf)),
        )

    forward_ms = time_cuda(forward_step, warmup, iters)
    backward_ms = time_backward_cuda(assembly, inputs, warmup, iters)
    together_ms = time_cuda(together_step, warmup, iters)
    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "together_ms": together_ms,
    }


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    fieldnames = ["batch", "method", "forward_ms", "backward_ms", "together_ms"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict[str, float | int | str]], metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=240)
    batches = sorted({int(row["batch"]) for row in rows})

    for method in ("custom", "compiled", "uncompiled"):
        ys = [
            float(next(row[metric] for row in rows if row["method"] == method and int(row["batch"]) == b))
            for b in batches
        ]
        ax.plot(
            batches,
            ys,
            color=COLORS[method],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=LABELS[method],
        )

    ax.set_title(f"VarAxisSF {TITLES[metric]} Timing", pad=10)
    ax.set_xlabel("Number of primitives")
    ax.set_ylabel("Time per iteration (ms)")
    ax.set_xticks(batches)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_batches(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def set_custom_threshold(threshold: int | None) -> None:
    env_name = "SUPERFIT_CUSTOM_VASF_REDUCED_BACKWARD_MIN_K"
    if threshold is None:
        os.environ.pop(env_name, None)
    else:
        os.environ[env_name] = str(threshold)


def run_threshold_sweep(args: argparse.Namespace) -> None:
    rows: list[dict[str, float | int | str]] = []
    for threshold in args.threshold_sweep:
        set_custom_threshold(threshold)
        for idx, batch in enumerate(args.batches):
            inputs = make_inputs(batch, args.points, coord_batch=1, seed=args.seed + idx)
            torch.cuda.empty_cache()
            timing = benchmark_method(custom_fused_assembly, inputs, args.warmup, args.iters)
            row = {
                "threshold": threshold,
                "batch": batch,
                "method": "custom",
                **timing,
            }
            rows.append(row)
            print(
                f"threshold={threshold},batch={batch},custom,"
                f"forward={timing['forward_ms']:.4f},"
                f"backward={timing['backward_ms']:.4f},"
                f"together={timing['together_ms']:.4f}"
            )

    csv_path = args.out_dir / "custom_vasf_threshold_sweep.csv"
    fieldnames = ["threshold", "batch", "method", "forward_ms", "backward_ms", "together_ms"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=200_000)
    parser.add_argument("--batches", type=parse_batches, default=parse_batches("2,4,8,16,32,64"))
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--out-dir", type=Path, default=Path("assets"))
    parser.add_argument(
        "--custom-threshold",
        type=int,
        default=None,
        help=(
            "Override the CustomVASF reduced params-backward cutoff. "
            "K values below this use the direct path; K values at or above it use the reduced path."
        ),
    )
    parser.add_argument(
        "--threshold-sweep",
        type=parse_batches,
        default=None,
        help=(
            "Comma-separated CustomVASF threshold candidates to benchmark. "
            "When set, only the custom path is timed and results are written to "
            "custom_vasf_threshold_sweep.csv."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run this benchmark on a GPU machine.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_custom_threshold(args.custom_threshold)
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    print(
        f"device,name={torch.cuda.get_device_name(device)},capability=sm_{major}{minor},"
        f"points={args.points},warmup={args.warmup},iters={args.iters},"
        f"custom_threshold={args.custom_threshold or 'default'}"
    )

    if args.threshold_sweep is not None:
        run_threshold_sweep(args)
        return

    assemblies = {
        "custom": custom_fused_assembly,
        "compiled": make_compiled_dynamic_assembly(),
        "uncompiled": make_uncompiled_assembly(),
    }

    rows: list[dict[str, float | int | str]] = []
    for idx, batch in enumerate(args.batches):
        inputs = make_inputs(batch, args.points, coord_batch=1, seed=args.seed + idx)
        for method, assembly in assemblies.items():
            torch.cuda.empty_cache()
            timing = benchmark_method(assembly, inputs, args.warmup, args.iters)
            row = {"batch": batch, "method": method, **timing}
            rows.append(row)
            print(
                f"{batch},{method},"
                f"forward={timing['forward_ms']:.4f},"
                f"backward={timing['backward_ms']:.4f},"
                f"together={timing['together_ms']:.4f}"
            )

    csv_path = args.out_dir / "custom_vasf_scaling.csv"
    write_csv(rows, csv_path)
    plot_metric(rows, "forward_ms", args.out_dir / "custom_vasf_forward.png")
    plot_metric(rows, "backward_ms", args.out_dir / "custom_vasf_backward.png")
    plot_metric(rows, "together_ms", args.out_dir / "custom_vasf_together.png")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
