<!--
SPDX-FileCopyrightText: 2026 Aditya Ganeshan
SPDX-License-Identifier: MIT
-->

# SuperFit Custom Ops

This directory contains optional CUDA kernels used by the `CustomVASF` optimization path. The current kernels fuse the forward pass and custom backward pass for packed batched `VarAxisSF` evaluation, plus an optimizer-style assembly helper that fuses the outer stochastic keep/drop blend and smooth-union assembly.

This custom kernel was built with GPT-5.5 assistance.

## Requirements

- CUDA-capable PyTorch
- CUDA toolkit compatible with the installed PyTorch build
- `ninja`
- A working C++ compiler

The rest of `superfit` does not require this extension. `AlgorithmConfig.USE_CUSTOM_OP = False` works without building it.

## Build

From the repository's `superfit/` project root:

```bash
python -m superfit.custom_ops.build
```

This builds `superfit.custom_ops.varaxis_sf_cuda_ext` in place next to the Python wrapper.

When `TORCH_CUDA_ARCH_LIST` is unset, the builder uses a broad architecture list filtered to what the local `nvcc` supports. With CUDA 12.9 this includes Pascal/Volta/Turing/Ampere/Ada/Hopper/Blackwell targets: `6.0;6.1;7.0;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;12.0+PTX`.

Useful target mappings:

- A40 and RTX 3090: `8.6`
- B200: `10.0`
- V100: `7.0`
- T4 / RTX 20xx: `7.5`

Set `TORCH_CUDA_ARCH_LIST` before running the build command if you want a narrower build. If you see `cudaErrorNoKernelImageForDevice`, remove the stale `.so` and rebuild with an arch list that includes your GPU.

## Test

From the same `superfit/` project root:

```bash
python -m pytest tests/custom_ops/test_custom_vasf_optim_integration.py -q
python -m pytest tests/custom_ops/test_varaxis_sf_cuda.py -q
```

The CUDA parity tests skip cleanly when CUDA is unavailable.

## Benchmark

Run the native CUDA op against a deterministic version of the optimizer baseline used in `superfit.optim.compile_function.compile_cached_with_dummy_opt`: compiled primitive eval plus compiled smooth-union assembly.

```bash
python -m superfit.custom_ops.benchmark_varaxis_sf_cuda --case 3x200000 --case 8x200000 --case 16x200000 --case 30x200000
```

The benchmark reports the fast-opt forward path and optimizer-style fused assembly params forward/backward timings as CSV rows. It also logs parity diagnostics and peak allocated CUDA memory for each path. Add `--include-primitive` for primitive-only diagnostic timings.

Add `--diagnostics` to split the custom assembly backward into prep, per-tile VJP, and reduction timings. That is useful when checking threshold behavior at small primitive counts.

## Runtime Behavior

The custom optimization path is opt-in. Set `AlgorithmConfig.USE_CUSTOM_OP = True` or use ablation `8` to enable it for `VarAxisSF`.

If `USE_CUSTOM_OP` is enabled but the extension has not been built, SuperFit raises a `RuntimeError` with the build command above. Disable `USE_CUSTOM_OP` to use the default PyTorch implementation.

The params-only custom backward paths use the direct kernel below `K=32` and the reduced partials path at `K>=32` by default. Set `SUPERFIT_CUSTOM_VASF_REDUCED_BACKWARD_MIN_K=16` before running to try an alternate cutoff on a specific machine.
