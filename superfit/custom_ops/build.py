# SPDX-FileCopyrightText: 2026 Aditya Ganeshan
# SPDX-License-Identifier: MIT

"""Build the optional SuperFit CustomVASF CUDA extension in place."""

from __future__ import annotations

import sys
import os
import re
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


DESIRED_CUDA_ARCHES = (
    "6.0",   # Pascal P100
    "6.1",   # Pascal GTX 10xx / P40
    "7.0",   # Volta V100
    "7.5",   # Turing T4 / RTX 20xx
    "8.0",   # Ampere A100
    "8.6",   # Ampere A40 / RTX 3090
    "8.7",   # Ampere embedded/workstation variants
    "8.9",   # Ada L4 / L40 / RTX 40xx
    "9.0",   # Hopper H100
    "10.0",  # Blackwell B100/B200
    "10.3",  # Blackwell family variants
    "12.0",  # Blackwell RTX/pro workstation variants
)
FALLBACK_CUDA_ARCH_LIST = "7.0;7.5;8.0;8.6;8.9;9.0+PTX"


def _arch_to_sm(arch: str) -> str:
    major, minor = arch.split(".")
    return f"sm_{major}{minor}"


def _nvcc_supported_sms() -> set[str]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return set()
    try:
        result = subprocess.run(
            [nvcc, "--list-gpu-code"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return set(re.findall(r"\bsm_\d+\b", result.stdout))


def default_cuda_arch_list() -> str:
    supported_sms = _nvcc_supported_sms()
    if not supported_sms:
        return FALLBACK_CUDA_ARCH_LIST

    arches = [arch for arch in DESIRED_CUDA_ARCHES if _arch_to_sm(arch) in supported_sms]
    if not arches:
        return FALLBACK_CUDA_ARCH_LIST

    arches[-1] = f"{arches[-1]}+PTX"
    return ";".join(arches)


def main() -> None:
    custom_ops_dir = Path(__file__).resolve().parent
    package_root = custom_ops_dir.parent
    sources = [
        custom_ops_dir / "csrc" / "varaxis_sf.cpp",
        custom_ops_dir / "csrc" / "varaxis_sf_kernel.cu",
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CustomVASF CUDA source files: {missing}")

    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        arch_list = default_cuda_arch_list()
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
        print(
            f"Using default TORCH_CUDA_ARCH_LIST={arch_list}. "
            "Set TORCH_CUDA_ARCH_LIST explicitly for a narrower or custom build."
        )

    script_args = sys.argv[1:] or ["build_ext", "--inplace"]
    setup(
        name="superfit-custom-ops",
        packages=["superfit", "superfit.custom_ops"],
        package_dir={"superfit": str(package_root)},
        ext_modules=[
            CUDAExtension(
                name="superfit.custom_ops.varaxis_sf_cuda_ext",
                sources=[str(path) for path in sources],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3"],
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
        script_args=script_args,
    )


if __name__ == "__main__":
    main()
