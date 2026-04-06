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
import argparse
import os
import tempfile
import time
import traceback

import trimesh

from superfit.algos.resfit import resfit
import superfit.utils.config as config_options
from superfit.utils.config import AlgorithmConfig as AlgConf, initialize_seeds


def _reset_to_fast_baseline(seed: int) -> None:
    config_options.main_setting()
    config_options.set_config_ablation(0, fastmode=True)
    # Match mesh_to_pa baseline behavior for ablation 0.
    AlgConf.PRIM_TYPE = "VarAxisSF"
    AlgConf.OPT_POST_PRUNE = True
    AlgConf.BIDIR = True
    config_options.fast_test_override()
    AlgConf.AOT_ARTIFACT_FILE = os.path.join(tempfile.gettempdir(), "superfit_config_test_aot.pt")
    AlgConf.COMPILED_FUNCTIONS = None
    initialize_seeds(seed=seed)


def _run_single_case(axis: str, value):
    _reset_to_fast_baseline(seed=42)

    # Per-axis overrides
    if axis == "PRUNE_METRIC":
        AlgConf.PRUNE_METRIC = value
    elif axis == "DECOMPOSE_MODE":
        AlgConf.DECOMPOSE_MODE = value
        AlgConf.DECOMPOSE_CONFIG = {}
    elif axis == "OPTIMIZER":
        AlgConf.OPTIMIZER = value
    elif axis == "LOWER_SP":
        AlgConf.LOWER_SP = value
    elif axis == "DO_PRUNE":
        AlgConf.DO_PRUNE = value
    elif axis == "PRIM_TYPE":
        AlgConf.PRIM_TYPE = value
    elif axis == "SMOOTHEN":
        AlgConf.SMOOTHEN = value
    elif axis == "TARGET_MODE":
        AlgConf.TARGET_MODE = value
    elif axis == "SAVE_JIT_CACHE":
        AlgConf.TorchCompile = True
        AlgConf.SAVE_JIT_CACHE = value
    elif axis == "TorchCompile":
        AlgConf.SAVE_JIT_CACHE = False
        AlgConf.TorchCompile = value
    elif axis == "USE_CURVATURE_WEIGHTS":
        AlgConf.USE_CURVATURE_WEIGHTS = value
    elif axis == "STOCHASTIC_DROPOUT":
        AlgConf.STOCHASTIC_DROPOUT = value
    elif axis == "TVERSKY_MODE":
        AlgConf.TVERSKY_MODE = value
    else:
        raise ValueError(f"Unknown axis: {axis}")

    # Keep compile cache isolated across runs.
    AlgConf.COMPILED_FUNCTIONS = None

    mesh = trimesh.primitives.Box(extents=[1.0, 1.0, 1.0])
    st = time.time()
    resfit(mesh, perform_eval=False)
    elapsed = time.time() - st
    return elapsed


def _print_summary(results):
    print("\n=== Summary ===")
    print("| Axis | Value | Status | Time (s) |")
    print("|---|---|---|---|")
    for row in results:
        print(f"| {row['axis']} | {row['value']} | {row['status']} | {row['time_s']:.2f} |")

    failures = [r for r in results if r["status"] != "PASS"]
    print(f"\nTotal: {len(results)} | Passed: {len(results) - len(failures)} | Failed: {len(failures)}")
    if failures:
        print("\n=== Failures (tracebacks) ===")
        for row in failures:
            print(f"\n--- {row['axis']}={row['value']} ---")
            print(row["error"])


def main():
    parser = argparse.ArgumentParser(description="Run fast config compatibility checks for Superfit.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop after first failing config.")
    args = parser.parse_args()

    axes_to_values = [
        ("PRUNE_METRIC", ["surface_iou", "surface_iou_wt_curvature", "cd", "surface_iou_wt_curvature_and_vox_iou"]),
        ("DECOMPOSE_MODE", ["MSD", "COACD", "VHACD"]),
        ("OPTIMIZER", ["ADAM", "ADAMW"]),
        ("LOWER_SP", [True, False]),
        ("DO_PRUNE", [True, False]),
        ("PRIM_TYPE", ["Cuboid", "SuperQuadric", "VarAxisSQ", "SuperFrustum", "VarAxisSF", "SuperGeon", "VarAxisSG", "SolidSF", "SPProto", "VarAxisSPP"]),
        ("SMOOTHEN", [True, False]),
        ("TARGET_MODE", ["bboxed", "dilated"]),
        ("SAVE_JIT_CACHE", [False, True]),
        ("TorchCompile", [False, True]),
        ("USE_CURVATURE_WEIGHTS", [True, False]),
        ("STOCHASTIC_DROPOUT", [True, False]),
        ("TVERSKY_MODE", [False, True]),
    ]

    results = []
    for axis, values in axes_to_values:
        for value in values:
            print(f"Running {axis}={value} ...")
            try:
                elapsed = _run_single_case(axis, value)
                results.append(
                    {"axis": axis, "value": value, "status": "PASS", "time_s": elapsed, "error": ""}
                )
            except Exception:
                results.append(
                    {
                        "axis": axis,
                        "value": value,
                        "status": "FAIL",
                        "time_s": 0.0,
                        "error": traceback.format_exc(),
                    }
                )
                if args.fail_fast:
                    _print_summary(results)
                    raise

    _print_summary(results)


if __name__ == "__main__":
    main()
