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

Batch script to generate optimization videos from shape folders under input_path.
"""
import os
import argparse
import _pickle as cPickle
from superfit.utils.render_seq import generate_renders
from superfit.utils.logger import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate optimization videos for folders under input_path."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Directory containing one subdir per folder_name, each with primitive_assembly.pkl (e.g. path/to/toys4k/ablation_0_param)",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Base name for saved video files (default: opt_video)",
    )
    parser.add_argument("--ablation", type=int, default=0, help="Ablation number for output naming.")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help="Mode: 'all' to render all iterations, or an integer for a specific iteration index",
    )
    parser.add_argument(
        "--no-save-separately",
        dest="save_separately",
        action="store_false",
        default=True,
        help="Save all iterations in a single video (default: save each iteration separately)",
    )
    parser.add_argument(
        "--folders",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit folder names. If omitted, all subdirectories under input_path are processed.",
    )
    args = parser.parse_args()

    # Validate mode
    if args.mode != "all":
        try:
            mode_int = int(args.mode)
            if mode_int < 0:
                raise ValueError("Mode must be 'all' or a non-negative integer")
            args.mode = mode_int
        except ValueError:
            raise ValueError(f"Mode must be 'all' or an integer, got: {args.mode}")

    if args.save_name is None:
        args.save_name = f"opt_video_ablation_{args.ablation}"

    return args


def main(args: argparse.Namespace):
    """Generate optimization videos for each selected folder under input_path."""
    if args.folders is None:
        folder_names = sorted(
            d for d in os.listdir(args.input_path)
            if os.path.isdir(os.path.join(args.input_path, d)) and not d.startswith(".")
        )
    else:
        folder_names = args.folders

    if not folder_names:
        logger.warning("No folders selected for video generation.")
        return

    for folder_name in folder_names:
        shape_dir = os.path.join(args.input_path, folder_name)
        input_path = os.path.join(shape_dir, "primitive_assembly.pkl")

        if not os.path.exists(input_path):
            logger.warning(f"Skipping {folder_name}: pkl not found at {input_path}")
            continue

        logger.info(f"Generating videos for: {folder_name}")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Save dir: {shape_dir}")

        try:
            info_dict = cPickle.load(open(input_path, "rb"))
        except Exception as e:
            logger.error(f"Failed to load {input_path}: {e}")
            continue

        n_iters = info_dict.get("n_iters", 0)
        if n_iters == 0:
            logger.warning(f"Skipping {folder_name}: n_iters is 0 or missing in pkl")
            continue

        if isinstance(args.mode, int) and args.mode >= n_iters:
            logger.warning(
                f"Skipping {folder_name}: mode {args.mode} out of range (0-{n_iters - 1})"
            )
            continue

        os.makedirs(shape_dir, exist_ok=True)

        generate_renders(
            info_dict,
            shape_dir,
            args.save_name,
            mode=args.mode,
            save_seperately=args.save_separately,
        )
        logger.info(f"  Done. Videos saved to {shape_dir}")

    logger.info("Batch video generation complete.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
