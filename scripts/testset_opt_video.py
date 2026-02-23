"""
Batch script to generate optimization videos for a list of folder names.

Edit FOLDER_NAMES at the top of this script to the list of folder_name values
you want videos for. Folder names are the same as in generate_on_testset.py:
- toys4k: e.g. "truck_028", "apple_003" (basename of mesh directory)
- partobjaverse: e.g. filename without extension

Each shape's videos are saved in that shape's directory (input_dir/folder_name)
alongside primitive_assembly.pkl. All iterations are saved as separate video files.
"""
import os
import argparse
import _pickle as cPickle
from superfit.utils.render_seq import generate_renders
from superfit.utils.logger import logger


# -----------------------------------------------------------------------------
# Edit this list: folder_name for each shape you want optimization videos for.
# Same convention as generate_on_testset.py (e.g. "truck_028", "apple_003").
# -----------------------------------------------------------------------------
FOLDER_NAMES = [
    "truck_028",
    # "apple_003",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate optimization videos for a list of folder names (see FOLDER_NAMES at top of script)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing one subdir per folder_name, each with primitive_assembly.pkl (e.g. path/to/toys4k/ablation_0_param)",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="opt_video",
        help="Base name for saved video files (default: opt_video)",
    )
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

    return args


def main(args: argparse.Namespace):
    """Generate optimization videos for each folder name in FOLDER_NAMES."""
    if not FOLDER_NAMES:
        logger.warning("FOLDER_NAMES is empty. Edit the list at the top of this script.")
        return

    for folder_name in FOLDER_NAMES:
        shape_dir = os.path.join(args.input_dir, folder_name)
        input_file = os.path.join(shape_dir, "primitive_assembly.pkl")

        if not os.path.exists(input_file):
            logger.warning(f"Skipping {folder_name}: pkl not found at {input_file}")
            continue

        logger.info(f"Generating videos for: {folder_name}")
        logger.info(f"  Input: {input_file}")
        logger.info(f"  Save dir: {shape_dir}")

        try:
            info_dict = cPickle.load(open(input_file, "rb"))
        except Exception as e:
            logger.error(f"Failed to load {input_file}: {e}")
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
