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

Script to generate optimization videos from primitive assembly pkl files.
Loads the pkl file, extracts parameter sequences, and renders videos showing
the optimization process with camera movement.
"""
import os
import argparse
import _pickle as cPickle
from superfit.utils.render_seq import generate_renders
from superfit.utils.logger import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate optimization videos from primitive assembly pkl file")
    parser.add_argument("--input_path", type=str, required=True, 
                       help="Path to primitive_assembly.pkl file")
    parser.add_argument("--mode", type=str, default="all",
                       help="Mode: 'all' to render all iterations, or an integer for a specific iteration index")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save videos. Defaults to the same directory as input_path")
    parser.add_argument("--save_name", type=str, default="opt_video",
                       help="Base name for saved video files (default: opt_video)")
    parser.add_argument("--no-save-seperately", dest="save_seperately", action="store_false", default=True,
                       help="Save all iterations in a single video file (default: save separately)")
    
    args = parser.parse_args()
    
    # Validate pkl file exists
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"File not found: {args.input_path}")
    
    # Validate mode
    if args.mode != "all":
        try:
            mode_int = int(args.mode)
            if mode_int < 0:
                raise ValueError("Mode must be 'all' or a non-negative integer")
            args.mode = mode_int
        except ValueError:
            raise ValueError(f"Mode must be 'all' or an integer, got: {args.mode}")
    
    # Set save_dir to input file directory if not specified
    if args.save_dir is None:
        args.save_dir = os.path.dirname(os.path.abspath(args.input_path))
    
    # Create save_dir if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    return args


def main(args: argparse.Namespace):
    """Main function to generate optimization videos."""
    logger.info(f"Loading pkl file: {args.input_path}")
    info_dict = cPickle.load(open(args.input_path, "rb"))
    
    # Validate n_iters exists
    n_iters = info_dict.get("n_iters", 0)
    if n_iters == 0:
        raise ValueError("n_iters not found or is 0 in pkl file")
    
    logger.info(f"Number of iterations in pkl file: {n_iters}")
    
    # Validate mode if it's an integer
    if isinstance(args.mode, int):
        if args.mode >= n_iters:
            raise ValueError(f"Mode {args.mode} is out of range. File has {n_iters} iterations (0-{n_iters-1})")
        logger.info(f"Rendering iteration {args.mode}")
    else:
        logger.info(f"Rendering all {n_iters} iterations")
    
    logger.info(f"Save directory: {args.save_dir}")
    logger.info(f"Save name: {args.save_name}")
    
    # Generate renders
    generate_renders(
        info_dict, 
        args.save_dir, 
        args.save_name, 
        mode=args.mode, 
        save_seperately=args.save_seperately
    )
    
    logger.info(f"Video generation complete. Files saved to: {args.save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
