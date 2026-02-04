#!/usr/bin/env python
"""
CLI-oriented Cellpose inference script for batch processing
Processes 32-bit TIFF images and generates segmentation masks
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import logging
from datetime import datetime
from tifffile import imread, imwrite

from cellpose import models, io, plot


def setup_logging(output_dir, verbose=False):
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"inference_{timestamp}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run Cellpose inference on 32-bit TIFF images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file or directory')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save output masks and visualizations')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File pattern for batch processing (e.g., *.tif, *.tiff)')
    
    # Model selection
    parser.add_argument('--model', type=str, default='cyto3',
                        help='Model to use: cyto3, cyto2, nuclei, or path to custom model')
    parser.add_argument('--model_type', type=str, default=None,
                        help='Alias for --model (for compatibility)')
    
    # Segmentation parameters
    parser.add_argument('--diameter', type=float, default=0,
                        help='Cell diameter in pixels (0 = auto-estimate)')
    parser.add_argument('--flow_threshold', type=float, default=0.4,
                        help='Flow error threshold (0-1, lower = more cells)')
    parser.add_argument('--cellprob_threshold', type=float, default=0.0,
                        help='Cell probability threshold (-6 to 6, higher = fewer cells)')
    parser.add_argument('--min_size', type=int, default=15,
                        help='Minimum cell size in pixels')
    
    # Channels
    parser.add_argument('--channels', type=int, nargs=2, default=[0, 0],
                        help='Channels: [cytoplasm, nucleus] (0=gray/none, 1=red, 2=green, 3=blue)')
    parser.add_argument('--channel_axis', type=int, default=None,
                        help='Axis of channels in image')
    
    # 3D options
    parser.add_argument('--do_3D', action='store_true',
                        help='Process as 3D volume')
    parser.add_argument('--stitch_threshold', type=float, default=0.0,
                        help='Stitch 2D masks into 3D (0 = no stitching)')
    parser.add_argument('--anisotropy', type=float, default=None,
                        help='Z-axis anisotropy for 3D (e.g., 2.0 if Z half as dense)')
    
    # Processing options
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for inference')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of tiles processed simultaneously')
    parser.add_argument('--no_resample', action='store_true',
                        help='Skip resampling (faster but less accurate)')
    parser.add_argument('--no_interp', action='store_true',
                        help='Disable interpolation in dynamics')
    
    # Normalization
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize images')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false',
                        help='Skip normalization')
    parser.add_argument('--invert', action='store_true',
                        help='Invert image intensities')
    
    # Output options
    parser.add_argument('--save_txt', action='store_true',
                        help='Save masks as text outlines')
    parser.add_argument('--save_png', action='store_true',
                        help='Save overlay visualization as PNG')
    parser.add_argument('--save_flows', action='store_true',
                        help='Save flow fields')
    parser.add_argument('--save_npy', action='store_true',
                        help='Save results as NPY file')
    parser.add_argument('--no_mask_tif', action='store_true',
                        help='Do not save mask as TIFF')
    
    # Image restoration (Cellpose 3.0)
    parser.add_argument('--restore', type=str, default=None,
                        help='Image restoration model (denoise, deblur, upsample)')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def get_image_files(input_path, pattern='*.tif'):
    """Get list of image files to process"""
    input_path = Path(input_path)
    
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob(pattern))
        return files
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def process_image(image_path, model, args, logger):
    """Process a single image"""
    logger.info(f"Processing: {image_path.name}")
    
    try:
        # Read image (handles 32-bit TIFF)
        img = imread(str(image_path))
        logger.info(f"  Image shape: {img.shape}, dtype: {img.dtype}")
        
        # Run segmentation
        masks, flows, styles = model.eval(
            img,
            diameter=args.diameter if args.diameter > 0 else None,
            channels=args.channels,
            channel_axis=args.channel_axis,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            min_size=args.min_size,
            do_3D=args.do_3D,
            stitch_threshold=args.stitch_threshold,
            anisotropy=args.anisotropy,
            batch_size=args.batch_size,
            resample=not args.no_resample,
            interp=not args.no_interp,
            normalize=args.normalize,
            invert=args.invert,
        )
        
        # Count cells
        n_cells = len(np.unique(masks)) - 1  # Subtract background
        logger.info(f"  Found {n_cells} cells")
        
        # Prepare output filename
        base_name = image_path.stem
        
        # Save masks as TIFF (16-bit for compatibility)
        if not args.no_mask_tif:
            mask_path = args.output_dir / f"{base_name}_cp_masks.tif"
            # Convert to uint16 if possible, otherwise uint32
            if masks.max() < 65536:
                imwrite(mask_path, masks.astype(np.uint16), compression='zlib')
            else:
                imwrite(mask_path, masks.astype(np.uint32), compression='zlib')
            logger.info(f"  Saved mask: {mask_path.name}")
        
        # Save NPY file with all results
        if args.save_npy:
            npy_path = args.output_dir / f"{base_name}_seg.npy"
            np.save(npy_path, {
                'masks': masks,
                'flows': flows,
                'styles': styles,
                'diameter': args.diameter,
                'img': img
            })
            logger.info(f"  Saved NPY: {npy_path.name}")
        
        # Save PNG overlay
        if args.save_png:
            png_path = args.output_dir / f"{base_name}_overlay.png"
            if img.ndim == 2 or (img.ndim == 3 and img.shape[0] <= 3):
                # Convert to RGB for visualization
                if img.ndim == 2:
                    img_rgb = np.stack([img] * 3, axis=-1)
                else:
                    img_rgb = np.moveaxis(img, 0, -1)
                
                # Normalize for display
                img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
                img_rgb = (img_rgb * 255).astype(np.uint8)
                
                # Create overlay using plot.mask_overlay
                import matplotlib.pyplot as plt
                from cellpose import plot
                overlay = plot.mask_overlay(img_rgb, masks)
                
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(overlay)
                ax.axis('off')
                ax.set_title(f'{base_name}: {n_cells} cells')
                fig.savefig(png_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"  Saved overlay: {png_path.name}")
        
        # Save flows
        if args.save_flows:
            flow_path = args.output_dir / f"{base_name}_flows.tif"
            imwrite(flow_path, np.stack([flows[1][0], flows[1][1], flows[2]], axis=0).astype(np.float32))
            logger.info(f"  Saved flows: {flow_path.name}")
        
        # Save text outlines
        if args.save_txt:
            txt_path = args.output_dir / f"{base_name}_cp_outlines.txt"
            io.outlines_to_text(txt_path, io.masks_to_outlines(masks))
            logger.info(f"  Saved outlines: {txt_path.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"  Error processing {image_path.name}: {e}", exc_info=True)
        return False


def main():
    args = parse_args()
    
    # Handle model_type alias
    if args.model_type is not None:
        args.model = args.model_type
    
    # Convert output_dir to Path object
    args.output_dir = Path(args.output_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.verbose)
    logger.info("=" * 80)
    logger.info("Cellpose Inference Script")
    logger.info("=" * 80)
    
    # Log configuration
    logger.info("Configuration:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 80)
    
    # Get image files
    try:
        image_files = get_image_files(args.input, args.pattern)
        logger.info(f"Found {len(image_files)} image(s) to process")
    except Exception as e:
        logger.error(f"Error finding images: {e}")
        return 1
    
    if len(image_files) == 0:
        logger.error("No images found to process")
        return 1
    
    # Initialize model
    logger.info(f"Loading model: {args.model}")
    try:
        # Use CellposeModel directly to avoid requiring size model
        # (useful for custom models without size model files)
        model = models.CellposeModel(
            gpu=args.use_gpu,
            pretrained_model=args.model
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return 1
    
    # Process images
    logger.info("-" * 80)
    logger.info("Starting inference...")
    logger.info("-" * 80)
    
    success_count = 0
    fail_count = 0
    
    for img_file in image_files:
        if process_image(Path(img_file), model, args, logger):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    logger.info("=" * 80)
    logger.info(f"Inference complete!")
    logger.info(f"  Successful: {success_count}/{len(image_files)}")
    logger.info(f"  Failed: {fail_count}/{len(image_files)}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
