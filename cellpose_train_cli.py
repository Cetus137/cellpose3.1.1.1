#!/usr/bin/env python
"""
CLI-oriented Cellpose training script
Provides command-line interface for training custom Cellpose models
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from cellpose import models, train, io, core, plot
from training_callbacks import TrainingVisualizer

# Suppress verbose debugging
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


def setup_logging(output_dir, verbose=False):
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"training_{timestamp}.log")
    
    # Configure logging
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
        description='Train a Cellpose model with custom data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing training images and masks')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Directory containing test images and masks (optional)')
    parser.add_argument('--mask_filter', type=str, default='_masks',
                        help='String in mask filenames to identify them')
    parser.add_argument('--look_one_level_down', action='store_true',
                        help='Search for images in subdirectories')
    
    # Model arguments
    parser.add_argument('--pretrained_model', type=str, default='cyto3',
                        help='Pretrained model to start from (cyto3, cyto2, nuclei, or path to model)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Name for saved model (default: pretrained_model + timestamp)')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save trained models')
    
    # Training hyperparameters
    parser.add_argument('--n_epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='Weight decay for optimizer')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (number of 224x224 patches)')
    parser.add_argument('--bsize', type=int, default=224,
                        help='Size of training crops/tiles (default 224). Set to your image size to avoid cropping.')
    parser.add_argument('--min_train_masks', type=int, default=5,
                        help='Minimum number of masks per image for training')
    
    # Data augmentation
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--no_resample', action='store_true',
                        help='Disable resampling of images to training diameter')
    
    # Channels
    parser.add_argument('--channels', type=int, nargs=2, default=[0, 0],
                        help='Channels to use: [cytoplasm, nucleus] (0=gray/none, 1=red, 2=green, 3=blue)')
    parser.add_argument('--channel_axis', type=int, default=None,
                        help='Axis of image which corresponds to channels')
    
    # Training schedule
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save model every N epochs')
    parser.add_argument('--save_each', action='store_true',
                        help='Save model at each epoch')
    parser.add_argument('--nimg_per_epoch', type=int, default=None,
                        help='Number of images per epoch (default: all images)')
    
    # GPU/device
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for training')
    parser.add_argument('--gpu_device', type=int, default=0,
                        help='GPU device number to use')
    
    # Size model
    parser.add_argument('--train_size', action='store_true',
                        help='Train size model alongside segmentation model')
    
    # Advanced options
    parser.add_argument('--SGD', type=int, default=0,
                        help='Use SGD optimizer instead of Adam (1=SGD with cosine annealing, 2=SGD)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize images')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false',
                        help='Do not normalize images')
    parser.add_argument('--residual_on', type=int, default=1,
                        help='Use residual connections')
    parser.add_argument('--style_on', type=int, default=1,
                        help='Use style vector')
    
    # Boundary head training
    parser.add_argument('--lambda_boundary', type=float, default=1.0,
                        help='Weight for boundary probability loss (default 1.0)')
    parser.add_argument('--freeze_base', action='store_true',
                        help='Freeze base network (encoder/decoder) and only train boundary head')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def load_data(train_dir, test_dir=None, mask_filter='_masks', 
              look_one_level_down=False, channels=[0,0], channel_axis=None, logger=None):
    """Load training and test data"""
    if logger:
        logger.info(f"Loading training data from: {train_dir}")
    
    # Get training files
    train_output = io.load_train_test_data(
        train_dir, 
        test_dir=test_dir,
        mask_filter=mask_filter,
        look_one_level_down=look_one_level_down
    )
    
    train_data, train_labels, train_files, test_data, test_labels, test_files = train_output
    
    if logger:
        logger.info(f"Found {len(train_data)} training images")
        if test_data is not None:
            logger.info(f"Found {len(test_data)} test images")
    
    return train_data, train_labels, train_files, test_data, test_labels, test_files


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.verbose)
    logger.info("=" * 80)
    logger.info("Cellpose Training Script")
    logger.info("=" * 80)
    
    # Log arguments
    logger.info("Training configuration:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 80)
    
    # Check GPU availability
    if args.use_gpu:
        use_gpu = core.use_gpu()
        if use_gpu:
            logger.info(f"GPU available and will be used (device {args.gpu_device})")
            import torch
            device = torch.device(f'cuda:{args.gpu_device}')
        else:
            logger.warning("GPU requested but not available, using CPU")
            device = None
    else:
        logger.info("Using CPU")
        device = None
    
    # Load data
    train_data, train_labels, train_files, test_data, test_labels, test_files = load_data(
        args.train_dir,
        args.test_dir,
        mask_filter=args.mask_filter,
        look_one_level_down=args.look_one_level_down,
        channels=args.channels,
        channel_axis=args.channel_axis,
        logger=logger
    )
    
    # Initialize model
    logger.info(f"Initializing model from: {args.pretrained_model}")
    model = models.CellposeModel(
        gpu=args.use_gpu,
        pretrained_model=args.pretrained_model,
        device=device,
        nchan=2  # Standard for cellpose models
    )
    
    # Reinitialize logdist_head ONLY if loading from a pretrained model without boundary head
    # (i.e., starting fresh training, not continuing from checkpoint)
    should_reinit = False
    if hasattr(model.net, 'logdist_head') and model.net.logdist_head is not None:
        # Check if we're loading from a model that already has boundary head weights
        # by checking if the pretrained_model path contains our custom model name pattern
        is_continuing_training = 'cellpose_boundary' in str(args.pretrained_model)
        
        if not is_continuing_training:
            logger.info("Reinitializing logdist_head for NEW training (starting from base model)")
            should_reinit = True
        else:
            logger.info("Keeping existing logdist_head weights (continuing from checkpoint)")
            should_reinit = False
    
    if should_reinit:
        logger.info("Reinitializing logdist_head and upsample_logdist weights for training")
        import torch
        
        # Reinitialize logdist decoder (separate from flow decoder)
        if hasattr(model.net, 'upsample_logdist'):
            logger.info("Reinitializing separate logdist decoder branch")
            for m in model.net.upsample_logdist.modules():
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
        
        # Reinitialize logdist head
        for m in model.net.logdist_head.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    # Freeze base network if requested (only train boundary head)
    if args.freeze_base:
        logger.info("\n" + "="*60)
        logger.info("FREEZING BASE NETWORK - Only training boundary head")
        logger.info("="*60)
        
        # Freeze all parameters first
        for param in model.net.parameters():
            param.requires_grad = False
        
        # Unfreeze only logdist_head parameters
        if hasattr(model.net, 'logdist_head') and model.net.logdist_head is not None:
            for param in model.net.logdist_head.parameters():
                param.requires_grad = True
            logger.info(f"Unfroze logdist_head parameters")
        
        # Count trainable vs frozen parameters
        trainable_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.net.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
        logger.info("="*60 + "\n")
    
    # Debug: Check weight stats after initialization
    if should_reinit:
        # Debug: Check weight stats after initialization
        logger.info("Checking logdist components weight initialization:")
        for name, param in model.net.logdist_head.named_parameters():
            logger.info(f"  logdist_head.{name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
        
        # Compute dataset-wide boundary prior and initialize final layer bias
        logger.info("Computing dataset-wide boundary pixel fraction for bias initialization...")
        from cellpose.train import make_boundary_gt
        boundary_fractions = []
        n_samples = min(50, len(train_labels))  # Sample subset for efficiency
        for i in range(n_samples):
            mask = train_labels[i]
            if mask.ndim == 3:
                mask = mask[0]  # Take first channel if 3D
            boundary_gt, _ = make_boundary_gt(mask, mean_diameter=30.0)
            # Compute fraction of all pixels that are boundary (not fraction relative to cell pixels)
            boundary_fraction = boundary_gt.sum() / boundary_gt.size
            boundary_fractions.append(boundary_fraction)
        
        if len(boundary_fractions) > 0:
            mean_boundary_fraction = np.mean(boundary_fractions)
            logger.info(f"Dataset boundary fraction: {mean_boundary_fraction:.4f} ({100*mean_boundary_fraction:.1f}%)")
            
            # Initialize final conv bias to log(p / (1-p))
            # This sets the initial prediction to match the dataset prior
            p = np.clip(mean_boundary_fraction, 0.01, 0.99)
            bias_init = np.log(p / (1 - p))
            
            # Find the final conv layer and set its bias
            final_conv = None
            for m in model.net.logdist_head.modules():
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)):
                    final_conv = m
            
            if final_conv is not None and final_conv.bias is not None:
                torch.nn.init.constant_(final_conv.bias, bias_init)
                logger.info(f"Initialized final boundary head bias to {bias_init:.4f} (p={p:.4f})")
            else:
                logger.warning("Could not find final conv layer with bias for initialization")
        else:
            logger.warning("Could not compute boundary fraction from training data")
    
    # Set model name
    if args.model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(args.pretrained_model).stem if os.path.exists(args.pretrained_model) else args.pretrained_model
        model_name = f"{base_name}_custom_{timestamp}"
    else:
        model_name = args.model_name
    
    model_path = os.path.join(args.output_dir, model_name)
    logger.info(f"Model will be saved to: {model_path}")
    
    # Training parameters (only params accepted by train_seg)
    train_params = {
        'channels': args.channels,
        'channel_axis': args.channel_axis,
        'normalize': args.normalize,
        'save_path': args.output_dir,
        'save_every': args.save_every,
        'save_each': args.save_each,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'bsize': args.bsize,
        'min_train_masks': args.min_train_masks,
        'model_name': model_name,
        'nimg_per_epoch': args.nimg_per_epoch,
        'SGD': args.SGD,
        'lambda_boundary': args.lambda_boundary,
    }
    
    if not args.no_resample:
        train_params['rescale'] = True
    
    # Disable augmentation if requested (but keep diameter normalization)
    if args.no_augmentation:
        train_params['scale_range'] = 0.0  # No random scaling, but diameter normalization still active
        logger.info("Augmentation disabled: scale_range=0.0 (diameter normalization still active)")
    
    logger.info("Starting training...")
    logger.info("-" * 80)
    
    # Select a random validation image for visualization
    val_idx = np.random.randint(0, len(test_data)) if test_data is not None and len(test_data) > 0 else None
    visualizer = None
    if val_idx is not None:
        # Load the actual image and mask data
        # test_data and test_labels are numpy arrays from io.load_train_test_data
        val_image = test_data[val_idx]
        
        # Load the instance mask from the mask file (not from test_labels which contains flows)
        # test_files contains the original image paths
        test_file = test_files[val_idx]
        # Get corresponding mask file
        mask_file = test_file.replace('.tif', '_masks.tif').replace('.png', '_masks.png').replace('.jpg', '_masks.jpg')
        if not os.path.exists(mask_file):
            # Try with _masks suffix in different locations
            base_path = Path(test_file)
            mask_file = str(base_path.parent / f"{base_path.stem}_masks{base_path.suffix}")
        
        if os.path.exists(mask_file):
            val_mask = io.imread(mask_file)
            logger.info(f"Loaded mask from: {mask_file}")
        else:
            logger.warning(f"Could not find mask file for {test_file}, using labels data")
            val_mask = test_labels[val_idx]
            # If mask is from flows (3D), extract the actual mask
            if val_mask.ndim == 3 and val_mask.shape[0] > 1:
                val_mask = val_mask[0]  # First channel is the mask
        
        # If val_image or val_mask are Path objects, load them
        if isinstance(val_image, (str, Path)):
            val_image = io.imread(str(val_image))
        if isinstance(val_mask, (str, Path)):
            val_mask = io.imread(str(val_mask))
        
        logger.info(f"Selected validation image {val_idx} for visualization")
        logger.info(f"Validation image shape: {val_image.shape}, mask shape: {val_mask.shape}")
        logger.info(f"Validation mask stats: min={val_mask.min()}, max={val_mask.max()}, unique_values={len(np.unique(val_mask))}")
        
        # Create visualizer callback
        visualizer = TrainingVisualizer(
            model=model,
            val_image=val_image,
            val_mask=val_mask,
            output_dir=args.output_dir,
            model_name=model_name,
            channels=args.channels,
            device=device,
            save_every=args.save_every
        )
        logger.info("Visualization callback configured")
    
    try:
        # Train model using train_seg function with callback
        cpmodel_path, train_losses, test_losses = train.train_seg(
            model.net,
            train_data=train_data,
            train_labels=train_labels,
            train_files=train_files,
            test_data=test_data,
            test_labels=test_labels,
            test_files=test_files,
            epoch_callback=visualizer.on_epoch_end if visualizer else None,
            **train_params
        )
        
        logger.info("-" * 80)
        logger.info(f"Training completed! Model saved to: {cpmodel_path}")
        logger.info(f"All training visualizations and loss plots saved to: {os.path.join(args.output_dir, 'training_plots')}")
        
        # Train size model if requested
        if args.train_size:
            logger.info("Training size model...")
            sz_model = models.SizeModel(model, device=device)
            sz_model.params = train.train_size(
                model.net, cpmodel_path,
                train_data=train_data, train_labels=train_labels,
                test_data=test_data, test_labels=test_labels,
                channels=args.channels
            )
            
            size_model_path = cpmodel_path + '_size.npy'
            logger.info(f"Size model saved to: {size_model_path}")
        
        logger.info("=" * 80)
        logger.info("Training finished successfully!")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
