"""
Utilities for 2.5D processing with z-context windows.

These utilities help prepare data for 2.5D boundary detection training,
where each sample consists of a context window of adjacent z-slices.
"""

import numpy as np
from skimage import io
from pathlib import Path
from typing import Union, Literal


def extract_z_context(volume, z_idx, context_slices, pad_mode='edge'):
    """
    Extract a context window of z-slices around a target slice.
    
    For 2.5D processing, this creates a multi-channel input where each channel
    is a different z-slice. The target slice is in the center of the window.
    
    Args:
        volume: (Z, H, W) 3D volume
        z_idx: Target z-slice index (0-based)
        context_slices: Total number of context slices (must be odd, e.g., 5 for ±2)
        pad_mode: Padding strategy for edge slices
                 - 'edge': Replicate edge slices (default)
                 - 'reflect': Mirror reflection
                 - 'wrap': Circular wrap
                 - 'zeros': Pad with zeros
    
    Returns:
        context_window: (context_slices, H, W) array with z-context
        
    Example:
        >>> volume = np.random.rand(100, 512, 512)  # 100 z-slices
        >>> context = extract_z_context(volume, z_idx=50, context_slices=5)
        >>> context.shape  # (5, 512, 512) - slices [48, 49, 50, 51, 52]
    """
    if context_slices % 2 == 0:
        raise ValueError(f"context_slices must be odd, got {context_slices}")
    
    Z, H, W = volume.shape
    half_context = context_slices // 2
    
    # Collect slices for context window
    context_window = np.zeros((context_slices, H, W), dtype=volume.dtype)
    
    for i, offset in enumerate(range(-half_context, half_context + 1)):
        target_z = z_idx + offset
        
        # Handle boundary conditions
        if target_z < 0:
            if pad_mode == 'edge':
                target_z = 0
            elif pad_mode == 'reflect':
                target_z = abs(target_z)
            elif pad_mode == 'wrap':
                target_z = Z + target_z
            elif pad_mode == 'zeros':
                context_window[i] = 0
                continue
            else:
                raise ValueError(f"Unknown pad_mode: {pad_mode}")
        elif target_z >= Z:
            if pad_mode == 'edge':
                target_z = Z - 1
            elif pad_mode == 'reflect':
                target_z = 2 * Z - target_z - 2
            elif pad_mode == 'wrap':
                target_z = target_z - Z
            elif pad_mode == 'zeros':
                context_window[i] = 0
                continue
            else:
                raise ValueError(f"Unknown pad_mode: {pad_mode}")
        
        context_window[i] = volume[target_z]
    
    return context_window


def prepare_2_5d_dataset(image_volume, mask_volume, context_slices=5, 
                         output_dir=None, pad_mode='edge', prefix='slice'):
    """
    Prepare 2.5D training data from 3D volumes.
    
    Converts a 3D volume into a series of 2.5D samples (C, H, W) where C is the
    context window size. Each sample is centered on a different z-slice.
    
    Args:
        image_volume: (Z, H, W) or path to 3D image volume
        mask_volume: (Z, H, W) or path to 3D mask volume
        context_slices: Number of z-slices in context window (must be odd)
        output_dir: Directory to save prepared samples (if None, returns arrays)
        pad_mode: Padding strategy for edge slices
        prefix: Prefix for saved files
    
    Returns:
        If output_dir is None: (images_list, masks_list)
        If output_dir is set: None (saves files to disk)
        
    Example:
        >>> # Prepare from arrays
        >>> images, masks = prepare_2_5d_dataset(
        ...     img_vol, mask_vol, context_slices=5
        ... )
        >>> 
        >>> # Or save to disk
        >>> prepare_2_5d_dataset(
        ...     'raw.tif', 'mask.tif', 
        ...     context_slices=5,
        ...     output_dir='./train_2_5d'
        ... )
    """
    # Load volumes if paths are provided
    if isinstance(image_volume, (str, Path)):
        image_volume = io.imread(str(image_volume))
    if isinstance(mask_volume, (str, Path)):
        mask_volume = io.imread(str(mask_volume))
    
    # Validate inputs
    if image_volume.shape != mask_volume.shape:
        raise ValueError(f"Image shape {image_volume.shape} != mask shape {mask_volume.shape}")
    
    if image_volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {image_volume.shape}")
    
    Z, H, W = image_volume.shape
    
    # Prepare output
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = output_dir / 'images'
        mask_dir = output_dir / 'masks'
        image_dir.mkdir(exist_ok=True)
        mask_dir.mkdir(exist_ok=True)
    
    images_list = []
    masks_list = []
    
    # Process each z-slice
    for z in range(Z):
        # Extract context windows
        img_context = extract_z_context(image_volume, z, context_slices, pad_mode)
        mask_context = extract_z_context(mask_volume, z, context_slices, pad_mode)
        
        if output_dir is not None:
            # Save to disk
            img_path = image_dir / f"{prefix}_{z:04d}.tif"
            mask_path = mask_dir / f"{prefix}_{z:04d}_masks.tif"
            
            io.imsave(str(img_path), img_context.astype(np.float32))
            io.imsave(str(mask_path), mask_context.astype(np.uint16))
        else:
            # Collect in memory
            images_list.append(img_context)
            masks_list.append(mask_context)
    
    if output_dir is not None:
        print(f"Saved {Z} samples to {output_dir}")
        print(f"  Images: {image_dir}")
        print(f"  Masks: {mask_dir}")
        print(f"  Shape per sample: ({context_slices}, {H}, {W})")
        return None
    else:
        return images_list, masks_list


def predict_volume_2_5d(predictor, volume, context_slices=None, pad_mode='edge', 
                        batch_size=8, device='cuda'):
    """
    Predict boundaries for an entire 3D volume using 2.5D processing.
    
    Applies the boundary predictor to each z-slice with its context window.
    
    Args:
        predictor: BoundaryPredictor instance
        volume: (Z, H, W) 3D volume or path to volume
        context_slices: Number of context slices (if None, uses predictor.in_channels)
        pad_mode: Padding strategy for edge slices
        batch_size: Batch size for inference
        device: Device to run on
    
    Returns:
        boundary_volume: (Z, H, W) 3D boundary probability volume
        
    Example:
        >>> from predict_boundary import BoundaryPredictor
        >>> predictor = BoundaryPredictor('checkpoint.pth')
        >>> volume = io.imread('stack.tif')
        >>> boundaries = predict_volume_2_5d(predictor, volume)
    """
    import torch
    
    # Load volume if path is provided
    if isinstance(volume, (str, Path)):
        volume = io.imread(str(volume))
    
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
    
    # Determine context slices
    if context_slices is None:
        context_slices = predictor.in_channels
    
    Z, H, W = volume.shape
    boundary_volume = np.zeros((Z, H, W), dtype=np.float32)
    
    # Process in batches
    for z_start in range(0, Z, batch_size):
        z_end = min(z_start + batch_size, Z)
        batch_contexts = []
        
        # Prepare batch of context windows
        for z in range(z_start, z_end):
            context = extract_z_context(volume, z, context_slices, pad_mode)
            batch_contexts.append(context)
        
        # Stack into batch tensor
        batch_tensor = np.stack(batch_contexts, axis=0)  # (B, C, H, W)
        batch_tensor = torch.from_numpy(batch_tensor).float().to(device)
        
        # Predict
        with torch.no_grad():
            logits = predictor.model(batch_tensor)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        
        # Store results
        boundary_volume[z_start:z_end] = probs
    
    return boundary_volume


if __name__ == "__main__":
    # Example usage
    print("2.5D Utilities")
    print("=" * 50)
    
    # Test extract_z_context
    print("\nTest: extract_z_context")
    test_volume = np.arange(10*5*5).reshape(10, 5, 5)
    context = extract_z_context(test_volume, z_idx=5, context_slices=5, pad_mode='edge')
    print(f"Volume shape: {test_volume.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Context z-indices: [3, 4, 5, 6, 7]")
    
    # Test edge case (near boundary)
    context_edge = extract_z_context(test_volume, z_idx=1, context_slices=5, pad_mode='edge')
    print(f"\nEdge case (z=1, pad='edge'): {context_edge.shape}")
    print("  First slice is replicated for z < 0")
    
    print("\n✓ 2.5D utilities ready!")
