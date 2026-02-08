"""
Prepare 2.5D training data from 3D volumes.

Takes paired image and mask volumes and splits them into non-overlapping
(context, Y, X) samples for 2.5D training.

Input:
    - Directory with paired volumes (e.g., img.tif and img_masks.tif)
    - Volumes of shape (Z, Y, X)

Output:
    - Directory with (context, Y, X) samples ready for training
    - Non-overlapping chunks extracted from volumes
"""

import numpy as np
from skimage import io
from pathlib import Path
import argparse
from tqdm import tqdm


def split_volume_into_chunks(volume, context_slices, overlap=False):
    """
    Split a 3D volume into non-overlapping chunks.
    
    Args:
        volume: (Z, Y, X) 3D array
        context_slices: Number of z-slices per chunk
        overlap: If True, create overlapping chunks (for validation)
                If False, non-overlapping chunks (for training)
    
    Returns:
        chunks: List of (context_slices, Y, X) arrays
        z_indices: List of starting z-indices for each chunk
    """
    Z, Y, X = volume.shape
    chunks = []
    z_indices = []
    
    if overlap:
        # Overlapping chunks (sliding window with step=1)
        # Used for validation to include all slices
        for z in range(0, Z - context_slices + 1):
            chunk = volume[z:z+context_slices]
            chunks.append(chunk)
            z_indices.append(z)
    else:
        # Non-overlapping chunks (step=context_slices)
        # Used for training to avoid data leakage
        for z in range(0, Z, context_slices):
            # Check if we have enough slices for a full chunk
            if z + context_slices <= Z:
                chunk = volume[z:z+context_slices]
                chunks.append(chunk)
                z_indices.append(z)
            elif z < Z:
                # Handle remainder: take last context_slices
                # This creates a small overlap but ensures we don't lose data
                chunk = volume[Z-context_slices:Z]
                chunks.append(chunk)
                z_indices.append(Z - context_slices)
                break
    
    return chunks, z_indices


def prepare_dataset(input_dir, output_dir, context_slices=5, mask_suffix='_masks',
                   overlap=False, min_z_size=None):
    """
    Prepare 2.5D training data from 3D volumes.
    
    Args:
        input_dir: Directory containing paired image and mask volumes
        output_dir: Output directory for prepared samples
        context_slices: Number of z-slices per sample
        mask_suffix: Suffix for mask files
        overlap: Whether to create overlapping samples (False for training)
        min_z_size: Minimum Z size to process (skip smaller volumes)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    image_dir = output_dir / 'images'
    mask_dir = output_dir / 'masks'
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = (list(input_dir.glob('*.tif')) + 
                   list(input_dir.glob('*.tiff')) +
                   list(input_dir.glob('*.png')))
    
    # Filter to only images (not masks)
    image_files = sorted([f for f in image_files if mask_suffix not in f.stem])
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {input_dir}")
    
    print(f"Found {len(image_files)} volumes to process")
    print(f"Context slices: {context_slices}")
    print(f"Overlap mode: {overlap}")
    print(f"Output directory: {output_dir}")
    print()
    
    total_samples = 0
    skipped_volumes = 0
    
    for img_file in tqdm(image_files, desc="Processing volumes"):
        # Find corresponding mask
        base_name = img_file.stem
        mask_found = False
        mask_path = None
        
        for ext in ['.tif', '.tiff', '.png']:
            candidate = input_dir / f"{base_name}{mask_suffix}{ext}"
            if candidate.exists():
                mask_path = candidate
                mask_found = True
                break
        
        if not mask_found:
            print(f"Warning: No mask found for {img_file.name}, skipping")
            skipped_volumes += 1
            continue
        
        # Load volumes
        try:
            image_vol = io.imread(str(img_file))
            mask_vol = io.imread(str(mask_path))
        except Exception as e:
            print(f"Error loading {img_file.name}: {e}")
            skipped_volumes += 1
            continue
        
        # Handle extra dimensions (e.g., (1, Z, Y, X) -> (Z, Y, X))
        if image_vol.ndim == 4 and image_vol.shape[0] == 1:
            image_vol = image_vol[0]
        if mask_vol.ndim == 4 and mask_vol.shape[0] == 1:
            mask_vol = mask_vol[0]
        
        # Validate
        if image_vol.shape != mask_vol.shape:
            print(f"Warning: Shape mismatch for {img_file.name}: "
                  f"image {image_vol.shape} vs mask {mask_vol.shape}, skipping")
            skipped_volumes += 1
            continue
        
        if image_vol.ndim != 3:
            print(f"Warning: {img_file.name} is not 3D (shape: {image_vol.shape}), skipping")
            skipped_volumes += 1
            continue
        
        Z, Y, X = image_vol.shape
        
        # Check minimum Z size
        if min_z_size is not None and Z < min_z_size:
            print(f"Warning: {img_file.name} Z-size {Z} < minimum {min_z_size}, skipping")
            skipped_volumes += 1
            continue
        
        if Z < context_slices:
            print(f"Warning: {img_file.name} has Z={Z} < context_slices={context_slices}, skipping")
            skipped_volumes += 1
            continue
        
        # Split into chunks
        img_chunks, z_indices = split_volume_into_chunks(image_vol, context_slices, overlap)
        mask_chunks, _ = split_volume_into_chunks(mask_vol, context_slices, overlap)
        
        # Save chunks
        for i, (img_chunk, mask_chunk, z_start) in enumerate(zip(img_chunks, mask_chunks, z_indices)):
            # Generate unique filename
            # Format: volumename_zXXXX-XXXX_sampleXXX.tif
            chunk_name = f"{base_name}_z{z_start:04d}-{z_start+context_slices-1:04d}_s{i:03d}"
            
            img_output = image_dir / f"{chunk_name}.tif"
            mask_output = mask_dir / f"{chunk_name}_masks.tif"
            
            # Save as float32 for images, uint16 for masks
            io.imsave(str(img_output), img_chunk.astype(np.float32), check_contrast=False)
            io.imsave(str(mask_output), mask_chunk.astype(np.uint16), check_contrast=False)
            
            total_samples += 1
    
    print()
    print("="*70)
    print("Dataset Preparation Complete!")
    print("="*70)
    print(f"Processed volumes: {len(image_files) - skipped_volumes}/{len(image_files)}")
    print(f"Skipped volumes: {skipped_volumes}")
    print(f"Total samples created: {total_samples}")
    print(f"Output directory: {output_dir}")
    print(f"  - Images: {image_dir} ({len(list(image_dir.glob('*.tif')))} files)")
    print(f"  - Masks: {mask_dir} ({len(list(mask_dir.glob('*.tif')))} files)")
    print()
    print("Sample shape per file: ({}, {}, {})".format(context_slices, Y, X))
    print()
    print("Ready for training with:")
    print(f"  python train_boundary.py --train_dir {output_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Prepare 2.5D training data from 3D volumes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - split volumes into non-overlapping chunks
  python prepare_training_data_2_5d.py \\
    --input_dir ./raw_volumes \\
    --output_dir ./train_data \\
    --context_slices 5

  # For validation data - use overlapping to include all slices
  python prepare_training_data_2_5d.py \\
    --input_dir ./raw_volumes \\
    --output_dir ./val_data \\
    --context_slices 5 \\
    --overlap

  # Custom mask suffix
  python prepare_training_data_2_5d.py \\
    --input_dir ./raw_volumes \\
    --output_dir ./train_data \\
    --context_slices 7 \\
    --mask_suffix _seg
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing paired image and mask volumes')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for prepared samples')
    parser.add_argument('--context_slices', type=int, default=5,
                        help='Number of z-slices per sample (default: 5)')
    parser.add_argument('--mask_suffix', type=str, default='_masks',
                        help='Suffix for mask files (default: _masks)')
    parser.add_argument('--overlap', action='store_true',
                        help='Create overlapping samples (for validation, not training)')
    parser.add_argument('--min_z_size', type=int, default=None,
                        help='Minimum Z size to process (skip smaller volumes)')
    
    args = parser.parse_args()
    
    # Validate context_slices is odd
    if args.context_slices % 2 == 0:
        print(f"Warning: context_slices={args.context_slices} is even. "
              f"Using odd number is recommended for centered context.")
    
    if args.context_slices < 1:
        raise ValueError(f"context_slices must be >= 1, got {args.context_slices}")
    
    # Prepare dataset
    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        context_slices=args.context_slices,
        mask_suffix=args.mask_suffix,
        overlap=args.overlap,
        min_z_size=args.min_z_size
    )


if __name__ == '__main__':
    main()
