"""
Dataset and data loading utilities for boundary detection.

Loads images and instance masks, generates boundary ground truth
using PlantSeg-inspired approach.

Supports both 2D and 2.5D data:
- 2D: Images and masks with shape (H, W)
- 2.5D: Images and masks with shape (C, H, W) where C is the z-context window
        (e.g., C=5 for ±2 adjacent slices around the center slice)

The dataset automatically detects the input format and handles both cases.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
from skimage.segmentation import find_boundaries
from skimage import io
import random


def make_boundary_gt(instance_mask, connectivity=2, mode='thick'):
    """
    Generate boundary ground truth using PlantSeg approach.
    
    Uses scikit-image find_boundaries with:
    - connectivity=2: 8-connectivity in 2D (includes diagonals)  
    - mode='thick': Robust boundaries considering all interface pixels
    
    Args:
        instance_mask: (H, W) array with unique integer per cell, 0=background
        connectivity: Connectivity for boundary detection (1=4-conn, 2=8-conn)
        mode: 'thick' (robust) or 'inner'/'outer'
    
    Returns:
        boundary_map: (H, W) binary array, 1=boundary, 0=interior
    """
    if instance_mask.max() == 0:
        return np.zeros_like(instance_mask, dtype=np.float32)
    
    # PlantSeg-inspired approach
    boundaries = find_boundaries(
        instance_mask,
        connectivity=connectivity,
        mode=mode
    )
    
    return boundaries.astype(np.float32)


class BoundaryDataset(Dataset):
    """
    Dataset for boundary detection training.
    
    Loads image-mask pairs and generates boundary ground truth on-the-fly.
    Automatically handles both 2D and 2.5D data formats:
    - 2D: Images/masks with shape (H, W)
    - 2.5D: Images/masks with shape (C, H, W) for z-context
    
    Directory structure (Cellpose-style):
        data_dir/
            img_001.tif          # Can be (H, W) or (C, H, W)
            img_001_masks.tif    # Same shape as image
            img_002.tif
            img_002_masks.tif
            ...
    
    Alternative structure:
        data_dir/
            images/
                img_001.png
                ...
            masks/
                img_001_masks.png
                ...
    """
    def __init__(self, data_dir, mask_suffix='_masks', 
                 transform=None, augment=True):
        """
        Args:
            data_dir: Path to data directory (can be flat or with images/masks subdirs)
            mask_suffix: Suffix for mask files (e.g., '_masks', '_seg')
            transform: Optional transforms
            augment: Whether to apply data augmentation
        """
        self.is_2_5d = None  # Will be detected from first sample
        self.data_dir = Path(data_dir)
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.augment = augment
        
        # Check if using subdirectory structure or flat structure
        image_dir = self.data_dir / 'images'
        if image_dir.exists():
            # Subdirectory structure
            self.image_dir = image_dir
            self.mask_dir = self.data_dir / 'masks'
            self.use_subdirs = True
        else:
            # Flat structure (Cellpose-style)
            self.image_dir = self.data_dir
            self.mask_dir = self.data_dir
            self.use_subdirs = False
        
        # Find all images (excluding masks)
        all_files = (list(self.image_dir.glob('*.png')) + 
                     list(self.image_dir.glob('*.tif')) +
                     list(self.image_dir.glob('*.tiff')))
        
        # Filter out mask files
        candidate_images = sorted([
            f for f in all_files 
            if mask_suffix not in f.stem
        ])
        
        # Filter to only keep images that have corresponding masks
        self.image_files = []
        skipped_count = 0
        for image_path in candidate_images:
            base_name = image_path.stem
            mask_found = False
            
            # Try different mask file extensions and naming patterns
            for ext in ['.tif', '.tiff', '.png']:
                # Pattern 1: base_name + suffix + ext
                candidate = self.mask_dir / f"{base_name}{mask_suffix}{ext}"
                if candidate.exists():
                    mask_found = True
                    break
                
                # Pattern 2: base_name (without original extension) + suffix + ext
                base_without_ext = base_name.split('.')[0] if '.' in base_name else base_name
                candidate = self.mask_dir / f"{base_without_ext}{mask_suffix}{ext}"
                if candidate.exists():
                    mask_found = True
                    break
            
            if mask_found:
                self.image_files.append(image_path)
            else:
                skipped_count += 1
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images with corresponding masks found in {self.image_dir}")
        
        print(f"Found {len(self.image_files)} images with masks in {self.image_dir}")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} images without corresponding masks")
        if self.use_subdirs:
            print(f"Using subdirectory structure (images/ and masks/)")
        else:
            print(f"Using flat structure (Cellpose-style)")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = io.imread(str(image_path))
        
        # Find corresponding mask
        base_name = image_path.stem
        mask_path = None
        
        # Try different mask file extensions and naming patterns
        for ext in ['.tif', '.tiff', '.png']:
            # Pattern 1: base_name + suffix + ext
            candidate = self.mask_dir / f"{base_name}{self.mask_suffix}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
            
            # Pattern 2: base_name (without original extension) + suffix + ext
            base_without_ext = base_name.split('.')[0] if '.' in base_name else base_name
            candidate = self.mask_dir / f"{base_without_ext}{self.mask_suffix}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        
        if mask_path is None or not mask_path.exists():
            raise ValueError(f"Mask not found for {image_path.name}. "
                           f"Tried: {base_name}{self.mask_suffix}.[tif|tiff|png]")
        
        mask = io.imread(str(mask_path))
        
        # Handle 2D vs 2.5D format
        # Detect format from first sample
        if self.is_2_5d is None:
            if image.ndim == 3 and image.shape[0] <= 32:  # Assume (C, H, W) if C is small
                self.is_2_5d = True
                print(f"Detected 2.5D data format with {image.shape[0]} context slices")
            else:
                self.is_2_5d = False
                print(f"Detected 2D data format")
        
        # Process based on detected format
        if self.is_2_5d:
            # Expected format: (C, H, W) for both image and mask
            if image.ndim == 2:
                raise ValueError(f"Expected 2.5D data (C, H, W) but got 2D image: {image.shape}")
            if mask.ndim == 2:
                raise ValueError(f"Expected 2.5D data (C, H, W) but got 2D mask: {mask.shape}")
            
            # Verify shapes match
            if image.shape != mask.shape:
                raise ValueError(f"Image shape {image.shape} does not match mask shape {mask.shape}")
            
            # For 2.5D, generate boundary from center slice
            center_idx = image.shape[0] // 2
            boundary_gt = make_boundary_gt(mask[center_idx])
            
        else:
            # 2D mode: Ensure single channel
            if image.ndim == 3:
                # Check if it's (C, H, W) format with C=1
                if image.shape[0] == 1:
                    image = image[0]  # Take first channel: (1, H, W) -> (H, W)
                elif image.shape[2] <= 4:  # Likely (H, W, C) RGB format
                    image = image.mean(axis=-1)  # Convert RGB to grayscale: (H, W, C) -> (H, W)
                else:
                    image = image[0]  # Assume (C, H, W), take first
                    
            if mask.ndim == 3:
                # Check if it's (C, H, W) format with C=1
                if mask.shape[0] == 1:
                    mask = mask[0]  # Take first channel: (1, H, W) -> (H, W)
                elif mask.shape[2] <= 4:  # Likely (H, W, C) format
                    mask = mask[:, :, 0]  # Take first channel: (H, W, C) -> (H, W)
                else:
                    mask = mask[0]  # Assume (C, H, W), take first
            
            # Generate boundary ground truth
            boundary_gt = make_boundary_gt(mask)
        
        # Apply augmentation
        if self.augment:
            image, boundary_gt = self._augment(image, boundary_gt)
        
        # Normalize image to [0, 1] using 99th percentile
        image = image.astype(np.float32)
        percentile_99 = np.percentile(image, 99)
        if percentile_99 > 0:
            image = image / percentile_99
            image = np.clip(image, 0, 1)  # Clip values above 99th percentile to 1
        
        # Ensure contiguous arrays (augmentation can create negative strides)
        image = np.ascontiguousarray(image)
        boundary_gt = np.ascontiguousarray(boundary_gt.astype(np.float32))
        
        # Convert to tensors
        if self.is_2_5d:
            # Already (C, H, W) format
            image = torch.from_numpy(image).float()  # (C, H, W)
        else:
            # Add channel dimension for 2D
            image = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)
        
        boundary_gt = torch.from_numpy(boundary_gt).unsqueeze(0).float()  # (1, H, W)
        
        # Debug final tensor shapes
        if idx == 0 and not hasattr(self, '_tensor_printed'):
            print(f"DEBUG: Final tensor shapes - image: {image.shape}, boundary_gt: {boundary_gt.shape}")
            self._tensor_printed = True
        
        return {
            'image': image,
            'boundary_gt': boundary_gt,
            'filename': image_path.name
        }
    
    def _augment(self, image, boundary):
        """
        Simple data augmentation: flips and rotations.
        
        Handles both 2D (H, W) and 2.5D (C, H, W) formats.
        For 2.5D, applies the same transformation to all z-slices.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            if image.ndim == 3:  # 2.5D: (C, H, W)
                image = np.flip(image, axis=2)  # Flip width
            else:  # 2D: (H, W)
                image = np.fliplr(image)
            boundary = np.fliplr(boundary)
        
        # Random vertical flip
        if random.random() > 0.5:
            if image.ndim == 3:  # 2.5D: (C, H, W)
                image = np.flip(image, axis=1)  # Flip height
            else:  # 2D: (H, W)
                image = np.flipud(image)
            boundary = np.flipud(boundary)
        
        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            if image.ndim == 3:  # 2.5D: (C, H, W)
                image = np.rot90(image, k, axes=(1, 2))  # Rotate H, W axes
            else:  # 2D: (H, W)
                image = np.rot90(image, k)
            boundary = np.rot90(boundary, k)
        
        return image, boundary


def get_dataloaders(train_dir, val_dir=None, batch_size=4, num_workers=4, augment=False, samples_per_epoch=None):
    """
    Create training and validation dataloaders.
    
    Args:
        train_dir: Path to training data
        val_dir: Path to validation data (optional)
        batch_size: Batch size
        num_workers: Number of workers for data loading
        augment: Whether to use data augmentation (default: False)
        samples_per_epoch: Number of random samples to use per epoch (default: None = use all)
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader (or None)
    """
    train_dataset = BoundaryDataset(train_dir, augment=augment)
    
    # Create sampler if samples_per_epoch is specified
    if samples_per_epoch is not None and samples_per_epoch < len(train_dataset):
        indices = np.random.permutation(len(train_dataset))[:samples_per_epoch]
        sampler = SubsetRandomSampler(indices)
        shuffle = False
        print(f"Using {samples_per_epoch} random samples per epoch (out of {len(train_dataset)} total)")
    else:
        sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dir is not None:
        val_dataset = BoundaryDataset(val_dir, augment=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing BoundaryDataset...")
    
    # You would replace this with your actual data directory
    # dataset = BoundaryDataset('/path/to/data')
    # sample = dataset[0]
    # print(f"Image shape: {sample['image'].shape}")
    # print(f"Boundary GT shape: {sample['boundary_gt'].shape}")
    # print(f"Boundary pixels: {sample['boundary_gt'].sum().item()}")
    
    print("Dataset module ready!")
