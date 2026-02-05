"""
Dataset and data loading utilities for boundary detection.

Loads images and instance masks, generates boundary ground truth
using PlantSeg-inspired approach.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
    
    Directory structure expected:
        data_dir/
            images/
                img_001.png
                img_002.png
                ...
            masks/
                img_001_masks.png
                img_002_masks.png
                ...
    """
    def __init__(self, data_dir, image_suffix='', mask_suffix='_masks', 
                 transform=None, augment=True):
        """
        Args:
            data_dir: Path to data directory
            image_suffix: Suffix for image files (e.g., '.tif', '')
            mask_suffix: Suffix for mask files (e.g., '_masks', '_seg')
            transform: Optional transforms
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.mask_dir = self.data_dir / 'masks'
        self.transform = transform
        self.augment = augment
        
        # Find all images
        self.image_files = sorted(list(self.image_dir.glob(f'*{image_suffix}.png')) + 
                                   list(self.image_dir.glob(f'*{image_suffix}.tif')) +
                                   list(self.image_dir.glob(f'*{image_suffix}.tiff')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        print(f"Found {len(self.image_files)} images in {self.image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = io.imread(str(image_path))
        
        # Load corresponding mask
        mask_name = image_path.stem.replace(image_path.suffix, '') + '_masks.png'
        mask_path = self.mask_dir / mask_name
        
        if not mask_path.exists():
            # Try alternative naming
            mask_name = image_path.stem + '_masks.png'
            mask_path = self.mask_dir / mask_name
        
        if not mask_path.exists():
            raise ValueError(f"Mask not found for {image_path.name}")
        
        mask = io.imread(str(mask_path))
        
        # Ensure 2D
        if image.ndim == 3:
            image = image.mean(axis=-1)  # Convert to grayscale
        if mask.ndim == 3:
            mask = mask[:, :, 0]  # Take first channel
        
        # Generate boundary ground truth
        boundary_gt = make_boundary_gt(mask)
        
        # Apply augmentation
        if self.augment:
            image, boundary_gt = self._augment(image, boundary_gt)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)
        boundary_gt = torch.from_numpy(boundary_gt).unsqueeze(0).float()  # (1, H, W)
        
        return {
            'image': image,
            'boundary_gt': boundary_gt,
            'filename': image_path.name
        }
    
    def _augment(self, image, boundary):
        """Simple data augmentation: flips and rotations."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image)
            boundary = np.fliplr(boundary)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = np.flipud(image)
            boundary = np.flipud(boundary)
        
        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k)
            boundary = np.rot90(boundary, k)
        
        return image, boundary


def get_dataloaders(train_dir, val_dir=None, batch_size=4, num_workers=4):
    """
    Create training and validation dataloaders.
    
    Args:
        train_dir: Path to training data
        val_dir: Path to validation data (optional)
        batch_size: Batch size
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader (or None)
    """
    train_dataset = BoundaryDataset(train_dir, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
