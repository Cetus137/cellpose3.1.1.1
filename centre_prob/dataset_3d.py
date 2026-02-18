"""
3D dataset for cell center detection.

Extracts 3D patches from volumes and creates Gaussian center targets.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import tifffile
from scipy.ndimage import gaussian_filter, center_of_mass
import random


class CentreDataset3D(Dataset):
    """
    3D Dataset for cell center detection.
    
    Creates Gaussian peaks at cell centroids as targets.
    Extracts random 3D patches from volumes during training.
    """
    
    def __init__(
        self,
        data_dir,
        patch_size=(256, 128, 128),
        augment=True,
        center_sigma=5.0,
        patches_per_volume=10
    ):
        """
        Args:
            data_dir: Directory containing normalized volumes and GT masks
            patch_size: (D, H, W) size of extracted patches
            augment: Whether to apply data augmentation
            center_sigma: Gaussian sigma for center targets (in pixels)
            patches_per_volume: Number of random patches to extract per volume per epoch
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.center_sigma = center_sigma
        self.patches_per_volume = patches_per_volume
        
        # Find all image files
        self.image_files = sorted(self.data_dir.glob('*_normalized.tif'))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No normalized images found in {data_dir}")
        
        print(f"Found {len(self.image_files)} volumes in {data_dir}")
        
        # Total dataset size = volumes * patches_per_volume
        self.dataset_size = len(self.image_files) * patches_per_volume
    
    def __len__(self):
        return self.dataset_size
    
    def _load_volume_pair(self, volume_idx):
        """Load image and GT mask for a volume."""
        image_path = self.image_files[volume_idx]
        
        # Construct GT mask path
        gt_path = image_path.parent / image_path.name.replace('_normalized.tif', '_GT.tif')
        
        if not gt_path.exists():
            raise FileNotFoundError(f"GT mask not found: {gt_path}")
        
        # Load volumes
        image = tifffile.imread(image_path).astype(np.float32)
        gt_mask = tifffile.imread(gt_path).astype(np.float32)
        
        return image, gt_mask
    
    def _extract_random_patch(self, image, gt_mask):
        """Extract random 3D patch from volume."""
        D, H, W = image.shape
        pd, ph, pw = self.patch_size
        
        # Random crop coordinates
        d_start = np.random.randint(0, max(1, D - pd + 1))
        h_start = np.random.randint(0, max(1, H - ph + 1))
        w_start = np.random.randint(0, max(1, W - pw + 1))
        
        # Extract patches
        image_patch = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        gt_patch = gt_mask[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        # Pad if needed (edge cases)
        if image_patch.shape != self.patch_size:
            image_patch = np.pad(
                image_patch,
                [(0, pd - image_patch.shape[0]),
                 (0, ph - image_patch.shape[1]),
                 (0, pw - image_patch.shape[2])],
                mode='constant'
            )
            gt_patch = np.pad(
                gt_patch,
                [(0, pd - gt_patch.shape[0]),
                 (0, ph - gt_patch.shape[1]),
                 (0, pw - gt_patch.shape[2])],
                mode='constant'
            )
        
        return image_patch, gt_patch
    
    def _apply_augmentation(self, image, gt_mask):
        """Apply 3D data augmentation preserving Z-dependent structure."""
        # Random flip in X and Y only (NOT Z - preserves depth gradient)
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()  # Flip Y
            gt_mask = np.flip(gt_mask, axis=1).copy()
        
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()  # Flip X
            gt_mask = np.flip(gt_mask, axis=2).copy()
        
        # Random 90-degree rotations in XY plane ONLY (preserves Z depth)
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            gt_mask = np.rot90(gt_mask, k=k, axes=(1, 2)).copy()
        
        # Random intensity scaling (subtle)
        if random.random() > 0.5:
            scale = np.random.uniform(0.85, 1.15)
            image = np.clip(image * scale, 0, 1)
        
        # Random gamma correction (subtle)
        if random.random() > 0.5:
            gamma = np.random.uniform(0.85, 1.15)
            image = np.power(image, gamma)
        
        # Random brightness adjustment (subtle)
        if random.random() > 0.5:
            brightness = np.random.uniform(-0.05, 0.05)
            image = np.clip(image + brightness, 0, 1)
        
        # Random contrast adjustment (subtle)
        if random.random() > 0.5:
            contrast = np.random.uniform(0.9, 1.1)
            mean = image.mean()
            image = np.clip((image - mean) * contrast + mean, 0, 1)
        
        # Additive Gaussian noise (subtle)
        if random.random() > 0.6:
            noise_std = np.random.uniform(0.005, 0.03)
            image = image + np.random.normal(0, noise_std, image.shape)
            image = np.clip(image, 0, 1)
        
        # Random Gaussian blur (subtle - simulate defocus)
        if random.random() > 0.7:
            sigma = np.random.uniform(0.2, 0.6)
            image = gaussian_filter(image, sigma=sigma)
        
        return image, gt_mask
    
    def _compute_center_mask(self, instance_mask):
        """
        Compute center probability mask from instance segmentation.
        Creates Gaussian peaks at cell centroids.
        
        Args:
            instance_mask: (D, H, W) instance segmentation (0=background, 1,2,3...=cell IDs)
        
        Returns:
            center_mask: (D, H, W) center probability [0, 1] with Gaussian peaks at centroids
        """
        D, H, W = instance_mask.shape
        center_mask = np.zeros_like(instance_mask, dtype=np.float32)
        
        # Get unique cell IDs (exclude background=0)
        cell_ids = np.unique(instance_mask)
        cell_ids = cell_ids[cell_ids > 0]
        
        if len(cell_ids) == 0:
            return center_mask
        
        # For each cell, find centroid and create Gaussian peak
        for cell_id in cell_ids:
            # Get mask for this cell
            cell_region = (instance_mask == cell_id)
            
            # Find centroid
            centroid = center_of_mass(cell_region)
            
            # Round to nearest pixel
            z_c, y_c, x_c = [int(round(c)) for c in centroid]
            
            # Check bounds
            if not (0 <= z_c < D and 0 <= y_c < H and 0 <= x_c < W):
                continue
            
            # Create coordinate grids centered at centroid
            z_grid, y_grid, x_grid = np.ogrid[:D, :H, :W]
            
            # Compute Gaussian
            # G(r) = exp(-r^2 / (2*sigma^2))
            dist_sq = (z_grid - z_c)**2 + (y_grid - y_c)**2 + (x_grid - x_c)**2
            gaussian = np.exp(-dist_sq / (2 * self.center_sigma**2))
            
            # Accumulate (max in case of overlap)
            center_mask = np.maximum(center_mask, gaussian.astype(np.float32))
        
        return center_mask
    
    def __getitem__(self, idx):
        """
        Get a random 3D patch.
        
        Returns:
            image: (1, D, H, W) tensor
            center_gt: (1, D, H, W) tensor - center probability
        """
        # Determine which volume to sample from
        volume_idx = idx // self.patches_per_volume
        
        # Load volume
        image, gt_instance = self._load_volume_pair(volume_idx)
        
        # Extract random patch
        image_patch, gt_instance_patch = self._extract_random_patch(image, gt_instance)
        
        # Augmentation
        if self.augment:
            image_patch, gt_instance_patch = self._apply_augmentation(image_patch, gt_instance_patch)
        
        # Compute center mask from instance labels
        center_mask = self._compute_center_mask(gt_instance_patch)
        
        # Ensure in [0, 1]
        center_mask = np.clip(center_mask, 0, 1)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_patch).unsqueeze(0).float()  # (1, D, H, W)
        center_tensor = torch.from_numpy(center_mask).unsqueeze(0).float()  # (1, D, H, W)
        
        return image_tensor, center_tensor


def test_dataset():
    """Test dataset with dummy data."""
    import tempfile
    import os
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy data
        print("Creating dummy data...")
        image = np.random.rand(128, 256, 256).astype(np.float32)
        
        # Create instance mask with a few cells
        gt_mask = np.zeros((128, 256, 256), dtype=np.float32)
        gt_mask[40:80, 50:100, 50:100] = 1  # Cell 1
        gt_mask[60:100, 150:200, 150:200] = 2  # Cell 2
        
        # Save
        tifffile.imwrite(os.path.join(tmpdir, 'vol1_normalized.tif'), image)
        tifffile.imwrite(os.path.join(tmpdir, 'vol1_GT.tif'), gt_mask)
        
        # Create dataset
        dataset = CentreDataset3D(
            data_dir=tmpdir,
            patch_size=(64, 128, 128),
            augment=True,
            center_sigma=5.0,
            patches_per_volume=5
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Get sample
        image_patch, center_patch = dataset[0]
        
        print(f"Image patch shape: {image_patch.shape}")
        print(f"Center patch shape: {center_patch.shape}")
        print(f"Image range: [{image_patch.min():.3f}, {image_patch.max():.3f}]")
        print(f"Center range: [{center_patch.min():.3f}, {center_patch.max():.3f}]")
        print(f"Center max (should be ~1.0): {center_patch.max():.3f}")


if __name__ == '__main__':
    test_dataset()
