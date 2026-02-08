"""
Inference script for BoundaryUNet.

Load a trained model and predict boundaries on new images.
Supports both 2D (single slices) and 2.5D (z-context windows).
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from skimage import io
import matplotlib.pyplot as plt

from boundary_unet import BoundaryUNet


class BoundaryPredictor:
    """Predictor class for boundary detection."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            device: Device to run inference on
        """
        self.device = device
        
        # Load checkpoint (weights_only=False since we trust our own checkpoints)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Detect in_channels from checkpoint if available
        # Try to infer from model state dict
        first_conv_weight = checkpoint['model_state_dict']['init_conv.conv1.weight']
        in_channels = first_conv_weight.shape[1]  # (out_ch, in_ch, H, W)
        
        # Create model
        self.model = BoundaryUNet(in_channels=in_channels, base_channels=32, depth=4)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        self.in_channels = in_channels
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Trained for {checkpoint['epoch']} epochs")
        print(f"Model expects {in_channels} input channels")
        if in_channels == 1:
            print("  -> 2D mode (single slice)")
        else:
            print(f"  -> 2.5D mode ({in_channels}-slice z-context window)")
    
    def preprocess(self, image):
        """
        Preprocess image for inference.
        
        Args:
            image: (H, W) for 2D or (C, H, W) for 2.5D numpy array
        
        Returns:
            image_tensor: (1, C, H, W) torch tensor
        """
        # Normalize to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        
        # Handle 2D vs 2.5D
        if image.ndim == 2:
            # 2D: (H, W) -> (1, 1, H, W)
            if self.in_channels != 1:
                raise ValueError(f"Model expects {self.in_channels} channels but got single-channel 2D image")
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        elif image.ndim == 3:
            # 2.5D: (C, H, W) -> (1, C, H, W)
            if image.shape[0] != self.in_channels:
                # Check if it's (H, W, C) format and convert
                if image.shape[2] == self.in_channels:
                    image = np.transpose(image, (2, 0, 1))
                else:
                    raise ValueError(f"Model expects {self.in_channels} channels but got {image.shape[0]} channels")
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        elif image.ndim == 4 and image.shape[0] == 1:
            #squueeze singleton dimension (1, C, H, W) -> (C, H, W)
            image = image.squeeze(0)
            if image.shape[0] != self.in_channels:
                raise ValueError(f"Model expects {self.in_channels} channels but got {image.shape[0]} channels after squeezing")
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()

        elif image.ndim == 4 and image.shape[0] == 3:
            # handle 3View images
            image = image[0,...]  # Take the first view (assuming all views are the same)
            if image.shape[0] != self.in_channels:
                raise ValueError(f"Model expects {self.in_channels} channels but got {image.shape[0]} channels after squeezing 3View")
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()

        else:
            raise ValueError(f"Invalid image dimensions: {image.shape}")
        
        return image_tensor
    
    def predict(self, image, threshold=0.5):
        """
        Predict boundaries for a single image.
        
        Args:
            image: (H, W) or (H, W, C) numpy array
            threshold: Threshold for binary boundary map (unused, kept for compatibility)
        
        Returns:
            boundary_prob: (H, W) probability map [0, 1]
        """
        # Preprocess
        image_tensor = self.preprocess(image).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.sigmoid(logits)
        
        # Convert to numpy
        boundary_prob = probs.squeeze().cpu().numpy()
        
        return boundary_prob
    
    def predict_batch(self, images, threshold=0.5):
        """
        Predict boundaries for a batch of images.
        
        Args:
            images: List of (H, W) numpy arrays
            threshold: Threshold for binary boundary map
        
        Returns:
            boundary_probs: List of (H, W) probability maps
            boundary_binaries: List of (H, W) binary maps
        """
        boundary_probs = []
        boundary_binaries = []
        
        for image in images:
            prob, binary = self.predict(image, threshold)
            boundary_probs.append(prob)
            boundary_binaries.append(binary)
        
        return boundary_probs, boundary_binaries
    
    def predict_volume(self, volume, threshold=0.5, pad_mode='edge', batch_size=8):
        """
        Predict boundaries for a full 3D volume using sliding window.
        
        For 2D models (in_channels=1): Processes each z-slice independently.
        For 2.5D models (in_channels>1): Uses sliding window with z-context,
        padding edges with specified mode.
        
        Args:
            volume: (Z, Y, X) numpy array - 3D volume
            threshold: Threshold for binary boundary map
            pad_mode: Padding mode for z-context at edges ('edge', 'reflect', 'wrap', 'zeros')
            batch_size: Number of slices to process in parallel
        
        Returns:
            boundary_prob: (Z, Y, X) probability volume [0, 1]
            boundary_binary: (Z, Y, X) binary boundary volume {0, 1}
        """
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume (Z, Y, X), got shape {volume.shape}")
        
        Z, Y, X = volume.shape
        
        # Normalize volume
        volume = volume.astype(np.float32)
        if volume.max() > 1.0:
            volume = volume / 255.0
        
        # Initialize output
        boundary_prob = np.zeros((Z, Y, X), dtype=np.float32)
        
        if self.in_channels == 1:
            # 2D mode: process each slice independently
            print(f"Processing {Z} slices with 2D model...")
            
            for z_start in range(0, Z, batch_size):
                z_end = min(z_start + batch_size, Z)
                batch_slices = []
                
                for z in range(z_start, z_end):
                    slice_2d = volume[z]
                    batch_slices.append(torch.from_numpy(slice_2d).unsqueeze(0).float())
                
                # Stack into batch: (B, 1, H, W)
                batch_tensor = torch.stack(batch_slices).to(self.device)
                
                with torch.no_grad():
                    logits = self.model(batch_tensor)
                    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                
                # Store results
                boundary_prob[z_start:z_end] = probs
                
                if (z_start // batch_size) % 10 == 0:
                    print(f"  Processed {z_end}/{Z} slices...")
        
        else:
            # 2.5D mode: sliding window with z-context
            print(f"Processing {Z} slices with 2.5D model (context={self.in_channels})...")
            
            # Pad volume along z-axis for edge handling
            half_context = self.in_channels // 2
            
            if pad_mode == 'zeros':
                padded = np.pad(volume, ((half_context, half_context), (0, 0), (0, 0)), mode='constant')
            else:
                padded = np.pad(volume, ((half_context, half_context), (0, 0), (0, 0)), mode=pad_mode)
            
            # Process in batches
            for z_start in range(0, Z, batch_size):
                z_end = min(z_start + batch_size, Z)
                batch_samples = []
                
                for z in range(z_start, z_end):
                    # Extract z-context window (centered at z + half_context due to padding)
                    z_center = z + half_context
                    context_window = padded[z_center - half_context : z_center + half_context + 1]
                    
                    # Should be (in_channels, Y, X)
                    if context_window.shape[0] != self.in_channels:
                        raise ValueError(f"Context window has wrong size: {context_window.shape[0]} vs expected {self.in_channels}")
                    
                    batch_samples.append(torch.from_numpy(context_window).float())
                
                # Stack into batch: (B, C, H, W)
                batch_tensor = torch.stack(batch_samples).to(self.device)
                
                with torch.no_grad():
                    logits = self.model(batch_tensor)
                    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                
                # Store results
                boundary_prob[z_start:z_end] = probs
                
                if (z_start // batch_size) % 10 == 0:
                    print(f"  Processed {z_end}/{Z} slices...")
        
        print(f"✓ Completed prediction for {Z} slices")
        
        return boundary_prob


def visualize_prediction(image, boundary_prob, boundary_binary=None, save_path=None):
    """
    Visualize prediction results.
    
    Args:
        image: Input image (H, W) for 2D or (C, H, W) for 2.5D
        boundary_prob: Boundary probability map
        boundary_binary: Binary boundary map (optional, can be None)
        save_path: Path to save visualization (optional)
    """
    # Determine number of subplots
    n_plots = 2 if boundary_binary is None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 2:
        axes = [axes[0], axes[1]]  # Ensure axes is always a list
    
    # Handle 2.5D: show center slice
    if image.ndim == 3:
        center_idx = image.shape[0] // 2
        display_image = image[center_idx]
    else:
        display_image = image
    
    # Original image
    axes[0].imshow(display_image, cmap='gray')
    axes[0].set_title('Input Image' + (' (center slice)' if image.ndim == 3 else ''))
    axes[0].axis('off')
    
    # Boundary probability
    im1 = axes[1].imshow(boundary_prob, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f'Boundary Probability (μ={boundary_prob.mean():.3f})')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Binary boundary (only if provided)
    if boundary_binary is not None:
        axes[2].imshow(boundary_binary, cmap='binary')
        axes[2].set_title(f'Binary Boundary ({boundary_binary.sum()} pixels)')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Predict boundaries using BoundaryUNet')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary boundary map')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualizations')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for volume prediction')
    parser.add_argument('--pad_mode', type=str, default='edge',
                        choices=['edge', 'reflect', 'wrap', 'zeros'],
                        help='Padding mode for z-context at volume edges')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Create predictor
    predictor = BoundaryPredictor(args.checkpoint, device=device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process image(s)
    image_path = Path(args.image)
    
    if image_path.is_file():
        # Single image or volume
        print(f"Processing {image_path.name}...")
        image = io.imread(str(image_path))
        
        # Handle multi-channel volumes (C, Z, Y, X) - take first channel
        # Common case: 3-view images stored as (3, Z, Y, X)
        if image.ndim == 4:
            if image.shape[0] == 1:
                # Singleton channel dimension
                image = image.squeeze(0)
                print(f"Squeezed singleton channel dimension, new shape: {image.shape}")
            elif image.shape[0] <= 3 and image.shape[1] > 32:
                # Multi-channel volume (C, Z, Y, X) - take first channel
                print(f"Detected multi-channel volume: {image.shape}, taking first channel")
                image = image[0]
                print(f"New shape: {image.shape}")
        
        # Check if it's a 3D volume
        if image.ndim == 3 and (image.shape[0] > 32 or predictor.in_channels == 1):
            # 3D volume: use sliding window prediction
            print(f"Detected 3D volume: {image.shape}")
            boundary_prob = predictor.predict_volume(
                image, 
                threshold=args.threshold,
                pad_mode=args.pad_mode,
                batch_size=args.batch_size
            )
            
            # Save probability volume
            prob_path = output_dir / f"{image_path.stem}_boundary_prob.tif"
            io.imsave(str(prob_path), (boundary_prob * 255).astype(np.uint8))
            print(f"Saved probability volume to {prob_path}")
            
            if args.visualize:
                # Visualize middle slice
                mid_z = image.shape[0] // 2
                vis_path = output_dir / f"{image_path.stem}_visualization_z{mid_z:04d}.png"
                display_img = image[mid_z]
                visualize_prediction(display_img, boundary_prob[mid_z], None, vis_path)
                print(f"Saved visualization of middle slice (z={mid_z}) to {vis_path}")
        else:
            # 2D or 2.5D single sample
            print(f"Detected {'2.5D sample' if image.ndim == 3 else '2D image'}: {image.shape}")
            boundary_prob = predictor.predict(image, args.threshold)
            
            # Save probability map
            prob_path = output_dir / f"{image_path.stem}_boundary_prob.tif"
            io.imsave(str(prob_path), (boundary_prob * 255).astype(np.uint8))
            print(f"Saved probability map to {prob_path}")
            
            if args.visualize:
                vis_path = output_dir / f"{image_path.stem}_visualization.png"
                visualize_prediction(image, boundary_prob, None, vis_path)
    
    elif image_path.is_dir():
        # Directory of images
        image_files = list(image_path.glob('*.png')) + \
                      list(image_path.glob('*.tif')) + \
                      list(image_path.glob('*.tiff'))
        
        print(f"Found {len(image_files)} images in {image_path}")
        
        for img_file in image_files:
            print(f"\nProcessing {img_file.name}...")
            image = io.imread(str(img_file))
            
            # Handle multi-channel volumes (C, Z, Y, X) - take first channel
            if image.ndim == 4:
                if image.shape[0] == 1:
                    # Singleton channel dimension
                    image = image.squeeze(0)
                    print(f"  Squeezed singleton channel dimension, new shape: {image.shape}")
                elif image.shape[0] <= 3 and image.shape[1] > 32:
                    # Multi-channel volume (C, Z, Y, X) - take first channel
                    print(f"  Detected multi-channel volume: {image.shape}, taking first channel")
                    image = image[0]
                    print(f"  New shape: {image.shape}")
            
            # Check if it's a 3D volume
            if image.ndim == 3 and (image.shape[0] > 32 or predictor.in_channels == 1):
                # 3D volume
                print(f"  Detected 3D volume: {image.shape}")
                boundary_prob = predictor.predict_volume(
                    image,
                    threshold=args.threshold,
                    pad_mode=args.pad_mode,
                    batch_size=args.batch_size
                )
                
                # Save probability volume
                prob_path = output_dir / f"{img_file.stem}_boundary_prob.tif"
                io.imsave(str(prob_path), (boundary_prob * 255).astype(np.uint8))
                
                if args.visualize:
                    mid_z = image.shape[0] // 2
                    vis_path = output_dir / f"{img_file.stem}_visualization_z{mid_z:04d}.png"
                    display_img = image[mid_z]
                    visualize_prediction(display_img, boundary_prob[mid_z], None, vis_path)
            else:
                # 2D or 2.5D single sample
                boundary_prob = predictor.predict(image, args.threshold)
                
                # Save probability map
                prob_path = output_dir / f"{img_file.stem}_boundary_prob.tif"
                io.imsave(str(prob_path), (boundary_prob * 255).astype(np.uint8))
                
                if args.visualize:
                    vis_path = output_dir / f"{img_file.stem}_visualization.png"
                    visualize_prediction(image, boundary_prob, None, vis_path)
        
        print(f"\nProcessed {len(image_files)} images")
    
    else:
        raise ValueError(f"Invalid path: {image_path}")
    
    print(f"\nAll predictions saved to {output_dir}")


if __name__ == '__main__':
    main()
