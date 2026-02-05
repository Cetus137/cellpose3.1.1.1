"""
Inference script for BoundaryUNet.

Load a trained model and predict boundaries on new images.
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
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model (assuming default architecture)
        self.model = BoundaryUNet(in_channels=1, base_channels=32, depth=4)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Trained for {checkpoint['epoch']} epochs")
    
    def preprocess(self, image):
        """
        Preprocess image for inference.
        
        Args:
            image: (H, W) or (H, W, C) numpy array
        
        Returns:
            image_tensor: (1, 1, H, W) torch tensor
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            image = image.mean(axis=-1)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        
        return image_tensor
    
    def predict(self, image, threshold=0.5):
        """
        Predict boundaries for a single image.
        
        Args:
            image: (H, W) or (H, W, C) numpy array
            threshold: Threshold for binary boundary map
        
        Returns:
            boundary_prob: (H, W) probability map [0, 1]
            boundary_binary: (H, W) binary boundary map {0, 1}
        """
        # Preprocess
        image_tensor = self.preprocess(image).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.sigmoid(logits)
        
        # Convert to numpy
        boundary_prob = probs.squeeze().cpu().numpy()
        boundary_binary = (boundary_prob > threshold).astype(np.uint8)
        
        return boundary_prob, boundary_binary
    
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


def visualize_prediction(image, boundary_prob, boundary_binary, save_path=None):
    """
    Visualize prediction results.
    
    Args:
        image: Input image
        boundary_prob: Boundary probability map
        boundary_binary: Binary boundary map
        save_path: Path to save visualization (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Boundary probability
    im1 = axes[1].imshow(boundary_prob, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f'Boundary Probability (μ={boundary_prob.mean():.3f})')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Binary boundary
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
        # Single image
        print(f"Processing {image_path.name}...")
        image = io.imread(str(image_path))
        
        boundary_prob, boundary_binary = predictor.predict(image, args.threshold)
        
        # Save results
        prob_path = output_dir / f"{image_path.stem}_boundary_prob.tif"
        binary_path = output_dir / f"{image_path.stem}_boundary_binary.png"
        
        io.imsave(str(prob_path), (boundary_prob * 255).astype(np.uint8))
        io.imsave(str(binary_path), (boundary_binary * 255).astype(np.uint8))
        
        print(f"Saved probability map to {prob_path}")
        print(f"Saved binary map to {binary_path}")
        
        if args.visualize:
            vis_path = output_dir / f"{image_path.stem}_visualization.png"
            visualize_prediction(image, boundary_prob, boundary_binary, vis_path)
    
    elif image_path.is_dir():
        # Directory of images
        image_files = list(image_path.glob('*.png')) + \
                      list(image_path.glob('*.tif')) + \
                      list(image_path.glob('*.tiff'))
        
        print(f"Found {len(image_files)} images in {image_path}")
        
        for img_file in image_files:
            print(f"Processing {img_file.name}...")
            image = io.imread(str(img_file))
            
            boundary_prob, boundary_binary = predictor.predict(image, args.threshold)
            
            # Save results
            prob_path = output_dir / f"{img_file.stem}_boundary_prob.tif"
            binary_path = output_dir / f"{img_file.stem}_boundary_binary.png"
            
            io.imsave(str(prob_path), (boundary_prob * 255).astype(np.uint8))
            io.imsave(str(binary_path), (boundary_binary * 255).astype(np.uint8))
            
            if args.visualize:
                vis_path = output_dir / f"{img_file.stem}_visualization.png"
                visualize_prediction(image, boundary_prob, boundary_binary, vis_path)
        
        print(f"\nProcessed {len(image_files)} images")
    
    else:
        raise ValueError(f"Invalid path: {image_path}")
    
    print(f"\nAll predictions saved to {output_dir}")


if __name__ == '__main__':
    main()
