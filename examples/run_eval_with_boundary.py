"""
Demo script for running Cellpose eval() with boundary probability output.

This script demonstrates how to:
1. Load an image
2. Run inference with boundary_prob extraction
3. Save and visualize the outputs
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add cellpose to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from cellpose import models, io


def run_eval_with_boundary(image_path, model_type='cyto3', diameter=30.0, channels=[0, 0]):
    """
    Run Cellpose eval with boundary probability extraction.
    
    Parameters
    ----------
    image_path : str
        Path to input image
    model_type : str
        Cellpose model type
    diameter : float
        Expected cell diameter in pixels
    channels : list
        Channel configuration [cytoplasm, nucleus]
    
    Returns
    -------
    masks : ndarray
        Segmentation masks
    cellprob : ndarray
        Cell probability map
    boundary_prob : ndarray or None
        Boundary probability map (if model has boundary head)
    """
    # Load image
    print(f"Loading image: {image_path}")
    img = io.imread(image_path)
    
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    
    # Load model
    print(f"Loading model: {model_type}")
    model = models.CellposeModel(gpu=True, model_type=model_type)
    
    # Check if model has logdist head
    has_boundary = hasattr(model.net, 'logdist_head')
    print(f"Model has logdist head: {has_boundary}")
    
    # Run evaluation with boundary enabled
    print("\nRunning inference...")
    masks, flows, styles = model.eval(
        img,
        channels=channels,
        diameter=diameter,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        return_boundary=True
    )
    
    # Extract outputs
    flow_hsv = flows[0]
    dP = flows[1]
    cellprob = flows[2]
    boundary_prob = flows[3] if len(flows) > 3 else None
    
    print(f"\nOutputs:")
    print(f"  masks: {masks.shape}, unique cells: {len(np.unique(masks)) - 1}")
    print(f"  cellprob: {cellprob.shape}, range: [{cellprob.min():.3f}, {cellprob.max():.3f}]")
    
    if boundary_prob is not None:
        print(f"  boundary_prob: {boundary_prob.shape}, range: [{boundary_prob.min():.3f}, {boundary_prob.max():.3f}]")
    else:
        print(f"  boundary_prob: None (model has no boundary head)")
    
    return masks, cellprob, boundary_prob, img


def save_outputs(image_path, masks, cellprob, boundary_prob, img):
    """Save outputs as images."""
    output_dir = Path(image_path).parent / "cellpose_outputs"
    output_dir.mkdir(exist_ok=True)
    
    basename = Path(image_path).stem
    
    # Save masks
    mask_path = output_dir / f"{basename}_masks.png"
    io.imsave(mask_path, masks.astype(np.uint16))
    print(f"\nSaved masks: {mask_path}")
    
    # Save cellprob
    cellprob_path = output_dir / f"{basename}_cellprob.png"
    cellprob_img = (cellprob * 255).astype(np.uint8)
    io.imsave(cellprob_path, cellprob_img)
    print(f"Saved cellprob: {cellprob_path}")
    
    # Save boundary_prob if available
    if boundary_prob is not None:
        boundary_path = output_dir / f"{basename}_boundary.png"
        boundary_img = (boundary_prob * 255).astype(np.uint8)
        io.imsave(boundary_path, boundary_img)
        print(f"Saved boundary_prob: {boundary_path}")


def visualize_outputs(img, masks, cellprob, boundary_prob):
    """Create visualization of all outputs."""
    if boundary_prob is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if img.ndim == 2:
        axes[0].imshow(img, cmap='gray')
    else:
        axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Masks
    from cellpose import plot
    axes[1].imshow(plot.mask_overlay(img, masks))
    axes[1].set_title(f'Masks ({len(np.unique(masks)) - 1} cells)')
    axes[1].axis('off')
    
    # Cell probability
    axes[2].imshow(cellprob, cmap='viridis')
    axes[2].set_title('Cell Probability')
    axes[2].axis('off')
    
    # Boundary probability
    if boundary_prob is not None:
        axes[3].imshow(boundary_prob, cmap='magma')
        axes[3].set_title('Boundary Probability')
        axes[3].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Cellpose eval with boundary probability')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, default='cyto3', help='Model type')
    parser.add_argument('--diameter', type=float, default=30.0, help='Cell diameter')
    parser.add_argument('--channels', type=int, nargs=2, default=[0, 0], help='Channels [cyto, nucleus]')
    parser.add_argument('--save', action='store_true', help='Save outputs')
    parser.add_argument('--plot', action='store_true', help='Display plot')
    
    args = parser.parse_args()
    
    # Run inference
    masks, cellprob, boundary_prob, img = run_eval_with_boundary(
        args.image,
        model_type=args.model,
        diameter=args.diameter,
        channels=args.channels
    )
    
    # Save outputs
    if args.save:
        save_outputs(args.image, masks, cellprob, boundary_prob, img)
    
    # Visualize
    if args.plot:
        fig = visualize_outputs(img, masks, cellprob, boundary_prob)
        plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
