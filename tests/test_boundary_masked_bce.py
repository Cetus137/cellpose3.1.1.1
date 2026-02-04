"""
Test boundary GT generation and masked BCE loss with visual debugging.

This test validates the boundary head fixes:
- Thicker boundary GT (radius=2)
- Boundary neighborhood mask (dilation radius=3)
- Masked BCE loss computation
"""

import numpy as np
import torch
import pytest
from scipy import ndimage as ndi
from cellpose.train import generate_boundary_gt, _loss_fn_seg


def create_touching_disks(size=128, radius1=25, radius2=20, center1=None, center2=None):
    """Create synthetic instance mask with two touching circular cells.
    
    Args:
        size (int): Image size (square)
        radius1 (int): Radius of first disk
        radius2 (int): Radius of second disk
        center1 (tuple): Center of first disk (y, x), defaults to left side
        center2 (tuple): Center of second disk (y, x), defaults to right side
        
    Returns:
        numpy.ndarray: Instance mask with values 1 and 2 for the two cells
    """
    if center1 is None:
        center1 = (size // 2, size // 3)
    if center2 is None:
        center2 = (size // 2, 2 * size // 3)
    
    mask = np.zeros((size, size), dtype=np.int32)
    
    # Create coordinate grids
    y, x = np.ogrid[:size, :size]
    
    # First disk
    dist1 = np.sqrt((y - center1[0])**2 + (x - center1[1])**2)
    mask[dist1 <= radius1] = 1
    
    # Second disk (overwrites overlapping region)
    dist2 = np.sqrt((y - center2[0])**2 + (x - center2[1])**2)
    mask[dist2 <= radius2] = 2
    
    return mask


def test_boundary_gt_thickness():
    """Test that boundary GT with radius=2 produces thicker boundaries than radius=1."""
    instance_mask = create_touching_disks()
    
    boundary_r1 = generate_boundary_gt(instance_mask, boundary_width=1)
    boundary_r2 = generate_boundary_gt(instance_mask, boundary_width=2)
    
    # Boundary with radius=2 should have more positive pixels
    assert boundary_r2.sum() > boundary_r1.sum(), \
        f"Boundary r=2 ({boundary_r2.sum()}) should have more pixels than r=1 ({boundary_r1.sum()})"
    
    # Both should have some boundary pixels
    assert boundary_r1.sum() > 0, "Boundary r=1 should have some positive pixels"
    assert boundary_r2.sum() > 0, "Boundary r=2 should have some positive pixels"
    
    print(f"✓ Boundary r=1: {int(boundary_r1.sum())} pixels")
    print(f"✓ Boundary r=2: {int(boundary_r2.sum())} pixels (thicker)")


def test_boundary_neighborhood_mask():
    """Test that boundary neighborhood mask covers pixels on both sides of touching region."""
    instance_mask = create_touching_disks()
    
    # Generate boundary GT with radius=2
    boundary_target = generate_boundary_gt(instance_mask, boundary_width=2)
    
    # Create neighborhood mask with dilation radius=3
    neighborhood_radius = 3
    boundary_mask = ndi.binary_dilation(boundary_target, iterations=neighborhood_radius).astype(np.float32)
    
    # Neighborhood mask should be larger than boundary GT
    assert boundary_mask.sum() > boundary_target.sum(), \
        "Neighborhood mask should cover more pixels than GT boundary"
    
    # Both should have positive pixels
    assert boundary_target.sum() > 0, "Boundary GT should have positive pixels"
    assert boundary_mask.sum() > 0, "Neighborhood mask should have positive pixels"
    
    # Check that touching region is covered
    # Find the touching region (pixels where both disks meet)
    # This should be covered by the neighborhood mask
    center_y, center_x = instance_mask.shape[0] // 2, instance_mask.shape[1] // 2
    touching_region_covered = boundary_mask[center_y-5:center_y+5, center_x-5:center_x+5].sum() > 0
    assert touching_region_covered, "Neighborhood mask should cover the touching region"
    
    print(f"✓ Boundary GT: {int(boundary_target.sum())} pixels")
    print(f"✓ Neighborhood mask: {int(boundary_mask.sum())} pixels")
    print(f"✓ Coverage ratio: {boundary_mask.sum()/boundary_target.sum():.2f}x")


def test_masked_bce_loss():
    """Test that masked BCE loss works correctly and handles edge cases."""
    # Create synthetic batch
    batch_size = 2
    height, width = 128, 128
    device = torch.device('cpu')
    
    # Create instance masks and boundary GT
    instance_masks = [create_touching_disks(size=128) for _ in range(batch_size)]
    boundary_gt_batch = np.array([generate_boundary_gt(m, boundary_width=2) for m in instance_masks])
    
    # Create dummy flow labels (cellprob, flowsY, flowsX)
    lbl = np.zeros((batch_size, 3, height, width), dtype=np.float32)
    lbl[:, 0] = (np.array(instance_masks) > 0).astype(np.float32)  # cellprob
    
    # Create dummy predictions
    y = torch.randn(batch_size, 3, height, width)
    boundary_logits = torch.randn(batch_size, 1, height, width)
    
    # Compute loss
    loss, flow_loss, cellprob_loss, boundary_loss = _loss_fn_seg(
        lbl, y, device,
        boundary_gt=boundary_gt_batch,
        boundary_logits=boundary_logits,
        boundary_weight=1.0,
        boundary_pos_weight=10.0,
        boundary_neighborhood_radius=3
    )
    
    # Check that all losses are valid tensors
    assert isinstance(loss, torch.Tensor), "Total loss should be a tensor"
    assert isinstance(boundary_loss, torch.Tensor), "Boundary loss should be a tensor"
    assert not torch.isnan(loss), "Total loss should not be NaN"
    assert not torch.isnan(boundary_loss), "Boundary loss should not be NaN"
    assert boundary_loss.item() > 0, "Boundary loss should be positive (random predictions)"
    
    print(f"✓ Total loss: {loss.item():.4f}")
    print(f"✓ Boundary loss: {boundary_loss.item():.4f}")
    print(f"✓ Flow loss: {flow_loss.item():.4f}")
    print(f"✓ Cellprob loss: {cellprob_loss.item():.4f}")


def test_empty_boundary_gt():
    """Test that loss function handles empty boundary GT gracefully."""
    batch_size = 2
    height, width = 128, 128
    device = torch.device('cpu')
    
    # Create empty boundary GT
    boundary_gt_batch = np.zeros((batch_size, height, width), dtype=np.float32)
    
    # Create dummy labels and predictions
    lbl = np.zeros((batch_size, 3, height, width), dtype=np.float32)
    y = torch.randn(batch_size, 3, height, width)
    boundary_logits = torch.randn(batch_size, 1, height, width)
    
    # Compute loss - should not crash
    loss, flow_loss, cellprob_loss, boundary_loss = _loss_fn_seg(
        lbl, y, device,
        boundary_gt=boundary_gt_batch,
        boundary_logits=boundary_logits,
        boundary_weight=1.0,
        boundary_pos_weight=10.0,
        boundary_neighborhood_radius=3
    )
    
    # Boundary loss should be zero (no boundary pixels)
    assert boundary_loss.item() == 0.0, "Boundary loss should be 0 for empty GT"
    assert not torch.isnan(loss), "Total loss should not be NaN even with empty boundary GT"
    
    print("✓ Empty boundary GT handled correctly (boundary_loss = 0)")


def visualize_boundary_debug(image, instance_mask, boundary_target, boundary_mask, 
                             output_path='boundary_debug.png'):
    """
    Visualize boundary GT and neighborhood mask overlaid on image for debugging.
    
    Args:
        image (numpy.ndarray): Input image (H, W) or (H, W, C)
        instance_mask (numpy.ndarray): Instance segmentation mask (H, W)
        boundary_target (numpy.ndarray): Boundary GT (H, W) with values in [0, 1]
        boundary_mask (numpy.ndarray): Boundary neighborhood mask (H, W) with values in [0, 1]
        output_path (str): Path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    # Normalize image to grayscale
    if image.ndim == 3:
        image_gray = image.mean(axis=-1)
    else:
        image_gray = image
    
    # Normalize to [0, 1]
    image_gray = (image_gray - image_gray.min()) / (image_gray.max() - image_gray.min() + 1e-8)
    
    # Create RGB visualization
    vis = np.stack([image_gray, image_gray, image_gray], axis=-1)
    
    # Overlay boundary GT in green
    green_overlay = np.zeros_like(vis)
    green_overlay[:, :, 1] = boundary_target  # Green channel
    vis = np.clip(vis * 0.6 + green_overlay * 0.8, 0, 1)
    
    # Overlay neighborhood mask in red (semi-transparent)
    red_overlay = np.zeros_like(vis)
    red_overlay[:, :, 0] = boundary_mask  # Red channel
    vis = np.clip(vis + red_overlay * 0.3, 0, 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image_gray, cmap='gray')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Instance mask
    axes[0, 1].imshow(instance_mask, cmap='tab20')
    axes[0, 1].set_title('Instance Mask')
    axes[0, 1].axis('off')
    
    # Boundary GT
    axes[0, 2].imshow(boundary_target, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Boundary GT\n({int(boundary_target.sum())} pixels)')
    axes[0, 2].axis('off')
    
    # Neighborhood mask
    axes[1, 0].imshow(boundary_mask, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Neighborhood Mask\n({int(boundary_mask.sum())} pixels)')
    axes[1, 0].axis('off')
    
    # Combined overlay
    axes[1, 1].imshow(vis)
    axes[1, 1].set_title('Overlay\n(Green=GT, Red=Neighborhood)')
    axes[1, 1].axis('off')
    
    # Statistics
    stats_text = (
        f"Boundary GT pixels: {int(boundary_target.sum())}\n"
        f"Neighborhood pixels: {int(boundary_mask.sum())}\n"
        f"Coverage ratio: {boundary_mask.sum()/(boundary_target.sum()+1e-8):.2f}x\n"
        f"GT density: {100*boundary_target.sum()/boundary_target.size:.2f}%\n"
        f"Mask density: {100*boundary_mask.sum()/boundary_mask.size:.2f}%"
    )
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    family='monospace')
    axes[1, 2].set_title('Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    plt.close()


def test_visual_debug():
    """Test the visual debugging function with synthetic data."""
    # Create synthetic data
    size = 128
    instance_mask = create_touching_disks(size=size)
    
    # Create synthetic image (add some noise)
    image = np.random.rand(size, size) * 0.2
    image[instance_mask > 0] += 0.6  # Cells are brighter
    
    # Generate boundary GT and neighborhood mask
    boundary_target = generate_boundary_gt(instance_mask, boundary_width=2)
    boundary_mask = ndi.binary_dilation(boundary_target, iterations=3).astype(np.float32)
    
    # Create visualization
    output_path = '/users/kir-fritzsche/aif490/devel/tissue_analysis/cellpose/tests/boundary_debug_test.png'
    visualize_boundary_debug(image, instance_mask, boundary_target, boundary_mask, output_path)
    
    print(f"✓ Visual debug test completed")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Testing Boundary Head Fixes")
    print("="*60 + "\n")
    
    print("Test 1: Boundary GT Thickness")
    print("-" * 60)
    test_boundary_gt_thickness()
    
    print("\nTest 2: Boundary Neighborhood Mask")
    print("-" * 60)
    test_boundary_neighborhood_mask()
    
    print("\nTest 3: Masked BCE Loss")
    print("-" * 60)
    test_masked_bce_loss()
    
    print("\nTest 4: Empty Boundary GT (Edge Case)")
    print("-" * 60)
    test_empty_boundary_gt()
    
    print("\nTest 5: Visual Debug Function")
    print("-" * 60)
    test_visual_debug()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
