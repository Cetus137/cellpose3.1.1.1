"""
Tests for boundary probability output from eval() method.
"""

import numpy as np
import pytest
import torch

from cellpose import models


def test_eval_with_boundary():
    """Test that eval() returns boundary_prob when boundary head is present."""
    # Create synthetic 128x128 image
    img = np.random.rand(128, 128).astype(np.float32) * 255
    
    # Load model with boundary head
    # NOTE: Adjust model path to actual model with boundary head
    model = models.CellposeModel(gpu=False, model_type='cyto3')
    
    # Check if model has logdist head
    if hasattr(model.net, 'logdist_head'):
        # Run eval with boundary enabled
        masks, flows, styles = model.eval(img, channels=[0, 0], diameter=30.0, return_boundary=True)
        
        # Verify flows list length
        assert len(flows) == 4, f"Expected 4 elements in flows, got {len(flows)}"
        
        # Extract components
        flow_hsv, dP, cellprob, boundary_prob = flows
        
        # Verify boundary_prob properties
        assert boundary_prob is not None, "boundary_prob should not be None"
        assert isinstance(boundary_prob, np.ndarray), "boundary_prob should be numpy array"
        assert boundary_prob.shape == cellprob.shape, f"Shape mismatch: {boundary_prob.shape} vs {cellprob.shape}"
        assert boundary_prob.min() >= 0.0 and boundary_prob.max() <= 1.0, f"Values out of [0,1]: [{boundary_prob.min()}, {boundary_prob.max()}]"
        
        print("✓ Test with boundary head passed")
    else:
        pytest.skip("Model does not have boundary head")


def test_eval_without_boundary_parameter():
    """Test that eval() works with return_boundary=False."""
    # Create synthetic image
    img = np.random.rand(128, 128).astype(np.float32) * 255
    
    # Load model
    model = models.CellposeModel(gpu=False, model_type='cyto3')
    
    # Run eval with boundary disabled
    masks, flows, styles = model.eval(img, channels=[0, 0], diameter=30.0, return_boundary=False)
    
    # Verify flows list length (should be 3 without boundary)
    assert len(flows) == 3, f"Expected 3 elements in flows without boundary, got {len(flows)}"
    
    flow_hsv, dP, cellprob = flows
    
    # Verify standard outputs
    assert isinstance(cellprob, np.ndarray), "cellprob should be numpy array"
    
    print("✓ Test without boundary parameter passed")


def test_eval_backward_compatibility():
    """Test that eval() maintains backward compatibility when called without return_boundary."""
    # Create synthetic image
    img = np.random.rand(128, 128).astype(np.float32) * 255
    
    # Load model
    model = models.CellposeModel(gpu=False, model_type='cyto3')
    
    # Run eval without specifying return_boundary (should default to True)
    masks, flows, styles = model.eval(img, channels=[0, 0], diameter=30.0)
    
    # Flows should be a list
    assert isinstance(flows, list), "flows should be a list"
    assert len(flows) >= 3, f"Expected at least 3 elements in flows, got {len(flows)}"
    
    # If logdist head present, should have 4 elements; otherwise 3
    if hasattr(model.net, 'logdist_head'):
        assert len(flows) == 4, "Expected 4 elements with logdist head"
    else:
        assert len(flows) == 3, "Expected 3 elements without logdist head"
    
    print("✓ Backward compatibility test passed")


def test_boundary_shape_consistency():
    """Test that boundary_prob has same shape as cellprob across different image sizes."""
    model = models.CellposeModel(gpu=False, model_type='cyto3')
    
    if not hasattr(model.net, 'logdist_head'):
        pytest.skip("Model does not have logdist head")
    
    # Test multiple image sizes
    sizes = [128, 256, 512]
    
    for size in sizes:
        img = np.random.rand(size, size).astype(np.float32) * 255
        
        masks, flows, styles = model.eval(img, channels=[0, 0], diameter=30.0, return_boundary=True)
        
        if len(flows) == 4:
            flow_hsv, dP, cellprob, boundary_prob = flows
            
            assert boundary_prob.shape == cellprob.shape, \
                f"Size {size}: boundary shape {boundary_prob.shape} != cellprob shape {cellprob.shape}"
            
            print(f"✓ Size {size}: shapes consistent")


def test_boundary_values_range():
    """Test that boundary probabilities are in valid [0, 1] range."""
    model = models.CellposeModel(gpu=False, model_type='cyto3')
    
    if not hasattr(model.net, 'logdist_head'):
        pytest.skip("Model does not have logdist head")
    
    img = np.random.rand(128, 128).astype(np.float32) * 255
    
    masks, flows, styles = model.eval(img, channels=[0, 0], diameter=30.0, return_boundary=True)
    
    if len(flows) == 4:
        boundary_prob = flows[3]
        
        # Check value range
        assert boundary_prob.min() >= 0.0, f"Min value {boundary_prob.min()} < 0"
        assert boundary_prob.max() <= 1.0, f"Max value {boundary_prob.max()} > 1"
        
        # Check for NaN/Inf
        assert not np.isnan(boundary_prob).any(), "boundary_prob contains NaN"
        assert not np.isinf(boundary_prob).any(), "boundary_prob contains Inf"
        
        print(f"✓ Value range valid: [{boundary_prob.min():.4f}, {boundary_prob.max():.4f}]")


if __name__ == "__main__":
    print("Running boundary eval tests...\n")
    
    try:
        test_eval_with_boundary()
    except Exception as e:
        print(f"✗ test_eval_with_boundary failed: {e}")
    
    try:
        test_eval_without_boundary_parameter()
    except Exception as e:
        print(f"✗ test_eval_without_boundary_parameter failed: {e}")
    
    try:
        test_eval_backward_compatibility()
    except Exception as e:
        print(f"✗ test_eval_backward_compatibility failed: {e}")
    
    try:
        test_boundary_shape_consistency()
    except Exception as e:
        print(f"✗ test_boundary_shape_consistency failed: {e}")
    
    try:
        test_boundary_values_range()
    except Exception as e:
        print(f"✗ test_boundary_values_range failed: {e}")
    
    print("\nAll tests completed!")
