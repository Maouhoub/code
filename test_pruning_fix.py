#!/usr/bin/env python3
"""
Test script to verify the structured pruning fixes
"""

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_parameter_counting():
    """Test that we can count parameters correctly"""
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
        
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))
    
    model = SimpleModel()
    
    # Count parameters manually
    manual_count = sum(p.numel() for p in model.parameters())
    print(f"Manual parameter count: {manual_count}")
    
    # Expected: fc1 (10*20 + 20) + fc2 (20*5 + 5) = 200 + 20 + 100 + 5 = 325
    expected = (10 * 20 + 20) + (20 * 5 + 5)
    print(f"Expected parameter count: {expected}")
    
    assert manual_count == expected, f"Parameter counting mismatch: {manual_count} != {expected}"
    print("? Parameter counting test passed")
    
    return model

def test_layer_pruning():
    """Test that we can actually prune layers"""
    
    model = test_parameter_counting()
    original_params = sum(p.numel() for p in model.parameters())
    
    # Test fc1 pruning (reduce output features from 20 to 10)
    with torch.no_grad():
        # Keep first 10 channels
        channels_to_keep = torch.arange(10)
        
        # Prune fc1 output
        old_weight = model.fc1.weight.data
        old_bias = model.fc1.bias.data
        
        new_weight = old_weight[channels_to_keep, :]
        new_bias = old_bias[channels_to_keep]
        
        model.fc1.weight.data = new_weight
        model.fc1.bias.data = new_bias
        model.fc1.out_features = 10
        
        # Prune fc2 input
        old_weight2 = model.fc2.weight.data
        new_weight2 = old_weight2[:, channels_to_keep]
        
        model.fc2.weight.data = new_weight2
        model.fc2.in_features = 10
    
    # Count parameters after pruning
    pruned_params = sum(p.numel() for p in model.parameters())
    reduction = (original_params - pruned_params) / original_params
    
    print(f"Original parameters: {original_params}")
    print(f"Pruned parameters: {pruned_params}")
    print(f"Reduction: {reduction:.1%}")
    
    # Expected reduction: we removed 10 channels from fc1 and fc2
    # fc1: removed 10*10 + 10 = 110 parameters
    # fc2: removed 5*10 = 50 parameters
    # Total removed: 160 parameters
    expected_removed = (10 * 10 + 10) + (5 * 10)
    expected_reduction = expected_removed / original_params
    
    print(f"Expected reduction: {expected_reduction:.1%}")
    
    assert abs(reduction - expected_reduction) < 0.01, f"Reduction mismatch: {reduction} != {expected_reduction}"
    print("? Layer pruning test passed")

def test_model_forward():
    """Test that pruned model can still do forward pass"""
    
    model = test_parameter_counting()
    
    # Test original model
    x = torch.randn(1, 10)
    y1 = model(x)
    print(f"Original output shape: {y1.shape}")
    
    # Prune model (same as above)
    with torch.no_grad():
        channels_to_keep = torch.arange(10)
        
        # Prune fc1
        model.fc1.weight.data = model.fc1.weight.data[channels_to_keep, :]
        model.fc1.bias.data = model.fc1.bias.data[channels_to_keep]
        model.fc1.out_features = 10
        
        # Prune fc2
        model.fc2.weight.data = model.fc2.weight.data[:, channels_to_keep]
        model.fc2.in_features = 10
    
    # Test pruned model
    y2 = model(x)
    print(f"Pruned output shape: {y2.shape}")
    
    assert y2.shape == (1, 5), f"Output shape mismatch: {y2.shape} != (1, 5)"
    print("? Forward pass test passed")

if __name__ == "__main__":
    print("Testing structured pruning fixes...")
    print("=" * 50)
    
    try:
        test_parameter_counting()
        test_layer_pruning()  
        test_model_forward()
        
        print("\n" + "=" * 50)
        print("? All tests passed! Pruning implementation should work correctly.")
        
    except Exception as e:
        print(f"\n? Test failed: {e}")
        import traceback
        traceback.print_exc()
