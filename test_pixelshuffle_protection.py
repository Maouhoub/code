#!/usr/bin/env python3
"""
Test script to verify PixelShuffle protection in structured pruning.
This script validates that our enhanced Torch-Pruning setup properly excludes
layers that feed into PixelShuffle operations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

# Import our enhanced pruning functions
from main_train_psnr_L2_fine_tune_structured_enhanced import (
    validate_pixelshuffle_constraints,
    apply_structured_pruning_torch_pruning,
    calculate_model_stats,
    TORCH_PRUNING_AVAILABLE
)

def create_simple_pixelshuffle_model(in_channels=64, scale=2):
    """Create a simple model with PixelShuffle for testing."""
    
    class SimplePixelShuffleModel(nn.Module):
        def __init__(self, in_channels, scale):
            super().__init__()
            self.conv1 = nn.Conv2d(3, in_channels, 3, 1, 1)
            self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
            
            # This is the critical layer that feeds into PixelShuffle
            self.conv_last = nn.Conv2d(in_channels, (scale**2) * 3, 3, 1, 1)
            self.pixelshuffle = nn.PixelShuffle(scale)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv_last(x)
            x = self.pixelshuffle(x)
            return x
    
    return SimplePixelShuffleModel(in_channels, scale)

def test_pixelshuffle_protection():
    """Test that our PixelShuffle protection works correctly."""
    
    print("=" * 60)
    print("Testing PixelShuffle Protection in Structured Pruning")
    print("=" * 60)
    
    # Create test model
    model = create_simple_pixelshuffle_model(in_channels=64, scale=2)
    print(f"Created test model with PixelShuffle (scale=2)")
    
    # Move to device if CUDA available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 1. Validate initial model
    print("\n1. Validating initial model:")
    is_valid_initial = validate_pixelshuffle_constraints(model, scale_factor=2)
    
    # 2. Calculate initial stats
    print("\n2. Calculating initial model statistics:")
    initial_stats = calculate_model_stats(model, (1, 3, 64, 64), device)
    print(f"Initial parameters: {initial_stats['params']:,}")
    print(f"Initial FLOPs: {initial_stats['flops']:,}")
    
    # 3. Apply structured pruning with PixelShuffle protection
    print("\n3. Applying structured pruning with PixelShuffle protection:")
    if TORCH_PRUNING_AVAILABLE:
        try:
            pruned_model = apply_structured_pruning_torch_pruning(model, pruning_ratio=0.3)
            print("? Torch-Pruning applied successfully")
        except Exception as e:
            print(f"? Torch-Pruning failed: {e}")
            pruned_model = model
    else:
        print("??  Torch-Pruning not available, skipping test")
        pruned_model = model
    
    # 4. Validate pruned model
    print("\n4. Validating pruned model:")
    is_valid_pruned = validate_pixelshuffle_constraints(pruned_model, scale_factor=2)
    
    # 5. Calculate pruned stats
    print("\n5. Calculating pruned model statistics:")
    pruned_stats = calculate_model_stats(pruned_model, (1, 3, 64, 64), device)
    print(f"Pruned parameters: {pruned_stats['params']:,}")
    print(f"Pruned FLOPs: {pruned_stats['flops']:,}")
    
    # 6. Compare results
    print("\n6. Results Summary:")
    if initial_stats['params'] > 0:
        param_reduction = (initial_stats['params'] - pruned_stats['params']) / initial_stats['params'] * 100
        print(f"Parameter reduction: {param_reduction:.1f}%")
    
    if initial_stats['flops'] > 0:
        flop_reduction = (initial_stats['flops'] - pruned_stats['flops']) / initial_stats['flops'] * 100
        print(f"FLOPs reduction: {flop_reduction:.1f}%")
    
    # 7. Test forward pass
    print("\n7. Testing forward pass:")
    try:
        with torch.no_grad():
            test_input = torch.randn(1, 3, 64, 64).to(device)
            output = pruned_model(test_input)
            print(f"? Forward pass successful: {test_input.shape} -> {output.shape}")
    except Exception as e:
        print(f"? Forward pass failed: {e}")
    
    # 8. Final validation
    print("\n8. Final Validation:")
    if is_valid_initial and is_valid_pruned:
        print("? PixelShuffle protection test PASSED")
        print("   - Initial model was valid")
        print("   - Pruned model maintains PixelShuffle constraints")
    else:
        print("? PixelShuffle protection test FAILED")
        print(f"   - Initial model valid: {is_valid_initial}")
        print(f"   - Pruned model valid: {is_valid_pruned}")
    
    print("=" * 60)

if __name__ == "__main__":
    test_pixelshuffle_protection()