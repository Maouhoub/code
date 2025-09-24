#!/usr/bin/env python3
"""
Quick Test Script for Structured Channel Pruning
=================================================

This script provides a simplified example for testing the structured channel pruning
functionality without requiring a full dataset setup.

Usage:
    python test_structured_pruning.py
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# Add the main directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Try importing from the enhanced script
    from main_train_psnr_L2_fine_tune_structured_enhanced import (
        calculate_model_stats,
        apply_structured_pruning_torch_pruning,
        apply_structured_pruning_native,
        print_results_table,
        TORCH_PRUNING_AVAILABLE
    )
    print("? Successfully imported pruning functions")
except ImportError as e:
    print(f"? Import error: {e}")
    print("Please ensure the enhanced script is in the same directory")
    sys.exit(1)

# Simple test model (mimicking SwinIR structure)
class TestSwinIRLite(nn.Module):
    def __init__(self, embed_dim=60, num_layers=4):
        super(TestSwinIRLite, self).__init__()
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, 3, 1, 1)
        
        # Several conv layers to mimic transformer blocks
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim*2, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers)
        ])
        
        # Upsampling (for 2x SR)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(embed_dim, 3, 3, 1, 1)
        )
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Process through layers
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
        
        # Upsample
        x = self.upsample(x)
        return x

def test_structured_pruning():
    """Test the structured pruning functionality."""
    
    print("="*80)
    print(" STRUCTURED CHANNEL PRUNING TEST")
    print("="*80)
    
    # Create test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TestSwinIRLite(embed_dim=60, num_layers=4).to(device)
    print("? Test model created")
    
    # Calculate baseline statistics
    input_shape = (3, 64, 64)
    print(f"\nCalculating baseline statistics for input shape: {input_shape}")
    
    baseline_stats = calculate_model_stats(model, input_shape, device)
    print(f"Baseline Parameters: {baseline_stats['total_params']:,}")
    print(f"Baseline FLOPs: {baseline_stats['flops']:,}")
    
    # Test inference time (baseline)
    model.eval()
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure baseline inference time
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            output = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    baseline_time = (time.time() - start_time) / 100
    
    print(f"Baseline Inference Time: {baseline_time:.6f} s")
    print(f"Output shape: {output.shape}")
    
    # Apply structured pruning
    print(f"\n{'='*60}")
    print(" APPLYING STRUCTURED PRUNING")
    print(f"{'='*60}")
    
    pruning_ratio = 0.3  # 30% pruning
    print(f"Pruning ratio: {pruning_ratio:.1%}")
    
    # Clone model for pruning
    pruned_model = TestSwinIRLite(embed_dim=60, num_layers=4).to(device)
    pruned_model.load_state_dict(model.state_dict())
    
    # Apply pruning
    if TORCH_PRUNING_AVAILABLE:
        print("Using Torch-Pruning...")
        pruned_model = apply_structured_pruning_torch_pruning(pruned_model, pruning_ratio)
    else:
        print("Using PyTorch native structured pruning...")
        pruned_model = apply_structured_pruning_native(pruned_model, pruning_ratio)
    
    # Calculate pruned statistics
    pruned_stats = calculate_model_stats(pruned_model, input_shape, device)
    print(f"Pruned Parameters: {pruned_stats['total_params']:,}")
    print(f"Pruned FLOPs: {pruned_stats['flops']:,}")
    
    # Test pruned model inference
    pruned_model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = pruned_model(dummy_input)
    
    # Measure pruned inference time
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            pruned_output = pruned_model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    pruned_time = (time.time() - start_time) / 100
    
    print(f"Pruned Inference Time: {pruned_time:.6f} s")
    print(f"Pruned Output shape: {pruned_output.shape}")
    
    # Calculate output similarity (as a proxy for quality preservation)
    with torch.no_grad():
        mse = nn.MSELoss()(output, pruned_output).item()
        psnr_proxy = -10 * np.log10(mse) if mse > 0 else float('inf')
    
    print(f"Output MSE: {mse:.8f}")
    print(f"Output PSNR (proxy): {psnr_proxy:.4f} dB")
    
    # Create results summary
    results = {
        'Parameters': {'baseline': baseline_stats['total_params'], 'pruned': pruned_stats['total_params']},
        'FLOPs': {'baseline': baseline_stats['flops'], 'pruned': pruned_stats['flops']},
        'Inference Time (s)': {'baseline': baseline_time, 'pruned': pruned_time},
        'Output PSNR (proxy)': {'baseline': float('inf'), 'pruned': psnr_proxy}
    }
    
    # Print results table
    print_results_table(results)
    
    print("\n?? Structured pruning test completed successfully!")
    print("="*80)
    
    return results

if __name__ == '__main__':
    try:
        results = test_structured_pruning()
        print("\n? All tests passed!")
        
        # Simple verification
        param_reduction = (1 - results['Parameters']['pruned'] / results['Parameters']['baseline']) * 100
        time_improvement = (1 - results['Inference Time (s)']['pruned'] / results['Inference Time (s)']['baseline']) * 100
        
        print(f"\n?? Summary:")
        print(f"   Parameter reduction: {param_reduction:.1f}%")
        print(f"   Speed improvement: {time_improvement:.1f}%")
        
        if param_reduction > 10:
            print("   ? Significant parameter reduction achieved")
        if time_improvement > 5:
            print("   ? Meaningful speed improvement achieved")
            
    except Exception as e:
        print(f"\n? Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)