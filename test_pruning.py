#!/usr/bin/env python3
"""
Test script for the structured pruning implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    import torch
    print("? PyTorch available")
except ImportError:
    print("? PyTorch not available")

try:
    import torch_pruning as tp
    print("? torch-pruning available")
    TORCH_PRUNING_AVAILABLE = True
except ImportError:
    print("? torch-pruning not available")
    TORCH_PRUNING_AVAILABLE = False

try:
    from fvcore.nn import flop_count
    print("? fvcore available")
    FVCORE_AVAILABLE = True
except ImportError:
    print("? fvcore not available")
    FVCORE_AVAILABLE = False

try:
    from thop import profile
    print("? thop available")
    THOP_AVAILABLE = True
except ImportError:
    print("? thop not available")
    THOP_AVAILABLE = False

# Test basic functionality
def test_basic_functions():
    print("\nTesting basic functions...")
    
    # Create a simple CNN model
    import torch.nn as nn
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
            self.conv4 = nn.Conv2d(64, 3, 3, padding=1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.conv4(x)
            return x
    
    model = SimpleCNN()
    
    # Test parameter counting
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    total, trainable = count_parameters(model)
    print(f"Model parameters: {total:,} (trainable: {trainable:,})")
    
    # Test FLOP counting
    if FVCORE_AVAILABLE:
        try:
            from fvcore.nn import flop_count
            dummy_input = torch.randn(1, 3, 64, 64)
            flops_dict = flop_count(model, (dummy_input,))
            total_flops = sum(flops_dict.values())
            print(f"FLOPs (fvcore): {total_flops/1e6:.2f}M")
        except Exception as e:
            print(f"FvCore FLOP counting failed: {e}")
    
    if THOP_AVAILABLE:
        try:
            from thop import profile
            import copy
            dummy_input = torch.randn(1, 3, 64, 64)
            model_copy = copy.deepcopy(model)
            flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
            print(f"FLOPs (thop): {flops/1e6:.2f}M, Params: {params/1e3:.1f}K")
        except Exception as e:
            print(f"THOP FLOP counting failed: {e}")
    
    # Test basic structured pruning
    print("\nTesting structured pruning...")
    import torch.nn.utils.prune as prune
    
    # Apply basic pruning
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=0.2, n=1, dim=0)
            print(f"Applied pruning to {name}")
    
    # Count parameters after pruning
    total_after, trainable_after = count_parameters(model)
    print(f"Parameters after pruning: {total_after:,} (trainable: {trainable_after:,})")
    
    # Remove pruning masks
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig'):
            prune.remove(module, 'weight')
            print(f"Removed pruning mask from {name}")
    
    print("? Basic functionality test completed!")

if __name__ == "__main__":
    print("Structured Pruning Test Script")
    print("=" * 40)
    test_basic_functions()