"""
SwinIR Structured Pruning Test Script
=====================================

This script tests the SwinIR-specific structured pruning functionality
without running the full training pipeline.
"""

import sys
import os
import torch
import torch.nn as nn

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_swinir_pruning():
    """Test SwinIR-specific structured pruning."""
    print("="*60)
    print("TESTING SWINIR STRUCTURED PRUNING")
    print("="*60)
    
    # Check environment
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test imports
    try:
        from swinir_pruning_utils import (
            swinir_structured_pruning, 
            identify_swinir_conv_layers,
            calculate_swinir_compression_stats
        )
        print("? SwinIR pruning utilities imported successfully")
    except ImportError as e:
        print(f"? Failed to import SwinIR pruning utilities: {e}")
        return False
    
    # Create a simple test model that mimics SwinIR structure
    class TestSwinIRModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Regular conv layers (should be prunable)
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
            self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
            
            # Attention layers (should be skipped)
            self.attn_conv = nn.Conv2d(128, 64, 1, 1, 0)  # Attention-related
            self.attention_qkv = nn.Conv2d(64, 192, 1, 1, 0)  # QKV projection
            
            # More regular layers
            self.conv3 = nn.Conv2d(128, 64, 3, 1, 1)
            self.conv_out = nn.Conv2d(64, 3, 3, 1, 1)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv_out(x)
            return x
    
    # Create test model
    print("\nCreating test model...")
    model = TestSwinIRModel()
    
    # Count parameters before pruning
    params_before = sum(p.numel() for p in model.parameters())
    print(f"Parameters before pruning: {params_before:,}")
    
    # Test layer identification
    print("\nTesting layer identification...")
    safe_layers = identify_swinir_conv_layers(model)
    print(f"Identified {len(safe_layers)} safe Conv2d layers:")
    for name, module in safe_layers:
        print(f"  - {name}: {module.out_channels} channels")
    
    # Test pruning
    print("\nApplying structured pruning...")
    try:
        model_pruned = swinir_structured_pruning(model, pruning_ratio=0.2)
        print("? Structured pruning completed successfully")
        
        # Count parameters after pruning
        params_after = sum(p.numel() for p in model_pruned.parameters())
        print(f"Parameters after pruning: {params_after:,}")
        
        # Calculate compression
        compression_ratio = (params_before - params_after) / params_before * 100
        print(f"Compression achieved: {compression_ratio:.2f}%")
        
        # Test model forward pass
        print("\nTesting forward pass...")
        with torch.no_grad():
            test_input = torch.randn(1, 3, 64, 64)
            if torch.cuda.is_available():
                test_input = test_input.cuda()
                model_pruned = model_pruned.cuda()
            
            output = model_pruned(test_input)
            print(f"? Forward pass successful. Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"? Pruning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_script_import():
    """Test if the main script can be imported without errors."""
    print("\n" + "="*60)
    print("TESTING MAIN SCRIPT IMPORT")
    print("="*60)
    
    try:
        # Test importing the main functions
        from main_train_psnr_L2_fine_tune_structured_enhanced import (
            apply_structured_pruning_native,
            calculate_model_stats
        )
        print("? Main script functions imported successfully")
        
        # Test creating a simple model and applying pruning
        test_model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1)
        )
        
        print("Testing native pruning function...")
        pruned_model = apply_structured_pruning_native(test_model, pruning_ratio=0.1)
        print("? Native pruning function works")
        
        return True
        
    except Exception as e:
        print(f"? Import/test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("SwinIR Structured Pruning Test Suite")
    print("=" * 60)
    
    # Test 1: SwinIR-specific pruning utilities
    test1_success = test_swinir_pruning()
    
    # Test 2: Main script functionality
    test2_success = test_main_script_import()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"SwinIR Pruning Utils: {'? PASS' if test1_success else '? FAIL'}")
    print(f"Main Script Import:   {'? PASS' if test2_success else '? FAIL'}")
    
    if test1_success and test2_success:
        print("\n?? All tests passed! The structured pruning setup is ready.")
        print("\nYou can now run:")
        print("python main_train_psnr_L2_fine_tune_structured_enhanced.py --opt options/swinir/train_swinir_sr_lightweight_structured_pruning.json")
    else:
        print("\n?? Some tests failed. Please check the error messages above.")
    
    print("="*60)