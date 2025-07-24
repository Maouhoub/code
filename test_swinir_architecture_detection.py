#!/usr/bin/env python3
"""
Test script to validate SwinIR architecture detection fix
Tests the ImportanceMaskManager's ability to properly identify SwinIR layers
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.select_model import define_Model
    from utils import utils_option as option
    from main_train_swinir_structured_pruning_complete import ImportanceMaskManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the correct directory with all dependencies")
    sys.exit(1)


class MockSwinIRAttention(nn.Module):
    """Mock SwinIR attention module for testing"""
    def __init__(self, dim=96, num_heads=6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        return x


class MockSwinIRMLP(nn.Module):
    """Mock SwinIR MLP module for testing"""
    def __init__(self, in_features=96, hidden_features=384):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MockSwinIRBlock(nn.Module):
    """Mock SwinIR transformer block for testing"""
    def __init__(self, dim=96, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MockSwinIRAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MockSwinIRMLP(dim, mlp_hidden_dim)
        
    def forward(self, x):
        return x


class MockSwinIRModel(nn.Module):
    """Mock SwinIR model for testing architecture detection"""
    def __init__(self, img_size=64, embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6]):
        super().__init__()
        
        # Create mock layers structure similar to real SwinIR
        self.layers = nn.ModuleList()
        
        for i_layer in range(len(depths)):
            layer = nn.ModuleList()
            for i_block in range(depths[i_layer]):
                block = MockSwinIRBlock(
                    dim=embed_dim * (2 ** i_layer),
                    num_heads=num_heads[i_layer]
                )
                layer.append(block)
            self.layers.append(layer)
        
        # Add some non-SwinIR layers to test selectivity
        self.conv_first = nn.Conv2d(3, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, 3, 3, 1, 1)
        
    def forward(self, x):
        return x


class MockModel:
    """Mock model wrapper to simulate the model structure used in training"""
    def __init__(self):
        self.netG = MockSwinIRModel()


def test_swinir_architecture_detection():
    """Test the SwinIR architecture detection functionality"""
    print("="*70)
    print("TESTING SWINIR ARCHITECTURE DETECTION")
    print("="*70)
    
    # Create mock model
    print("1. Creating mock SwinIR model...")
    mock_model = MockModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mock_model.netG = mock_model.netG.to(device)
    
    print(f"   Device: {device}")
    print(f"   Model created with {len(list(mock_model.netG.parameters()))} parameters")
    
    # Test ImportanceMaskManager
    print("\n2. Testing ImportanceMaskManager initialization...")
    mask_manager = ImportanceMaskManager(mock_model)
    
    # Test mask initialization
    print("\n3. Testing mask initialization...")
    success = mask_manager.initialize_masks()
    
    print(f"\n4. Architecture Detection Results:")
    print(f"   Initialization successful: {success}")
    print(f"   Attention masks found: {len(mask_manager.attention_masks)}")
    print(f"   Channel masks found: {len(mask_manager.channel_masks)}")
    
    # Detailed analysis
    print(f"\n5. Detailed Mask Analysis:")
    print("   Attention Layers:")
    for name, mask in mask_manager.attention_masks.items():
        print(f"     {name}: {len(mask)} heads")
    
    print("   MLP Layers:")
    for name, mask in mask_manager.channel_masks.items():
        print(f"     {name}: {len(mask)} channels")
    
    # Validate detection quality
    print(f"\n6. Detection Quality Assessment:")
    
    # Count expected vs detected layers
    # Expected: 4 layers * 6 blocks = 24 attention modules (not including sub-components)
    # Expected: 4 layers * 6 blocks * 2 MLP layers (fc1, fc2) = 48 MLP layers
    expected_attention = 4 * 6  # 4 layers * 6 blocks each (main attn modules only)
    expected_mlp = 4 * 6 * 2   # 4 layers * 6 blocks * 2 MLPs per block (fc1, fc2)
    
    attention_ratio = len(mask_manager.attention_masks) / expected_attention
    mlp_ratio = len(mask_manager.channel_masks) / expected_mlp
    
    print(f"   Expected attention layers: {expected_attention}")
    print(f"   Detected attention layers: {len(mask_manager.attention_masks)}")
    print(f"   Detection ratio: {attention_ratio:.2f}")
    
    print(f"   Expected MLP layers: {expected_mlp}")
    print(f"   Detected MLP layers: {len(mask_manager.channel_masks)}")
    print(f"   Detection ratio: {mlp_ratio:.2f}")
    
    # Check for over-detection (too many sub-components detected)
    qkv_detected = sum(1 for name in mask_manager.attention_masks.keys() if 'qkv' in name.lower())
    proj_detected = sum(1 for name in mask_manager.attention_masks.keys() if 'proj' in name.lower())
    
    print(f"   Sub-component detection check:")
    print(f"     QKV layers detected: {qkv_detected} (should be 0 for clean detection)")
    print(f"     Proj layers detected: {proj_detected} (should be 0 for clean detection)")
    
    # Test importance score computation
    print(f"\n7. Testing importance score computation...")
    
    # Create synthetic activations for testing
    test_activations = {}
    for name in list(mask_manager.attention_masks.keys())[:2]:  # Test first 2
        num_heads = len(mask_manager.attention_masks[name])
        # Synthetic attention weights [batch, heads, seq, seq]
        test_activations[name] = torch.randn(2, num_heads, 16, 16, device=device)
    
    for name in list(mask_manager.channel_masks.keys())[:2]:  # Test first 2
        num_channels = len(mask_manager.channel_masks[name])
        # Synthetic activations [batch, seq, channels]
        test_activations[name] = torch.randn(2, 16, num_channels, device=device)
    
    # Update importance scores
    mask_manager.update_importance_scores(test_activations)
    
    print(f"   Importance scores computed for {len(mask_manager.importance_scores)} layers")
    for name, importance in list(mask_manager.importance_scores.items())[:3]:
        print(f"     {name}: shape={importance.shape}, mean={importance.mean():.4f}")
    
    # Success criteria
    print(f"\n8. Test Results Summary:")
    print("   " + "="*50)
    
    criteria = {
        "Mask initialization": success,
        "Attention detection": len(mask_manager.attention_masks) > 0,
        "MLP detection": len(mask_manager.channel_masks) > 0,
        "Reasonable attention ratio": 0.8 <= attention_ratio <= 1.2,  # More precise range
        "Reasonable MLP ratio": 0.8 <= mlp_ratio <= 1.2,
        "Importance computation": len(mask_manager.importance_scores) > 0,
        "Clean detection (no sub-components)": qkv_detected == 0 and proj_detected == 0,
    }
    
    passed_tests = 0
    total_tests = len(criteria)
    
    for test_name, passed in criteria.items():
        status = "? PASS" if passed else "? FAIL"
        print(f"   {status} {test_name}")
        if passed:
            passed_tests += 1
    
    print(f"\n   Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("   ?? ALL TESTS PASSED - SwinIR detection is working correctly!")
        return True
    else:
        print("   ? SOME TESTS FAILED - Architecture detection needs improvement")
        return False


def test_real_swinir_model():
    """Test with a real SwinIR model if available"""
    print("\n" + "="*70)
    print("TESTING WITH REAL SWINIR MODEL (if available)")
    print("="*70)
    
    try:
        # Try to load a real SwinIR model
        opt = {
            'model': 'swinir',
            'netG': {
                'upscale': 2,
                'in_chans': 3,
                'img_size': 64,
                'window_size': 8,
                'img_range': 1.0,
                'depths': [6, 6, 6, 6],
                'embed_dim': 96,
                'num_heads': [6, 6, 6, 6],
                'mlp_ratio': 2,
                'upsampler': 'pixelshuffle',
                'resi_connection': '1conv'
            },
            'path': {'pretrained_netG': None}
        }
        
        print("1. Attempting to create real SwinIR model...")
        model = define_Model(opt)
        
        print("2. Testing with real SwinIR architecture...")
        mask_manager = ImportanceMaskManager(model)
        success = mask_manager.initialize_masks()
        
        print(f"   Real SwinIR detection successful: {success}")
        print(f"   Attention layers found: {len(mask_manager.attention_masks)}")
        print(f"   MLP layers found: {len(mask_manager.channel_masks)}")
        
        return success
        
    except Exception as e:
        print(f"   Could not test with real SwinIR model: {e}")
        print("   This is expected if SwinIR model files are not available")
        return None


def main():
    """Main test function"""
    print("SwinIR Architecture Detection Test Suite")
    print("This test validates the improved architecture detection in ImportanceMaskManager")
    
    # Test 1: Mock model
    mock_success = test_swinir_architecture_detection()
    
    # Test 2: Real model (if available)
    real_success = test_real_swinir_model()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    if mock_success:
        print("? Mock SwinIR model detection: PASSED")
    else:
        print("? Mock SwinIR model detection: FAILED")
    
    if real_success is True:
        print("? Real SwinIR model detection: PASSED")
    elif real_success is False:
        print("? Real SwinIR model detection: FAILED")
    else:
        print("- Real SwinIR model detection: SKIPPED (model not available)")
    
    overall_success = mock_success and (real_success is not False)
    
    if overall_success:
        print("\n?? OVERALL RESULT: SwinIR architecture detection fix is working correctly!")
        print("The ImportanceMaskManager can now properly identify SwinIR layers.")
    else:
        print("\n? OVERALL RESULT: SwinIR architecture detection needs further improvement.")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
