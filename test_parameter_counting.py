#!/usr/bin/env python3
"""
Test Script for Validating Parameter Counting Accuracy
Tests if parameter counting reflects actual model size reduction after pruning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main classes
from main_train_swinir_structured_pruning_complete import (
    ImportanceMaskManager, StructuredPruner, IterativePruningPipeline
)

class SimpleMockSwinIRAttention(nn.Module):
    """Simplified mock SwinIR attention for testing"""
    def __init__(self, dim=96, num_heads=6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SimpleMockSwinIRMLP(nn.Module):
    """Simplified mock SwinIR MLP for testing"""
    def __init__(self, in_features=96, hidden_features=384):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class SimpleMockSwinIRBlock(nn.Module):
    """Simplified mock SwinIR block for testing"""
    def __init__(self, dim=96, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.attn = SimpleMockSwinIRAttention(dim, num_heads)
        self.mlp = SimpleMockSwinIRMLP(dim, int(dim * mlp_ratio))
        
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class SimpleMockSwinIR(nn.Module):
    """Simplified mock SwinIR model for testing"""
    def __init__(self, img_size=64, embed_dim=96, depths=[2, 2], num_heads=[6, 6]):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        
        # Create layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_blocks = nn.ModuleList()
            for _ in range(depths[i_layer]):
                block = SimpleMockSwinIRBlock(
                    dim=embed_dim * (2 ** i_layer), 
                    num_heads=num_heads[i_layer]
                )
                layer_blocks.append(block)
            self.layers.append(layer_blocks)
        
        # Simple head
        self.head = nn.Linear(embed_dim * (2 ** (self.num_layers - 1)), 3)
        
    def forward(self, x):
        # Simple forward pass
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C] for transformer processing
        
        # Process through layers
        for i_layer, layer in enumerate(self.layers):
            # Adjust dimensions for each layer
            current_dim = self.embed_dim * (2 ** i_layer)
            if x.shape[-1] != current_dim:
                if x.shape[-1] > current_dim:
                    x = x[:, :, :current_dim]
                else:
                    pad_size = current_dim - x.shape[-1]
                    x = F.pad(x, (0, pad_size))
            
            for block in layer:
                x = block(x)
        
        # Final projection and reshape back
        x = x.mean(dim=1)  # Global average pooling [B, C]
        if x.shape[-1] != self.embed_dim * (2 ** (self.num_layers - 1)):
            # Adjust for head input dimension
            target_dim = self.embed_dim * (2 ** (self.num_layers - 1))
            if x.shape[-1] > target_dim:
                x = x[:, :target_dim]
            else:
                x = F.pad(x, (0, target_dim - x.shape[-1]))
        
        x = self.head(x)  # [B, 3]
        x = x.unsqueeze(-1).unsqueeze(-1).expand(B, 3, H, W)  # Expand back to [B, 3, H, W]
        return x

class MockModel:
    """Mock model wrapper that mimics the structure expected by the pruning code"""
    def __init__(self):
        self.netG = SimpleMockSwinIR()
        self.device = 'cpu'
        
    def feed_data(self, data):
        pass
        
    def test(self):
        pass
        
    def current_visuals(self):
        return {'E': torch.randn(1, 3, 64, 64), 'H': torch.randn(1, 3, 64, 64)}
    
    def parameters(self):
        return self.netG.parameters()
    
    def named_parameters(self):
        return self.netG.named_parameters()

def analyze_model_parameters(model, name="Model"):
    """Comprehensive parameter analysis"""
    if hasattr(model, 'netG'):
        network = model.netG
    else:
        network = model
    
    total_params = 0
    effective_params = 0
    zero_params = 0
    
    layer_analysis = {}
    
    for param_name, param in network.named_parameters():
        if param.requires_grad:
            param_total = param.numel()
            param_nonzero = torch.count_nonzero(param).item()
            param_zero = param_total - param_nonzero
            
            total_params += param_total
            effective_params += param_nonzero
            zero_params += param_zero
            
            sparsity = param_zero / param_total if param_total > 0 else 0
            layer_analysis[param_name] = {
                'total': param_total,
                'effective': param_nonzero,
                'zero': param_zero,
                'sparsity': sparsity,
                'shape': list(param.shape)
            }
    
    analysis = {
        'total_params': total_params,
        'effective_params': effective_params,
        'zero_params': zero_params,
        'sparsity': zero_params / total_params if total_params > 0 else 0,
        'size_reduction': zero_params / total_params if total_params > 0 else 0,
        'layer_analysis': layer_analysis
    }
    
    print(f"\n?? {name} Parameter Analysis:")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Effective parameters: {effective_params:,}")
    print(f"   Zero parameters:      {zero_params:,}")
    print(f"   Overall sparsity:     {analysis['sparsity']:.2%}")
    print(f"   Size reduction:       {analysis['size_reduction']:.2%}")
    
    return analysis

def test_parameter_counting_methods(model):
    """Test different parameter counting methods"""
    print("\n?? Testing Parameter Counting Methods:")
    
    # Method 1: Total parameter count (traditional - WRONG for pruned models)
    if hasattr(model, 'netG'):
        network = model.netG
    else:
        network = model
    
    total_params_traditional = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    # Method 2: Effective parameter count (correct for pruned models)
    effective_params_correct = sum(torch.count_nonzero(p).item() for p in network.parameters() if p.requires_grad)
    
    # Method 3: Using our fixed counting methods
    mask_manager = ImportanceMaskManager(model)
    mask_manager.initialize_masks()
    pruner = StructuredPruner(model, mask_manager)
    
    method3_count = pruner._count_parameters()
    
    print(f"   Method 1 (Traditional): {total_params_traditional:,} - WRONG for pruned models")
    print(f"   Method 2 (Effective):   {effective_params_correct:,} - CORRECT")
    print(f"   Method 3 (Our Fixed):   {method3_count:,} - SHOULD MATCH Method 2")
    
    # Validation
    method2_matches_method3 = effective_params_correct == method3_count
    
    print(f"   ? Method 2 == Method 3: {method2_matches_method3}")
    
    return {
        'traditional': total_params_traditional,
        'effective': effective_params_correct,
        'our_method': method3_count,
        'methods_match': method2_matches_method3
    }

def simulate_manual_pruning(model):
    """Manually zero some weights to test counting"""
    print("\n?? Simulating Manual Pruning:")
    
    if hasattr(model, 'netG'):
        network = model.netG
    else:
        network = model
    
    # Manually zero some weights in first layer to test counting
    pruned_count = 0
    for name, param in network.named_parameters():
        if 'layers.0.0' in name and 'weight' in name:
            with torch.no_grad():
                # Zero out 25% of weights
                flat_param = param.view(-1)
                num_to_zero = len(flat_param) // 4
                indices_to_zero = torch.randperm(len(flat_param))[:num_to_zero]
                flat_param[indices_to_zero] = 0.0
                pruned_count += num_to_zero
                print(f"   Manually zeroed {num_to_zero} weights in {name}")
    
    return pruned_count

def test_parameter_counting_accuracy():
    """Main test function for parameter counting accuracy"""
    print("="*70)
    print("PARAMETER COUNTING ACCURACY TEST")
    print("="*70)
    
    # 1. Create original model
    print("1. Creating original mock SwinIR model...")
    model = MockModel()
    original_model = copy.deepcopy(model)
    
    # 2. Analyze original model
    original_analysis = analyze_model_parameters(original_model, "Original")
    
    # 3. Test counting methods on original model
    original_counting = test_parameter_counting_methods(original_model)
    
    # 4. Simulate manual pruning
    manual_pruned = simulate_manual_pruning(model)
    
    # 5. Analyze manually pruned model
    manual_analysis = analyze_model_parameters(model, "Manually Pruned")
    
    # 6. Test counting methods on manually pruned model  
    manual_counting = test_parameter_counting_methods(model)
    
    # 7. Apply structured pruning
    print("\n?? Applying Structured Pruning:")
    mask_manager = ImportanceMaskManager(model)
    mask_manager.initialize_masks()
    
    # Generate importance scores
    for name in mask_manager.attention_masks.keys():
        num_heads = len(mask_manager.attention_masks[name])
        importance = torch.randn(num_heads)
        importance[:num_heads//2] += 2.0  # Make first half more important
        mask_manager.importance_scores[name] = importance
    
    for name in mask_manager.channel_masks.keys():
        num_channels = len(mask_manager.channel_masks[name])
        importance = torch.randn(num_channels)
        importance[:num_channels//2] += 1.5  # Make first half more important
        mask_manager.importance_scores[name] = importance
    
    # Apply structured pruning
    pruner = StructuredPruner(model, mask_manager)
    pruning_plan = pruner.create_pruning_plan(0.3, 0.5)  # 30% pruning
    actual_reduction = pruner.apply_pruning_plan(pruning_plan)
    
    # 8. Analyze structured pruned model
    structured_analysis = analyze_model_parameters(model, "Structured Pruned")
    
    # 9. Test counting methods on structured pruned model
    structured_counting = test_parameter_counting_methods(model)
    
    # 10. Comprehensive validation
    print("\n" + "="*70)
    print("PARAMETER COUNTING VALIDATION")
    print("="*70)
    
    # Test 1: Method consistency
    original_consistent = original_counting['methods_match']
    manual_consistent = manual_counting['methods_match']
    structured_consistent = structured_counting['methods_match']
    
    print("Test 1: Method Consistency")
    print(f"   Original model:     {'? PASS' if original_consistent else '? FAIL'}")
    print(f"   Manually pruned:    {'? PASS' if manual_consistent else '? FAIL'}")
    print(f"   Structured pruned:  {'? PASS' if structured_consistent else '? FAIL'}")
    
    # Test 2: Size reduction detection
    original_size = original_analysis['effective_params']
    manual_size = manual_analysis['effective_params']
    structured_size = structured_analysis['effective_params']
    
    manual_reduction = (original_size - manual_size) / original_size
    structured_reduction = (original_size - structured_size) / original_size
    
    manual_detected = manual_reduction > 0.01  # At least 1% reduction
    structured_detected = structured_reduction > 0.1  # At least 10% reduction
    
    print("\nTest 2: Size Reduction Detection")
    print(f"   Manual pruning reduction:     {manual_reduction:.2%} ({'? PASS' if manual_detected else '? FAIL'})")
    print(f"   Structured pruning reduction: {structured_reduction:.2%} ({'? PASS' if structured_detected else '? FAIL'})")
    
    # Test 3: Traditional vs Effective counting divergence
    original_traditional = original_counting['traditional']
    original_effective = original_counting['effective']
    
    manual_traditional = manual_counting['traditional']
    manual_effective = manual_counting['effective']
    
    structured_traditional = structured_counting['traditional']
    structured_effective = structured_counting['effective']
    
    # Original should have traditional == effective
    original_match = original_traditional == original_effective
    
    # Pruned should have traditional > effective
    manual_diverge = manual_traditional > manual_effective
    structured_diverge = structured_traditional > structured_effective
    
    print("\nTest 3: Traditional vs Effective Counting")
    print(f"   Original (should match):      {original_traditional:,} == {original_effective:,} ({'? PASS' if original_match else '? FAIL'})")
    print(f"   Manual (should diverge):      {manual_traditional:,} > {manual_effective:,} ({'? PASS' if manual_diverge else '? FAIL'})")
    print(f"   Structured (should diverge):  {structured_traditional:,} > {structured_effective:,} ({'? PASS' if structured_diverge else '? FAIL'})")
    
    # Test 4: Model size estimation accuracy
    original_size_mb = original_size * 4 / 1024 / 1024  # float32 = 4 bytes
    structured_size_mb = structured_size * 4 / 1024 / 1024
    
    size_reduction_mb = original_size_mb - structured_size_mb
    size_reduction_percent = size_reduction_mb / original_size_mb
    
    size_meaningful = size_reduction_mb > 0.1  # At least 0.1 MB reduction
    
    print("\nTest 4: Model Size Estimation")
    print(f"   Original model size:    {original_size_mb:.2f} MB")
    print(f"   Pruned model size:      {structured_size_mb:.2f} MB")
    print(f"   Size reduction:         {size_reduction_mb:.2f} MB ({size_reduction_percent:.1%})")
    print(f"   Meaningful reduction:   {'? PASS' if size_meaningful else '? FAIL'}")
    
    # Overall assessment
    all_tests_passed = (
        original_consistent and manual_consistent and structured_consistent and
        manual_detected and structured_detected and
        original_match and manual_diverge and structured_diverge and
        size_meaningful
    )
    
    print(f"\n{'='*70}")
    if all_tests_passed:
        print("?? OVERALL RESULT: ? PARAMETER COUNTING IS ACCURATE!")
        print("? Parameter counting correctly reflects actual model size reduction")
    else:
        print("? OVERALL RESULT: ? PARAMETER COUNTING ISSUES DETECTED!")
        print("?? Parameter counting needs improvements")
    print(f"{'='*70}")
    
    return all_tests_passed

if __name__ == "__main__":
    success = test_parameter_counting_accuracy()
    sys.exit(0 if success else 1)
