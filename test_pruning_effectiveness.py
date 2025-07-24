#!/usr/bin/env python3
"""
Test Script for Validating Pruning Effectiveness
Tests if structural pruning actually zeros weights permanently and reduces parameters
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
        
        # Ensure we have the right sequence length
        seq_len = x.shape[1]
        
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
        """Expose the underlying network parameters"""
        return self.netG.parameters()
    
    def named_parameters(self):
        """Expose the underlying network named parameters"""
        return self.netG.named_parameters()

def count_zero_parameters(model):
    """Count the number of zero parameters in a model"""
    if hasattr(model, 'netG'):
        network = model.netG
    else:
        network = model
        
    total_params = 0
    zero_params = 0
    
    for name, param in network.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    
    return total_params, zero_params

def count_effective_parameters(model):
    """Count non-zero (effective) parameters"""
    if hasattr(model, 'netG'):
        network = model.netG
    else:
        network = model
        
    effective_params = 0
    for param in network.parameters():
        if param.requires_grad:
            effective_params += torch.count_nonzero(param).item()
    
    return effective_params

def analyze_weight_sparsity(model, layer_names=None):
    """Analyze sparsity in specific layers"""
    if hasattr(model, 'netG'):
        network = model.netG
    else:
        network = model
    
    sparsity_analysis = {}
    
    for name, param in network.named_parameters():
        if param.requires_grad:
            total_weights = param.numel()
            zero_weights = (param == 0).sum().item()
            sparsity = zero_weights / total_weights if total_weights > 0 else 0
            
            if layer_names is None or any(layer_name in name for layer_name in layer_names):
                sparsity_analysis[name] = {
                    'total_weights': total_weights,
                    'zero_weights': zero_weights,
                    'sparsity': sparsity,
                    'shape': list(param.shape)
                }
    
    return sparsity_analysis

def test_pruning_effectiveness():
    """Main test function for pruning effectiveness"""
    print("="*70)
    print("PRUNING EFFECTIVENESS TEST")
    print("="*70)
    
    # 1. Create mock model
    print("1. Creating mock SwinIR model...")
    model = MockModel()
    original_model = copy.deepcopy(model)
    
    # Count initial parameters
    total_params_before, zero_params_before = count_zero_parameters(model)
    effective_params_before = count_effective_parameters(model)
    
    print(f"   Total parameters: {total_params_before:,}")
    print(f"   Zero parameters: {zero_params_before:,}")
    print(f"   Effective parameters: {effective_params_before:,}")
    print(f"   Initial sparsity: {zero_params_before/total_params_before*100:.2f}%")
    
    # 2. Initialize mask manager and pruner
    print("\n2. Initializing pruning components...")
    mask_manager = ImportanceMaskManager(model)
    mask_init_success = mask_manager.initialize_masks()
    
    if not mask_init_success:
        print("   ? ERROR: Mask initialization failed!")
        return False
    
    print(f"   ? Found {len(mask_manager.attention_masks)} attention layers")
    print(f"   ? Found {len(mask_manager.channel_masks)} MLP layers")
    
    # 3. Create synthetic importance scores
    print("\n3. Creating synthetic importance scores...")
    
    # Generate realistic importance scores with clear differences
    for name in mask_manager.attention_masks.keys():
        num_heads = len(mask_manager.attention_masks[name])
        # Create importance scores where some heads are clearly less important
        importance = torch.randn(num_heads)
        importance[:num_heads//2] += 2.0  # Make first half more important
        mask_manager.importance_scores[name] = importance
        print(f"   Attention {name}: importance range [{importance.min():.2f}, {importance.max():.2f}]")
    
    for name in mask_manager.channel_masks.keys():
        num_channels = len(mask_manager.channel_masks[name])
        # Create importance scores where some channels are clearly less important
        importance = torch.randn(num_channels)
        importance[:num_channels//2] += 1.5  # Make first half more important
        mask_manager.importance_scores[name] = importance
        print(f"   MLP {name}: importance range [{importance.min():.2f}, {importance.max():.2f}]")
    
    # 4. Create pruner and apply pruning
    print("\n4. Creating pruner and applying structured pruning...")
    pruner = StructuredPruner(model, mask_manager)
    
    # Create pruning plan with moderate aggressiveness
    target_ratio = 0.3  # 30% pruning
    importance_threshold = 0.5
    
    print(f"   Target pruning ratio: {target_ratio:.1%}")
    print(f"   Importance threshold: {importance_threshold}")
    
    pruning_plan = pruner.create_pruning_plan(target_ratio, importance_threshold)
    
    print("\n   Pruning plan summary:")
    print(f"   - Attention heads to prune: {len(pruning_plan['attention_heads'])}")
    print(f"   - MLP channels to prune: {len(pruning_plan['mlp_channels'])}")
    
    for layer_name, heads_to_prune in pruning_plan['attention_heads'].items():
        print(f"     {layer_name}: {heads_to_prune} heads")
    
    for layer_name, channels_to_prune in pruning_plan['mlp_channels'].items():
        print(f"     {layer_name}: {channels_to_prune} channels")
    
    # 5. Apply pruning and measure effectiveness
    print("\n5. Applying pruning and measuring effectiveness...")
    
    # Get weight analysis before pruning
    sparsity_before = analyze_weight_sparsity(model)
    
    # Apply the pruning plan
    actual_reduction = pruner.apply_pruning_plan(pruning_plan)
    
    # Get weight analysis after pruning
    sparsity_after = analyze_weight_sparsity(model)
    
    # Count parameters after pruning
    total_params_after, zero_params_after = count_zero_parameters(model)
    effective_params_after = count_effective_parameters(model)
    
    print(f"\n   ?? PRUNING RESULTS:")
    print(f"   Total parameters: {total_params_before:,} ? {total_params_after:,}")
    print(f"   Zero parameters: {zero_params_before:,} ? {zero_params_after:,}")
    print(f"   Effective parameters: {effective_params_before:,} ? {effective_params_after:,}")
    print(f"   Sparsity: {zero_params_before/total_params_before*100:.2f}% ? {zero_params_after/total_params_after*100:.2f}%")
    print(f"   Parameter reduction: {(effective_params_before - effective_params_after)/effective_params_before*100:.2f}%")
    
    # 6. Detailed layer analysis
    print("\n6. Detailed layer analysis...")
    
    # Find layers that should have been pruned
    pruned_layers = []
    for layer_name in pruning_plan['attention_heads'].keys():
        if pruning_plan['attention_heads'][layer_name] > 0:
            pruned_layers.append(layer_name)
    
    for layer_name in pruning_plan['mlp_channels'].keys():
        if pruning_plan['mlp_channels'][layer_name] > 0:
            pruned_layers.append(layer_name)
    
    effective_pruning_detected = False
    
    for layer_name in pruned_layers[:5]:  # Check first 5 pruned layers
        # Find corresponding weight parameters
        relevant_params = []
        for param_name in sparsity_after.keys():
            if any(part in param_name for part in layer_name.split('.')):
                relevant_params.append(param_name)
        
        for param_name in relevant_params:
            before = sparsity_before.get(param_name, {})
            after = sparsity_after.get(param_name, {})
            
            if before and after:
                sparsity_increase = after['sparsity'] - before['sparsity']
                if sparsity_increase > 0.01:  # At least 1% increase in sparsity
                    print(f"   ? {param_name}: sparsity {before['sparsity']:.3f} ? {after['sparsity']:.3f} (+{sparsity_increase:.3f})")
                    effective_pruning_detected = True
                elif sparsity_increase > 0:
                    print(f"   ~ {param_name}: sparsity {before['sparsity']:.3f} ? {after['sparsity']:.3f} (+{sparsity_increase:.3f})")
                else:
                    print(f"   ? {param_name}: no sparsity change")
    
    # 7. Forward pass test
    print("\n7. Testing forward pass after pruning...")
    
    try:
        # Test that the model still works
        test_input = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            output_before = original_model.netG(test_input)
            output_after = model.netG(test_input)
        
        output_difference = torch.norm(output_after - output_before).item()
        print(f"   ? Forward pass successful")
        print(f"   Output difference: {output_difference:.6f}")
        
        forward_pass_success = True
        
    except Exception as e:
        print(f"   ? Forward pass failed: {e}")
        forward_pass_success = False
    
    # 8. Final assessment
    print("\n" + "="*70)
    print("PRUNING EFFECTIVENESS ASSESSMENT")
    print("="*70)
    
    # Check effectiveness criteria
    parameter_reduction_achieved = (effective_params_before - effective_params_after) / effective_params_before
    sparsity_increase = (zero_params_after - zero_params_before) / total_params_after
    
    criteria_results = {
        'parameter_reduction': {
            'value': parameter_reduction_achieved,
            'target': 0.1,  # At least 10% reduction
            'passed': parameter_reduction_achieved >= 0.1
        },
        'sparsity_increase': {
            'value': sparsity_increase,
            'target': 0.05,  # At least 5% increase in sparsity
            'passed': sparsity_increase >= 0.05
        },
        'effective_pruning': {
            'value': effective_pruning_detected,
            'target': True,
            'passed': effective_pruning_detected
        },
        'forward_pass': {
            'value': forward_pass_success,
            'target': True,
            'passed': forward_pass_success
        }
    }
    
    print("Test Results:")
    for criterion, result in criteria_results.items():
        status = "? PASS" if result['passed'] else "? FAIL"
        print(f"  {criterion:20}: {status} ({result['value']} vs target {result['target']})")
    
    # Overall result
    all_passed = all(result['passed'] for result in criteria_results.values())
    
    print(f"\n{'='*70}")
    if all_passed:
        print("?? OVERALL RESULT: ? PRUNING IS EFFECTIVE!")
        print("? Structural pruning successfully reduces parameters permanently")
    else:
        print("? OVERALL RESULT: ? PRUNING ISSUES DETECTED!")
        print("?? Structural pruning needs improvements")
    
    print(f"{'='*70}")
    
    return all_passed

if __name__ == "__main__":
    success = test_pruning_effectiveness()
    sys.exit(0 if success else 1)
