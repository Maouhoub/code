#!/usr/bin/env python3
"""
Validation Script for Optimized Structured Pruning
Tests: Aggressive pruning (65% target), Enhanced KD, Better importance metrics
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from models.select_model import define_Model
from data.select_dataset import define_Dataset
from torch.utils.data import DataLoader
from utils import utils_option as option
from utils import utils_image as util

# Import our optimized structured pruning components
from main_train_swinir_structured_pruning_complete import (
    ImportanceMaskManager, 
    StructuredPruner, 
    KnowledgeDistillationTrainer,
    IterativePruningPipeline,
    ComprehensiveEvaluator
)

def create_test_model():
    """Create a test SwinIR model"""
    # Use minimal config for fast testing
    opt = {
        'model': 'swinir',
        'netG': {
            'upscale': 2,
            'in_chans': 3,
            'img_size': 64,
            'window_size': 8,
            'img_range': 1.0,
            'depths': [4, 4],  # Reduced for testing
            'embed_dim': 96,
            'num_heads': [6, 6],
            'mlp_ratio': 2,
            'upsampler': 'pixelshuffle',
            'resi_connection': '1conv'
        },
        'path': {'pretrained_netG': None},
        'is_train': True
    }
    
    from utils import utils_option as option
    opt = option.dict_to_nonedict(opt)
    
    model = define_Model(opt)
    model.init_train()
    return model

def create_test_data():
    """Create minimal test data"""
    # Create synthetic test data
    test_data = []
    for i in range(3):  # Small test set
        L = torch.randn(1, 3, 32, 32)  # Low resolution
        H = torch.randn(1, 3, 64, 64)  # High resolution (2x)
        test_data.append({
            'L': L,
            'H': H,
            'L_path': [f'test_{i}.png']
        })
    return test_data

def test_importance_metrics():
    """Test enhanced importance metrics (gradient + magnitude based)"""
    print("?? Testing Enhanced Importance Metrics...")
    
    model = create_test_model()
    mask_manager = ImportanceMaskManager(model)
    
    # Initialize masks
    success = mask_manager.initialize_masks()
    print(f"  ? Mask initialization: {'Success' if success else 'Failed'}")
    
    # Test with synthetic activations and layer modules
    test_activations = {}
    layer_modules = {}
    
    # Create test data for attention layers
    for name in list(mask_manager.attention_masks.keys())[:2]:  # Test first 2
        num_heads = len(mask_manager.attention_masks[name])
        # Synthetic attention weights: [batch, heads, seq_len, seq_len]
        attention_weights = torch.randn(2, num_heads, 16, 16)
        test_activations[name] = attention_weights
        
        # Create mock layer module with QKV weights
        class MockAttentionLayer:
            def __init__(self, num_heads):
                self.num_heads = num_heads
                embed_dim = 96
                self.qkv = torch.nn.Linear(embed_dim, 3 * embed_dim)
                
        layer_modules[name] = MockAttentionLayer(num_heads)
    
    # Create test data for MLP layers  
    for name in list(mask_manager.channel_masks.keys())[:2]:  # Test first 2
        num_channels = len(mask_manager.channel_masks[name])
        # Synthetic MLP activations: [batch, seq_len, channels]
        mlp_activations = torch.randn(2, 16, num_channels)
        test_activations[name] = mlp_activations
        
        # Create mock layer module with weights
        class MockMLPLayer:
            def __init__(self, num_channels):
                self.weight = torch.randn(num_channels, 96)
                
        layer_modules[name] = MockMLPLayer(num_channels)
    
    # Test importance computation with layer modules
    mask_manager.update_importance_scores(test_activations, layer_modules)
    
    print(f"  ? Importance scores computed for {len(mask_manager.importance_scores)} layers")
    
    # Verify importance scores are reasonable
    for name, importance in mask_manager.importance_scores.items():
        if torch.any(torch.isnan(importance)) or torch.any(torch.isinf(importance)):
            print(f"  ? Invalid importance scores in {name}")
            return False
        print(f"  ? {name}: importance range [{importance.min():.4f}, {importance.max():.4f}]")
    
    return True

def test_aggressive_pruning():
    """Test aggressive pruning (60-70% target)"""
    print("?? Testing Aggressive Pruning...")
    
    model = create_test_model()
    mask_manager = ImportanceMaskManager(model)
    mask_manager.initialize_masks()
    
    # Add some importance scores
    for name in mask_manager.attention_masks.keys():
        importance = torch.rand(len(mask_manager.attention_masks[name]))
        mask_manager.importance_scores[name] = importance
        
    for name in mask_manager.channel_masks.keys():
        importance = torch.rand(len(mask_manager.channel_masks[name]))
        mask_manager.importance_scores[name] = importance
    
    pruner = StructuredPruner(model, mask_manager)
    
    # Test aggressive pruning plan
    aggressive_target = 0.65  # 65% target
    plan = pruner.create_pruning_plan(aggressive_target, importance_threshold=0.3)
    
    print(f"  ? Created aggressive pruning plan with target {aggressive_target:.1%}")
    print(f"  ? Attention heads to prune: {len(plan['attention_heads'])} layers")
    print(f"  ? MLP channels to prune: {len(plan['mlp_channels'])} layers")
    print(f"  ? Estimated reduction: {plan['estimated_reduction']:.1%}")
    
    # Count parameters before pruning
    params_before = pruner._count_parameters()
    
    # Apply pruning
    actual_reduction = pruner.apply_pruning_plan(plan)
    
    # Count parameters after pruning
    params_after = pruner._count_parameters()
    
    print(f"  ? Parameters before: {params_before:,}")
    print(f"  ? Parameters after: {params_after:,}")
    print(f"  ? Actual reduction: {actual_reduction:.1%}")
    
    # Verify aggressive pruning worked
    if actual_reduction > 0.20:  # At least 20% reduction achieved
        print(f"  ? Aggressive pruning successful: {actual_reduction:.1%} reduction")
        return True
    else:
        print(f"  ? Aggressive pruning insufficient: only {actual_reduction:.1%} reduction")
        return False

def test_enhanced_knowledge_distillation():
    """Test enhanced KD with feature distillation"""
    print("?? Testing Enhanced Knowledge Distillation...")
    
    model = create_test_model()
    teacher_model = create_test_model()
    
    # Create KD trainer with enhanced settings
    kd_trainer = KnowledgeDistillationTrainer(
        teacher_model=teacher_model,
        student_model=model,
        temperature=6.0,  # Higher temperature
        alpha=0.6         # Balanced loss
    )
    
    print(f"  ? KD trainer initialized with T={kd_trainer.temperature}, ?={kd_trainer.alpha}")
    print(f"  ? Feature hooks registered: {len(kd_trainer.teacher_hooks)} teacher + {len(kd_trainer.student_hooks)} student")
    
    # Test forward pass with feature extraction
    device = next(model.netG.parameters()).device
    test_input = torch.randn(1, 3, 32, 32).to(device)
    test_target = torch.randn(1, 3, 64, 64).to(device)
    
    # Clear previous features
    kd_trainer.teacher_features.clear()
    kd_trainer.student_features.clear()
    
    # Forward pass through both models
    with torch.no_grad():
        teacher_output = teacher_model.netG(test_input)
    
    student_output = model.netG(test_input)
    
    # Test enhanced distillation loss
    total_loss, hard_loss, output_distill_loss, feature_loss = kd_trainer.distillation_loss(
        student_output, teacher_output, test_target, F.mse_loss
    )
    
    print(f"  ? Loss computation successful:")
    print(f"    - Hard loss: {hard_loss.item():.6f}")
    print(f"    - Output distillation: {output_distill_loss.item():.6f}")
    print(f"    - Feature distillation: {feature_loss.item():.6f}")
    print(f"    - Total loss: {total_loss.item():.6f}")
    
    # Test feature extraction
    print(f"  ? Features extracted: {len(kd_trainer.teacher_features)} teacher, {len(kd_trainer.student_features)} student")
    
    # Cleanup
    kd_trainer.cleanup_hooks()
    print(f"  ? Hooks cleaned up")
    
    # Verify losses are reasonable
    if total_loss.item() > 0 and not torch.isnan(total_loss) and not torch.isinf(total_loss):
        print("  ? Enhanced KD working correctly")
        return True
    else:
        print("  ? Enhanced KD has issues")
        return False

def test_full_pipeline():
    """Test the complete optimized pipeline with small data"""
    print("?? Testing Complete Optimized Pipeline...")
    
    model = create_test_model()
    test_data = create_test_data()
    
    # Create data loader
    train_loader = test_data  # Use as simple list for testing
    
    # Aggressive pipeline configuration
    config = {
        'target_ratio': 0.65,        # Aggressive 65% target
        'num_iterations': 2,         # Reduced for testing
        'schedule_type': 'exponential',
        'fine_tune_epochs': 3,       # Reduced for testing
        'patience': 2
    }
    
    print(f"  ? Pipeline config: {config['target_ratio']:.1%} target, {config['num_iterations']} iterations")
    
    # Initialize pipeline
    pipeline = IterativePruningPipeline(model, config)
    
    # Count initial parameters
    initial_params = pipeline._count_parameters(model)
    print(f"  ? Initial parameters: {initial_params:,}")
    
    # Test mask initialization
    mask_count = len(pipeline.mask_manager.attention_masks) + len(pipeline.mask_manager.channel_masks)
    print(f"  ? Masks initialized: {mask_count} layers")
    
    if mask_count == 0:
        print("  ? No masks found - pipeline cannot proceed")
        return False
    
    # Test importance collection (simplified)
    print("  ? Testing importance collection...")
    test_activations = {}
    
    # Add synthetic activations for mask layers
    for name in pipeline.mask_manager.attention_masks.keys():
        num_heads = len(pipeline.mask_manager.attention_masks[name])
        test_activations[name] = torch.randn(1, num_heads, 16, 16)
        
    for name in pipeline.mask_manager.channel_masks.keys():
        num_channels = len(pipeline.mask_manager.channel_masks[name])
        test_activations[name] = torch.randn(1, 16, num_channels)
    
    pipeline.mask_manager.update_importance_scores(test_activations)
    print(f"  ? Importance scores updated for {len(pipeline.mask_manager.importance_scores)} layers")
    
    # Test pruning plan creation
    plan = pipeline.pruner.create_pruning_plan(config['target_ratio'])
    print(f"  ? Pruning plan created with {plan['estimated_reduction']:.1%} estimated reduction")
    
    # Test pruning application
    actual_reduction = pipeline.pruner.apply_pruning_plan(plan)
    final_params = pipeline._count_parameters(model)
    
    print(f"  ? Pruning applied:")
    print(f"    - Final parameters: {final_params:,}")
    print(f"    - Actual reduction: {actual_reduction:.1%}")
    print(f"    - Parameter ratio: {final_params/initial_params:.3f}")
    
    # Verify significant pruning occurred
    if actual_reduction > 0.15:  # At least 15% reduction
        print("  ? Complete pipeline test successful")
        return True
    else:
        print(f"  ? Insufficient pruning: only {actual_reduction:.1%}")
        return False

def main():
    """Run all validation tests"""
    print("="*70)
    print("VALIDATION: OPTIMIZED STRUCTURED PRUNING")
    print("="*70)
    print("Testing improvements:")
    print("  1. Enhanced importance metrics (gradient + magnitude)")
    print("  2. Aggressive pruning (65% target)")
    print("  3. Enhanced knowledge distillation (feature distillation)")
    print("="*70)
    
    results = {}
    
    # Test 1: Enhanced importance metrics
    try:
        results['importance_metrics'] = test_importance_metrics()
    except Exception as e:
        print(f"  ? Importance metrics test failed: {e}")
        results['importance_metrics'] = False
    
    print()
    
    # Test 2: Aggressive pruning
    try:
        results['aggressive_pruning'] = test_aggressive_pruning()
    except Exception as e:
        print(f"  ? Aggressive pruning test failed: {e}")
        results['aggressive_pruning'] = False
    
    print()
    
    # Test 3: Enhanced knowledge distillation
    try:
        results['enhanced_kd'] = test_enhanced_knowledge_distillation()
    except Exception as e:
        print(f"  ? Enhanced KD test failed: {e}")
        results['enhanced_kd'] = False
    
    print()
    
    # Test 4: Complete pipeline
    try:
        results['full_pipeline'] = test_full_pipeline()
    except Exception as e:
        print(f"  ? Full pipeline test failed: {e}")
        results['full_pipeline'] = False
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "? PASS" if passed else "? FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("?? All optimizations working correctly!")
        print("\nNext steps:")
        print("  1. Run full training with: python main_train_swinir_structured_pruning_complete.py")
        print("  2. Monitor for 40%+ parameter reduction with <0.5dB PSNR drop")
        print("  3. Check speedup improvements")
    else:
        print("??  Some optimizations need attention")
        print("\nFailed tests require debugging before full training")
    
    print("="*70)

if __name__ == "__main__":
    main()
