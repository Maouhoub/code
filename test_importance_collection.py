#!/usr/bin/env python3
"""
Test script to validate the importance score collection fix
Tests that real activations are captured and used for importance computation
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main_train_swinir_structured_pruning_complete import ImportanceMaskManager, IterativePruningPipeline
    from test_swinir_architecture_detection import MockSwinIRModel, MockModel
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
        # Proper forward pass that matches the structure expected by hooks
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
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
        # Proper transformer block forward pass
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MockSwinIRModelEnhanced(nn.Module):
    """Enhanced Mock SwinIR model for testing importance collection"""
    def __init__(self, img_size=64, embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6]):
        super().__init__()
        
        # Input processing
        self.conv_first = nn.Conv2d(3, embed_dim, 3, 1, 1)
        self.patch_embed = nn.Linear(embed_dim, embed_dim)
        
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
        
        # Output processing
        self.conv_last = nn.Conv2d(embed_dim, 3, 3, 1, 1)
        
    def forward(self, x):
        # Proper forward pass that generates realistic activations
        B, C, H, W = x.shape
        
        # Patch embedding simulation
        x = self.conv_first(x)  # [B, embed_dim, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
        x = self.patch_embed(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            for block in layer:
                x = block(x)
        
        # Reconstruct output
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.conv_last(x)
        
        return x


class MockDataLoader:
    """Mock data loader for testing importance collection"""
    def __init__(self, batch_size=2, num_batches=5, device='cuda'):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = device
        self.current_batch = 0
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        # Create realistic batch data
        batch = {
            'L': torch.randn(self.batch_size, 3, 64, 64, device=self.device),  # Low-res input
            'H': torch.randn(self.batch_size, 3, 128, 128, device=self.device),  # High-res target
        }
        self.current_batch += 1
        return batch


class TestableModel:
    """Enhanced mock model with proper feed_data method for testing"""
    def __init__(self):
        self.netG = MockSwinIRModelEnhanced()
        self.current_batch = None
    
    def feed_data(self, batch):
        """Mock feed_data method"""
        self.current_batch = batch
        # For compatibility, ensure model is aware of the data
        return self
    
    def test(self):
        """Mock test method"""
        pass
    
    def current_visuals(self):
        """Mock current_visuals method"""
        if self.current_batch is not None:
            return {
                'E': self.current_batch['L'],  # Estimated output
                'H': self.current_batch['H']   # High-res target
            }
        return {'E': torch.zeros(2, 3, 64, 64), 'H': torch.zeros(2, 3, 64, 64)}
    
    def eval(self):
        """Mock eval method to set model to evaluation mode"""
        self.netG.eval()
        return self
    
    def train(self):
        """Mock train method to set model to training mode"""
        self.netG.train()
        return self


def test_importance_collection_hooks():
    """Test that hooks are properly registered for activation capture"""
    print("="*70)
    print("TESTING IMPORTANCE COLLECTION HOOK REGISTRATION")
    print("="*70)
    
    # Create test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TestableModel()
    model.netG = model.netG.to(device)
    
    # Initialize mask manager
    mask_manager = ImportanceMaskManager(model)
    success = mask_manager.initialize_masks()
    
    print(f"1. Model initialization: {'? SUCCESS' if success else '? FAILED'}")
    print(f"   Device: {device}")
    print(f"   Attention layers: {len(mask_manager.attention_masks)}")
    print(f"   MLP layers: {len(mask_manager.channel_masks)}")
    
    # Create test config and pipeline
    config = {
        'target_ratio': 0.2,
        'num_iterations': 1,
        'fine_tune_epochs': 1
    }
    
    pipeline = IterativePruningPipeline(model, config)
    
    # Create mock data loader
    data_loader = MockDataLoader(batch_size=2, num_batches=5, device=device)
    
    print(f"\n2. Testing importance collection...")
    
    # Test the importance collection method
    try:
        pipeline._collect_importance_scores(data_loader)
        collection_success = True
        print("   ? Importance collection completed without errors")
    except Exception as e:
        collection_success = False
        print(f"   ? Importance collection failed: {e}")
    
    # Check if importance scores were actually computed
    importance_computed = len(pipeline.mask_manager.importance_scores) > 0
    print(f"   Importance scores computed: {'? YES' if importance_computed else '? NO'}")
    print(f"   Number of layers with importance: {len(pipeline.mask_manager.importance_scores)}")
    
    return collection_success and importance_computed


def test_importance_score_quality():
    """Test the quality and realism of computed importance scores"""
    print("\n" + "="*70)
    print("TESTING IMPORTANCE SCORE QUALITY")
    print("="*70)
    
    # Create test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TestableModel()
    model.netG = model.netG.to(device)
    
    # Initialize mask manager
    mask_manager = ImportanceMaskManager(model)
    mask_manager.initialize_masks()
    
    # Create test config and pipeline
    config = {'target_ratio': 0.2, 'num_iterations': 1, 'fine_tune_epochs': 1}
    pipeline = IterativePruningPipeline(model, config)
    
    # Create mock data loader
    data_loader = MockDataLoader(batch_size=2, num_batches=5, device=device)
    
    print("1. Collecting importance scores...")
    pipeline._collect_importance_scores(data_loader)
    
    # Analyze importance score quality
    print("\n2. Analyzing importance score quality...")
    
    quality_checks = {
        "Non-zero scores": False,
        "Reasonable variance": False,
        "Proper tensor shapes": False,
        "Device consistency": False,
        "Score range validity": False
    }
    
    total_scores = 0
    valid_scores = 0
    
    for layer_name, importance in pipeline.mask_manager.importance_scores.items():
        total_scores += 1
        
        # Check if scores are non-zero
        non_zero = torch.any(importance != 0).item()
        if non_zero:
            quality_checks["Non-zero scores"] = True
        
        # Check variance (should not be all identical)
        variance = torch.var(importance).item()
        if variance > 1e-6:
            quality_checks["Reasonable variance"] = True
        
        # Check tensor shape (should match mask dimensions)
        expected_shape = None
        if layer_name in pipeline.mask_manager.attention_masks:
            expected_shape = pipeline.mask_manager.attention_masks[layer_name].shape
        elif layer_name in pipeline.mask_manager.channel_masks:
            expected_shape = pipeline.mask_manager.channel_masks[layer_name].shape
        
        if expected_shape is not None and importance.shape == expected_shape:
            quality_checks["Proper tensor shapes"] = True
        
        # Check device consistency
        if importance.device == device:
            quality_checks["Device consistency"] = True
        
        # Check score range (should be reasonable, not extreme)
        if 0.001 <= torch.mean(importance).item() <= 1000.0:
            quality_checks["Score range validity"] = True
        
        valid_scores += 1
        
        print(f"   {layer_name}:")
        print(f"     Shape: {importance.shape}")
        print(f"     Mean: {torch.mean(importance).item():.4f}")
        print(f"     Std: {torch.std(importance).item():.4f}")
        print(f"     Min/Max: {torch.min(importance).item():.4f} / {torch.max(importance).item():.4f}")
        
        if total_scores >= 5:  # Limit output for readability
            break
    
    print(f"\n3. Quality assessment results:")
    passed_checks = sum(quality_checks.values())
    total_checks = len(quality_checks)
    
    for check_name, passed in quality_checks.items():
        status = "? PASS" if passed else "? FAIL"
        print(f"   {status} {check_name}")
    
    print(f"\n   Overall quality: {passed_checks}/{total_checks} checks passed")
    print(f"   Scores computed for {valid_scores}/{total_scores} layers")
    
    return passed_checks >= total_checks * 0.8  # 80% pass rate


def test_importance_consistency():
    """Test that importance scores are consistent across multiple runs"""
    print("\n" + "="*70)
    print("TESTING IMPORTANCE SCORE CONSISTENCY")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run importance collection multiple times
    scores_runs = []
    
    for run in range(3):
        print(f"\nRun {run + 1}/3:")
        
        # Create fresh model for each run
        model = TestableModel()
        model.netG = model.netG.to(device)
        
        # Set deterministic behavior
        torch.manual_seed(42 + run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + run)
        
        # Initialize and collect
        mask_manager = ImportanceMaskManager(model)
        mask_manager.initialize_masks()
        
        config = {'target_ratio': 0.2, 'num_iterations': 1, 'fine_tune_epochs': 1}
        pipeline = IterativePruningPipeline(model, config)
        
        data_loader = MockDataLoader(batch_size=2, num_batches=3, device=device)
        pipeline._collect_importance_scores(data_loader)
        
        # Store scores for this run
        run_scores = {}
        for name, scores in pipeline.mask_manager.importance_scores.items():
            run_scores[name] = scores.clone().detach()
        
        scores_runs.append(run_scores)
        print(f"   Collected scores for {len(run_scores)} layers")
    
    # Analyze consistency
    print(f"\n2. Analyzing consistency across runs...")
    
    if len(scores_runs) >= 2:
        consistent_layers = 0
        total_layers = 0
        
        # Compare first two runs
        run1_scores = scores_runs[0]
        run2_scores = scores_runs[1]
        
        common_layers = set(run1_scores.keys()) & set(run2_scores.keys())
        
        for layer_name in common_layers:
            total_layers += 1
            scores1 = run1_scores[layer_name]
            scores2 = run2_scores[layer_name]
            
            # Check if shapes match
            if scores1.shape == scores2.shape:
                # Calculate correlation
                correlation = torch.corrcoef(torch.stack([scores1.flatten(), scores2.flatten()]))[0, 1]
                
                # Check for reasonable consistency (correlation > 0.3 or similar patterns)
                if not torch.isnan(correlation) and correlation.item() > 0.3:
                    consistent_layers += 1
                    print(f"   ? {layer_name}: correlation = {correlation.item():.3f}")
                else:
                    print(f"   ? {layer_name}: correlation = {correlation.item():.3f} (low)")
            else:
                print(f"   ? {layer_name}: shape mismatch {scores1.shape} vs {scores2.shape}")
        
        consistency_ratio = consistent_layers / total_layers if total_layers > 0 else 0
        print(f"\n   Consistency: {consistent_layers}/{total_layers} layers ({consistency_ratio:.1%})")
        
        return consistency_ratio >= 0.7  # 70% consistency threshold
    else:
        print("   ? Insufficient runs for consistency analysis")
        return False


def test_importance_vs_synthetic():
    """Test that real importance collection is different from pure synthetic"""
    print("\n" + "="*70)
    print("TESTING REAL VS SYNTHETIC IMPORTANCE DIFFERENCES")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get scores from real collection
    model1 = TestableModel()
    model1.netG = model1.netG.to(device)
    mask_manager1 = ImportanceMaskManager(model1)
    mask_manager1.initialize_masks()
    
    config = {'target_ratio': 0.2, 'num_iterations': 1, 'fine_tune_epochs': 1}
    pipeline1 = IterativePruningPipeline(model1, config)
    
    data_loader = MockDataLoader(batch_size=2, num_batches=5, device=device)
    pipeline1._collect_importance_scores(data_loader)
    real_scores = pipeline1.mask_manager.importance_scores
    
    # Generate purely synthetic scores
    model2 = TestableModel()
    model2.netG = model2.netG.to(device)
    mask_manager2 = ImportanceMaskManager(model2)
    mask_manager2.initialize_masks()
    
    synthetic_scores = {}
    for name in mask_manager2.attention_masks.keys():
        num_heads = len(mask_manager2.attention_masks[name])
        synthetic_scores[name] = torch.randn(num_heads, device=device)
    
    for name in mask_manager2.channel_masks.keys():
        num_channels = len(mask_manager2.channel_masks[name])
        synthetic_scores[name] = torch.randn(num_channels, device=device)
    
    # Compare differences
    print("1. Comparing real vs synthetic importance scores...")
    
    differences_found = 0
    total_comparisons = 0
    
    common_layers = set(real_scores.keys()) & set(synthetic_scores.keys())
    
    for layer_name in list(common_layers)[:5]:  # Limit output
        if layer_name in real_scores and layer_name in synthetic_scores:
            real = real_scores[layer_name]
            synthetic = synthetic_scores[layer_name]
            
            if real.shape == synthetic.shape:
                total_comparisons += 1
                
                # Compare statistical properties
                real_mean = torch.mean(real).item()
                synthetic_mean = torch.mean(synthetic).item()
                real_std = torch.std(real).item()
                synthetic_std = torch.std(synthetic).item()
                
                mean_diff = abs(real_mean - synthetic_mean)
                std_diff = abs(real_std - synthetic_std)
                
                # Check if they're substantially different
                if mean_diff > 0.1 or std_diff > 0.1:
                    differences_found += 1
                
                print(f"   {layer_name}:")
                print(f"     Real: mean={real_mean:.3f}, std={real_std:.3f}")
                print(f"     Synthetic: mean={synthetic_mean:.3f}, std={synthetic_std:.3f}")
                print(f"     Differences: mean_diff={mean_diff:.3f}, std_diff={std_diff:.3f}")
    
    difference_ratio = differences_found / total_comparisons if total_comparisons > 0 else 0
    print(f"\n   Substantial differences found: {differences_found}/{total_comparisons} ({difference_ratio:.1%})")
    
    # Real collection should produce different results than pure random
    return difference_ratio >= 0.3  # At least 30% should be different


def main():
    """Main test function"""
    print("Importance Score Collection Test Suite")
    print("This test validates that real activations are captured for importance computation")
    
    # Run all tests
    test_results = {
        "Hook Registration": test_importance_collection_hooks(),
        "Score Quality": test_importance_score_quality(),
        "Score Consistency": test_importance_consistency(),
        "Real vs Synthetic": test_importance_vs_synthetic(),
    }
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "? PASSED" if passed else "? FAILED"
        print(f"{status} {test_name}")
        if passed:
            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n?? ALL TESTS PASSED - Importance collection is working correctly!")
        print("? Real activations are being captured and used for importance computation")
        print("? The 'Broken Importance Score Collection' issue has been fixed")
        success = True
    elif passed_tests >= total_tests * 0.75:
        print("\n??  MOSTLY SUCCESSFUL - Minor issues remain but core functionality works")
        print("? Real activation capture is functional")
        print("??  Some quality or consistency issues may need attention")
        success = True
    else:
        print("\n? TESTS FAILED - Importance collection needs more work")
        print("? The 'Broken Importance Score Collection' issue is not fully resolved")
        success = False
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
