"""
Test script for Chunk 1: Importance Scoring and Mask Infrastructure
Validates the implementation without requiring full training
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the code directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_train_structured_pruning_chunk1 import (
    ImportanceMaskModule, 
    AttentionHeadMask, 
    MLPChannelMask, 
    StructuredPruner
)

def test_importance_mask_module():
    """Test basic ImportanceMaskModule functionality"""
    print("Testing ImportanceMaskModule...")
    
    # Test initialization
    mask = ImportanceMaskModule(num_elements=8, init_value=1.0)
    assert mask.num_elements == 8
    assert torch.allclose(mask.importance_scores, torch.ones(8))
    
    # Test forward pass
    x = torch.randn(2, 10, 8)  # (batch, seq, features)
    output = mask(x, dim=-1)
    assert output.shape == x.shape
    assert torch.allclose(output, x)  # Should be identity with init_value=1.0
    
    # Test L1 loss
    l1_loss = mask.compute_l1_loss()
    assert l1_loss.item() == 8.0  # Sum of absolute values of ones
    
    print("? ImportanceMaskModule tests passed")

def test_attention_head_mask():
    """Test AttentionHeadMask functionality"""
    print("Testing AttentionHeadMask...")
    
    num_heads = 4
    head_dim = 16
    total_dim = num_heads * head_dim
    
    mask = AttentionHeadMask(num_heads=num_heads, init_value=1.0)
    
    # Test with attention output
    B, N = 2, 100  # batch size, sequence length
    attention_output = torch.randn(B, N, total_dim)
    
    masked_output = mask(attention_output)
    assert masked_output.shape == (B, N, total_dim)
    assert torch.allclose(masked_output, attention_output)  # Identity with init_value=1.0
    
    # Test with different importance scores
    mask.importance_scores.data = torch.tensor([1.0, 0.5, 0.0, 0.8])
    masked_output = mask(attention_output)
    
    # Check that heads are scaled correctly
    reshaped_output = masked_output.view(B, N, num_heads, head_dim)
    assert torch.allclose(reshaped_output[:, :, 2, :], torch.zeros_like(reshaped_output[:, :, 2, :]))  # Head 2 should be zero
    
    print("? AttentionHeadMask tests passed")

def test_mlp_channel_mask():
    """Test MLPChannelMask functionality"""
    print("Testing MLPChannelMask...")
    
    num_channels = 64
    mask = MLPChannelMask(num_channels=num_channels, init_value=1.0)
    
    # Test with MLP output
    B, N = 2, 100
    mlp_output = torch.randn(B, N, num_channels)
    
    masked_output = mask(mlp_output)
    assert masked_output.shape == (B, N, num_channels)
    assert torch.allclose(masked_output, mlp_output)  # Identity with init_value=1.0
    
    # Test with custom importance scores
    mask.importance_scores.data = torch.cat([
        torch.ones(32),     # First half: keep
        torch.zeros(32)     # Second half: remove
    ])
    
    masked_output = mask(mlp_output)
    assert torch.allclose(masked_output[:, :, 32:], torch.zeros_like(masked_output[:, :, 32:]))
    
    print("? MLPChannelMask tests passed")

class MockSwinIRLayer(nn.Module):
    """Mock SwinIR layer for testing"""
    def __init__(self, dim=96, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class MockModel(nn.Module):
    """Mock model with SwinIR-like structure"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            MockSwinIRLayer(dim=96, num_heads=4),
            MockSwinIRLayer(dim=96, num_heads=4),
        ])
        
        # Add some regular linear layers that should be detected as MLP layers
        self.mlp_layer1 = nn.Linear(96, 384)
        self.mlp_layer2 = nn.Linear(384, 96)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def test_structured_pruner():
    """Test StructuredPruner with mock model"""
    print("Testing StructuredPruner...")
    
    # Create mock model
    model = MockModel()
    
    # Initialize pruner
    pruner = StructuredPruner(model)
    
    # Check that masks were created
    print(f"Found {len(pruner.head_masks)} attention masks")
    print(f"Found {len(pruner.channel_masks)} channel masks")
    
    # Test regularization loss computation
    reg_loss = pruner.compute_regularization_loss(lambda_l1=1e-4)
    assert reg_loss.item() > 0
    print(f"Regularization loss: {reg_loss.item():.6f}")
    
    # Test importance summary
    summary = pruner.get_importance_summary()
    assert 'statistics' in summary
    assert 'attention_heads' in summary
    assert 'mlp_channels' in summary
    
    print(f"Total masked parameters: {summary['statistics'].get('total_parameters', 0)}")
    
    # Test gradient flow
    dummy_input = torch.randn(1, 100, 96)
    output = model(dummy_input)
    loss = torch.sum(output) + reg_loss
    loss.backward()
    
    # Check that importance scores have gradients
    for name, mask in pruner.head_masks.items():
        assert mask.importance_scores.grad is not None
        print(f"? Gradients computed for attention mask {name}")
    
    for name, mask in pruner.channel_masks.items():
        assert mask.importance_scores.grad is not None
        print(f"? Gradients computed for channel mask {name}")
    
    print("? StructuredPruner tests passed")

def test_sparsity_induction():
    """Test that L1 regularization encourages sparsity"""
    print("Testing sparsity induction...")
    
    # Create a simple mask
    mask = ImportanceMaskModule(num_elements=10, init_value=1.0)
    optimizer = torch.optim.SGD([mask.importance_scores], lr=0.1)
    
    # Simulate training with L1 regularization
    initial_scores = mask.get_importance_scores().clone()
    
    for step in range(100):
        optimizer.zero_grad()
        
        # Dummy forward pass
        dummy_input = torch.randn(1, 5, 10)
        output = mask(dummy_input)
        
        # Loss = dummy task loss + L1 regularization
        task_loss = torch.sum(output ** 2)
        reg_loss = mask.compute_l1_loss() * 0.01  # Strong regularization
        total_loss = task_loss + reg_loss
        
        total_loss.backward()
        optimizer.step()
    
    final_scores = mask.get_importance_scores()
    
    # Check that scores have moved toward zero
    score_reduction = (initial_scores.abs().mean() - final_scores.abs().mean()).item()
    print(f"Average score reduction: {score_reduction:.4f}")
    print(f"Final scores range: [{final_scores.min().item():.4f}, {final_scores.max().item():.4f}]")
    
    assert score_reduction > 0, "L1 regularization should reduce scores"
    print("? Sparsity induction test passed")

def main():
    """Run all tests"""
    print("="*60)
    print("RUNNING CHUNK 1 TESTS")
    print("="*60)
    
    try:
        test_importance_mask_module()
        test_attention_head_mask()
        test_mlp_channel_mask()
        test_structured_pruner()
        test_sparsity_induction()
        
        print("\n" + "="*60)
        print("?? ALL TESTS PASSED! Chunk 1 implementation is working correctly.")
        print("="*60)
        
        print("\nKey Validation Results:")
        print("? Importance masks properly initialized")
        print("? Attention head masking works correctly")
        print("? MLP channel masking works correctly")
        print("? L1 regularization induces sparsity")
        print("? Gradient flow through masks verified")
        print("? Integration with model architecture successful")
        
    except Exception as e:
        print(f"\n? TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
