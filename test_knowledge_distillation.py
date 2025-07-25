#!/usr/bin/env python3
"""
Knowledge Distillation Validation Script
Tests and fixes KD implementation issues: weak loss weighting and limited feature matching
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
    ImportanceMaskManager, StructuredPruner, KnowledgeDistillationTrainer
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
    """Simplified mock SwinIR model with feature extraction"""
    def __init__(self, img_size=64, embed_dim=96, depths=[2, 2], num_heads=[6, 6]):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.features = []  # Store intermediate features
        
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
        
    def forward(self, x, extract_features=False):
        # Simple forward pass
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C] for transformer processing
        
        features = []
        
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
                if extract_features:
                    features.append(x.clone())
        
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
        output = x.unsqueeze(-1).unsqueeze(-1).expand(B, 3, H, W)  # Expand back to [B, 3, H, W]
        
        if extract_features:
            return output, features
        return output

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
    
    def eval(self):
        self.netG.eval()

class ImprovedKnowledgeDistillationTrainer:
    """Improved KD trainer with better loss weighting and feature matching"""
    
    def __init__(self, teacher_model, student_model, 
                 temperature=4.0, alpha=0.7, feature_weight=0.1, 
                 adaptive_weighting=True):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.adaptive_weighting = adaptive_weighting
        
        # Loss tracking for adaptive weighting
        self.loss_history = {'hard': [], 'soft': [], 'feature': []}
        
        # Ensure teacher model is on the same device as student
        if hasattr(student_model, 'netG'):
            device = next(student_model.netG.parameters()).device
        else:
            device = next(student_model.parameters()).device
            
        # Move teacher to same device
        if hasattr(teacher_model, 'netG'):
            self.teacher_model.netG = self.teacher_model.netG.to(device)
        else:
            self.teacher_model = self.teacher_model.to(device)
        
        # Set teacher to eval mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_adaptive_weights(self):
        """Compute adaptive weights based on loss magnitudes"""
        if not self.adaptive_weighting or len(self.loss_history['hard']) < 3:
            return self.alpha, 1 - self.alpha, self.feature_weight
        
        # Get recent loss magnitudes
        recent_hard = np.mean(self.loss_history['hard'][-3:])
        recent_soft = np.mean(self.loss_history['soft'][-3:])
        recent_feature = np.mean(self.loss_history['feature'][-3:]) if self.loss_history['feature'] else 0.0
        
        # Balance based on relative magnitudes
        total_magnitude = recent_hard + recent_soft + recent_feature
        if total_magnitude > 0:
            # Adjust weights to balance loss contributions
            soft_weight = min(0.8, max(0.2, recent_soft / total_magnitude * 3))
            hard_weight = min(0.8, max(0.2, recent_hard / total_magnitude * 3))
            feature_weight = min(0.3, max(0.05, recent_feature / total_magnitude * 3))
            
            # Normalize
            total_weight = soft_weight + hard_weight + feature_weight
            soft_weight /= total_weight
            hard_weight /= total_weight
            feature_weight /= total_weight
            
            return soft_weight, hard_weight, feature_weight
        
        return self.alpha, 1 - self.alpha, self.feature_weight
    
    def improved_distillation_loss(self, student_output, teacher_output, target, 
                                 student_features=None, teacher_features=None):
        """Improved knowledge distillation with multiple loss components"""
        
        # 1. Hard loss (student vs ground truth) - L1 + MSE combination
        hard_mse = F.mse_loss(student_output, target)
        hard_l1 = F.l1_loss(student_output, target)
        hard_loss = 0.7 * hard_mse + 0.3 * hard_l1
        
        # 2. Soft loss (student vs teacher) - Multiple approaches
        # Output-level distillation (pixel-wise)
        soft_mse = F.mse_loss(student_output, teacher_output)
        
        # Structural similarity loss
        def ssim_loss(pred, target):
            mu1 = F.avg_pool2d(pred, 3, 1, 1)
            mu2 = F.avg_pool2d(target, 3, 1, 1)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.avg_pool2d(pred * pred, 3, 1, 1) - mu1_sq
            sigma2_sq = F.avg_pool2d(target * target, 3, 1, 1) - mu2_sq
            sigma12 = F.avg_pool2d(pred * target, 3, 1, 1) - mu1_mu2
            
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return 1 - ssim_map.mean()
        
        soft_ssim = ssim_loss(student_output, teacher_output)
        soft_loss = 0.8 * soft_mse + 0.2 * soft_ssim
        
        # 3. Feature-level distillation
        feature_loss = torch.tensor(0.0, device=student_output.device)
        if student_features is not None and teacher_features is not None:
            feature_losses = []
            min_len = min(len(student_features), len(teacher_features))
            
            for i in range(min_len):
                s_feat, t_feat = student_features[i], teacher_features[i]
                
                # Match dimensions if needed
                if s_feat.shape != t_feat.shape:
                    # Simple interpolation to match shapes
                    if s_feat.numel() > t_feat.numel():
                        s_feat = F.adaptive_avg_pool1d(s_feat.flatten(1), t_feat.numel()).view(t_feat.shape)
                    else:
                        t_feat = F.adaptive_avg_pool1d(t_feat.flatten(1), s_feat.numel()).view(s_feat.shape)
                
                # Multiple feature matching strategies
                feat_mse = F.mse_loss(s_feat, t_feat)
                feat_cosine = 1 - F.cosine_similarity(s_feat.flatten(1), t_feat.flatten(1), dim=1).mean()
                
                feat_loss = 0.7 * feat_mse + 0.3 * feat_cosine
                feature_losses.append(feat_loss)
            
            if feature_losses:
                feature_loss = torch.stack(feature_losses).mean()
        
        # Track loss history for adaptive weighting
        self.loss_history['hard'].append(hard_loss.item())
        self.loss_history['soft'].append(soft_loss.item())
        self.loss_history['feature'].append(feature_loss.item())
        
        # Keep only recent history
        max_history = 10
        for key in self.loss_history:
            if len(self.loss_history[key]) > max_history:
                self.loss_history[key] = self.loss_history[key][-max_history:]
        
        # Get adaptive weights
        soft_weight, hard_weight, feature_weight = self.compute_adaptive_weights()
        
        # Combined loss
        total_loss = (soft_weight * soft_loss + 
                     hard_weight * hard_loss + 
                     feature_weight * feature_loss)
        
        return {
            'total_loss': total_loss,
            'hard_loss': hard_loss,
            'soft_loss': soft_loss,
            'feature_loss': feature_loss,
            'weights': {
                'soft': soft_weight,
                'hard': hard_weight,
                'feature': feature_weight
            }
        }

def test_kd_loss_weighting():
    """Test KD loss weighting mechanisms"""
    print("?? Testing KD Loss Weighting...")
    
    # Create mock data
    batch_size, channels, height, width = 2, 3, 64, 64
    student_output = torch.randn(batch_size, channels, height, width)
    teacher_output = torch.randn(batch_size, channels, height, width) 
    target = torch.randn(batch_size, channels, height, width)
    
    # Test original KD trainer
    teacher_model = MockModel()
    student_model = MockModel()
    
    original_kd = KnowledgeDistillationTrainer(teacher_model, student_model)
    
    # Test original loss
    try:
        mse_fn = nn.MSELoss()
        total_loss, hard_loss, soft_loss = original_kd.distillation_loss(
            student_output, teacher_output, target, mse_fn
        )
        
        print(f"   Original KD - Total: {total_loss.item():.6f}, Hard: {hard_loss.item():.6f}, Soft: {soft_loss.item():.6f}")
        original_working = True
    except Exception as e:
        print(f"   Original KD - ERROR: {e}")
        original_working = False
    
    # Test improved KD trainer
    improved_kd = ImprovedKnowledgeDistillationTrainer(
        copy.deepcopy(teacher_model), copy.deepcopy(student_model)
    )
    
    # Generate mock features
    feature_dim = 256
    student_features = [torch.randn(batch_size, 100, feature_dim) for _ in range(3)]
    teacher_features = [torch.randn(batch_size, 100, feature_dim) for _ in range(3)]
    
    # Test improved loss
    try:
        loss_dict = improved_kd.improved_distillation_loss(
            student_output, teacher_output, target,
            student_features, teacher_features
        )
        
        print(f"   Improved KD - Total: {loss_dict['total_loss'].item():.6f}")
        print(f"                 Hard: {loss_dict['hard_loss'].item():.6f}")
        print(f"                 Soft: {loss_dict['soft_loss'].item():.6f}")
        print(f"                 Feature: {loss_dict['feature_loss'].item():.6f}")
        print(f"                 Weights: {loss_dict['weights']}")
        improved_working = True
    except Exception as e:
        print(f"   Improved KD - ERROR: {e}")
        improved_working = False
    
    return original_working, improved_working

def test_feature_matching():
    """Test feature-level distillation capabilities"""
    print("\n?? Testing Feature Matching...")
    
    # Create models with feature extraction
    teacher_model = MockModel()
    student_model = MockModel()
    
    # Test data
    batch_size = 2
    input_data = torch.randn(batch_size, 3, 64, 64)
    
    # Extract features from both models
    with torch.no_grad():
        teacher_output, teacher_features = teacher_model.netG(input_data, extract_features=True)
        student_output, student_features = student_model.netG(input_data, extract_features=True)
    
    print(f"   Teacher features: {len(teacher_features)} layers")
    print(f"   Student features: {len(student_features)} layers")
    
    # Test feature shapes
    for i, (t_feat, s_feat) in enumerate(zip(teacher_features, student_features)):
        print(f"   Layer {i}: Teacher {list(t_feat.shape)} vs Student {list(s_feat.shape)}")
    
    # Test improved KD with features
    improved_kd = ImprovedKnowledgeDistillationTrainer(teacher_model, student_model)
    
    target = torch.randn_like(teacher_output)
    
    try:
        loss_dict = improved_kd.improved_distillation_loss(
            student_output, teacher_output, target,
            student_features, teacher_features
        )
        
        feature_effectiveness = loss_dict['feature_loss'].item() > 0
        print(f"   Feature loss computed: {feature_effectiveness} ({loss_dict['feature_loss'].item():.6f})")
        return feature_effectiveness
        
    except Exception as e:
        print(f"   Feature matching ERROR: {e}")
        return False

def test_adaptive_weighting():
    """Test adaptive loss weighting mechanism"""
    print("\n?? Testing Adaptive Loss Weighting...")
    
    teacher_model = MockModel()
    student_model = MockModel()
    
    # Test with adaptive weighting enabled
    adaptive_kd = ImprovedKnowledgeDistillationTrainer(
        teacher_model, student_model, adaptive_weighting=True
    )
    
    # Simulate multiple training steps
    batch_size = 2
    weights_evolution = []
    
    for step in range(10):
        # Generate data with different characteristics each step
        student_output = torch.randn(batch_size, 3, 64, 64) * (0.5 + step * 0.1)
        teacher_output = torch.randn(batch_size, 3, 64, 64)
        target = torch.randn(batch_size, 3, 64, 64)
        
        loss_dict = adaptive_kd.improved_distillation_loss(
            student_output, teacher_output, target
        )
        
        weights_evolution.append(loss_dict['weights'].copy())
        
        if step % 3 == 0:
            print(f"   Step {step}: Weights = {loss_dict['weights']}")
    
    # Check if weights actually adapt
    initial_weights = weights_evolution[0]
    final_weights = weights_evolution[-1]
    
    weight_changed = any(
        abs(initial_weights[key] - final_weights[key]) > 0.01 
        for key in initial_weights
    )
    
    print(f"   Weights adapted over time: {weight_changed}")
    print(f"   Initial weights: {initial_weights}")
    print(f"   Final weights: {final_weights}")
    
    return weight_changed

def test_comprehensive_kd_validation():
    """Comprehensive KD validation including performance comparison"""
    print("\n?? Comprehensive KD Performance Test...")
    
    # Setup models and data
    teacher_model = MockModel()
    student_model = MockModel()
    
    # Create test dataset
    num_samples = 5
    test_data = []
    for _ in range(num_samples):
        input_img = torch.randn(1, 3, 64, 64)
        target_img = torch.randn(1, 3, 64, 64)
        test_data.append((input_img, target_img))
    
    # Test Original KD
    print("   Testing Original KD...")
    original_kd = KnowledgeDistillationTrainer(
        copy.deepcopy(teacher_model), copy.deepcopy(student_model)
    )
    
    original_losses = []
    for input_img, target_img in test_data:
        try:
            with torch.no_grad():
                teacher_out = original_kd.teacher_model.netG(input_img)
                student_out = original_kd.student_model.netG(input_img)
            
            total_loss, hard_loss, soft_loss = original_kd.distillation_loss(
                student_out, teacher_out, target_img, nn.MSELoss()
            )
            original_losses.append(total_loss.item())
        except Exception as e:
            print(f"     Original KD error: {e}")
            original_losses.append(float('inf'))
    
    # Test Improved KD
    print("   Testing Improved KD...")
    improved_kd = ImprovedKnowledgeDistillationTrainer(
        copy.deepcopy(teacher_model), copy.deepcopy(student_model)
    )
    
    improved_losses = []
    for input_img, target_img in test_data:
        try:
            with torch.no_grad():
                teacher_out, teacher_feats = improved_kd.teacher_model.netG(input_img, extract_features=True)
                student_out, student_feats = improved_kd.student_model.netG(input_img, extract_features=True)
            
            loss_dict = improved_kd.improved_distillation_loss(
                student_out, teacher_out, target_img,
                student_feats, teacher_feats
            )
            improved_losses.append(loss_dict['total_loss'].item())
        except Exception as e:
            print(f"     Improved KD error: {e}")
            improved_losses.append(float('inf'))
    
    # Analysis
    original_avg = np.mean(original_losses) if original_losses else float('inf')
    improved_avg = np.mean(improved_losses) if improved_losses else float('inf')
    
    print(f"   Original KD avg loss: {original_avg:.6f}")
    print(f"   Improved KD avg loss: {improved_avg:.6f}")
    
    # Check if improved version provides more stable losses
    original_std = np.std(original_losses) if len(original_losses) > 1 else float('inf')
    improved_std = np.std(improved_losses) if len(improved_losses) > 1 else float('inf')
    
    print(f"   Original KD loss std: {original_std:.6f}")
    print(f"   Improved KD loss std: {improved_std:.6f}")
    
    stability_improved = improved_std < original_std
    print(f"   Loss stability improved: {stability_improved}")
    
    return original_avg, improved_avg, stability_improved

def main():
    """Main validation function"""
    print("="*70)
    print("KNOWLEDGE DISTILLATION VALIDATION")
    print("="*70)
    
    # Run all tests
    print("1. Testing KD Loss Weighting Mechanisms...")
    original_working, improved_working = test_kd_loss_weighting()
    
    print("\n2. Testing Feature-Level Distillation...")
    feature_matching_works = test_feature_matching()
    
    print("\n3. Testing Adaptive Loss Weighting...")
    adaptive_weighting_works = test_adaptive_weighting()
    
    print("\n4. Comprehensive Performance Comparison...")
    original_avg, improved_avg, stability_improved = test_comprehensive_kd_validation()
    
    # Overall assessment
    print(f"\n{'='*70}")
    print("KNOWLEDGE DISTILLATION ASSESSMENT")
    print(f"{'='*70}")
    
    print("Test Results:")
    print(f"   ? Original KD working:        {'? PASS' if original_working else '? FAIL'}")
    print(f"   ? Improved KD working:        {'? PASS' if improved_working else '? FAIL'}")
    print(f"   ? Feature matching:           {'? PASS' if feature_matching_works else '? FAIL'}")
    print(f"   ? Adaptive weighting:         {'? PASS' if adaptive_weighting_works else '? FAIL'}")
    print(f"   ? Loss stability improved:    {'? PASS' if stability_improved else '? FAIL'}")
    
    # Issues identified and recommendations
    print(f"\n?? IDENTIFIED ISSUES:")
    
    if not original_working:
        print("   ? Original KD implementation has critical errors")
    
    if original_working and not improved_working:
        print("   ?? Improved KD has implementation issues")
    elif improved_working:
        print("   ? Improved KD implementation working")
    
    if not feature_matching_works:
        print("   ? Feature-level distillation not working properly")
    else:
        print("   ? Feature-level distillation functioning")
    
    if not adaptive_weighting_works:
        print("   ? Adaptive loss weighting not functioning")
    else:
        print("   ? Adaptive loss weighting working")
    
    print(f"\n?? RECOMMENDATIONS:")
    print("   1. Replace original KD with improved version")
    print("   2. Enable feature-level distillation for better knowledge transfer")
    print("   3. Use adaptive weighting to balance loss components")
    print("   4. Add multiple loss types (MSE + L1 + SSIM) for robustness")
    print("   5. Implement proper feature dimension matching")
    
    # Overall success
    all_improvements_work = (improved_working and feature_matching_works and 
                           adaptive_weighting_works and stability_improved)
    
    if all_improvements_work:
        print(f"\n?? OVERALL RESULT: ? KD IMPROVEMENTS SUCCESSFUL!")
        print("   Knowledge distillation now has proper loss weighting and feature matching")
    else:
        print(f"\n?? OVERALL RESULT: ? KD IMPROVEMENTS NEEDED!")
        print("   Knowledge distillation still has weak loss weighting or limited feature matching")
    
    print(f"{'='*70}")
    
    return all_improvements_work

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
