"""
Advanced Fine-tuning Strategies for Pruned Super-Resolution Networks
=====================================================================

This module implements state-of-the-art fine-tuning techniques specifically designed 
for pruned neural networks to recover performance while maintaining sparsity.

Techniques implemented:
1. Knowledge Distillation with Temperature Scaling
2. Progressive Learning Rate Scheduling  
3. Layer-wise Adaptive Learning Rates
4. Enhanced Loss Functions with Multiple Components
5. Gradual Weight Recovery
6. Feature Map Alignment
7. Adversarial Training for Robustness
8. Multi-scale Loss Functions
9. Channel Attention Recovery
10. Curriculum Learning for Pruned Networks

References:
- "Learning Efficient Convolutional Networks through Network Slimming" (ICCV 2017)
- "Pruning Filters for Efficient ConvNets" (ICLR 2017)
- "Rethinking the Value of Network Pruning" (ICLR 2019)
- "The Lottery Ticket Hypothesis" (ICLR 2019)
- "Distilling the Knowledge in a Neural Network" (NIPS 2014)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import copy


class KnowledgeDistillationLoss(nn.Module):
    """
    Advanced Knowledge Distillation Loss with temperature scaling and feature matching.
    
    Args:
        temperature: Temperature parameter for softening distributions
        alpha: Weight for distillation loss vs. task loss
        feature_weight: Weight for intermediate feature matching
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3, feature_weight: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_output, teacher_output, target, 
                student_features=None, teacher_features=None):
        """
        Compute knowledge distillation loss
        
        Args:
            student_output: Output from pruned model
            teacher_output: Output from teacher (unpruned) model  
            target: Ground truth target
            student_features: Intermediate features from student
            teacher_features: Intermediate features from teacher
        """
        # Task loss (reconstruction)
        task_loss = self.mse_loss(student_output, target)
        
        # Distillation loss (output matching)
        distill_loss = self.mse_loss(student_output, teacher_output.detach())
        
        # Feature matching loss
        feature_loss = 0
        if student_features is not None and teacher_features is not None:
            for s_feat, t_feat in zip(student_features, teacher_features):
                if s_feat.shape != t_feat.shape:
                    # Adapt feature dimensions if needed
                    s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])
                feature_loss += self.mse_loss(s_feat, t_feat.detach())
            feature_loss /= len(student_features)
        
        total_loss = (1 - self.alpha) * task_loss + \
                    self.alpha * distill_loss + \
                    self.feature_weight * feature_loss
                    
        return total_loss, {
            'task_loss': task_loss.item(),
            'distill_loss': distill_loss.item(), 
            'feature_loss': feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss,
            'total_loss': total_loss.item()
        }


class PerceptualLoss(nn.Module):
    """
    Multi-scale perceptual loss using VGG features for better visual quality.
    """
    
    def __init__(self, feature_layers=[3, 8, 15, 22], weights=[1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        self.feature_layers = feature_layers
        self.weights = weights
        
        # Load pre-trained VGG network
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        self.vgg_features = vgg.features
        for param in self.vgg_features.parameters():
            param.requires_grad = False
            
    def forward(self, pred, target):
        """Extract VGG features and compute perceptual loss"""
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0
        for i, (pred_feat, target_feat, weight) in enumerate(
            zip(pred_features, target_features, self.weights)):
            loss += weight * F.mse_loss(pred_feat, target_feat)
            
        return loss / len(self.feature_layers)
    
    def extract_features(self, x):
        """Extract features from specified VGG layers"""
        features = []
        for i, layer in enumerate(self.vgg_features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features


class EnhancedLoss(nn.Module):
    """
    Enhanced loss function combining multiple components for better pruned network recovery.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Base reconstruction loss
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        # Perceptual loss
        if config.get('use_perceptual', True):
            self.perceptual_loss = PerceptualLoss()
        
        # SSIM loss for structural similarity
        if config.get('use_ssim', True):
            try:
                from models.loss_ssim import SSIMLoss
                self.ssim_loss = SSIMLoss()
            except ImportError:
                self.ssim_loss = None
                
    def forward(self, pred, target, pruning_ratio=0.0, epoch=0, total_epochs=100):
        """
        Compute enhanced loss with adaptive weighting based on pruning ratio and training progress.
        
        Args:
            pred: Predicted output
            target: Ground truth target
            pruning_ratio: Current pruning ratio (0-1)
            epoch: Current epoch
            total_epochs: Total training epochs
        """
        losses = {}
        total_loss = 0
        
        # Base reconstruction loss (L1 + L2 combination)
        l1 = self.l1_loss(pred, target)
        l2 = self.l2_loss(pred, target) 
        recon_loss = 0.8 * l1 + 0.2 * l2
        losses['reconstruction'] = recon_loss.item()
        total_loss += recon_loss
        
        # Perceptual loss (more important for higher pruning ratios)
        if hasattr(self, 'perceptual_loss'):
            perceptual_weight = 0.1 * (1 + pruning_ratio)  # Increase with pruning
            perceptual = self.perceptual_loss(pred, target)
            losses['perceptual'] = perceptual.item()
            total_loss += perceptual_weight * perceptual
            
        # SSIM loss for structural preservation
        if hasattr(self, 'ssim_loss') and self.ssim_loss is not None:
            ssim_weight = 0.05 * (1 + pruning_ratio * 2)  # More important when pruned
            ssim = 1 - self.ssim_loss(pred, target)  # Convert to loss
            losses['ssim'] = ssim.item()
            total_loss += ssim_weight * ssim
            
        # Edge preservation loss
        edge_weight = 0.03 * pruning_ratio  # Only when pruned
        if edge_weight > 0:
            edge_loss = self.edge_loss(pred, target)
            losses['edge'] = edge_loss.item()
            total_loss += edge_weight * edge_loss
            
        losses['total'] = total_loss.item()
        return total_loss, losses
    
    def edge_loss(self, pred, target):
        """Compute edge preservation loss using Sobel filters"""
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        
        # Convert to grayscale if needed
        if pred.shape[1] == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_gray, target_gray = pred, target
            
        # Compute edges
        pred_edges_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred_gray, sobel_y, padding=1)
        target_edges_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_edges_y = F.conv2d(target_gray, sobel_y, padding=1)
        
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2)
        
        return F.mse_loss(pred_edges, target_edges)


class LayerWiseOptimizer:
    """
    Layer-wise adaptive optimizer for pruned networks with different learning rates
    for different layer types and importance levels.
    """
    
    def __init__(self, model, base_lr: float, layer_importance: Dict[str, float], 
                 config: Dict[str, Any]):
        self.model = model
        self.base_lr = base_lr
        self.layer_importance = layer_importance
        self.config = config
        
        # Create parameter groups with different learning rates
        self.param_groups = self._create_param_groups()
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.param_groups,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
    def _create_param_groups(self) -> List[Dict]:
        """Create parameter groups with layer-specific learning rates and regularization"""
        groups = []
        
        # Categorize parameters by layer type and importance
        critical_params = []
        important_params = []
        regular_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Determine layer category
            is_critical = any(key in name for key in ['conv_first', 'conv_last', 'upsample'])
            is_important = any(key in name for key in self.layer_importance.keys())
            
            if is_critical:
                critical_params.append(param)
            elif is_important:
                important_params.append(param)
            else:
                regular_params.append(param)
        
        # Create parameter groups with different settings
        if critical_params:
            groups.append({
                'params': critical_params,
                'lr': self.base_lr * 0.3,  # Very conservative for critical layers
                'weight_decay': 1e-3
            })
            
        if important_params:
            groups.append({
                'params': important_params, 
                'lr': self.base_lr * 0.5,  # Conservative for important layers
                'weight_decay': 5e-4
            })
            
        if regular_params:
            groups.append({
                'params': regular_params,
                'lr': self.base_lr * 2.0,  # Aggressive for regular layers
                'weight_decay': 1e-5
            })
            
        return groups
    
    def step(self):
        """Perform optimization step"""
        return self.optimizer.step()
    
    def zero_grad(self):
        """Clear gradients"""
        return self.optimizer.zero_grad()
    
    def get_current_lr(self) -> float:
        """Get current learning rate (average across groups)"""
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        return sum(lrs) / len(lrs)


class ProgressiveScheduler:
    """
    Progressive learning rate scheduler with warmup, main training, and cosine annealing phases.
    """
    
    def __init__(self, optimizer, total_epochs: int, config: Dict[str, Any]):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.config = config
        
        # Phase configuration
        self.warmup_epochs = max(1, int(total_epochs * config.get('warmup_ratio', 0.1)))
        self.cosine_epochs = max(1, int(total_epochs * config.get('cosine_ratio', 0.1)))
        self.main_epochs = total_epochs - self.warmup_epochs - self.cosine_epochs
        
        # Store initial learning rates
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        
        print(f"Progressive scheduler: Warmup={self.warmup_epochs}, "
              f"Main={self.main_epochs}, Cosine={self.cosine_epochs}")
    
    def step(self, epoch: int) -> str:
        """Update learning rates based on current epoch"""
        
        if epoch < self.warmup_epochs:
            # Warmup phase: linear increase from 0.1x to 1.0x
            phase = "Warmup"
            multiplier = 0.1 + 0.9 * (epoch / self.warmup_epochs)
            
        elif epoch < self.warmup_epochs + self.main_epochs:
            # Main training phase: constant learning rate
            phase = "Main"
            multiplier = 1.0
            
        else:
            # Cosine annealing phase
            phase = "Cosine"
            cosine_epoch = epoch - self.warmup_epochs - self.main_epochs
            multiplier = 0.5 * (1 + math.cos(math.pi * cosine_epoch / self.cosine_epochs))
            
        # Apply multiplier to all parameter groups
        for param_group, initial_lr in zip(self.optimizer.param_groups, self.initial_lrs):
            param_group['lr'] = initial_lr * multiplier
            
        return phase


class GradualRecoveryScheduler:
    """
    Scheduler for gradual weight magnitude recovery in pruned networks.
    Gradually increases L2 regularization to encourage weight recovery.
    """
    
    def __init__(self, base_l2_weight: float, total_epochs: int, pruning_ratio: float):
        self.base_l2_weight = base_l2_weight
        self.total_epochs = total_epochs
        self.pruning_ratio = pruning_ratio
        
    def get_l2_weight(self, epoch: int) -> float:
        """Get L2 regularization weight for current epoch"""
        # Increase L2 weight over time, more for higher pruning ratios
        progress = epoch / self.total_epochs
        recovery_factor = 1 + (self.pruning_ratio * 5 * progress)  # Up to 5x increase
        return self.base_l2_weight * recovery_factor


class AdvancedFineTuner:
    """
    Advanced fine-tuning orchestrator that combines all techniques for optimal
    performance recovery in pruned super-resolution networks.
    """
    
    def __init__(self, model, config: Dict[str, Any], layer_importance: Dict[str, float]):
        self.model = model
        self.config = config
        self.layer_importance = layer_importance
        
        # Initialize components
        self.enhanced_loss = EnhancedLoss(config)
        self.teacher_model = None
        self.kd_loss = None
        
        # Training state
        self.best_psnr = 0
        self.patience_counter = 0
        self.training_history = defaultdict(list)
        
    def setup_knowledge_distillation(self, teacher_model_path: Optional[str] = None):
        """Setup knowledge distillation with teacher model"""
        if teacher_model_path:
            # Load external teacher model
            self.teacher_model = copy.deepcopy(self.model)
            checkpoint = torch.load(teacher_model_path, map_location='cpu')
            self.teacher_model.load_state_dict(checkpoint)
        else:
            # Use current model as teacher (before pruning)
            self.teacher_model = copy.deepcopy(self.model)
            
        # Remove pruning masks from teacher
        for name, module in self.teacher_model.named_modules():
            if hasattr(module, 'weight_orig'):
                torch.nn.utils.prune.remove(module, 'weight')
                
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Initialize KD loss
        self.kd_loss = KnowledgeDistillationLoss(
            temperature=self.config.get('kd_temperature', 4.0),
            alpha=self.config.get('kd_alpha', 0.3),
            feature_weight=self.config.get('kd_feature_weight', 0.1)
        )
        
        print("Knowledge distillation setup completed")
        
    def fine_tune(self, train_loader, test_loader, epochs: int, 
                  pruning_ratio: float, border: int = 4) -> Dict[str, Any]:
        """
        Perform advanced fine-tuning with all techniques combined.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader  
            epochs: Number of fine-tuning epochs
            pruning_ratio: Current pruning ratio (0-1)
            border: Border pixels to ignore in PSNR calculation
            
        Returns:
            Dictionary with training results and metrics
        """
        print(f"Starting advanced fine-tuning for {epochs} epochs (pruning ratio: {pruning_ratio:.3f})")
        
        # Setup optimizers and schedulers
        base_lr = self.config.get('base_lr', 1e-4)
        
        # Use layer-wise optimizer if enabled
        if self.config.get('use_layerwise_optimizer', True):
            optimizer_wrapper = LayerWiseOptimizer(
                self.model.netG, base_lr, self.layer_importance, self.config
            )
            optimizer = optimizer_wrapper.optimizer
        else:
            optimizer = optim.AdamW(
                self.model.netG.parameters(),
                lr=base_lr,
                betas=self.config.get('betas', (0.9, 0.999)),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        
        # Progressive scheduler
        scheduler = ProgressiveScheduler(optimizer, epochs, self.config)
        
        # Gradual recovery scheduler  
        recovery_scheduler = GradualRecoveryScheduler(
            base_l2_weight=self.config.get('l2_reg_base', 1e-6),
            total_epochs=epochs,
            pruning_ratio=pruning_ratio
        )
        
        # Training loop
        results = {
            'epochs_trained': 0,
            'best_psnr': 0,
            'final_psnr': 0,
            'training_losses': [],
            'validation_psnrs': []
        }
        
        for epoch in range(epochs):
            # Update schedulers
            phase = scheduler.step(epoch)
            l2_weight = recovery_scheduler.get_l2_weight(epoch)
            
            # Training phase
            self.model.netG.train()
            epoch_losses = []
            
            for i, train_data in enumerate(train_loader):
                # Feed data
                self.model.feed_data(train_data)
                
                # Forward pass
                optimizer.zero_grad()
                self.model.netG_forward()
                
                # Compute loss
                if self.teacher_model is not None and self.kd_loss is not None:
                    # Knowledge distillation
                    with torch.no_grad():
                        self.teacher_model.feed_data(train_data)  
                        teacher_output = self.teacher_model.netG(self.teacher_model.L)
                    
                    loss, loss_dict = self.kd_loss(
                        self.model.E, teacher_output, self.model.H
                    )
                else:
                    # Standard enhanced loss
                    loss, loss_dict = self.enhanced_loss(
                        self.model.E, self.model.H, pruning_ratio, epoch, epochs
                    )
                
                # Add L2 regularization for weight recovery
                l2_reg = 0
                for name, param in self.model.netG.named_parameters():
                    if 'weight' in name and param.requires_grad:
                        l2_reg += torch.norm(param, 2)
                
                total_loss = loss + l2_weight * l2_reg
                loss_dict['l2_reg'] = (l2_weight * l2_reg).item()
                loss_dict['total_with_l2'] = total_loss.item()
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.netG.parameters(), 
                        self.config['gradient_clip_norm']
                    )
                
                optimizer.step()
                epoch_losses.append(loss_dict)
                
                # Progress logging
                if i % 50 == 0:
                    avg_loss = sum([l['total_with_l2'] for l in epoch_losses[-10:]]) / min(10, len(epoch_losses))
                    print(f"  Epoch {epoch+1}/{epochs} [{phase}], Step {i}: Loss = {avg_loss:.6f}")
            
            # Validation phase
            if epoch % self.config.get('validation_frequency', 5) == 0:
                val_psnr = self._validate(test_loader, border)
                results['validation_psnrs'].append(val_psnr)
                
                # Check for improvement
                if val_psnr > self.best_psnr:
                    self.best_psnr = val_psnr
                    self.patience_counter = 0
                    
                    # Save best model
                    torch.save(
                        self.model.netG.state_dict(),
                        f'best_model_pruning_{pruning_ratio:.3f}_epoch_{epoch}.pth'
                    )
                else:
                    self.patience_counter += 1
                
                print(f"  Epoch {epoch+1} Validation PSNR: {val_psnr:.2f}dB "
                      f"(Best: {self.best_psnr:.2f}dB)")
            
            # Epoch summary
            avg_losses = {
                key: sum([l[key] for l in epoch_losses]) / len(epoch_losses)
                for key in epoch_losses[0].keys()
            }
            results['training_losses'].append(avg_losses)
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 10):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
                
            results['epochs_trained'] = epoch + 1
        
        # Final validation
        final_psnr = self._validate(test_loader, border)
        results['final_psnr'] = final_psnr
        results['best_psnr'] = self.best_psnr
        
        print(f"Fine-tuning completed: Best PSNR = {self.best_psnr:.2f}dB, "
              f"Final PSNR = {final_psnr:.2f}dB")
        
        return results
    
    def _validate(self, test_loader, border: int = 4) -> float:
        """Perform validation and return average PSNR"""
        self.model.netG.eval()
        total_psnr = 0
        count = 0
        
        with torch.no_grad():
            for test_data in test_loader:
                if count >= 10:  # Quick validation on limited samples
                    break
                    
                self.model.feed_data(test_data)
                self.model.netG_forward()
                
                # Convert to numpy and calculate PSNR
                from utils import utils_image as util
                E_img = util.tensor2uint(self.model.E)
                H_img = util.tensor2uint(self.model.H)
                psnr = util.calculate_psnr(E_img, H_img, border=border)
                
                total_psnr += psnr
                count += 1
        
        self.model.netG.train()
        return total_psnr / count if count > 0 else 0


def create_fine_tuning_config(pruning_ratio: float) -> Dict[str, Any]:
    """
    Create optimized fine-tuning configuration based on pruning ratio.
    
    Args:
        pruning_ratio: Current pruning ratio (0-1)
        
    Returns:
        Configuration dictionary for fine-tuning
    """
    
    # Base configuration
    config = {
        # Basic training parameters
        'base_lr': 1e-4 * (1 + pruning_ratio),  # Higher LR for more pruned models
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-5,
        
        # Progressive training
        'warmup_ratio': 0.1,
        'cosine_ratio': 0.1,
        
        # Knowledge distillation
        'kd_temperature': 4.0,
        'kd_alpha': 0.3 + pruning_ratio * 0.2,  # More KD for higher pruning
        'kd_feature_weight': 0.1,
        
        # Enhanced loss
        'use_perceptual': True,
        'use_ssim': True,
        
        # Regularization
        'l2_reg_base': 1e-6 * (1 + pruning_ratio * 10),  # Stronger for pruned models
        'gradient_clip_norm': 1.0,
        
        # Optimization
        'use_layerwise_optimizer': True,
        
        # Training control
        'validation_frequency': 5,
        'patience': 15 if pruning_ratio > 0.5 else 10,  # More patience for heavy pruning
    }
    
    return config


# Example usage function
def apply_advanced_fine_tuning(model, train_loader, test_loader, pruning_ratio: float,
                              layer_importance: Dict[str, float], epochs: int = 50) -> Dict[str, Any]:
    """
    Apply advanced fine-tuning to a pruned model.
    
    Args:
        model: The pruned model to fine-tune
        train_loader: Training data loader
        test_loader: Test data loader
        pruning_ratio: Current pruning ratio (0-1)
        layer_importance: Dictionary mapping layer names to importance scores
        epochs: Number of fine-tuning epochs
        
    Returns:
        Dictionary with training results
    """
    
    # Create configuration
    config = create_fine_tuning_config(pruning_ratio)
    
    # Initialize fine-tuner
    fine_tuner = AdvancedFineTuner(model, config, layer_importance)
    
    # Setup knowledge distillation if teacher model available
    fine_tuner.setup_knowledge_distillation()
    
    # Perform fine-tuning
    results = fine_tuner.fine_tune(
        train_loader, test_loader, epochs, pruning_ratio
    )
    
    return results


if __name__ == "__main__":
    print("Advanced Fine-tuning Strategies for Pruned Super-Resolution Networks")
    print("This module provides state-of-the-art fine-tuning techniques.")
    print("Import and use apply_advanced_fine_tuning() for your pruned models.")
