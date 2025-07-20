"""
Utilities for structured pruning implementation
Based on the methodology described in contribution.txt
"""

import torch
import torch.nn as nn
import numpy as np

def count_model_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def compute_model_flops(model, input_size=(1, 3, 64, 64)):
    """Estimate model FLOPs (simplified)"""
    # This is a simplified version - for accurate FLOP counting,
    # you would use libraries like ptflops or fvcore
    total_flops = 0
    
    def flop_count_hook(module, input, output):
        nonlocal total_flops
        if isinstance(module, nn.Conv2d):
            # Conv2D FLOPs = output_elements * (kernel_size * input_channels + bias)
            output_elements = output.numel()
            kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            if module.bias is not None:
                kernel_flops += 1
            total_flops += output_elements * kernel_flops
        elif isinstance(module, nn.Linear):
            # Linear FLOPs = output_features * input_features + bias
            total_flops += module.in_features * module.out_features
            if module.bias is not None:
                total_flops += module.out_features
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(flop_count_hook))
    
    # Forward pass
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        if next(model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops

def analyze_attention_patterns(model, data_loader, num_samples=10):
    """Analyze attention patterns to identify important heads"""
    attention_stats = {}
    model.eval()
    
    def attention_hook(module, input, output, name):
        if hasattr(module, 'attn_weights'):  # If attention weights are stored
            weights = module.attn_weights.detach().cpu()
            if name not in attention_stats:
                attention_stats[name] = []
            attention_stats[name].append(weights.mean(dim=(0, 2, 3)))  # Average over batch and spatial dims
    
    # Register hooks for attention modules
    hooks = []
    for name, module in model.named_modules():
        if 'attn' in name and hasattr(module, 'num_heads'):
            hooks.append(module.register_forward_hook(
                lambda m, i, o, n=name: attention_hook(m, i, o, n)
            ))
    
    # Collect statistics
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_samples:
                break
            model.feed_data(data)
            model.test()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute average attention scores
    avg_attention = {}
    for name, stats in attention_stats.items():
        if stats:
            avg_attention[name] = torch.stack(stats).mean(dim=0)
    
    model.train()
    return avg_attention

def compute_gradient_based_importance(model, data_loader, criterion, num_samples=5):
    """Compute gradient-based importance scores for pruning"""
    model.train()
    importance_scores = {}
    
    # Initialize importance scores
    for name, param in model.named_parameters():
        if 'mask' in name:  # Only for mask parameters
            importance_scores[name] = torch.zeros_like(param)
    
    total_samples = 0
    for i, data in enumerate(data_loader):
        if i >= num_samples:
            break
            
        model.feed_data(data)
        
        # Forward pass
        model.optimize_parameters(0)  # This will compute gradients
        
        # Accumulate gradients
        for name, param in model.named_parameters():
            if 'mask' in name and param.grad is not None:
                importance_scores[name] += torch.abs(param.grad)
        
        total_samples += 1
    
    # Average importance scores
    for name in importance_scores:
        importance_scores[name] /= total_samples
    
    return importance_scores

def visualize_pruning_stats(masks, save_path=None):
    """Create visualization of pruning statistics"""
    import matplotlib.pyplot as plt
    
    # Collect mask statistics
    head_sparsity = []
    channel_sparsity = []
    
    for name, mask in masks.head_masks.items():
        sparsity = (torch.abs(mask) < 1e-6).float().mean().item()
        head_sparsity.append(sparsity)
    
    for name, mask in masks.channel_masks.items():
        sparsity = (torch.abs(mask) < 1e-6).float().mean().item()
        channel_sparsity.append(sparsity)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Attention head sparsity
    ax1.bar(range(len(head_sparsity)), head_sparsity)
    ax1.set_title('Attention Head Sparsity')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Sparsity Ratio')
    ax1.set_ylim(0, 1)
    
    # MLP channel sparsity
    ax2.bar(range(len(channel_sparsity)), channel_sparsity)
    ax2.set_title('MLP Channel Sparsity')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Sparsity Ratio')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Pruning statistics saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_pruning_results(results, save_path):
    """Save pruning experiment results"""
    import json
    
    # Convert tensors to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (int, float, str, list, dict)):
            serializable_results[key] = value
        else:
            serializable_results[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {save_path}")

def load_pruning_masks(masks, checkpoint_path):
    """Load saved pruning masks"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'pruning_masks' in checkpoint:
        mask_dict = checkpoint['pruning_masks']
        
        for name, mask in mask_dict.get('head_masks', {}).items():
            if name in masks.head_masks:
                masks.head_masks[name].data.copy_(mask)
        
        for name, mask in mask_dict.get('channel_masks', {}).items():
            if name in masks.channel_masks:
                masks.channel_masks[name].data.copy_(mask)
        
        print("Pruning masks loaded successfully")
    else:
        print("No pruning masks found in checkpoint")

def save_pruning_masks(masks, checkpoint_path):
    """Save pruning masks to checkpoint"""
    mask_dict = {
        'head_masks': {name: mask.cpu() for name, mask in masks.head_masks.items()},
        'channel_masks': {name: mask.cpu() for name, mask in masks.channel_masks.items()}
    }
    
    torch.save({'pruning_masks': mask_dict}, checkpoint_path)
    print(f"Pruning masks saved to {checkpoint_path}")
