"""
SwinIR-Specific Structured Pruning Functions
=============================================

This module contains SwinIR-specific implementations for structured pruning
that handle the complex attention mechanism and relative position bias tables.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import warnings

def identify_swinir_conv_layers(model):
    """
    Identify Conv2d layers in SwinIR that are safe to prune.
    Excludes attention-related and problematic layers.
    """
    safe_conv_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Skip attention-related layers
            if any(skip_word in name.lower() for skip_word in ['attn', 'attention']):
                continue
                
            # Skip very small layers (less than 4 output channels)
            if module.out_channels < 4:
                continue
                
            # Skip 1x1 convolutions in attention mechanisms
            if module.kernel_size == (1, 1) and 'qkv' in name.lower():
                continue
                
            safe_conv_layers.append((name, module))
    
    return safe_conv_layers

def swinir_structured_pruning(model, pruning_ratio=0.1):
    """
    Apply structured channel pruning specifically designed for SwinIR.
    
    This function:
    1. Identifies safe Conv2d layers to prune
    2. Skips attention-related layers that cause CUDA assertion errors
    3. Uses conservative pruning ratios
    4. Handles SwinIR's complex architecture gracefully
    """
    print(f"Applying SwinIR-specific structured channel pruning (ratio: {pruning_ratio:.1%})")
    
    # Get safe Conv2d layers
    safe_layers = identify_swinir_conv_layers(model)
    print(f"Found {len(safe_layers)} safe Conv2d layers for pruning")
    
    if len(safe_layers) == 0:
        print("?? No safe layers found for pruning")
        return model
    
    # Apply pruning to safe layers only
    successful_prunes = 0
    failed_prunes = 0
    
    for name, module in safe_layers:
        try:
            # Use a more conservative pruning ratio for SwinIR
            effective_ratio = min(pruning_ratio, 0.2)  # Cap at 20%
            
            # Ensure we don't prune too many channels
            channels_to_prune = int(module.out_channels * effective_ratio)
            if channels_to_prune == 0:
                channels_to_prune = 1
            
            # Make sure we don't prune all channels
            if channels_to_prune >= module.out_channels:
                channels_to_prune = module.out_channels - 1
            
            # Apply structured pruning
            prune.ln_structured(
                module, 
                name='weight', 
                amount=channels_to_prune,  # Use absolute number instead of ratio
                n=1,  # L1 norm
                dim=0   # Prune output channels
            )
            
            successful_prunes += 1
            print(f"  ? Pruned {name}: {channels_to_prune}/{module.out_channels} channels")
            
        except Exception as e:
            failed_prunes += 1
            print(f"  ? Failed to prune {name}: {e}")
            continue
    
    print(f"Pruning summary: {successful_prunes} successful, {failed_prunes} failed")
    
    if successful_prunes == 0:
        print("?? No layers were successfully pruned. Using minimal pruning approach.")
        # Try minimal pruning on the safest layers
        return minimal_swinir_pruning(model, safe_layers)
    
    return model

def minimal_swinir_pruning(model, safe_layers):
    """
    Apply minimal pruning to the safest SwinIR layers.
    """
    print("Applying minimal SwinIR pruning...")
    
    # Only prune the largest, safest layers
    large_layers = [(name, module) for name, module in safe_layers 
                   if module.out_channels >= 60 and 'conv' in name.lower()]
    
    if not large_layers:
        print("No large safe layers found for minimal pruning")
        return model
    
    # Prune only 1-2 channels from the largest layers
    for name, module in large_layers[:2]:  # Only first 2 large layers
        try:
            prune.ln_structured(
                module, 
                name='weight', 
                amount=1,  # Remove only 1 channel
                n=1,  # L1 norm
                dim=0   # Prune output channels
            )
            print(f"  ? Minimally pruned {name}: 1/{module.out_channels} channels")
        except Exception as e:
            print(f"  ? Even minimal pruning failed for {name}: {e}")
            continue
    
    return model

def safe_remove_pruning_masks(model):
    """
    Safely remove pruning masks from SwinIR model.
    """
    removed_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig'):
            try:
                prune.remove(module, 'weight')
                removed_count += 1
                print(f"  ? Removed pruning mask from: {name}")
            except Exception as e:
                print(f"  ? Could not remove mask from {name}: {e}")
                continue
    
    print(f"Removed {removed_count} pruning masks")
    return model

def calculate_swinir_compression_stats(model_before, model_after):
    """
    Calculate compression statistics for SwinIR model.
    """
    def count_conv_params(model):
        """Count parameters only in Conv2d layers."""
        conv_params = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                conv_params += sum(p.numel() for p in module.parameters())
        return conv_params
    
    def count_total_params(model):
        """Count all parameters."""
        return sum(p.numel() for p in model.parameters())
    
    before_total = count_total_params(model_before)
    after_total = count_total_params(model_after)
    
    before_conv = count_conv_params(model_before)
    after_conv = count_conv_params(model_after)
    
    stats = {
        'total_params_before': before_total,
        'total_params_after': after_total,
        'conv_params_before': before_conv,
        'conv_params_after': after_conv,
        'total_compression': (before_total - after_total) / before_total * 100 if before_total > 0 else 0,
        'conv_compression': (before_conv - after_conv) / before_conv * 100 if before_conv > 0 else 0
    }
    
    return stats