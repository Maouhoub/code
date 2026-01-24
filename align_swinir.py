import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import copy
import random

# Exact imports from the context file
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from models.select_model import define_Model

# Import Torch-Pruning for Dependency Analysis
try:
    import torch_pruning as tp
except ImportError:
    print("Error: Torch-Pruning is required for dependency graph analysis.")
    exit(1)

def get_pad_amount(channels, align=32):
    if channels % align == 0:
        return 0
    return (align - (channels % align))


def collect_unwrapped_parameters(model):
    """Match the pruning script behavior for SwinIR attention params."""
    unwrapped_parameters = []
    for name, param in model.named_parameters():
        if 'relative_position_bias_table' in name or 'attn_mask' in name:
            unwrapped_parameters.append((name, param))
    return unwrapped_parameters

def pad_module_weights(module, pad_amount, dim=0):
    """
    Pad weight/bias with ZEROS.
    dim=0 (Output channels), dim=1 (Input channels)
    """
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return

    # Pad Weights
    weight = module.weight.data
    pad_shape = list(weight.shape)
    pad_shape[dim] = pad_amount
    
    # Create zero padding
    padding = torch.zeros(pad_shape, dtype=weight.dtype, device=weight.device)
    
    # Concatenate
    new_weight = torch.cat([weight, padding], dim=dim)
    
    # Update parameter
    module.weight = nn.Parameter(new_weight)
    
    # Pad Bias (only for output channels)
    if dim == 0 and module.bias is not None:
        bias = module.bias.data
        bias_padding = torch.zeros((pad_amount,), dtype=bias.dtype, device=bias.device)
        new_bias = torch.cat([bias, bias_padding], dim=0)
        module.bias = nn.Parameter(new_bias)

    # Update attributes
    if isinstance(module, nn.Conv2d):
        if dim == 0: module.out_channels += pad_amount
        if dim == 1: module.in_channels += pad_amount
    elif isinstance(module, nn.Linear):
         if dim == 0: module.out_features += pad_amount
         if dim == 1: module.in_features += pad_amount

def is_target_layer(name, module, *, embed_dim=None, upsampler=None, scale=2):
    """
    Alignment targets must be a *safe subset* of the pruning targets.

    Why: padding the main embed-dim stream (e.g. conv_first/conv_after_body or any C==embed_dim path)
    changes tensor shapes through PatchEmbed/LayerNorm/Transformer blocks and will break the model
    unless you propagate the change through the entire network.

    We therefore align:
      - MLP expansion `mlp.fc1` (safe: extra hidden units can be made no-op)
      - Upsampler convs that are *not* the final image-producing conv for pixelshuffledirect
    """
    lower_name = name.lower()

    # Ignore attention-related modules entirely
    if 'attn' in lower_name or 'relative_position' in lower_name:
        return False

    # 1) MLP expansion is safe to pad
    if isinstance(module, nn.Linear) and 'mlp.fc1' in lower_name:
        return True

    if not isinstance(module, nn.Conv2d):
        return False

    # 2) Never touch the global embed_dim stream
    if 'conv_first' in lower_name or 'conv_after_body' in lower_name:
        return False
    if embed_dim is not None and module.out_channels == embed_dim:
        return False

    # 3) Upsampling / reconstruction convs (careful with pixelshuffledirect)
    if any(k in lower_name for k in ['conv_before_upsample', 'conv_up', 'upsample']):
        # For pixelshuffledirect, UpsampleOneStep is (1 conv -> pixelshuffle) producing the final image.
        # Padding that conv changes the output channels after pixelshuffle and breaks output shape.
        if upsampler == 'pixelshuffledirect':
            try:
                final_conv_out = (int(scale) ** 2) * 3
            except Exception:
                final_conv_out = 12
            if module.out_channels == final_conv_out:
                return False
        return True

    return False

def align_model_to_warp(model, example_inputs, align=32):
    print(f"\n{'='*60}")
    print(f" ALIGNING PRUNED LAYERS TO {align} (A100 WARP OPTIMIZATION)")
    print(f"{'='*60}")

    # Match pruning script: pass unwrapped_parameters so TP doesn't guess prune-dims for attention tables.
    unwrapped_parameters = collect_unwrapped_parameters(model)
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs, unwrapped_parameters=unwrapped_parameters)

    visited_modules = set()
    padded_count = 0
    
    modules_list = list(model.named_modules())
    embed_dim = getattr(model, 'embed_dim', None)
    upsampler = getattr(model, 'upsampler', None)
    scale = getattr(model, 'upscale', 2)
    
    for name, module in modules_list:
        if module in visited_modules:
            continue
            
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Skip output layer (keep RGB 3 channels)
            if isinstance(module, nn.Conv2d) and module.out_channels == 3:
                continue

            # CRITICAL: Only process layers that are safe alignment targets
            if not is_target_layer(name, module, embed_dim=embed_dim, upsampler=upsampler, scale=scale):
                continue

            # Determine specifics
            if isinstance(module, nn.Conv2d):
                pruning_fn = tp.prune_conv_out_channels
                current_ch = module.out_channels
            else:
                pruning_fn = tp.prune_linear_out_channels
                current_ch = module.out_features
            
            # Check alignment
            pad_val = get_pad_amount(current_ch, align)
            
            if pad_val == 0:
                # Mark group as visited
                try:
                    group = DG.get_pruning_group(module, pruning_fn, idxs=[0])
                    for dep, _ in group:
                        visited_modules.add(dep.target.module)
                except:
                    pass
                continue
                
            print(f"Padding Group via {name}: {current_ch} -> {current_ch + pad_val} (+{pad_val})")
            
            # Get dependency group
            try:
                group = DG.get_pruning_group(module, pruning_fn, idxs=[0])
            except Exception as e:
                print(f"  Warning: Could not get dependency group for {name}. Skipping. ({e})")
                continue
            
            # Apply Zero-Padding to the whole group
            for dep, _ in group:
                target_module = dep.target.module
                visited_modules.add(target_module)
                
                # Determine dimension (In vs Out) based on handler name
                handler_name = str(dep.handler)
                pad_dim = 1 if ('in_channel' in handler_name or 'in_feature' in handler_name) else 0
                
                # Apply Pad
                pad_module_weights(target_module, pad_val, pad_dim)
            
            padded_count += 1
            
    print(f"Alignment Complete. Modified {padded_count} groups.")
    return model

def main(json_path='options/swinir/train_swinir_sr_lightweight_structured_pruning.json'):
    # ----------------------------------------
    # Step--1 (prepare opt) from original script
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    print("Effective Options file used is : ", parser.parse_args().opt)
    # Using is_train=True to ensure compatibility with model definition
    opt = option.parse(parser.parse_args().opt, is_train=True) 
    opt['dist'] = parser.parse_args().dist

    # Distributed settings
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # Update opt - Finding Last Checkpoint
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    if init_path_G is not None:
        print(f"Loading checkpoint from: {init_path_G}")
        opt['path']['pretrained_netG'] = init_path_G
    
    current_step = init_iter_G

    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # Step--3 (initialize model) from original script
    # ----------------------------------------
    model = define_Model(opt)
    model.init_train()

    # Get the actual network
    netG = model.netG if hasattr(model, 'netG') else model
    
    # 3. Sanity Check (Before)
    device = next(netG.parameters()).device
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    
    print("Running initial sanity check...")
    with torch.no_grad():
        output_original = netG(dummy_input)

    # 4. Align Model (Zero Padding) with Filtering
    align_model_to_warp(netG, dummy_input, align=32)

    # 5. Sanity Check (After)
    print("Running post-alignment verification...")
    with torch.no_grad():
        output_aligned = netG(dummy_input)

    # Calculate difference
    diff = torch.abs(output_original - output_aligned).max().item()
    print(f"\nVerification Diff (Should be ~0.0): {diff:.8f}")
    if diff > 1e-6:
        print("WARNING: Output changed significantly. Zero-padding might be incorrect.")
    else:
        print("SUCCESS: Model output preserved perfectly.")

    # 6. Save Aligned Model using Exact Mechanism
    print("\n" + "="*80)
    print(" SAVING FINAL ALIGNED MODEL")
    print("="*80)
    
    # Update the model in the wrapper if needed
    if hasattr(model, 'netG'):
        model.netG = netG
    
    # Use exact save mechanism from context file
    model.save(current_step)
    
    print(f"Saved aligned model for step {current_step}")
    print("="*80)

if __name__ == '__main__':
    main()