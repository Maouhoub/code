import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import copy
from utils import utils_option as option
# We don't import define_Model blindly because we might need to load a Full Pruned Model object
# from models.select_model import define_Model 

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

def pad_module_weights(module, pad_amount, dim=0):
    """
    Pad weight/bias with ZEROS.
    This ensures the output is mathematically identical to the unpadded version.
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

def is_target_layer(name, module):
    """
    Check if this layer matches the pruning criteria from the original script.
    Original criteria:
    1. 'mlp.fc1'
    2. 'conv_first'
    3. 'conv_after_body'
    4. keywords: ['upsample', 'pixelshuffle', 'conv_before_upsample', 'conv_up']
    5. embed_dim match (we approximate this by ensuring we catch the Residual Body via dependencies of conv_first)
    """
    lower_name = name.lower()
    
    # 1. MLP Expansion
    if 'mlp.fc1' in lower_name:
        return True
        
    # 2. Body / Residual Stream Root
    if 'conv_first' in lower_name:
        return True
        
    # 3. Aggregation
    if 'conv_after_body' in lower_name:
        return True
        
    # 4. Upsampling / Reconstruction
    upsample_keywords = ['upsample', 'pixelshuffle', 'conv_before_upsample', 'conv_up']
    if any(k in lower_name for k in upsample_keywords):
        return True
        
    return False

def align_model_to_warp(model, example_inputs, align=32):
    print(f"\n{'='*60}")
    print(f" ALIGNING PRUNED LAYERS TO {align} (A100 WARP OPTIMIZATION)")
    print(f"{'='*60}")

    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs)

    visited_modules = set()
    padded_count = 0
    
    modules_list = list(model.named_modules())
    
    for name, module in modules_list:
        if module in visited_modules:
            continue
            
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Skip output layer (keep RGB 3 channels)
            if isinstance(module, nn.Conv2d) and module.out_channels == 3:
                continue

            # CRITICAL: Only process layers that were targets of the pruning script
            if not is_target_layer(name, module):
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

def load_pruned_model(opt):
    """
    Attempts to load the actual pruned model structure/weights.
    Auto-detects full model (pickle) vs state_dict.
    """
    path = opt['path'].get('pretrained_netG', None)
    if path is None:
        # Try to find last checkpoint
        _, path = option.find_last_checkpoint(opt['path']['models'], net_type='G')
        
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Could not find a pruned model checkpoint in: {opt['path']['models']}")
        
    print(f"Loading checkpoint: {path}")
    
    # 1. Try loading as Full Model (Architecture + Weights)
    # This is typical for 'netG_pruned_full_stepX.pth'
    try:
        model = torch.load(path)
        if isinstance(model, nn.Module):
            print("Successfully loaded Full Model object (Architecture included).")
            return model
    except Exception as e:
        pass # Not a full model, likely state_dict

    # 2. Try loading as Checkpoint Dict (Custom format from pruning script)
    # 'pruned_checkpoint = {'model': module_snapshot, ...}'
    try:
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
             if isinstance(checkpoint['model'], nn.Module):
                print("Successfully extracted Full Model from checkpoint dictionary.")
                return checkpoint['model']
    except:
        pass

    # 3. Fallback: Instantiate standard model and load state_dict
    # WARNING: This will fail if dimensions don't match (which is expected for pruned models)
    # We must assume the user provided a full model path if they want to load a pruned structure.
    print("Warning: Could not load as Full Model. Attempting standard instantiation (May fail if shapes mismatch)...")
    from models.select_model import define_Model
    model = define_Model(opt)
    model.init_train() # loads weights
    return model.netG

def main(json_path='options/swinir/train_swinir_sr_lightweight_structured_pruning.json'):
    # 1. Setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False) 
    opt = option.dict_to_nonedict(opt)

    # 2. Load Model
    # We use custom loader to handle pruned structures
    netG = load_pruned_model(opt)
    netG.eval()
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG.to(device)

    # 3. Sanity Check (Before)
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

    # 6. Save Aligned Model
    # Determine save path
    original_path = opt['path'].get('pretrained_netG', 'model.pth')
    if os.path.isdir(original_path): original_path = os.path.join(original_path, 'model.pth')
    
    dir_name = os.path.dirname(original_path)
    base_name = os.path.basename(original_path)
    save_path = os.path.join(dir_name, f"aligned32_{base_name}")
    
    print(f"Saving aligned model to: {save_path}")
    
    # Save Full Model object to preserve the new padded structure
    torch.save(netG, save_path)
    print("Saved as Full Model (Architecture + Weights). Load using torch.load()")

if __name__ == '__main__':
    main()
