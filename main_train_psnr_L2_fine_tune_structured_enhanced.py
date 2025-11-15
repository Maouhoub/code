# =============================================================================
# Colab-Ready Setup: Install required packages
# =============================================================================
"""
# Run this in Colab to install dependencies:
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install opencv-python
!pip install scikit-image
!pip install tensorboard
!pip install ptflops
!git clone https://github.com/VainF/Torch-Pruning.git
!cd Torch-Pruning && pip install -e .
"""

import os.path
import math
import argparse
import random
import numpy as np
import json
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn as nn
import torch.distributed as dist
import time
import copy
from collections import defaultdict

# Import Torch-Pruning for structured pruning
try:
    import torch_pruning as tp
    TORCH_PRUNING_AVAILABLE = True
    print("? Torch-Pruning is available")
except ImportError:
    TORCH_PRUNING_AVAILABLE = False
    print("? Torch-Pruning not available, using PyTorch native structured pruning")
    import torch.nn.utils.prune as prune

# FLOPs counting
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
    print("? ptflops is available for FLOPs calculation")
except ImportError:
    try:
        from utils.utils_modelsummary import get_model_flops
        PTFLOPS_AVAILABLE = False
        print("? Using local FLOPs calculation")
    except ImportError:
        print("? No FLOPs calculation available")
        PTFLOPS_AVAILABLE = False

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

'''
# =============================================================================
# Enhanced SwinIR Training with Structured Channel Pruning
# =============================================================================
# Original training code for MSRResNet adapted for SwinIR with structured pruning
# Uses Torch-Pruning for efficient structured channel pruning
# Measures FLOPs, parameters, PSNR, SSIM, and inference time
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

# =============================================================================
# Helper Functions for Model Analysis and Pruning
# =============================================================================


def load_cached_layer_sensitivities():
    cache_path = os.path.join('layer_sensitivities', 'layer_sensitivity_cache.json')
    if not os.path.isfile(cache_path):
        return {}

    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        layers = data.get('layers', {})
        return {name: info.get('scaled', 1.0) for name, info in layers.items()}
    except Exception as e:
        print(f"Failed to load cached layer sensitivities: {e}")
        return {}


def calculate_model_stats(model, input_shape=(3, 64, 64), device='cpu'):
    """
    Calculate FLOPs and parameter count for the model.
    """
    model.eval()
    stats = {}
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stats['total_params'] = total_params
    stats['trainable_params'] = trainable_params
    
    # Calculate FLOPs with robust error handling
    stats['flops'] = 0  # Default value
    
    try:
        if PTFLOPS_AVAILABLE:
            # Using ptflops with CUDA error handling
            try:
                # Move model to CPU temporarily for FLOPs calculation if CUDA errors occur
                original_device = next(model.parameters()).device
                if 'cuda' in str(original_device):
                    # Try GPU first
                    flops, params = get_model_complexity_info(model, input_shape, as_strings=False, 
                                                            print_per_layer_stat=False, verbose=False)
                    stats['flops'] = flops
                else:
                    flops, params = get_model_complexity_info(model, input_shape, as_strings=False, 
                                                            print_per_layer_stat=False, verbose=False)
                    stats['flops'] = flops
            except Exception as cuda_e:
                print(f"GPU FLOPs calculation failed: {cuda_e}")
                try:
                    # Try moving to CPU for FLOPs calculation
                    print("Attempting CPU FLOPs calculation...")
                    original_device = next(model.parameters()).device
                    model_cpu = model.cpu()
                    flops, params = get_model_complexity_info(model_cpu, input_shape, as_strings=False, 
                                                            print_per_layer_stat=False, verbose=False)
                    stats['flops'] = flops
                    model.to(original_device)  # Move back to original device
                except Exception as cpu_e:
                    print(f"CPU FLOPs calculation also failed: {cpu_e}")
                    stats['flops'] = 0
        else:
            # Using local utils
            try:
                flops = get_model_flops(model, input_shape, print_per_layer_stat=False)
                stats['flops'] = flops if flops is not None else 0
            except:
                stats['flops'] = 0
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs: {e}")
        stats['flops'] = 0
    
    # Ensure flops is never None
    if stats['flops'] is None:
        stats['flops'] = 0
    
    return stats

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_layer_sensitivity(layer_name):
    """Return a multiplicative sensitivity factor for pruning ratios based on layer name."""
    lname = layer_name.lower()

    # Highly sensitive layers
    if any(keyword in lname for keyword in ['conv_first', 'conv_after_body']):
        return 0.4  # prune much less aggressively

    # Moderately sensitive layers (e.g., upsampling pipeline)
    if any(keyword in lname for keyword in ['upsample', 'pixelshuffle', 'conv_before_upsample', 'conv_up']):
        return 0.6

    # Default sensitivity
    return 1.0

def validate_pixelshuffle_constraints(model, scale_factor=2):
    """
    Validate that all layers feeding into PixelShuffle have correct channel counts.
    For PixelShuffle with scale_factor, input channels must be divisible by scale_factor^2.
    """
    print(f"Validating PixelShuffle constraints for scale factor {scale_factor}...")
    required_divisor = scale_factor ** 2
    issues_found = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.PixelShuffle):
            # Find the layer that feeds into this PixelShuffle
            parent_name = '.'.join(name.split('.')[:-1])
            if parent_name:
                try:
                    parent_module = model
                    for part in parent_name.split('.'):
                        parent_module = getattr(parent_module, part)
                    
                    # Check if parent is Sequential (Upsample/UpsampleOneStep)
                    if isinstance(parent_module, nn.Sequential):
                        # Find the Conv2d that feeds into this PixelShuffle
                        conv_layers = [m for m in parent_module.children() if isinstance(m, nn.Conv2d)]
                        if conv_layers:
                            conv_layer = conv_layers[-1]  # Last conv before PixelShuffle
                            out_channels = conv_layer.out_channels
                            
                            if out_channels % required_divisor != 0:
                                issues_found.append(f"Layer {name}: Conv2d output channels ({out_channels}) not divisible by {required_divisor}")
                            else:
                                print(f"? {name}: Conv2d output channels ({out_channels}) properly divisible by {required_divisor}")
                except Exception as e:
                    print(f"Warning: Could not validate {name}: {e}")
    
    if issues_found:
        print("? PixelShuffle constraint violations found:")
        for issue in issues_found:
            print(f"  - {issue}")
        return False
    else:
        print("? All PixelShuffle constraints satisfied")
        return True

def apply_structured_pruning_torch_pruning(model, cached_layer_sensitivities, pruning_ratio=0.1, layer_ratio_cap=0.25, native_layer_ratio_cap=0.2):
    """
    Apply structured channel pruning using Torch-Pruning library.
    """
    print("Layer sensitivities : ", cached_layer_sensitivities)
    if not TORCH_PRUNING_AVAILABLE:
        print("Torch-Pruning not available, falling back to PyTorch native pruning")
        return apply_structured_pruning_native(model, pruning_ratio, max_layer_ratio=native_layer_ratio_cap)

    print(f"Applying structured channel pruning with ratio: {pruning_ratio}")

    try:
        # Create pruner with SwinIR-specific configurations
        example_inputs = torch.randn(1, 3, 64, 64)
        if next(model.parameters()).is_cuda:
            example_inputs = example_inputs.cuda()

        # Define importance metric (L1 norm for channels)
        imp = tp.importance.MagnitudeImportance(p=1)  # L1 norm

        # Identify modules for structured pruning while protecting critical components
        unwrapped_parameters = []
        prunable_modules = set()
        module_sensitivity = {}
        mlp_fc1_names = []
        conv_prunable_names = []
        fc2_names = []
        ignored_modules = set()
        pixelshuffle_container_names = []
        out_channel_groups = {}

        embed_dim = getattr(model, 'embed_dim', None)
        scale_factor = getattr(model, 'upscale', 2)
        try:
            pixelshuffle_group_size = int(scale_factor) ** 2
        except Exception:
            pixelshuffle_group_size = 4  # default protection for 2x upscale

        for name, module in model.named_modules():
            lower_name = name.lower()
            if 'attn' in lower_name or 'relative_position' in lower_name:
                ignored_modules.add(module)
                continue

            if isinstance(module, nn.PixelShuffle):
                ignored_modules.add(module)
                pixelshuffle_container_names.append(name)
                continue

            class_name = module.__class__.__name__
            if class_name in ['Upsample', 'UpsampleOneStep']:
                ignored_modules.add(module)
                pixelshuffle_container_names.append(name)
                continue

            if isinstance(module, nn.Sequential) and any(isinstance(child, nn.PixelShuffle) for child in module.children()):
                ignored_modules.add(module)
                pixelshuffle_container_names.append(name)
                continue

            if isinstance(module, nn.Linear):
                if 'mlp.fc1' in lower_name:
                    prunable_modules.add(module)
                    mlp_fc1_names.append(name)
                    module_sensitivity[module] = cached_layer_sensitivities.get(lower_name, 1)
                elif 'mlp.fc2' in lower_name:
                    fc2_names.append(name)
                else:
                    ignored_modules.add(module)
                continue

            if isinstance(module, nn.Conv2d):
                if 'conv_last' in lower_name or module.out_channels == 3:
                    ignored_modules.add(module)
                    continue

                if 'conv_first' in lower_name:
                    prunable_modules.add(module)
                    conv_prunable_names.append(name)
                    module_sensitivity[module] = cached_layer_sensitivities.get(lower_name, 1)
                    continue

                if 'conv_after_body' in lower_name:
                    prunable_modules.add(module)
                    conv_prunable_names.append(name)
                    module_sensitivity[module] = cached_layer_sensitivities.get(lower_name, 1)
                    continue

                if any(keyword in lower_name for keyword in ['upsample', 'pixelshuffle', 'conv_before_upsample', 'conv_up']):
                    prunable_modules.add(module)
                    conv_prunable_names.append(name)
                    module_sensitivity[module] = cached_layer_sensitivities.get(lower_name, 1)
                    out_channel_groups[module] = max(pixelshuffle_group_size, 1)
                    continue

                if any(keyword in lower_name for keyword in ['patch_embed', 'patch_unembed']):
                    ignored_modules.add(module)
                    continue

                if embed_dim is not None and module.out_channels == embed_dim:
                    prunable_modules.add(module)
                    conv_prunable_names.append(name)
                    module_sensitivity[module] = cached_layer_sensitivities.get(lower_name, 1)
                    continue

                ignored_modules.add(module)
                continue

        ignored_layers = [m for m in ignored_modules if m not in prunable_modules]

        if conv_prunable_names:
            print(f"Prunable Conv2d layers (output channels): {conv_prunable_names}")
        else:
            print("No Conv2d layers matched the pruning criteria.")

        if mlp_fc1_names:
            print(f"Prunable MLP fc1 layers (output channels): first={mlp_fc1_names[0]}, total={len(mlp_fc1_names)}")
        else:
            print("No MLP fc1 layers detected for pruning.")

        if fc2_names:
            print(f"Protected MLP fc2 layers (will receive input pruning via dependencies): first={fc2_names[0]}, total={len(fc2_names)}")

        for container_name in pixelshuffle_container_names:
            print(f"Excluding PixelShuffle container: {container_name}")

        for name, param in model.named_parameters():
            if 'relative_position_bias_table' in name or 'attn_mask' in name:
                unwrapped_parameters.append((name, param))

        if not prunable_modules:
            print("No eligible modules found for Torch-Pruning; returning model unchanged.")
            return model

        # Default sensitivity for modules that were not explicitly categorized
        #for module in prunable_modules:
        #    module_sensitivity.setdefault(module, 1.0)
        
        layer_pruning_ratios = {}
        for module in prunable_modules:
            sensitivity = module_sensitivity.get(module, 1.0)
            layer_ratio = min(pruning_ratio * sensitivity, layer_ratio_cap)
            layer_pruning_ratios[module] = layer_ratio

        # Initialize pruner with SwinIR-specific settings
        print(f"Using per layer ratios {layer_pruning_ratios}")
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            pruning_ratio=pruning_ratio,
            root_module_types=[nn.Conv2d, nn.Linear],
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
            pruning_ratio_dict=layer_pruning_ratios,
            out_channel_groups=out_channel_groups if out_channel_groups else None,
        )

        # Apply pruning
    

        try:
            pruner.step()
        except Exception as e:
            # Older Torch-Pruning versions may not accept pruning_ratios argument
            print(f"Torch-Pruning step failed  : {e}")
            #pruner.step()

        # Validate PixelShuffle divisibility after pruning
        try:
            validate_pixelshuffle_constraints(model)
        except Exception as pixelshuffle_error:
            print(f"Warning: PixelShuffle constraint check failed post-pruning: {pixelshuffle_error}")

        return model
    except Exception as e:
        print(f"Error with Torch-Pruning: {e}")
        print("SwinIR attention mechanism is complex for Torch-Pruning, falling back to PyTorch native pruning")
        return apply_structured_pruning_native(model, pruning_ratio, max_layer_ratio=native_layer_ratio_cap)


def apply_structured_pruning_native(model, pruning_ratio=0.1, max_layer_ratio=0.2):
    """
    Apply structured channel pruning using PyTorch native structured pruning.
    Uses SwinIR-specific pruning utilities.
    """
    # Import the SwinIR-specific pruning functions
    try:
        from swinir_pruning_utils import swinir_structured_pruning
        return swinir_structured_pruning(model, pruning_ratio)
    except ImportError:
        print("SwinIR pruning utils not found, falling back to basic pruning")

        import torch.nn.utils.prune as prune  # Import here to ensure it's available

        print(f"Applying PyTorch native structured channel pruning with ratio: {pruning_ratio}")

        # Collect all Conv2d layers for structured pruning, excluding problematic ones
        modules_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 1:  # Skip if only 1 channel
                # Skip attention-related Conv2d layers that might cause issues
                if not any(skip_word in name.lower() for skip_word in ['attn', 'attention', 'relative_position']):
                    modules_to_prune.append((name, module, 'weight'))

        print(f"Found {len(modules_to_prune)} Conv2d layers to prune (excluding attention layers)")

        # Apply structured pruning (L1 norm, dim=0 for channel pruning)
        successful_prunes = 0
        for name, module, param_name in modules_to_prune:
            try:
                # Check if the layer has enough channels to prune
                if module.out_channels > 2:  # Need at least 2 channels to prune
                    # Use conservative pruning - limit to configured maximum or 1 channel minimum
                    sensitivity = get_layer_sensitivity(name)
                    effective_ratio = min(pruning_ratio * sensitivity, max_layer_ratio)
                    channels_to_prune = max(1, int(module.out_channels * effective_ratio))
                    # Don't prune all channels
                    if channels_to_prune >= module.out_channels:
                        channels_to_prune = module.out_channels - 1

                    prune.ln_structured(module, name=param_name, amount=channels_to_prune, n=1, dim=0)
                    successful_prunes += 1
                    print(f"  ? Pruned {name}: {channels_to_prune}/{module.out_channels} channels")
                else:
                    print(f"Skipping {name}: insufficient channels ({module.out_channels})")
            except Exception as e:
                print(f"  ? Could not prune layer {name}: {e}")
                continue

        print(f"Successfully pruned {successful_prunes}/{len(modules_to_prune)} layers")
        return model

def remove_pruning_masks(model):
    """Remove pruning masks to make pruning permanent."""
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig'):
            try:
                prune.remove(module, 'weight')
                print(f"Removed pruning mask from: {name}")
            except:
                pass
    return model

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, data_range=255, multichannel=True, channel_axis=2)
    except ImportError:
        print("Warning: scikit-image not available, SSIM calculation skipped")
        return 0.0

def print_results_table(results):
    """Print a formatted results comparison table."""
    print("\n" + "="*100)
    print(" STRUCTURED CHANNEL PRUNING RESULTS COMPARISON")
    print("="*100)
    print(f"{'Metric':<25} {'Baseline':<20} {'Pruned':<20} {'Change':<20} {'Change %':<15}")
    print("-"*100)
    
    for metric, values in results.items():
        baseline = values['baseline']
        pruned = values['pruned']
        
        if baseline != 0:
            change = pruned - baseline
            change_pct = (change / baseline) * 100
        else:
            change = pruned
            change_pct = 0
        
        if metric in ['FLOPs', 'Parameters']:
            baseline_str = f"{baseline/1e6:.2f}M" if baseline > 1e6 else f"{baseline/1e3:.2f}K"
            pruned_str = f"{pruned/1e6:.2f}M" if pruned > 1e6 else f"{pruned/1e3:.2f}K"
            change_str = f"{change/1e6:.2f}M" if abs(change) > 1e6 else f"{change/1e3:.2f}K"
        elif metric in ['PSNR (dB)', 'SSIM']:
            baseline_str = f"{baseline:.4f}"
            pruned_str = f"{pruned:.4f}"
            change_str = f"{change:.4f}"
        elif metric == 'Inference Time (s)':
            baseline_str = f"{baseline:.4f}"
            pruned_str = f"{pruned:.4f}"
            change_str = f"{change:.4f}"
        else:
            baseline_str = f"{baseline:.2f}"
            pruned_str = f"{pruned:.2f}"
            change_str = f"{change:.2f}"
        
        print(f"{metric:<25} {baseline_str:<20} {pruned_str:<20} {change_str:<20} {change_pct:>13.2f}%")
    
    print("="*100)
    print("Note: Negative changes in FLOPs/Parameters indicate reduction (good for efficiency)")
    print("      Negative changes in PSNR/SSIM indicate quality degradation")
    print("="*100)

def evaluate_model(model, test_loader, opt, current_step, suffix="", max_images=22):
    """
    Comprehensive model evaluation function.
    Returns PSNR, SSIM, and average inference time.
    Limited to max_images for faster evaluation during research.
    """
    
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_inference_time = 0.0
    idx = 0
    border = opt['scale']

    model_network = model.netG if hasattr(model, 'netG') else model
    model_network.eval()

    try:
        timing_device = next(model_network.parameters()).device
    except StopIteration:
        timing_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        for test_data in test_loader:
            idx += 1
            
            # Limit evaluation to max_images for faster research
            if idx > max_images:
                print(f"  (Limiting evaluation to {max_images} images for speed)")
                break
                
            image_name_ext = os.path.basename(test_data['L_path'][0])
            img_name, ext = os.path.splitext(image_name_ext)

            # Create output directory
            if suffix:
                img_dir = os.path.join(opt['path']['images'], f"{img_name}_{suffix}")
            else:
                img_dir = os.path.join(opt['path']['images'], img_name)
            util.mkdir(img_dir)

            # Forward pass with timing
            model.feed_data(test_data)

            if torch.cuda.is_available() and isinstance(timing_device, torch.device) and timing_device.type == 'cuda':
                torch.cuda.synchronize(timing_device)

            start_time = time.time()
            model.test()

            if torch.cuda.is_available() and isinstance(timing_device, torch.device) and timing_device.type == 'cuda':
                torch.cuda.synchronize(timing_device)

            end_time = time.time()
            
            inference_time = end_time - start_time
            avg_inference_time += inference_time

            # Get results
            visuals = model.current_visuals()
            E_img = util.tensor2uint(visuals['E'])
            H_img = util.tensor2uint(visuals['H'])

            # Save estimated image
            save_img_path = os.path.join(img_dir, f'{img_name}_{current_step}.png')
            util.imsave(E_img, save_img_path)

            # Calculate PSNR
            current_psnr = util.calculate_psnr(E_img, H_img, border=border)
            avg_psnr += current_psnr
            
            # Calculate SSIM
            current_ssim = calculate_ssim(E_img, H_img)
            avg_ssim += current_ssim

            print(f'{idx:>4d} --> {image_name_ext:>15s} | PSNR: {current_psnr:<6.2f}dB | SSIM: {current_ssim:<6.4f} | Time: {inference_time:<6.4f}s')

    # Calculate averages
    avg_psnr /= idx
    avg_ssim /= idx
    avg_inference_time /= idx

    print(f"\n Evaluation Results ({suffix}):")
    print(f"  Average PSNR: {avg_psnr:.4f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    print(f"  Average Inference Time: {avg_inference_time:.4f} s")
    print(f"  Total Images: {idx}")
    
    return avg_psnr, avg_ssim, avg_inference_time


def main(json_path='options/swinir/train_swinir_sr_lightweight_structured_pruning.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    if init_path_G is not None:
        opt['path']['pretrained_netG'] = init_path_G
    if init_path_E is not None:
        opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    print("iterations : ", init_iter_optimizerG, init_path_optimizerG)
    if init_path_optimizerG is not None:
        opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    structured_opt = opt.get('structured_pruning', {}) or {}
    total_pruning_ratio = structured_opt.get('total_pruning_ratio', 0.7)
    pruning_steps = int(structured_opt.get('pruning_steps', 18))
    schedule_weight_start = structured_opt.get('schedule_weight_start', 0.6)
    schedule_weight_end = structured_opt.get('schedule_weight_end', 1.0)
    target_psnr_drop = structured_opt.get('target_psnr_drop', 0.2)
    max_layer_ratio = structured_opt.get('max_layer_ratio', 0.25)
    native_max_layer_ratio = structured_opt.get('native_max_layer_ratio', 0.2)
    max_eval_images = int(structured_opt.get('max_eval_images', 22))

    if pruning_steps <= 0:
        raise ValueError("structured_pruning.pruning_steps must be a positive integer")

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (create dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) create_dataloader for train and test
    # ----------------------------------------
    



    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            # Use the full fine-tuning dataset (no random subsampling)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                print('Number of train images for fine-tuning: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_ds =  dataset_opt[0]
            test_set = define_Dataset(test_ds)
            test_loader = DataLoader(test_set, batch_size=1,
                                        shuffle=False, num_workers=1,
                                        drop_last=False, pin_memory=True)
            
            test_datasets = [] if len(dataset_opt) <= 1 else dataset_opt[1:]

            
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)
        



    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    
    # =============================================================================
    # Baseline Model Evaluation (Before Pruning)
    # =============================================================================
    print("\n" + "="*80)
    print(" BASELINE MODEL EVALUATION (BEFORE PRUNING)")
    print("="*80)
    
    cached_layer_sensitivities = load_cached_layer_sensitivities()


    # Get device
    device = next(model.parameters()).device
    print(f"Device: {device}")
    
    # Calculate baseline statistics
    input_shape = (3, 64, 64)  # Adjust based on your input size
    baseline_stats = calculate_model_stats(model.netG if hasattr(model, 'netG') else model, 
                                         input_shape, device)
    
    print(f"Baseline Parameters: {baseline_stats['total_params']:,}")
    print(f"Baseline FLOPs: {baseline_stats['flops']:,}")
    
    # Store results for comparison
    results = {
        'Parameters': {'baseline': baseline_stats['total_params'], 'pruned': 0},
        'FLOPs': {'baseline': baseline_stats['flops'], 'pruned': 0},
        'PSNR (dB)': {'baseline': 0, 'pruned': 0},
        'SSIM': {'baseline': 0, 'pruned': 0},
        'Inference Time (s)': {'baseline': 0, 'pruned': 0}
    }

    # =============================================================================
    # Baseline Testing (Before Pruning)
    # =============================================================================
    print("\nEvaluating baseline model...")
    if opt['rank'] == 0:
        baseline_psnr, baseline_ssim, baseline_inference_time = evaluate_model(
            model, test_loader, opt, current_step, "baseline", max_images=max_eval_images)
        results['PSNR (dB)']['baseline'] = baseline_psnr
        results['SSIM']['baseline'] = baseline_ssim
        results['Inference Time (s)']['baseline'] = baseline_inference_time
        
        print(f"Baseline PSNR: {baseline_psnr:.4f} dB")
        print(f"Baseline SSIM: {baseline_ssim:.4f}")
        print(f"Baseline Inference Time: {baseline_inference_time:.4f} s")

    # =============================================================================
    # Structured Channel Pruning with Progressive Fine-tuning
    # =============================================================================
    print("\n" + "="*80)
    print(" STRUCTURED CHANNEL PRUNING")
    print("="*80)
    
    # Pruning configuration
    schedule_weights = np.linspace(schedule_weight_start, schedule_weight_end, pruning_steps)
    weight_sum = schedule_weights.sum()
    if weight_sum == 0:
        pruning_schedule = [total_pruning_ratio / pruning_steps] * pruning_steps
    else:
        pruning_schedule = [total_pruning_ratio * (w / weight_sum) for w in schedule_weights]
    pruning_ratio_per_step = total_pruning_ratio / pruning_steps
    target_psnr_threshold = baseline_psnr - target_psnr_drop  # Stop if PSNR drops below this
    
    print(f"Target total pruning ratio: {total_pruning_ratio:.1%}")
    print(f"Pruning steps: {pruning_steps}")
    print(f"Pruning ratio per step: {pruning_ratio_per_step:.1%}")
    print(f"Step ratio range: {pruning_schedule[0]:.2%} ? {pruning_schedule[-1]:.2%}")
    print(f"PSNR threshold: {target_psnr_threshold} dB")
    print(f"Layer ratio caps | Torch-Pruning: {max_layer_ratio:.2%}, Native: {native_max_layer_ratio:.2%}")
    
    current_psnr = baseline_psnr if opt['rank'] == 0 else 1000  # Initialize with baseline
    pruning_iteration = 0
    
    # Progressive structured pruning loop
    while pruning_iteration < pruning_steps and current_psnr > target_psnr_threshold:
        pruning_iteration += 1
        print(f"\n{'-'*60}")
        print(f" PRUNING ITERATION {pruning_iteration}/{pruning_steps}")
        print(f"{'-'*60}")
        
        # Apply structured channel pruning
        current_pruning_ratio = pruning_schedule[pruning_iteration - 1]
        print(f"Applying structured channel pruning (ratio: {current_pruning_ratio:.2%})...")
        
        # Get the actual network (handle model wrapper)
        network = model.netG if hasattr(model, 'netG') else model
        
        # Apply structured pruning
        if TORCH_PRUNING_AVAILABLE:
            network = apply_structured_pruning_torch_pruning(
                network,
                cached_layer_sensitivities,
                pruning_ratio=current_pruning_ratio,
                layer_ratio_cap=max_layer_ratio,
                native_layer_ratio_cap=native_max_layer_ratio
            )
        else:
            network = apply_structured_pruning_native(
                network,
                pruning_ratio=current_pruning_ratio,
                max_layer_ratio=native_max_layer_ratio,
            )
        
        # Update model
        if hasattr(model, 'netG'):
            model.netG = network
        
        # Validate PixelShuffle constraints after pruning
        scale_factor = opt.get('scale', 2)  # Default to 2x upscaling
        if not validate_pixelshuffle_constraints(network, scale_factor):
            print("  PixelShuffle constraints violated! Model may not work correctly.")
        
        print("? Structured pruning applied successfully")
        
        # Calculate post-pruning statistics with robust error handling
        current_stats = calculate_model_stats(network, input_shape, device)
        
        # Safe calculation of compression ratios
        if baseline_stats['total_params'] > 0:
            compression_ratio = (1 - current_stats['total_params'] / baseline_stats['total_params']) * 100
        else:
            compression_ratio = 0
            
        if baseline_stats['flops'] > 0 and current_stats['flops'] > 0:
            flop_reduction = (1 - current_stats['flops'] / baseline_stats['flops']) * 100
        else:
            flop_reduction = 0
        
        print(f"Parameters after pruning: {current_stats['total_params']:,} ({compression_ratio:.1f}% reduction)")
        if current_stats['flops'] > 0:
            print(f"FLOPs after pruning: {current_stats['flops']:,} ({flop_reduction:.1f}% reduction)")
        else:
            print(f"FLOPs calculation unavailable due to model complexity")

        # =============================================================================
        # Fine-tuning after Pruning
        # =============================================================================
        print(f"\n Fine-tuning after pruning iteration {pruning_iteration}...")
        
        fine_tune_epochs = opt['fine_tune']['L2_ft_epochs']
        patience = opt['fine_tune'].get('early_stop_patience', 6)
        if patience is None or patience <= 0:
            patience = fine_tune_epochs
        min_delta = opt['fine_tune'].get('early_stop_min_delta', 0.0) or 0.0
        best_val_psnr = -float('inf')
        epochs_without_improvement = 0
        stop_early = False
        best_state = None

        print(f"Fine-tuning epochs: {fine_tune_epochs}")
        if opt['rank'] == 0:
            print(f"Early stopping patience: {patience} | min_delta: {min_delta}")

        for epoch in range(fine_tune_epochs):
            if opt['rank'] == 0:
                print(f"Fine-tuning epoch: {epoch+1}/{fine_tune_epochs}")
            
            if opt['dist']:
                train_sampler.set_epoch(epoch + seed)

            epoch_loss = 0.0
            num_batches = 0
            
            for i, train_data in enumerate(train_loader):
                current_step += 1

                # Update learning rate
                model.update_learning_rate(current_step)

                # Feed data and optimize
                model.feed_data(train_data)
                model.optimize_parameters(current_step)
                
                # Accumulate loss for logging
                if opt['rank'] == 0:
                    logs = model.current_log()
                    if 'G_loss' in logs:
                        epoch_loss += logs['G_loss']
                    num_batches += 1

            # Log training information & early stopping checks based on validation metrics
            if opt['rank'] == 0:
                if num_batches > 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"  Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")
                else:
                    print("  Warning: No batches processed during fine-tuning epoch")

                val_suffix = f"pruned_iter_{pruning_iteration}_epoch_{epoch+1}"
                epoch_psnr, epoch_ssim, epoch_inference_time = evaluate_model(
                    model, test_loader, opt, current_step, val_suffix, max_images=max_eval_images)
                print(f"    Validation PSNR: {epoch_psnr:.4f} dB | SSIM: {epoch_ssim:.4f} | Time: {epoch_inference_time:.4f} s")

                if epoch_psnr >= best_val_psnr + min_delta:
                    best_val_psnr = epoch_psnr
                    epochs_without_improvement = 0
                    best_state = copy.deepcopy(model.netG.state_dict())
                else:
                    epochs_without_improvement += 1
                    print(f"    No PSNR improvement ({epochs_without_improvement}/{patience})")
                    if epochs_without_improvement >= patience:
                        print("    Early stopping triggered: validation PSNR plateaued")
                        stop_early = True
                model.save(current_step)

            # Sync early stopping decision across processes if needed
            if opt['dist'] and dist.is_available() and dist.is_initialized():
                device_for_sync = next(model.parameters()).device
                stop_tensor = torch.tensor([1 if stop_early else 0], device=device_for_sync, dtype=torch.int)
                dist.broadcast(stop_tensor, src=0)
                stop_early = bool(stop_tensor.item())
            if stop_early:
                if best_state is not None:
                    model.netG.load_state_dict(best_state, strict=False)
                break

        # =============================================================================
        # Evaluation after Fine-tuning
        # =============================================================================
        print(f"\n Evaluating model after pruning iteration {pruning_iteration}...")
        if opt['rank'] == 0:
            current_psnr, current_ssim, current_inference_time = evaluate_model(
                model, test_loader, opt, current_step, f"pruned_iter_{pruning_iteration}", max_images=max_eval_images)
            
            print(f"  PSNR: {current_psnr:.4f} dB")
            print(f"  SSIM: {current_ssim:.4f}")
            print(f"  Inference Time: {current_inference_time:.4f} s")
            print(f"  PSNR vs Baseline: {current_psnr - baseline_psnr:+.4f} dB")
            
            # Check if we should continue pruning
            if current_psnr < target_psnr_threshold:
                print(f"? PSNR ({current_psnr:.4f}) below threshold ({target_psnr_threshold}). Stopping pruning.")
                break

    # =============================================================================
    # Final Model Statistics and Results
    # =============================================================================
    print("\n" + "="*80)
    print(" FINAL MODEL EVALUATION")
    print("="*80)
    
    if opt['rank'] == 0:
        # Calculate final statistics
        final_network = model.netG if hasattr(model, 'netG') else model
        final_stats = calculate_model_stats(final_network, input_shape, device)
        
        # Update results
        results['Parameters']['pruned'] = final_stats['total_params']
        results['FLOPs']['pruned'] = final_stats['flops']
        results['PSNR (dB)']['pruned'] = current_psnr
        results['SSIM']['pruned'] = current_ssim
        results['Inference Time (s)']['pruned'] = current_inference_time


        if len(test_datasets) == 0:
            print("No extra datasets to test on")
        for ds in test_datasets:
            testing_set = define_Dataset(ds)
            testing_loader = DataLoader(testing_set, batch_size=1,
                                        shuffle=False, num_workers=1,
                                        drop_last=False, pin_memory=True)
            psnr, ssim, inference_time = evaluate_model(
                model, testing_loader, opt, current_step, f"pruned_testset_{ds['name']}", max_images=max_eval_images // 2)
            print(f" Test Set {ds['name']} - PSNR: {psnr:.4f} dB | SSIM: {ssim:.4f} | Inference Time: {inference_time:.4f} s")
        
        # Print final comparison table
        print_results_table(results)

    # =============================================================================
    # Model Saving
    # =============================================================================
    if opt['rank'] == 0:
        print("\n" + "="*80)
        print(" SAVING FINAL PRUNED MODEL")
        print("="*80)
        
        print(' Saving the final pruned model...')
        
        # Remove pruning masks to make pruning permanent
        model_network = model.netG if hasattr(model, 'netG') else model
        #model_network = remove_pruning_masks(model_network)
        
        # Update the model
        if hasattr(model, 'netG'):
            model.netG = model_network
        '''
        # Build safe checkpoint for reloading the pruned network
        pruned_module = model_network.module if hasattr(model_network, 'module') else model_network
        module_snapshot = copy.deepcopy(pruned_module).cpu().eval()

        # 1) Full module serialization (architecture + weights)
        pruned_model_full_path = os.path.join(opt['path']['models'], f"netG_pruned_full_step{current_step}.pth")
        torch.save(module_snapshot, pruned_model_full_path)
        print(f" Full model (architecture + weights) saved: {pruned_model_full_path}")

        # 2) Lightweight checkpoint with explicit metadata/state_dict
        pruned_checkpoint = {
            'model': module_snapshot,
            'state_dict': module_snapshot.state_dict(),
            'metadata': {
                'step': current_step,
                'pruning_iterations_completed': pruning_iteration,
                'target_pruning_ratio_per_step': pruning_ratio_per_step,
                'baseline_stats': baseline_stats,
                'final_stats': final_stats,
                'fine_tune': opt.get('fine_tune', {}),
                'netG_config': opt.get('netG', {}),
                'scale': opt.get('scale')
            }
        }
        pruned_state_path = os.path.join(opt['path']['models'], f"netG_pruned_checkpoint_step{current_step}.pth")
        torch.save(pruned_checkpoint, pruned_state_path)
        print(f" Torch checkpoint saved: {pruned_state_path}")
        '''
        # Save the model using the framework's native routine as well
        model.save(current_step)

        
        # Save additional information
        results_path = os.path.join(opt['path']['models'], 'pruning_results.txt')
        with open(results_path, 'w') as f:
            f.write("Structured Channel Pruning Results\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Pruning Configuration:\n")
            f.write(f"- Total pruning steps: {pruning_iteration}\n")
            f.write(f"- Pruning ratio per step: {pruning_ratio_per_step:.1%}\n")
            f.write(f"- Target PSNR threshold: {target_psnr_threshold} dB\n\n")
            
            f.write("Final Results:\n")
            for metric, values in results.items():
                baseline = values['baseline']
                pruned = values['pruned']
                change = pruned - baseline if baseline != 0 else pruned
                change_pct = (change / baseline) * 100 if baseline != 0 else 0
                
                f.write(f"- {metric}: {baseline:.4f} ? {pruned:.4f} ({change_pct:+.2f}%)\n")

            #f.write(f"\nSaved pruned model (full): {pruned_model_full_path}\n")
            #f.write(f"Saved pruned checkpoint  : {pruned_state_path}\n")
        
        print(f' Results summary saved to: {results_path}')
        
        print("\n Structured channel pruning completed successfully!")
        print("="*80)


if __name__ == '__main__':
    main()