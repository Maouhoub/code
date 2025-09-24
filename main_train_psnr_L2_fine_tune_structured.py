import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import time
import json
import copy

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import torch.nn.utils.prune as prune

# Try to import torch_pruning, install if not available
try:
    import torch_pruning as tp
    TORCH_PRUNING_AVAILABLE = True
except ImportError:
    print("Warning: torch_pruning not installed. Using basic PyTorch pruning.")
    TORCH_PRUNING_AVAILABLE = False

# For FLOP and parameter counting
try:
    from fvcore.nn import FlopCountMode
    from fvcore.nn.flop_count import flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    try:
        from thop import profile
        THOP_AVAILABLE = True
        FVCORE_AVAILABLE = False
    except ImportError:
        print("Warning: Neither fvcore nor thop available for FLOP counting.")
        FVCORE_AVAILABLE = False
        THOP_AVAILABLE = False

'''
# --------------------------------------------
# training code for MSRResNet with structured pruning
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def count_flops(model, input_shape=(1, 3, 64, 64)):
    """Count FLOPs using available libraries."""
    device = next(model.parameters()).device
    
    if FVCORE_AVAILABLE:
        try:
            dummy_input = torch.randn(input_shape).to(device)
            flops_dict = flop_count(model, (dummy_input,), supported_ops=None)
            total_flops = sum(flops_dict.values())
            return total_flops
        except Exception as e:
            print(f"FvCore FLOP counting failed: {e}")
    
    if THOP_AVAILABLE:
        try:
            dummy_input = torch.randn(input_shape).to(device)
            model_copy = copy.deepcopy(model)
            flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
            return flops
        except Exception as e:
            print(f"THOP FLOP counting failed: {e}")
    
    print("No FLOP counting library available.")
    return None

def measure_inference_time(model, input_shape=(1, 3, 64, 64), num_runs=100):
    """Measure average inference time."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    
    model.eval()
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def evaluate_model_metrics(model, test_loader, border=0, max_test_samples=50):
    """Evaluate model performance (PSNR/SSIM) on test set."""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            if i >= max_test_samples:
                break
                
            model.feed_data(test_data)
            model.test()
            
            visuals = model.current_visuals()
            E_img = util.tensor2uint(visuals['E'])
            H_img = util.tensor2uint(visuals['H'])
            
            psnr = util.calculate_psnr(E_img, H_img, border=border)
            ssim = util.calculate_ssim(E_img, H_img, border=border)
            
            total_psnr += psnr
            total_ssim += ssim
            count += 1
    
    avg_psnr = total_psnr / count if count > 0 else 0
    avg_ssim = total_ssim / count if count > 0 else 0
    
    return avg_psnr, avg_ssim

def create_pruning_importance_dict(model):
    """Create importance dictionary for L1-norm based pruning."""
    importance_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.weight.data.numel() > 0:
            # Calculate L1 norm importance for each channel
            weight = module.weight.data
            # For output channels (dim=0)
            l1_norm = torch.norm(weight.view(weight.size(0), -1), p=1, dim=1)
            importance_dict[name] = l1_norm
    
    return importance_dict

def apply_structured_pruning_torch_pruning(model, pruning_ratio=0.3):
    """Apply structured pruning using torch_pruning library."""
    if not TORCH_PRUNING_AVAILABLE:
        print("torch_pruning not available, falling back to basic pruning")
        return apply_basic_structured_pruning(model, pruning_ratio)
    
    try:
        # Create example input
        device = next(model.parameters()).device
        example_inputs = torch.randn(1, 3, 64, 64).to(device)
        
        # Create dependency graph
        DG = tp.DependencyGraph()
        DG.build_dependency(model, example_inputs=example_inputs)
        
        # Define importance function (L1 norm)
        def importance_fn(layer):
            if hasattr(layer, 'weight') and layer.weight is not None:
                return torch.norm(layer.weight.data.view(layer.weight.size(0), -1), p=1, dim=1)
            return None
        
        # Apply pruning
        pruning_idxs = []
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                importance = importance_fn(m)
                if importance is not None:
                    num_channels = len(importance)
                    num_prune = int(pruning_ratio * num_channels)
                    if num_prune > 0:
                        _, sorted_idx = torch.sort(importance)
                        pruning_idx = sorted_idx[:num_prune].tolist()
                        pruning_idxs.extend(pruning_idx)
                        
                        # Apply pruning to this layer
                        pruning_plan = DG.get_pruning_plan(m, tp.prune_conv_out_channels, idxs=pruning_idx)
                        pruning_plan.exec()
        
        return model
        
    except Exception as e:
        print(f"Torch-Pruning failed: {e}, falling back to basic pruning")
        return apply_basic_structured_pruning(model, pruning_ratio)

def apply_basic_structured_pruning(model, pruning_ratio=0.3):
    """Apply basic structured pruning using PyTorch's built-in pruning."""
    importance_dict = create_pruning_importance_dict(model)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and name in importance_dict:
            importance = importance_dict[name]
            num_channels = len(importance)
            num_prune = int(pruning_ratio * num_channels)
            
            if num_prune > 0 and num_prune < num_channels:
                # Get indices of least important channels
                _, sorted_idx = torch.sort(importance)
                prune_idx = sorted_idx[:num_prune].tolist()
                
                # Apply structured pruning
                prune.ln_structured(module, name='weight', amount=pruning_ratio, n=1, dim=0)
    
    return model

def print_model_summary(model, title="Model Summary"):
    """Print comprehensive model summary."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Try to count FLOPs
    flops = count_flops(model)
    if flops is not None:
        if flops > 1e9:
            print(f"FLOPs: {flops/1e9:.2f} G")
        elif flops > 1e6:
            print(f"FLOPs: {flops/1e6:.2f} M")
        else:
            print(f"FLOPs: {flops:,}")
    
    # Model sparsity
    sparsity = util.compute_sparsity(model)
    print(f"Model sparsity: {sparsity:.2%}")
    
    print(f"{'='*50}\n")

def main(json_path='options/train_msrresnet_psnr.json'):

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
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    print("iterations : ", init_iter_optimizerG, init_path_optimizerG)
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
            # Randomly select 150 images
            train_set = torch.utils.data.Subset(train_set, random.sample(range(len(train_set)), min(150, len(train_set))))
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
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()

    # Results tracking
    results = {
        'before_pruning': {},
        'after_pruning': {},
        'iterations': []
    }
    
    # Initial model evaluation (before pruning)
    print("\n" + "="*60)
    print("INITIAL MODEL EVALUATION (BEFORE PRUNING)")
    print("="*60)
    
    print_model_summary(model.netG, "Original Model")
    
    # Evaluate original model performance
    original_psnr, original_ssim = evaluate_model_metrics(model, test_loader, border=border)
    original_inference_time = measure_inference_time(model.netG)
    original_params, _ = count_parameters(model.netG)
    original_flops = count_flops(model.netG)
    
    results['before_pruning'] = {
        'psnr': original_psnr,
        'ssim': original_ssim,
        'params': original_params,
        'flops': original_flops,
        'inference_time': original_inference_time,
        'sparsity': util.compute_sparsity(model.netG)
    }
    
    print(f"Original PSNR: {original_psnr:.4f} dB")
    print(f"Original SSIM: {original_ssim:.4f}")
    print(f"Original inference time: {original_inference_time:.4f}s")
    
    # Define target PSNR threshold (allow 0.2 dB drop)
    target_psnr_threshold = original_psnr - 0.2
    print(f"Target PSNR threshold: {target_psnr_threshold:.4f} dB")

    pruning_iteration = 0
    current_psnr = original_psnr
    max_pruning_iterations = 5  # Limit iterations to prevent infinite loop

    while current_psnr > target_psnr_threshold and pruning_iteration < max_pruning_iterations:
        pruning_iteration += 1

        print(f"\n" + "="*60)
        print(f"PRUNING ITERATION {pruning_iteration}")
        print("="*60)

        # Progressive pruning: start with smaller amounts and increase
        base_pruning_ratio = 0.2 + (pruning_iteration - 1) * 0.1  # 0.2, 0.3, 0.4, ...
        pruning_ratio = min(base_pruning_ratio, 0.4)  # Cap at 40%
        
        print(f"Applying structured pruning with ratio: {pruning_ratio:.1%}")
        
        # Apply structured pruning using Torch-Pruning or fallback
        if TORCH_PRUNING_AVAILABLE:
            print("Using Torch-Pruning for structured channel pruning...")
            model.netG = apply_structured_pruning_torch_pruning(model.netG, pruning_ratio)
        else:
            print("Using basic structured pruning...")
            model.netG = apply_basic_structured_pruning(model.netG, pruning_ratio)
        
        print_model_summary(model.netG, f"Model after pruning iteration {pruning_iteration}")
        
        # Re-initialize optimizer after pruning
        model.init_train()

        '''
        # ----------------------------------------
        # Step--4 (main training)
        # ----------------------------------------
        '''
        e_pochs = opt['fine_tune']['L2_ft_epochs']

        print("Fine-tuning epochs: ", e_pochs)

        for epoch in range(e_pochs):
            print("epoch : ", epoch)
            if opt['dist']:
                train_sampler.set_epoch(epoch + seed)

            for i, train_data in enumerate(train_loader):

                print("Current step: ", current_step)
                current_step += 1

                # -------------------------------
                # 1) Update learning rate
                # -------------------------------
                model.update_learning_rate(current_step)

                # -------------------------------
                # 2) Feed patch pairs
                # -------------------------------
                model.feed_data(train_data)

                # -------------------------------
                # 3) Optimize parameters
                # -------------------------------
                model.optimize_parameters(current_step)

            # -------------------------------
            # Training information
            # -------------------------------
            if opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = ''
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                print(message)

        # -------------------------------
        # Comprehensive Testing and Evaluation
        # -------------------------------
        if opt['rank'] == 0:
            print(f"\n" + "-"*50)
            print(f"EVALUATING PRUNED MODEL (Iteration {pruning_iteration})")
            print("-"*50)
            
            # Comprehensive evaluation
            current_psnr, current_ssim = evaluate_model_metrics(model, test_loader, border=border)
            current_inference_time = measure_inference_time(model.netG)
            current_params, _ = count_parameters(model.netG)
            current_flops = count_flops(model.netG)
            current_sparsity = util.compute_sparsity(model.netG)
            
            # Store iteration results
            iteration_results = {
                'iteration': pruning_iteration,
                'psnr': current_psnr,
                'ssim': current_ssim,
                'params': current_params,
                'flops': current_flops,
                'inference_time': current_inference_time,
                'sparsity': current_sparsity,
                'pruning_ratio': pruning_ratio
            }
            results['iterations'].append(iteration_results)
            
            # Print current results
            print(f"Current PSNR: {current_psnr:.4f} dB (Drop: {original_psnr - current_psnr:.4f} dB)")
            print(f"Current SSIM: {current_ssim:.4f}")
            print(f"Current inference time: {current_inference_time:.4f}s")
            print(f"Current parameters: {current_params:,}")
            if current_flops:
                print(f"Current FLOPs: {current_flops/1e6:.2f}M" if current_flops < 1e9 else f"{current_flops/1e9:.2f}G")
            print(f"Current sparsity: {current_sparsity:.2%}")
            
            # Calculate efficiency gains
            param_reduction = (original_params - current_params) / original_params * 100
            speed_up = original_inference_time / current_inference_time if current_inference_time > 0 else 1.0
            
            print(f"\nEfficiency gains so far:")
            print(f"Parameter reduction: {param_reduction:.1f}%")
            print(f"Speed up: {speed_up:.2f}x")
            
            if current_flops and original_flops:
                flop_reduction = (original_flops - current_flops) / original_flops * 100
                print(f"FLOP reduction: {flop_reduction:.1f}%")
            
            # Save some sample images from this iteration
            sample_count = 0
            for test_data in test_loader:
                if sample_count >= 5:  # Save only first 5 samples
                    break
                    
                image_name_ext = os.path.basename(test_data['L_path'][0])
                img_name, ext = os.path.splitext(image_name_ext)

                img_dir = os.path.join(opt['path']['images'], f"iteration_{pruning_iteration}")
                util.mkdir(img_dir)

                model.feed_data(test_data)
                model.test()

                visuals = model.current_visuals()
                E_img = util.tensor2uint(visuals['E'])

                # Save estimated image E
                save_img_path = os.path.join(img_dir, f'{img_name}_iter{pruning_iteration}.png')
                util.imsave(E_img, save_img_path)
                
                sample_count += 1

    # -------------------------------
    # Final Evaluation and Comparison
    # -------------------------------
    if opt['rank'] == 0:
        print("\n" + "="*80)
        print("FINAL PRUNING RESULTS AND COMPARISON")
        print("="*80)
        
        # Final model evaluation
        final_psnr, final_ssim = evaluate_model_metrics(model, test_loader, border=border)
        final_inference_time = measure_inference_time(model.netG)
        final_params, _ = count_parameters(model.netG)
        final_flops = count_flops(model.netG)
        final_sparsity = util.compute_sparsity(model.netG)
        
        results['after_pruning'] = {
            'psnr': final_psnr,
            'ssim': final_ssim,
            'params': final_params,
            'flops': final_flops,
            'inference_time': final_inference_time,
            'sparsity': final_sparsity
        }
        
        # Print comprehensive comparison
        print(f"\n{'Metric':<20} {'Original':<15} {'Pruned':<15} {'Change':<15} {'Improvement'}")
        print("-" * 80)
        
        # PSNR comparison
        psnr_drop = original_psnr - final_psnr
        print(f"{'PSNR (dB)':<20} {original_psnr:<15.4f} {final_psnr:<15.4f} {-psnr_drop:<15.4f} {'?' if psnr_drop <= 0.2 else '?'}")
        
        # SSIM comparison
        ssim_drop = original_ssim - final_ssim
        print(f"{'SSIM':<20} {original_ssim:<15.4f} {final_ssim:<15.4f} {-ssim_drop:<15.4f} {'?' if ssim_drop <= 0.01 else '?'}")
        
        # Parameters comparison
        param_reduction = (original_params - final_params) / original_params * 100
        print(f"{'Parameters':<20} {original_params:<15,} {final_params:<15,} {-param_reduction:<14.1f}% {'?' if param_reduction > 0 else '?'}")
        
        # FLOPs comparison
        if original_flops and final_flops:
            flop_reduction = (original_flops - final_flops) / original_flops * 100
            flop_orig_str = f"{original_flops/1e9:.2f}G" if original_flops > 1e9 else f"{original_flops/1e6:.1f}M"
            flop_final_str = f"{final_flops/1e9:.2f}G" if final_flops > 1e9 else f"{final_flops/1e6:.1f}M"
            print(f"{'FLOPs':<20} {flop_orig_str:<15} {flop_final_str:<15} {-flop_reduction:<14.1f}% {'?' if flop_reduction > 0 else '?'}")
        
        # Inference time comparison
        speed_up = original_inference_time / final_inference_time if final_inference_time > 0 else 1.0
        print(f"{'Inference Time (s)':<20} {original_inference_time:<15.4f} {final_inference_time:<15.4f} {speed_up:<14.2f}x {'?' if speed_up > 1.0 else '?'}")
        
        # Sparsity comparison
        sparsity_increase = final_sparsity - results['before_pruning']['sparsity']
        print(f"{'Sparsity (%)':<20} {results['before_pruning']['sparsity']*100:<14.1f}% {final_sparsity*100:<14.1f}% {sparsity_increase*100:<14.1f}% {'?' if sparsity_increase > 0 else '?'}")
        
        print("\n" + "="*80)
        
        # Summary
        success_criteria = []
        if psnr_drop <= 0.2:
            success_criteria.append("? PSNR drop ? 0.2 dB")
        else:
            success_criteria.append("? PSNR drop > 0.2 dB")
            
        if param_reduction > 0:
            success_criteria.append(f"? {param_reduction:.1f}% parameter reduction")
        else:
            success_criteria.append("? No parameter reduction")
            
        if speed_up > 1.0:
            success_criteria.append(f"? {speed_up:.2f}x speed improvement")
        else:
            success_criteria.append("? No speed improvement")
        
        print("PRUNING SUCCESS CRITERIA:")
        for criterion in success_criteria:
            print(f"  {criterion}")
        
        # Save results to JSON
        results_file = os.path.join(opt['path']['log'], 'pruning_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
        
        print_model_summary(model.netG, "Final Pruned Model")

    # -------------------------------
    # Save model
    # -------------------------------
    if opt['rank'] == 0:
        print('\nSaving the final pruned model...')
        
        # Remove pruning masks to make the pruning permanent
        modules_to_remove = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight_orig'):
                modules_to_remove.append((name, module))
        
        for name, module in modules_to_remove:
            print(f"Making pruning permanent for: {name}")
            prune.remove(module, 'weight')

        model.save(0)
        print("Final pruned model saved successfully!")

if __name__ == '__main__':
    main()
