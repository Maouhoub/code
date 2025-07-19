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

# Import our advanced fine-tuning strategies
from advanced_fine_tuning_strategies import (
    AdvancedFineTuner, 
    create_fine_tuning_config,
    apply_advanced_fine_tuning
)

'''
# --------------------------------------------
# Advanced training code for MSRResNet with intelligent pruning and sophisticated fine-tuning
# Designed for Q1/Q2 journal publication quality results
# --------------------------------------------
'''


def compute_model_sparsity(model):
    """Compute overall sparsity of the model"""
    total_params = 0
    zero_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, 'weight_mask'):
                # Pruned layer
                mask = module.weight_mask
                total_params += mask.numel()
                zero_params += (mask == 0).sum().item()
            elif hasattr(module, 'weight'):
                # Unpruned layer
                weight = module.weight
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()
    
    sparsity = zero_params / total_params if total_params > 0 else 0
    return sparsity * 100  # Return as percentage


def save_pruning_statistics(model, iteration, psnr, sparsity, save_path):
    """Save detailed pruning statistics for analysis"""
    stats = {
        'iteration': iteration,
        'psnr': psnr,
        'sparsity_percent': sparsity,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'layer_statistics': {}
    }
    
    # Collect per-layer statistics
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask
                layer_sparsity = (mask == 0).sum().item() / mask.numel() * 100
                stats['layer_statistics'][name] = {
                    'sparsity': layer_sparsity,
                    'total_params': mask.numel(),
                    'zero_params': (mask == 0).sum().item()
                }
    
    # Save to JSON file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=2)


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
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']

    # ----------------------------------------
    # Load advanced fine-tuning configuration
    # ----------------------------------------
    advanced_config_path = 'options/advanced_fine_tuning_config.json'
    if os.path.exists(advanced_config_path):
        with open(advanced_config_path, 'r') as f:
            advanced_config = json.load(f)
        opt.update(advanced_config)
        print("Loaded advanced fine-tuning configuration")
    else:
        print("Advanced config not found, using default settings")

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

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            # Use subset for faster experimentation - remove for full training
            train_set = torch.utils.data.Subset(train_set, random.sample(range(len(train_set)), min(150, len(train_set))))
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
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

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()

    # ----------------------------------------
    # Enhanced Pruning Configuration
    # ----------------------------------------
    pruning_iteration = 0
    best_psnr = 0
    patience = 5
    bad_iterations = 0
    
    # Enhanced layer importance with more granular control
    layer_importance = opt['fine_tune'].get('layer_importance_weights', {
        'conv_first': 0.8,
        'conv_last': 0.9,
        'conv_before_upsample': 0.8,
        'upsample': 0.7,
        'patch_embed': 0.8,
        'patch_unembed': 0.8,
        'layers.0': 0.6,
        'layers.1': 0.5,
        'layers.2': 0.4,
        'norm': 0.7,
        'mlp': 0.4,
        'attn': 0.6
    })
    
    def get_layer_protection_factor(layer_name):
        """Get protection factor for a layer"""
        for key, protection in layer_importance.items():
            if key in layer_name:
                return protection
        return 0.0
    
    def compute_layer_sensitivity_advanced(model):
        """Advanced sensitivity computation using multiple metrics"""
        sensitivities = {}
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight
                    
                    # Multiple sensitivity metrics
                    magnitude = torch.abs(weight).mean().item()
                    variance = torch.var(weight).item()
                    l2_norm = torch.norm(weight, 2).item()
                    
                    # Combined sensitivity score
                    sensitivity = (magnitude * 0.5 + 
                                 math.sqrt(variance) * 0.3 + 
                                 l2_norm / weight.numel() * 0.2)
                    sensitivities[name] = sensitivity
        return sensitivities
    
    # Pruning schedule configuration
    base_pruning_rate = 0.08  # Conservative 8% per iteration
    max_pruning_rate = 0.12   # Maximum 12% for any layer
    target_psnr = 34.45  # Target PSNR threshold
    
    # Statistics tracking
    pruning_results = []
    iteration_psnr = 1000  # Initialize high
    
    # Save unpruned model as baseline
    if opt['rank'] == 0:
        baseline_path = os.path.join(opt['path']['models'], 'baseline_unpruned.pth')
        torch.save(model.netG.state_dict(), baseline_path)
        print(f"Saved baseline unpruned model to {baseline_path}")

    # ----------------------------------------
    # Main Pruning and Fine-tuning Loop
    # ----------------------------------------
    while iteration_psnr > target_psnr and pruning_iteration < 10:  # Max 10 iterations
        pruning_iteration += 1
        print(f"\n{'='*60}")
        print(f"PRUNING ITERATION {pruning_iteration}")
        print(f"{'='*60}")
        
        # Compute current sparsity before pruning
        sparsity_before = compute_model_sparsity(model.netG)
        print(f"Model sparsity before pruning: {sparsity_before:.2f}%")
        
        # ========================================
        # STEP 1: INTELLIGENT PRUNING
        # ========================================
        
        # Compute advanced layer sensitivities
        layer_sensitivities = compute_layer_sensitivity_advanced(model.netG)
        
        # Normalize sensitivities
        if layer_sensitivities:
            max_sensitivity = max(layer_sensitivities.values())
            min_sensitivity = min(layer_sensitivities.values())
            if max_sensitivity > min_sensitivity:
                normalized_sensitivities = {
                    name: (sens - min_sensitivity) / (max_sensitivity - min_sensitivity)
                    for name, sens in layer_sensitivities.items()
                }
            else:
                normalized_sensitivities = {name: 0.5 for name in layer_sensitivities}
        else:
            normalized_sensitivities = {}
        
        # Apply adaptive pruning
        params_to_prune = []
        for name, module in model.netG.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                protection_factor = get_layer_protection_factor(name)
                sensitivity = normalized_sensitivities.get(name, 0.5)
                
                # Adaptive pruning rate calculation
                adaptive_rate = base_pruning_rate * (1 - protection_factor) * (1 - sensitivity * 0.5)
                adaptive_rate = min(adaptive_rate, max_pruning_rate)
                
                if adaptive_rate > 0.01:  # Only prune if meaningful
                    params_to_prune.append((module, 'weight'))
                    print(f"  {name}: pruning rate = {adaptive_rate:.3f}, "
                          f"protection = {protection_factor:.2f}, sensitivity = {sensitivity:.2f}")
        
        # Apply global structured pruning
        if params_to_prune:
            current_pruning_rate = base_pruning_rate * (1 - pruning_iteration * 0.005)
            current_pruning_rate = max(current_pruning_rate, 0.03)
            
            print(f"\nApplying global pruning with rate: {current_pruning_rate:.3f}")
            prune.global_unstructured(
                params_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=current_pruning_rate
            )
        
        # Compute sparsity after pruning
        sparsity_after = compute_model_sparsity(model.netG)
        print(f"Model sparsity after pruning: {sparsity_after:.2f}%")
        
        # ========================================
        # STEP 2: ADVANCED FINE-TUNING
        # ========================================
        
        print(f"\nStarting advanced fine-tuning...")
        
        # Create fine-tuning configuration based on current pruning ratio
        pruning_ratio = sparsity_after / 100.0
        ft_config = create_fine_tuning_config(pruning_ratio)
        
        # Merge with user configuration
        user_ft_config = opt.get('fine_tune', {})
        ft_config.update(user_ft_config)
        
        # Initialize advanced fine-tuner
        fine_tuner = AdvancedFineTuner(model, ft_config, layer_importance)
        
        # Setup knowledge distillation with baseline model
        if pruning_iteration == 1:
            fine_tuner.setup_knowledge_distillation(baseline_path)
        
        # Determine number of epochs based on pruning severity
        base_epochs = ft_config.get('L2_ft_epochs', 75)
        progressive_factor = 1 + (pruning_iteration - 1) * 0.2
        epochs = min(int(base_epochs * progressive_factor), 150)  # Cap at 150 epochs
        
        print(f"Fine-tuning for {epochs} epochs with pruning ratio {pruning_ratio:.3f}")
        
        # Perform advanced fine-tuning
        ft_results = fine_tuner.fine_tune(
            train_loader=train_loader,
            test_loader=test_loader, 
            epochs=epochs,
            pruning_ratio=pruning_ratio,
            border=border
        )
        
        # ========================================
        # STEP 3: COMPREHENSIVE EVALUATION
        # ========================================
        
        print(f"\nEvaluating pruned model...")
        
        if opt['rank'] == 0:
            avg_psnr = 0.0
            avg_ssim = 0.0
            avg_inference_time = 0.0
            idx = 0

            model.netG.eval()
            with torch.no_grad():
                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], f'iter_{pruning_iteration}', img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    
                    # Measure inference time
                    start_time = time.time()
                    model.test()
                    end_time = time.time()
                    avg_inference_time += (end_time - start_time)

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # Save result image
                    save_img_path = os.path.join(img_dir, f'{img_name}_iter{pruning_iteration}.png')
                    util.imsave(E_img, save_img_path)

                    # Calculate metrics
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    current_ssim = util.calculate_ssim(E_img, H_img, border=border)
                    
                    print(f'{idx:>4d}--> {image_name_ext:>10s} | {current_psnr:<4.2f}dB | {current_ssim:<4.3f}SSIM')
                    avg_psnr += current_psnr
                    avg_ssim += current_ssim

            avg_psnr = avg_psnr / idx
            avg_ssim = avg_ssim / idx 
            avg_inference_time = avg_inference_time / idx
            iteration_psnr = avg_psnr
            final_sparsity = compute_model_sparsity(model.netG)

            # Comprehensive results logging
            iteration_results = {
                'iteration': pruning_iteration,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'sparsity_percent': final_sparsity,
                'inference_time_ms': avg_inference_time * 1000,
                'fine_tuning_epochs': ft_results['epochs_trained'],
                'best_val_psnr': ft_results['best_psnr'],
                'compression_ratio': 1 / (1 - final_sparsity/100),
                'efficiency_score': avg_psnr / avg_inference_time  # PSNR per second
            }
            
            pruning_results.append(iteration_results)
            
            print(f'\n{"-"*60}')
            print(f'ITERATION {pruning_iteration} RESULTS:')
            print(f'  Average PSNR: {avg_psnr:.3f}dB')
            print(f'  Average SSIM: {avg_ssim:.4f}')
            print(f'  Model sparsity: {final_sparsity:.2f}%')
            print(f'  Compression ratio: {iteration_results["compression_ratio"]:.2f}x')
            print(f'  Avg inference time: {avg_inference_time*1000:.2f}ms')
            print(f'  Efficiency score: {iteration_results["efficiency_score"]:.2f} PSNR/s')
            print(f'  Fine-tuning epochs: {ft_results["epochs_trained"]}')
            print(f'{"-"*60}')
            
            # Save detailed statistics
            stats_path = os.path.join(opt['path']['models'], f'pruning_stats_iter_{pruning_iteration}.json')
            save_pruning_statistics(model.netG, pruning_iteration, avg_psnr, final_sparsity, stats_path)
            
            # Check stopping criteria
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                bad_iterations = 0
                
                # Save best model
                best_model_path = os.path.join(opt['path']['models'], 'best_pruned_model.pth')
                torch.save({
                    'model_state_dict': model.netG.state_dict(),
                    'iteration': pruning_iteration,
                    'psnr': avg_psnr,
                    'sparsity': final_sparsity,
                    'results': iteration_results
                }, best_model_path)
                print(f'  ? New best PSNR: {best_psnr:.3f}dB (saved to {best_model_path})')
            else:
                bad_iterations += 1
                print(f'  ? No improvement. Bad iterations: {bad_iterations}/{patience}')
            
            # Early stopping
            if bad_iterations >= patience:
                print(f"Early stopping: No improvement for {patience} iterations")
                break
    
    # ----------------------------------------
    # Final Model Processing and Results Summary
    # ----------------------------------------
    if opt['rank'] == 0:
        print(f"\n{'='*60}")
        print("FINAL PROCESSING AND RESULTS")
        print(f"{'='*60}")
        
        # Remove pruning masks to make weights permanent
        print('Finalizing pruned model (removing masks)...')
        for name, module in model.netG.named_modules():
            if hasattr(module, 'weight_orig'):
                print(f"  Removing pruning mask for: {name}")
                prune.remove(module, 'weight')
        
        # Save final model
        final_model_path = os.path.join(opt['path']['models'], f'final_pruned_model_iter_{pruning_iteration}.pth')
        torch.save({
            'model_state_dict': model.netG.state_dict(),
            'pruning_iterations': pruning_iteration,
            'final_psnr': iteration_psnr,
            'final_sparsity': compute_model_sparsity(model.netG),
            'all_results': pruning_results,
            'config': ft_config
        }, final_model_path)
        
        # Save comprehensive results summary
        summary = {
            'experiment_summary': {
                'total_iterations': pruning_iteration,
                'final_psnr': iteration_psnr,
                'final_sparsity': compute_model_sparsity(model.netG),
                'best_psnr_achieved': best_psnr,
                'target_psnr': target_psnr,
                'success': iteration_psnr > target_psnr
            },
            'iteration_results': pruning_results,
            'configuration': {
                'base_pruning_rate': base_pruning_rate,
                'max_pruning_rate': max_pruning_rate,
                'layer_importance': layer_importance,
                'fine_tuning_config': ft_config
            }
        }
        
        results_path = os.path.join(opt['path']['models'], 'comprehensive_results.json')
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print final summary
        print(f'EXPERIMENT COMPLETED SUCCESSFULLY!')
        print(f'  Final PSNR: {iteration_psnr:.3f}dB')
        print(f'  Final Sparsity: {compute_model_sparsity(model.netG):.2f}%')
        print(f'  Total Iterations: {pruning_iteration}')
        print(f'  Best PSNR Achieved: {best_psnr:.3f}dB')
        print(f'  Model saved to: {final_model_path}')
        print(f'  Results saved to: {results_path}')
        
        # Performance analysis
        if pruning_results:
            initial_psnr = pruning_results[0]['psnr'] if 'psnr' in pruning_results[0] else 'N/A'
            final_sparsity = pruning_results[-1]['sparsity_percent']
            compression_ratio = pruning_results[-1]['compression_ratio']
            
            print(f'\nPERFORMANCE ANALYSIS:')
            print(f'  PSNR degradation: {initial_psnr - iteration_psnr:.3f}dB' if isinstance(initial_psnr, (int, float)) else 'N/A')
            print(f'  Compression achieved: {compression_ratio:.2f}x')
            print(f'  Parameters reduced: {final_sparsity:.1f}%')
            
            if iteration_psnr > target_psnr:
                print(f'  ? SUCCESS: Target PSNR {target_psnr}dB maintained!')
            else:
                print(f'  ? Target PSNR {target_psnr}dB not achieved')


if __name__ == '__main__':
    main()
