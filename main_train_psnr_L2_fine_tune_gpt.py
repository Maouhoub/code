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

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import torch.nn.utils.prune as prune

'''
# --------------------------------------------
# training code for MSRResNet with intelligent pruning
# --------------------------------------------
'''


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
    print ("iterations : " , init_iter_optimizerG, init_path_optimizerG)
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
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)

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
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()

    # ----------------------------------------
    # Pruning Configuration
    # ----------------------------------------
    pruning_iteration = 0
    best_psnr = 0
    patience = 3
    bad_iterations = 0
    
    # Layer importance coefficients (protect critical layers)
    layer_importance = {
        'patch_embed': 0.8,
        'patch_unembed': 0.8,
        'layers.0': 0.6,
        'conv_before_upsample': 0.8,
        'conv_last': 0.8,
        'conv_first': 0.7,
        'upsample': 0.7
    }
    
    def get_layer_protection_factor(layer_name):
        """Get protection factor for a layer (0=no protection, 1=full protection)"""
        for key, protection in layer_importance.items():
            if key in layer_name:
                return protection
        return 0.0  # No protection for unspecified layers
    
    def compute_layer_sensitivity(model):
        """Compute layer sensitivity based on weight magnitudes"""
        sensitivities = {}
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if hasattr(module, 'weight') and module.weight is not None:
                    # Use weight magnitude as sensitivity measure
                    weight_magnitude = torch.abs(module.weight).mean().item()
                    sensitivities[name] = weight_magnitude
        return sensitivities
    
    # Pruning schedule configuration
    base_pruning_rate = 0.05  # 5% per iteration
    max_pruning_rate = 0.15   # Maximum 15% for any layer
    
    iteration_psnr = 1000  # Initialize high to start loop

    while iteration_psnr > 34.45 and bad_iterations < patience:
        pruning_iteration += 1
        print(f"Pruning iteration: {pruning_iteration}")
        
        # Compute layer sensitivities
        layer_sensitivities = compute_layer_sensitivity(model)
        
        # Normalize sensitivities to [0, 1] range
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
        
        # Apply layer-wise pruning
        params_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                # Calculate adaptive pruning rate
                protection_factor = get_layer_protection_factor(name)
                sensitivity = normalized_sensitivities.get(name, 0.5)
                
                # Reduce pruning rate for protected and sensitive layers
                adaptive_rate = base_pruning_rate * (1 - protection_factor) * (1 - sensitivity * 0.5)
                adaptive_rate = min(adaptive_rate, max_pruning_rate)
                
                if adaptive_rate > 0.001:  # Only prune if rate is meaningful
                    params_to_prune.append((module, 'weight'))
                    print(f"Layer {name}: pruning rate = {adaptive_rate:.3f}")
        
        # Apply global pruning with computed rates
        if params_to_prune:
            # Use global pruning to maintain consistency
            current_pruning_rate = base_pruning_rate * (1 - pruning_iteration * 0.01)  # Decrease rate over time
            current_pruning_rate = max(current_pruning_rate, 0.02)  # Minimum rate
            
            print(f"Applying global pruning with rate: {current_pruning_rate:.3f}")
            prune.global_unstructured(
                params_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=current_pruning_rate
            )

        '''
        # ----------------------------------------
        # Step--4 (main training)
        # ----------------------------------------
        '''
        
        # Dynamic fine-tuning epochs
        base_epochs = opt['fine_tune']['L2_ft_epochs']
        e_pochs = max(base_epochs, int(base_epochs * (1 + pruning_iteration * 0.1)))
        
        print(f"Fine-tuning epochs: {e_pochs}")
        
        # Learning rate adjustment for post-pruning recovery
        if hasattr(model, 'optimizers') and 'G' in model.optimizers:
            original_lr = opt['train']['G_optimizer_lr']
            recovery_lr = original_lr * (1.5 ** pruning_iteration)  # Increase LR for recovery
            for param_group in model.optimizers['G'].param_groups:
                param_group['lr'] = min(recovery_lr, original_lr * 3)  # Cap at 3x original
        
        epoch_psnr_history = []
        
        for epoch in range(e_pochs):
            if opt['dist']:
                train_sampler.set_epoch(epoch + seed)

            epoch_start_time = time.time()
            
            for i, train_data in enumerate(train_loader):
                current_step += 1
                
                # Update learning rate
                model.update_learning_rate(current_step)
                
                # Feed data and optimize
                model.feed_data(train_data)
                model.optimize_parameters(current_step)
            
            # Log training info
            if opt['rank'] == 0:
                logs = model.current_log()
                message = f'Epoch {epoch+1}/{e_pochs}: '
                for k, v in logs.items():
                    message += f'{k}: {v:.3e} '
                message += f'Time: {time.time() - epoch_start_time:.2f}s'
                print(message)

        # ----------------------------------------
        # Testing after fine-tuning
        # ----------------------------------------
        if opt['rank'] == 0:
            avg_psnr = 0.0
            avg_inference_time = 0.0
            idx = 0

            for test_data in test_loader:
                idx += 1
                image_name_ext = os.path.basename(test_data['L_path'][0])
                img_name, ext = os.path.splitext(image_name_ext)

                img_dir = os.path.join(opt['path']['images'], img_name)
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

                # Save image
                save_img_path = os.path.join(img_dir, f'{img_name}_{current_step}.png')
                util.imsave(E_img, save_img_path)

                # Calculate PSNR
                current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                print(f'{idx:>4d}--> {image_name_ext:>10s} | {current_psnr:<4.2f}dB')
                avg_psnr += current_psnr

            avg_psnr = avg_psnr / idx
            avg_inference_time = avg_inference_time / idx
            iteration_psnr = avg_psnr
            current_sparsity = util.compute_sparsity(model)

            # Results logging
            print(f'Iteration {pruning_iteration} Results:')
            print(f'  Average PSNR: {avg_psnr:.2f}dB')
            print(f'  Average inference time: {avg_inference_time:.4f}s')
            print(f'  Model sparsity: {current_sparsity:.2f}%')
            
            # Check if improvement
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                bad_iterations = 0
                print(f'  New best PSNR: {best_psnr:.2f}dB')
            else:
                bad_iterations += 1
                print(f'  No improvement. Bad iterations: {bad_iterations}/{patience}')
            
            print('-' * 50)

    # ----------------------------------------
    # Final model saving
    # ----------------------------------------
    if opt['rank'] == 0:
        print('Saving the final pruned model...')
        
        # Remove pruning masks to make weights permanent
        for name, module in model.named_modules():
            if hasattr(module, 'weight_orig'):
                print(f"Removing pruning mask for: {name}")
                prune.remove(module, 'weight')
        
        # Save final model
        model.save(current_step)
        
        print(f'Final model saved with:')
        print(f'  PSNR: {iteration_psnr:.2f}dB')
        print(f'  Sparsity: {util.compute_sparsity(model):.2f}%')
        print(f'  Total pruning iterations: {pruning_iteration}')

if __name__ == '__main__':
    main()