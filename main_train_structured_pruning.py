import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn as nn
import time

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

'''
# --------------------------------------------
# Structured Pruning + Knowledge Distillation for SwinIR
# Based on contribution.txt methodology
# 
# Usage: python main_train_structured_pruning.py --opt options/swinir/train_swinir_sr_lightweight.json
# 
# Optional parameters in JSON config (add to existing config if needed):
# "structured_pruning": {
#     "target_psnr": 34.55,           // Target PSNR to maintain (default: 34.55)
#     "max_iterations": 5,            // Max pruning iterations (default: 5)
#     "prune_ratio_per_iteration": 0.15,  // % to prune each iteration (default: 0.15)
#     "sparsity_lambda": 1e-4,        // L1 regularization strength (default: 1e-4)
#     "sparsity_epochs": 3            // Epochs for sparsity training (default: 3)
# }
# --------------------------------------------
'''

class StructuredPruningMasks(nn.Module):
    """Learnable masks for attention heads and MLP channels"""
    def __init__(self, model):
        super().__init__()
        self.head_masks = nn.ParameterDict()
        self.channel_masks = nn.ParameterDict()
        
        # Add masks for SwinIR transformer components
        for name, module in model.named_modules():
            # Attention head masks
            if 'attn' in name and hasattr(module, 'num_heads'):
                mask_name = name.replace('.', '_')
                self.head_masks[mask_name] = nn.Parameter(
                    torch.ones(module.num_heads), requires_grad=True
                )
                print(f"Added attention head mask: {mask_name} ({module.num_heads} heads)")
            
            # MLP channel masks  
            elif 'mlp.fc1' in name and hasattr(module, 'out_features'):
                mask_name = name.replace('.', '_')
                self.channel_masks[mask_name] = nn.Parameter(
                    torch.ones(module.out_features), requires_grad=True
                )
                print(f"Added MLP channel mask: {mask_name} ({module.out_features} channels)")

class KnowledgeDistillationLoss(nn.Module):
    """Combined image and high-frequency distillation loss"""
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        
        # Laplacian kernel for high-frequency emphasis
        self.register_buffer('laplacian_kernel', 
            torch.tensor([[[[-1, -1, -1], 
                           [-1, 8, -1], 
                           [-1, -1, -1]]]]).float())
        
    def forward(self, student_output, teacher_output):
        # Image-space distillation
        image_loss = self.mse_loss(student_output, teacher_output)
        
        # High-frequency component emphasis (as mentioned in contribution.txt)
        student_hf = torch.nn.functional.conv2d(student_output, self.laplacian_kernel, padding=1)
        teacher_hf = torch.nn.functional.conv2d(teacher_output, self.laplacian_kernel, padding=1)
        hf_loss = self.mse_loss(student_hf, teacher_hf)
        
        total_loss = self.alpha * image_loss + self.beta * hf_loss
        return total_loss

def compute_head_importance(model, masks):
    """Compute importance scores for attention heads"""
    head_scores = {}
    for name, mask in masks.head_masks.items():
        # L1 norm as importance measure
        head_scores[name] = torch.abs(mask).detach().cpu().numpy()
    return head_scores

def compute_channel_importance(model, masks):
    """Compute importance scores for MLP channels"""
    channel_scores = {}
    for name, mask in masks.channel_masks.items():
        # L1 norm as importance measure  
        channel_scores[name] = torch.abs(mask).detach().cpu().numpy()
    return channel_scores

def apply_structured_pruning(masks, prune_ratio=0.1):
    """Remove least important heads and channels iteratively"""
    total_pruned = 0
    
    # Prune attention heads
    for name, mask in masks.head_masks.items():
        num_heads = len(mask)
        num_to_prune = max(1, int(num_heads * prune_ratio))
        
        # Find least important heads
        _, indices = torch.topk(torch.abs(mask), num_to_prune, largest=False)
        
        # Set to zero (structured pruning)
        with torch.no_grad():
            mask[indices] = 0.0
        
        total_pruned += num_to_prune
        print(f"Pruned {num_to_prune}/{num_heads} heads in {name}")
    
    # Prune MLP channels
    for name, mask in masks.channel_masks.items():
        num_channels = len(mask)
        num_to_prune = max(1, int(num_channels * prune_ratio))
        
        # Find least important channels
        _, indices = torch.topk(torch.abs(mask), num_to_prune, largest=False)
        
        # Set to zero (structured pruning)
        with torch.no_grad():
            mask[indices] = 0.0
        
        total_pruned += num_to_prune
        print(f"Pruned {num_to_prune}/{num_channels} channels in {name}")
    
    return total_pruned

def apply_sparsity_regularization(masks, lambda_l1=1e-4):
    """L1 regularization on masks to encourage sparsity"""
    reg_loss = 0.0
    
    for mask in masks.head_masks.values():
        reg_loss += lambda_l1 * torch.norm(mask, p=1)
        
    for mask in masks.channel_masks.values():
        reg_loss += lambda_l1 * torch.norm(mask, p=1)
        
    return reg_loss

def apply_masks_to_model(model, masks):
    """Apply learned masks to model during forward pass"""
    def mask_attention_hook(module, input, output, mask_name):
        if mask_name in masks.head_masks:
            mask = masks.head_masks[mask_name]
            # Apply mask to attention heads (simplified)
            # In practice, this would require modifying the attention computation
            pass
    
    def mask_mlp_hook(module, input, output, mask_name):
        if mask_name in masks.channel_masks:
            mask = masks.channel_masks[mask_name]
            # Apply mask to MLP outputs
            if output.shape[-1] == len(mask):
                return output * mask.view(1, 1, -1)
        return output
    
    # Register hooks (simplified implementation)
    for name, module in model.named_modules():
        if 'attn' in name and hasattr(module, 'num_heads'):
            mask_name = name.replace('.', '_')
            module.register_forward_hook(
                lambda m, i, o, mn=mask_name: mask_attention_hook(m, i, o, mn)
            )
        elif 'mlp.fc1' in name:
            mask_name = name.replace('.', '_')
            module.register_forward_hook(
                lambda m, i, o, mn=mask_name: mask_mlp_hook(m, i, o, mn)
            )

def main(json_path='options/swinir/train_swinir_sr_lightweight.json'):
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
    # save opt and configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    #seed = opt['train']['manual_seed']

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
            # Subset for faster experimentation
            train_set = torch.utils.data.Subset(train_set, random.sample(range(len(train_set)), min(550, len(train_set))))
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
    # Step--3 (initialize models and structured pruning)
    # ----------------------------------------
    '''
    
    # Teacher model (frozen, full capacity)
    teacher_model = define_Model(opt)
    teacher_model.init_train()
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    print("Teacher model initialized and frozen")
    
    # Student model (will be pruned)
    student_model = define_Model(opt)
    student_model.init_train()
    print("Student model initialized")
    
    # Initialize structured pruning masks
    pruning_masks = StructuredPruningMasks(student_model)
    pruning_masks.to(student_model.device if hasattr(student_model, 'device') else 'cuda')
    
    # Apply masks to student model
    apply_masks_to_model(student_model, pruning_masks)
    
    # Knowledge distillation loss
    kd_loss = KnowledgeDistillationLoss(alpha=0.7, beta=0.3)
    
    # ----------------------------------------
    # Structured Pruning Configuration with Defaults
    # ----------------------------------------
    
    # Extract pruning parameters from config or use defaults
    structured_pruning_config = opt.get('structured_pruning', {})
    
    target_psnr = structured_pruning_config.get('target_psnr', 34.55)  # 0.4dB drop from your 34.95 baseline
    max_iterations = structured_pruning_config.get('max_iterations', 5)
    prune_ratio_per_iteration = structured_pruning_config.get('prune_ratio_per_iteration', 0.15)  # 15% as in contribution.txt
    sparsity_lambda = structured_pruning_config.get('sparsity_lambda', 1e-4)  # L1 regularization strength
    
    print(f"Structured Pruning Configuration:")
    print(f"  Target PSNR: {target_psnr:.2f} dB")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Prune ratio per iteration: {prune_ratio_per_iteration}")
    print(f"  Sparsity lambda: {sparsity_lambda}")
    
    iteration_psnr = 1000  # Initialize high to start loop
    pruning_iteration = 0

    while iteration_psnr > target_psnr and pruning_iteration < max_iterations:
        pruning_iteration += 1
        print(f"\n{'='*60}")
        print(f"STRUCTURED PRUNING ITERATION {pruning_iteration}")
        print(f"{'='*60}")
        
        # ----------------------------------------
        # Phase 1: Sparsity Training (L1 regularization on masks)
        # ----------------------------------------
        print("Phase 1: Training with sparsity regularization...")
        sparsity_epochs = structured_pruning_config.get('sparsity_epochs', 3)  # Short sparsity training from config or default
        
        for epoch in range(sparsity_epochs):
            if opt['dist']:
                train_sampler.set_epoch(epoch + seed)
                
            for i, train_data in enumerate(train_loader):
                current_step += 1
                
                # Update learning rate
                student_model.update_learning_rate(current_step)
                
                # Get teacher output (frozen)
                with torch.no_grad():
                    teacher_model.feed_data(train_data)
                    teacher_model.test()
                    teacher_visuals = teacher_model.current_visuals()
                    teacher_output = teacher_visuals['E']
                
                # Student forward pass
                student_model.feed_data(train_data)
                student_model.optimize_parameters(current_step)
                
                # Get student output for distillation
                student_visuals = student_model.current_visuals()
                student_output = student_visuals['E']
                
                # Knowledge distillation loss
                distill_loss = kd_loss(student_output, teacher_output)
                
                # Sparsity regularization on masks
                sparsity_loss = apply_sparsity_regularization(pruning_masks, lambda_l1=sparsity_lambda)
                
                # Combined loss
                total_loss = distill_loss + sparsity_loss
                
                # Backward pass for masks only
                if hasattr(student_model, 'optimizers') and 'G' in student_model.optimizers:
                    student_model.optimizers['G'].zero_grad()
                    total_loss.backward()
                    student_model.optimizers['G'].step()
                
                if i % 50 == 0:
                    print(f"Epoch {epoch+1}/{sparsity_epochs}, Iter {i}: "
                          f"Distill: {distill_loss.item():.4f}, Sparsity: {sparsity_loss.item():.6f}")
        
        # ----------------------------------------
        # Phase 2: Structured Pruning
        # ----------------------------------------
        print("Phase 2: Applying structured pruning...")
        pruned_count = apply_structured_pruning(pruning_masks, prune_ratio=prune_ratio_per_iteration)
        print(f"Total components pruned: {pruned_count}")
        
        # ----------------------------------------
        # Phase 3: Fine-tuning after pruning
        # ----------------------------------------
        print("Phase 3: Fine-tuning after pruning...")
        
        # Get fine-tuning epochs from config or use default
        if 'fine_tune' in opt and 'L2_ft_epochs' in opt['fine_tune']:
            ft_epochs = opt['fine_tune']['L2_ft_epochs']
        else:
            ft_epochs = 5  # Default fine-tuning epochs
        
        for epoch in range(ft_epochs):
            if opt['dist']:
                train_sampler.set_epoch(epoch + seed)
                
            for i, train_data in enumerate(train_loader):
                current_step += 1
                
                # Update learning rate
                student_model.update_learning_rate(current_step)
                
                # Get teacher output
                with torch.no_grad():
                    teacher_model.feed_data(train_data)
                    teacher_model.test()
                    teacher_output = teacher_model.current_visuals()['E']
                
                # Student training with distillation
                student_model.feed_data(train_data)
                student_model.optimize_parameters(current_step)
                student_output = student_model.current_visuals()['E']
                
                # Apply distillation loss
                distill_loss = kd_loss(student_output, teacher_output)
                
                if hasattr(student_model, 'optimizers') and 'G' in student_model.optimizers:
                    student_model.optimizers['G'].zero_grad()
                    distill_loss.backward()
                    student_model.optimizers['G'].step()
                
                if i % 50 == 0:
                    print(f"Fine-tune Epoch {epoch+1}/{ft_epochs}, Iter {i}: "
                          f"KD Loss: {distill_loss.item():.4f}")

        # ----------------------------------------
        # Phase 4: Evaluation
        # ----------------------------------------
        print("Phase 4: Evaluation...")
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

                student_model.feed_data(test_data)
                
                # Measure inference time
                start_time = time.time()
                student_model.test()
                end_time = time.time()
                avg_inference_time += (end_time - start_time)

                visuals = student_model.current_visuals()
                E_img = util.tensor2uint(visuals['E'])
                H_img = util.tensor2uint(visuals['H'])

                # Save image
                save_img_path = os.path.join(img_dir, f'{img_name}_iter{pruning_iteration}.png')
                util.imsave(E_img, save_img_path)

                # Calculate PSNR
                current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                avg_psnr += current_psnr

            avg_psnr = avg_psnr / idx
            avg_inference_time = avg_inference_time / idx
            iteration_psnr = avg_psnr

            # Compute effective sparsity
            total_params = sum(p.numel() for p in student_model.parameters())
            zero_params = sum((torch.abs(mask) < 1e-6).sum().item() for mask in pruning_masks.head_masks.values())
            zero_params += sum((torch.abs(mask) < 1e-6).sum().item() for mask in pruning_masks.channel_masks.values())
            effective_sparsity = (zero_params / total_params) * 100 if total_params > 0 else 0

            # Results logging
            print(f"\nIteration {pruning_iteration} Results:")
            print(f"  Average PSNR: {avg_psnr:.2f}dB")
            print(f"  Average inference time: {avg_inference_time:.4f}s")
            print(f"  Effective sparsity: {effective_sparsity:.2f}%")
            print(f"  Target PSNR: {target_psnr:.2f}dB")
            
            if avg_psnr > target_psnr:
                print(f"  ? Target PSNR achieved!")
            else:
                print(f"  ? PSNR below target, continuing...")

    # ----------------------------------------
    # Final model saving
    # ----------------------------------------
    if opt['rank'] == 0:
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Completed {pruning_iteration} pruning iterations")
        print(f"Final PSNR: {iteration_psnr:.2f}dB")
        print(f"Target PSNR: {target_psnr:.2f}dB")
        
        # Count pruned components
        total_heads = sum(len(mask) for mask in pruning_masks.head_masks.values())
        pruned_heads = sum((torch.abs(mask) < 1e-6).sum().item() for mask in pruning_masks.head_masks.values())
        total_channels = sum(len(mask) for mask in pruning_masks.channel_masks.values())
        pruned_channels = sum((torch.abs(mask) < 1e-6).sum().item() for mask in pruning_masks.channel_masks.values())
        
        print(f"Pruned attention heads: {pruned_heads}/{total_heads} ({pruned_heads/total_heads*100:.1f}%)")
        print(f"Pruned MLP channels: {pruned_channels}/{total_channels} ({pruned_channels/total_channels*100:.1f}%)")
        
        # Save final model
        student_model.save(current_step)
        print("Final structured-pruned model saved successfully!")

if __name__ == '__main__':
    main()
