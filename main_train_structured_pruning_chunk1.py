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
import torch.nn.functional as F
import time

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

'''
# --------------------------------------------
# Structured Pruning for SwinIR - Chunk 1: Importance Scoring & Mask Infrastructure
# Implementation of learnable importance masks for attention heads and MLP channels
# --------------------------------------------
'''

class ImportanceMaskModule(nn.Module):
    """
    Learnable importance mask for attention heads or MLP channels
    """
    def __init__(self, num_elements, init_value=1.0):
        super().__init__()
        # Learnable scaling coefficients
        self.importance_scores = nn.Parameter(torch.full((num_elements,), init_value, dtype=torch.float32))
        self.num_elements = num_elements
        
    def forward(self, x, dim=-1):
        """
        Apply importance scaling to input tensor
        Args:
            x: input tensor
            dim: dimension to apply scaling along
        """
        # Reshape importance scores to match input dimensions
        shape = [1] * len(x.shape)
        shape[dim] = self.num_elements
        mask = self.importance_scores.view(shape)
        return x * mask
    
    def get_importance_scores(self):
        """Get current importance scores"""
        return self.importance_scores.detach().clone()
    
    def compute_l1_loss(self):
        """Compute L1 regularization loss for sparsity"""
        return torch.sum(torch.abs(self.importance_scores))

class AttentionHeadMask(ImportanceMaskModule):
    """Specific mask for attention heads"""
    def __init__(self, num_heads, init_value=1.0):
        super().__init__(num_heads, init_value)
        self.num_heads = num_heads
    
    def forward(self, attention_output):
        """
        Apply head-wise importance scaling to attention output
        Assumes attention_output shape: (B, N, num_heads * head_dim)
        """
        B, N, total_dim = attention_output.shape
        head_dim = total_dim // self.num_heads
        
        # Reshape to separate heads: (B, N, num_heads, head_dim)
        attention_output = attention_output.view(B, N, self.num_heads, head_dim)
        
        # Apply importance scaling per head
        mask = self.importance_scores.view(1, 1, self.num_heads, 1)
        scaled_output = attention_output * mask
        
        # Reshape back: (B, N, total_dim)
        return scaled_output.view(B, N, total_dim)

class MLPChannelMask(ImportanceMaskModule):
    """Specific mask for MLP channels"""
    def __init__(self, num_channels, init_value=1.0):
        super().__init__(num_channels, init_value)
    
    def forward(self, mlp_output):
        """Apply channel-wise importance scaling to MLP output"""
        return super().forward(mlp_output, dim=-1)

class StructuredPruner:
    """
    Main pruning manager that handles importance scoring and mask infrastructure
    """
    def __init__(self, model):
        self.model = model
        self.head_masks = {}
        self.channel_masks = {}
        self.layer_info = {}
        self._initialize_masks()
    
    def _initialize_masks(self):
        """Initialize importance masks for all relevant layers"""
        print("Initializing importance masks...")
        
        for name, module in self.model.named_modules():
            # Handle SwinIR attention layers
            if hasattr(module, 'num_heads') and hasattr(module, 'dim'):
                # This is likely a WindowAttention module
                mask = AttentionHeadMask(module.num_heads)
                self.head_masks[name] = mask
                self.layer_info[name] = {
                    'type': 'attention',
                    'num_heads': module.num_heads,
                    'dim': module.dim
                }
                print(f"  Added attention head mask for {name}: {module.num_heads} heads")
            
            # Handle MLP layers (Linear layers in MLP blocks)
            elif isinstance(module, nn.Linear) and 'mlp' in name.lower():
                mask = MLPChannelMask(module.out_features)
                self.channel_masks[name] = mask
                self.layer_info[name] = {
                    'type': 'mlp',
                    'in_features': module.in_features,
                    'out_features': module.out_features
                }
                print(f"  Added MLP channel mask for {name}: {module.out_features} channels")
        
        print(f"Initialized {len(self.head_masks)} attention masks and {len(self.channel_masks)} channel masks")
        
        # Register masks as model parameters
        for name, mask in self.head_masks.items():
            safe_name = name.replace('.', '_').replace('[', '_').replace(']', '_')
            self.model.add_module(f"head_mask_{safe_name}", mask)
        
        for name, mask in self.channel_masks.items():
            safe_name = name.replace('.', '_').replace('[', '_').replace(']', '_')
            self.model.add_module(f"channel_mask_{safe_name}", mask)
    
    def compute_regularization_loss(self, lambda_l1=1e-4):
        """
        Compute L1 regularization loss to encourage sparsity
        """
        total_loss = 0.0
        num_masks = 0
        
        # L1 loss for attention head masks
        for mask in self.head_masks.values():
            total_loss += mask.compute_l1_loss()
            num_masks += 1
        
        # L1 loss for channel masks
        for mask in self.channel_masks.values():
            total_loss += mask.compute_l1_loss()
            num_masks += 1
        
        if num_masks > 0:
            return lambda_l1 * total_loss / num_masks
        return torch.tensor(0.0, device=total_loss.device if num_masks > 0 else 'cpu')
    
    def get_importance_summary(self):
        """Get summary of current importance scores"""
        summary = {
            'attention_heads': {},
            'mlp_channels': {},
            'statistics': {}
        }
        
        all_scores = []
        
        # Attention head importance
        for name, mask in self.head_masks.items():
            scores = mask.get_importance_scores()
            summary['attention_heads'][name] = {
                'scores': scores.cpu().numpy(),
                'mean': scores.mean().item(),
                'min': scores.min().item(),
                'max': scores.max().item(),
                'std': scores.std().item()
            }
            all_scores.extend(scores.cpu().numpy())
        
        # MLP channel importance
        for name, mask in self.channel_masks.items():
            scores = mask.get_importance_scores()
            summary['mlp_channels'][name] = {
                'scores': scores.cpu().numpy(),
                'mean': scores.mean().item(),
                'min': scores.min().item(),
                'max': scores.max().item(),
                'std': scores.std().item()
            }
            all_scores.extend(scores.cpu().numpy())
        
        # Overall statistics
        if all_scores:
            all_scores = np.array(all_scores)
            summary['statistics'] = {
                'total_parameters': len(all_scores),
                'mean': np.mean(all_scores),
                'min': np.min(all_scores),
                'max': np.max(all_scores),
                'std': np.std(all_scores),
                'near_zero_ratio': np.sum(np.abs(all_scores) < 0.1) / len(all_scores)
            }
        
        return summary
    
    def print_importance_summary(self):
        """Print a readable summary of importance scores"""
        summary = self.get_importance_summary()
        
        print("\n" + "="*60)
        print("IMPORTANCE SCORES SUMMARY")
        print("="*60)
        
        print(f"Total masked parameters: {summary['statistics'].get('total_parameters', 0)}")
        print(f"Overall statistics:")
        stats = summary['statistics']
        print(f"  Mean: {stats.get('mean', 0):.4f}")
        print(f"  Range: [{stats.get('min', 0):.4f}, {stats.get('max', 0):.4f}]")
        print(f"  Std: {stats.get('std', 0):.4f}")
        print(f"  Near-zero ratio: {stats.get('near_zero_ratio', 0):.2%}")
        
        print(f"\nAttention head masks: {len(summary['attention_heads'])}")
        for name, info in summary['attention_heads'].items():
            print(f"  {name}: mean={info['mean']:.4f}, std={info['std']:.4f}")
        
        print(f"\nMLP channel masks: {len(summary['mlp_channels'])}")
        for name, info in summary['mlp_channels'].items():
            print(f"  {name}: mean={info['mean']:.4f}, std={info['std']:.4f}")
        
        print("="*60)

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
    print("iterations :", init_iter_optimizerG, init_path_optimizerG)
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']

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
    # CHUNK 1: Initialize Structured Pruner with Importance Masks
    # ----------------------------------------
    print("\n" + "="*60)
    print("INITIALIZING STRUCTURED PRUNING - CHUNK 1")
    print("="*60)
    
    # Initialize the structured pruner
    pruner = StructuredPruner(model)
    
    # Regularization configuration
    lambda_l1 = 1e-4  # L1 regularization strength
    print(f"L1 regularization lambda: {lambda_l1}")
    
    '''
    # ----------------------------------------
    # Step--4 (training with importance scoring)
    # ----------------------------------------
    '''
    # Training configuration
    num_epochs = opt.get('train', {}).get('epochs', 10)
    print(f"Training epochs: {num_epochs}")
    
    # Training loop with importance scoring
    for epoch in range(num_epochs):
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)

        epoch_start_time = time.time()
        epoch_losses = []
        epoch_reg_losses = []
        
        for i, train_data in enumerate(train_loader):
            current_step += 1
            
            # Update learning rate
            model.update_learning_rate(current_step)
            
            # Feed data and get outputs
            model.feed_data(train_data)
            
            # Get the base model loss
            model.optimize_parameters(current_step)
            
            # Add importance regularization loss
            if hasattr(model, 'optimizer_G'):
                model.optimizer_G.zero_grad()
                
                # Forward pass to get outputs
                model.test()  # Get model outputs
                
                # Compute regularization loss
                reg_loss = pruner.compute_regularization_loss(lambda_l1)
                
                # Get the main loss from model
                logs = model.current_log()
                main_loss = logs.get('G_loss', 0)
                
                # Total loss = main loss + regularization
                if reg_loss.requires_grad:
                    total_loss = main_loss + reg_loss
                    total_loss.backward()
                    model.optimizer_G.step()
                    
                    epoch_losses.append(main_loss.item() if torch.is_tensor(main_loss) else main_loss)
                    epoch_reg_losses.append(reg_loss.item())
            
            # Periodic logging
            if current_step % 50 == 0:
                logs = model.current_log()
                recent_reg_loss = epoch_reg_losses[-1] if epoch_reg_losses else 0
                print(f"Step {current_step}: Main loss: {logs.get('G_loss', 0):.6f}, Reg loss: {recent_reg_loss:.6f}")
        
        # End of epoch logging
        if opt['rank'] == 0:
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            avg_reg_loss = np.mean(epoch_reg_losses) if epoch_reg_losses else 0
            
            print(f"\nEpoch {epoch+1}/{num_epochs} completed:")
            print(f"  Average main loss: {avg_loss:.6f}")
            print(f"  Average regularization loss: {avg_reg_loss:.6f}")
            print(f"  Epoch time: {time.time() - epoch_start_time:.2f}s")
            
            # Print importance summary every few epochs
            if (epoch + 1) % 3 == 0:
                pruner.print_importance_summary()

    # ----------------------------------------
    # Testing and final evaluation
    # ----------------------------------------
    if opt['rank'] == 0:
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        avg_psnr = 0.0
        idx = 0

        for test_data in test_loader:
            idx += 1
            image_name_ext = os.path.basename(test_data['L_path'][0])
            img_name, ext = os.path.splitext(image_name_ext)

            img_dir = os.path.join(opt['path']['images'], img_name)
            util.mkdir(img_dir)

            model.feed_data(test_data)
            model.test()

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
        
        print(f'\nFinal Results:')
        print(f'  Average PSNR: {avg_psnr:.2f}dB')
        
        # Final importance summary
        pruner.print_importance_summary()
        
        # Save model with importance masks
        model.save(current_step)
        print(f'Model saved with importance masks at step {current_step}')

if __name__ == '__main__':
    main()
