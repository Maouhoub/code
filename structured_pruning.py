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
import copy
from collections import OrderedDict

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import torch.nn.utils.prune as prune

'''
# --------------------------------------------
# Hardware-Aware Structured Pruning with Knowledge Distillation
# for Efficient Image Super-Resolution
# --------------------------------------------
'''

class ChannelPruner:
    """Structured channel pruning with importance scoring"""
    
    def __init__(self, model):
        self.model = model
        self.importance_scores = {}
        
    def compute_channel_importance(self, data_loader, num_batches=10):
        """Compute channel importance using gradient-based scoring"""
        self.model.eval()
        importance_scores = {}
        
        # Initialize importance scores
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                importance_scores[name] = torch.zeros(module.out_channels)
        
        batch_count = 0
        with torch.enable_grad():
            for batch_data in data_loader:
                if batch_count >= num_batches:
                    break
                    
                # Forward pass
                self.model.feed_data(batch_data)
                loss = self.model.optimize_parameters(0, compute_loss_only=True)
                
                # Backward pass
                loss.backward()
                
                # Accumulate gradients for importance scoring
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d) and name in importance_scores:
                        if module.weight.grad is not None:
                            # Use gradient magnitude as importance metric
                            grad_importance = torch.sum(torch.abs(module.weight.grad), dim=(1,2,3))
                            importance_scores[name] += grad_importance.detach().cpu()
                
                self.model.zero_grad()
                batch_count += 1
        
        # Normalize importance scores
        for name in importance_scores:
            importance_scores[name] = importance_scores[name] / batch_count
            
        self.importance_scores = importance_scores
        return importance_scores
    
    def get_pruning_plan(self, target_pruning_ratio=0.5):
        """Generate layer-wise pruning plan based on importance scores"""
        pruning_plan = {}
        
        # Hardware-aware layer protection
        protected_layers = {
            'conv_first': 0.1,  # Protect first conv
            'conv_last': 0.1,   # Protect final conv
            'conv_before_upsample': 0.2,  # Protect upsampling layers
            'upsample': 0.1
        }
        
        for name, scores in self.importance_scores.items():
            # Base pruning ratio
            layer_pruning_ratio = target_pruning_ratio
            
            # Apply protection factors
            for protected_name, protection_factor in protected_layers.items():
                if protected_name in name:
                    layer_pruning_ratio *= protection_factor
                    break
            
            # Calculate number of channels to prune
            num_channels = len(scores)
            num_prune = int(num_channels * layer_pruning_ratio)
            
            # Get indices of least important channels
            _, pruned_indices = torch.topk(scores, num_prune, largest=False)
            
            pruning_plan[name] = {
                'pruned_indices': pruned_indices.tolist(),
                'kept_indices': [i for i in range(num_channels) if i not in pruned_indices],
                'pruning_ratio': num_prune / num_channels
            }
            
        return pruning_plan
    
    def apply_structured_pruning(self, pruning_plan):
        """Apply structured channel pruning to the model"""
        print("Applying structured channel pruning...")
        
        for name, module in self.model.named_modules():
            if name in pruning_plan and isinstance(module, nn.Conv2d):
                plan = pruning_plan[name]
                kept_indices = torch.tensor(plan['kept_indices'])
                
                # Prune output channels
                with torch.no_grad():
                    module.weight = nn.Parameter(module.weight[kept_indices])
                    if module.bias is not None:
                        module.bias = nn.Parameter(module.bias[kept_indices])
                    module.out_channels = len(kept_indices)
                
                print(f"Layer {name}: pruned to {len(kept_indices)} channels "
                      f"(ratio: {plan['pruning_ratio']:.3f})")

class KnowledgeDistillationLoss(nn.Module):
    """Knowledge Distillation Loss for Super-Resolution"""
    
    def __init__(self, alpha=0.7, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_output, teacher_output, target):
        # Task-specific loss (L1/L2 reconstruction)
        task_loss = self.mse_loss(student_output, target)
        
        # Feature distillation loss
        distill_loss = self.mse_loss(student_output, teacher_output.detach())
        
        # Combined loss
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss
        
        return total_loss, task_loss, distill_loss

class HardwareProfiler:
    """Profile model performance on different hardware"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def profile_model(self, model, input_shape=(1, 3, 64, 64), num_runs=100):
        """Profile inference time and memory usage"""
        model.eval()
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Measure inference time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / num_runs
        
        # Measure memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(dummy_input)
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            peak_memory = 0
        
        return {
            'inference_time': avg_inference_time,
            'peak_memory_mb': peak_memory,
            'throughput_fps': 1.0 / avg_inference_time
        }

def compute_flops(model, input_shape=(1, 3, 64, 64)):
    """Compute FLOPs for the model"""
    def flop_count_hook(module, input, output):
        if isinstance(module, nn.Conv2d):
            # Convolution FLOPs: batch_size * output_elements * kernel_flops
            kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            output_elements = output.numel() // output.shape[0]  # Remove batch dimension
            flops = kernel_flops * output_elements
            module.__flops__ += flops
        elif isinstance(module, nn.Linear):
            flops = module.in_features * module.out_features
            if output.dim() > 2:
                flops *= output.numel() // output.shape[-1] // output.shape[0]
            module.__flops__ += flops
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.__flops__ = 0
            hooks.append(module.register_forward_hook(flop_count_hook))
    
    # Forward pass
    dummy_input = torch.randn(input_shape)
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Sum FLOPs
    total_flops = sum(getattr(module, '__flops__', 0) for module in model.modules())
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops

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
    parser.add_argument('--target_pruning_ratio', type=float, default=0.5, help='Target channel pruning ratio')
    parser.add_argument('--kd_alpha', type=float, default=0.7, help='Knowledge distillation weight')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    
    # Pruning configuration
    target_pruning_ratio = parser.parse_args().target_pruning_ratio
    kd_alpha = parser.parse_args().kd_alpha

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
    # save opt to a '../option.json' file
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
    # Step--3 (initialize models)
    # ----------------------------------------
    '''
    # Teacher model (original, unpruned)
    teacher_model = define_Model(opt)
    teacher_model.init_train()
    
    # Student model (to be pruned)
    student_model = define_Model(opt)
    student_model.init_train()
    
    # Copy teacher weights to student initially
    student_model.load_state_dict(teacher_model.state_dict())
    
    print("Models initialized successfully!")

    '''
    # ----------------------------------------
    # Step--4 (Hardware profiling - baseline)
    # ----------------------------------------
    '''
    if opt['rank'] == 0:
        profiler = HardwareProfiler()
        
        print("=" * 60)
        print("BASELINE MODEL ANALYSIS")
        print("=" * 60)
        
        # Profile original model
        baseline_profile = profiler.profile_model(teacher_model.netG)
        baseline_flops = compute_flops(teacher_model.netG)
        baseline_params = sum(p.numel() for p in teacher_model.netG.parameters())
        
        print(f"Original Model:")
        print(f"  Parameters: {baseline_params:,}")
        print(f"  FLOPs: {baseline_flops:,}")
        print(f"  Inference time: {baseline_profile['inference_time']:.4f}s")
        print(f"  Peak memory: {baseline_profile['peak_memory_mb']:.1f}MB")
        print(f"  Throughput: {baseline_profile['throughput_fps']:.1f}FPS")

    '''
    # ----------------------------------------
    # Step--5 (Structured Pruning with KD)
    # ----------------------------------------
    '''
    
    # Initialize components
    channel_pruner = ChannelPruner(student_model)
    kd_criterion = KnowledgeDistillationLoss(alpha=kd_alpha, temperature=4.0)
    
    print("=" * 60)
    print("STRUCTURED PRUNING WITH KNOWLEDGE DISTILLATION")
    print("=" * 60)
    
    # Phase 1: Compute channel importance
    print("Phase 1: Computing channel importance scores...")
    importance_scores = channel_pruner.compute_channel_importance(train_loader, num_batches=20)
    
    # Phase 2: Generate pruning plan
    print("Phase 2: Generating pruning plan...")
    pruning_plan = channel_pruner.get_pruning_plan(target_pruning_ratio)
    
    # Phase 3: Apply structured pruning
    print("Phase 3: Applying structured channel pruning...")
    channel_pruner.apply_structured_pruning(pruning_plan)
    
    # Phase 4: Knowledge Distillation Fine-tuning
    print("Phase 4: Knowledge distillation fine-tuning...")
    
    num_kd_epochs = 50  # Intensive fine-tuning after pruning
    teacher_model.eval()  # Teacher stays in eval mode
    
    for epoch in range(num_kd_epochs):
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)
        
        epoch_start_time = time.time()
        total_loss = 0.0
        total_task_loss = 0.0
        total_distill_loss = 0.0
        
        for i, train_data in enumerate(train_loader):
            current_step += 1
            
            # Update learning rate
            student_model.update_learning_rate(current_step)
            
            # Forward pass through teacher (no gradients)
            with torch.no_grad():
                teacher_model.feed_data(train_data)
                teacher_model.test()  # Get teacher output
                teacher_output = teacher_model.current_visuals()['E']
            
            # Forward pass through student
            student_model.feed_data(train_data)
            student_model.optimize_parameters(current_step)
            student_output = student_model.current_visuals()['E']
            target = train_data['H'].cuda()
            
            # Knowledge distillation loss
            kd_loss, task_loss, distill_loss = kd_criterion(student_output, teacher_output, target)
            
            # Backward pass for student
            student_model.optimizer_G.zero_grad()
            kd_loss.backward()
            student_model.optimizer_G.step()
            
            total_loss += kd_loss.item()
            total_task_loss += task_loss.item()
            total_distill_loss += distill_loss.item()
        
        # Log training info
        if opt['rank'] == 0:
            avg_loss = total_loss / len(train_loader)
            avg_task_loss = total_task_loss / len(train_loader)
            avg_distill_loss = total_distill_loss / len(train_loader)
            
            print(f'Epoch {epoch+1}/{num_kd_epochs}: '
                  f'Total Loss: {avg_loss:.6f}, '
                  f'Task Loss: {avg_task_loss:.6f}, '
                  f'Distill Loss: {avg_distill_loss:.6f}, '
                  f'Time: {time.time() - epoch_start_time:.2f}s')

    '''
    # ----------------------------------------
    # Step--6 (Comprehensive Evaluation)
    # ----------------------------------------
    '''
    
    if opt['rank'] == 0:
        print("=" * 60)
        print("COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        # Hardware profiling of pruned model
        pruned_profile = profiler.profile_model(student_model.netG)
        pruned_flops = compute_flops(student_model.netG)
        pruned_params = sum(p.numel() for p in student_model.netG.parameters())
        
        # Calculate improvements
        param_reduction = (1 - pruned_params / baseline_params) * 100
        flop_reduction = (1 - pruned_flops / baseline_flops) * 100
        speedup = baseline_profile['inference_time'] / pruned_profile['inference_time']
        memory_reduction = (1 - pruned_profile['peak_memory_mb'] / baseline_profile['peak_memory_mb']) * 100
        
        print("HARDWARE ANALYSIS:")
        print(f"  Parameter reduction: {param_reduction:.1f}% ({baseline_params:,} ? {pruned_params:,})")
        print(f"  FLOPs reduction: {flop_reduction:.1f}% ({baseline_flops:,} ? {pruned_flops:,})")
        print(f"  Speed improvement: {speedup:.2f}x ({baseline_profile['inference_time']:.4f}s ? {pruned_profile['inference_time']:.4f}s)")
        print(f"  Memory reduction: {memory_reduction:.1f}% ({baseline_profile['peak_memory_mb']:.1f}MB ? {pruned_profile['peak_memory_mb']:.1f}MB)")
        print(f"  Throughput improvement: {pruned_profile['throughput_fps'] / baseline_profile['throughput_fps']:.2f}x")
        
        # Quality evaluation on test set
        print("\nQUALITY ANALYSIS:")
        
        # Test teacher model
        teacher_psnr = 0.0
        teacher_ssim = 0.0
        
        # Test student model  
        student_psnr = 0.0
        student_ssim = 0.0
        
        idx = 0
        for test_data in test_loader:
            idx += 1
            
            # Teacher inference
            teacher_model.feed_data(test_data)
            teacher_model.test()
            teacher_visuals = teacher_model.current_visuals()
            teacher_img = util.tensor2uint(teacher_visuals['E'])
            
            # Student inference
            student_model.feed_data(test_data)
            student_model.test()  
            student_visuals = student_model.current_visuals()
            student_img = util.tensor2uint(student_visuals['E'])
            
            # Ground truth
            gt_img = util.tensor2uint(test_data['H'])
            
            # Calculate metrics
            teacher_psnr += util.calculate_psnr(teacher_img, gt_img, border=border)
            teacher_ssim += util.calculate_ssim(teacher_img, gt_img, border=border)
            student_psnr += util.calculate_psnr(student_img, gt_img, border=border) 
            student_ssim += util.calculate_ssim(student_img, gt_img, border=border)
            
            # Save comparison images for first few samples
            if idx <= 5:
                image_name_ext = os.path.basename(test_data['L_path'][0])
                img_name, ext = os.path.splitext(image_name_ext)
                img_dir = os.path.join(opt['path']['images'], img_name)
                util.mkdir(img_dir)
                
                # Save teacher, student, and GT images
                util.imsave(teacher_img, os.path.join(img_dir, f'{img_name}_teacher.png'))
                util.imsave(student_img, os.path.join(img_dir, f'{img_name}_student_pruned.png'))
                util.imsave(gt_img, os.path.join(img_dir, f'{img_name}_gt.png'))
        
        # Average metrics
        teacher_psnr /= idx
        teacher_ssim /= idx
        student_psnr /= idx
        student_ssim /= idx
        
        psnr_drop = teacher_psnr - student_psnr
        ssim_drop = teacher_ssim - student_ssim
        
        print(f"  Teacher PSNR: {teacher_psnr:.2f}dB")
        print(f"  Student PSNR: {student_psnr:.2f}dB (drop: {psnr_drop:.2f}dB)")
        print(f"  Teacher SSIM: {teacher_ssim:.4f}")
        print(f"  Student SSIM: {student_ssim:.4f} (drop: {ssim_drop:.4f})")
        
        # Summary for publication
        print("\n" + "=" * 60)
        print("PUBLICATION-READY RESULTS SUMMARY")
        print("=" * 60)
        print(f"? Parameter Reduction: {param_reduction:.1f}%")
        print(f"? FLOPs Reduction: {flop_reduction:.1f}%") 
        print(f"? Inference Speedup: {speedup:.2f}x")
        print(f"? Memory Reduction: {memory_reduction:.1f}%")
        print(f"? PSNR Drop: {psnr_drop:.2f}dB")
        print(f"? SSIM Drop: {ssim_drop:.4f}")
        
        # Check if results meet publication standards
        print("\nPUBLICATION QUALITY CHECK:")
        if param_reduction > 50 and flop_reduction > 40 and psnr_drop < 1.0:
            print("?? RESULTS MEET Q1/Q2 PUBLICATION STANDARDS!")
        else:
            print("??  Results may need improvement for top-tier publication")
            if param_reduction <= 50:
                print(f"   - Parameter reduction too low: {param_reduction:.1f}% (target: >50%)")
            if flop_reduction <= 40:
                print(f"   - FLOPs reduction too low: {flop_reduction:.1f}% (target: >40%)")
            if psnr_drop >= 1.0:
                print(f"   - PSNR drop too high: {psnr_drop:.2f}dB (target: <1.0dB)")

    '''
    # ----------------------------------------
    # Step--7 (Save final model)
    # ----------------------------------------
    '''
    
    if opt['rank'] == 0:
        print("Saving final pruned model...")
        student_model.save(current_step)
        
        # Save results to file
        results = {
            'parameter_reduction_percent': param_reduction,
            'flops_reduction_percent': flop_reduction,
            'inference_speedup': speedup,
            'memory_reduction_percent': memory_reduction,
            'teacher_psnr': teacher_psnr,
            'student_psnr': student_psnr,
            'psnr_drop': psnr_drop,
            'teacher_ssim': teacher_ssim,
            'student_ssim': student_ssim,
            'ssim_drop': ssim_drop,
            'pruning_ratio': target_pruning_ratio,
            'kd_alpha': kd_alpha
        }
        
        import json
        with open(os.path.join(opt['path']['log'], 'pruning_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print("? Training completed successfully!")
        print("?? Results saved to pruning_results.json")


if __name__ == '__main__':
    main()