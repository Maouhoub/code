"""
Enhanced SwinIR Structured Pruning with Torch-Pruning Library
==============================================================

This implementation replaces the custom pruning approach with the torch-pruning library
for improved efficiency and accuracy. The torch-pruning library provides:
- DepGraph algorithm for automatic dependency detection
- Professional importance scoring methods
- Robust group-based pruning
- Support for Vision Transformers including SwinIR

Key improvements over the original implementation:
1. Uses torch-pruning's DepGraph for automatic layer dependency detection
2. Employs proven importance metrics (GroupMagnitudeImportance, etc.)
3. Implements iterative pruning with knowledge distillation
4. Supports global pruning and isomorphic pruning for better results

Author: Research Team
Date: August 2025
Purpose: Academic publication in Q1/Q2 journals
"""

import os.path
import math
import argparse
import random
import numpy as np
import logging
import gc
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import sys

# Import torch-pruning library
try:
    import torch_pruning as tp
except ImportError:
    print("ERROR: torch-pruning library not found. Install with: pip install torch-pruning")
    sys.exit(1)

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

# Set CUDA debugging environment variables for better error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
print("CUDA debugging enabled: CUDA_LAUNCH_BLOCKING=1, TORCH_USE_CUDA_DSA=1")

'''
Enhanced Structured Pruning Training for SwinIR using Torch-Pruning Library
============================================================================
'''

class SwinIRCustomPruner(tp.BasePruningFunc):
    """
    Custom pruner for SwinIR WindowAttention layers
    Handles the specific architecture of SwinIR attention mechanisms
    """
    
    def prune_out_channels(self, layer: nn.Module, idxs: list):
        # Handle the qkv linear layer (3 * dim -> 3 * (dim - len(idxs)))
        if hasattr(layer, 'qkv'):
            # Calculate new dimensions after pruning
            old_dim = layer.dim
            new_dim = old_dim - len(idxs)
            
            # Create index mappings for qkv (query, key, value)
            qkv_idxs = []
            for i in range(3):  # q, k, v
                offset_idxs = [idx + i * old_dim for idx in idxs]
                qkv_idxs.extend(offset_idxs)
            
            # Prune qkv linear layer
            tp.prune_linear_out_channels(layer.qkv, qkv_idxs)
            
            # Update layer attributes
            layer.dim = new_dim
            layer.num_heads = min(layer.num_heads, new_dim // 32)  # Ensure valid head count
            if layer.num_heads == 0:
                layer.num_heads = 1
            
        # Handle projection layer
        if hasattr(layer, 'proj'):
            tp.prune_linear_in_channels(layer.proj, idxs)
            tp.prune_linear_out_channels(layer.proj, idxs)
        
        return layer
    
    def prune_in_channels(self, layer: nn.Module, idxs: list):
        if hasattr(layer, 'qkv'):
            tp.prune_linear_in_channels(layer.qkv, idxs)
        return layer
    
    def get_out_channels(self, layer):
        return getattr(layer, 'dim', 0)
    
    def get_in_channels(self, layer):
        return getattr(layer, 'dim', 0)


class TorchPruningManager:
    """
    Enhanced pruning manager using torch-pruning library
    Provides professional-grade pruning with automatic dependency detection
    """
    
    def __init__(self, model, example_inputs, config):
        self.model = model
        self.example_inputs = example_inputs
        self.config = config
        
        # Get the actual network (handle model wrappers)
        self.network = model.netG if hasattr(model, 'netG') else model
        
        # Initialize torch-pruning components
        self.dependency_graph = None
        self.pruner = None
        self.importance_metric = None
        
        # Performance tracking
        self.original_macs = 0
        self.original_params = 0
        self.current_macs = 0
        self.current_params = 0
        
        print("Initializing Torch-Pruning Manager...")
        self._setup_pruning_components()
    
    def _setup_pruning_components(self):
        """Setup torch-pruning components"""
        try:
            # Build dependency graph
            print("Building dependency graph...")
            self.dependency_graph = tp.DependencyGraph().build_dependency(
                self.network, 
                example_inputs=self.example_inputs
            )
            
            # Calculate baseline metrics
            self.original_macs, self.original_params = tp.utils.count_ops_and_params(
                self.network, self.example_inputs
            )
            self.current_macs = self.original_macs
            self.current_params = self.original_params
            
            print(f"Original Model - MACs: {self.original_macs/1e9:.2f}G, Params: {self.original_params/1e6:.2f}M")
            
            # Setup importance metric based on config
            importance_type = self.config.get('importance_metric', 'magnitude')
            if importance_type == 'magnitude':
                self.importance_metric = tp.importance.GroupMagnitudeImportance(p=2)
            elif importance_type == 'taylor':
                self.importance_metric = tp.importance.GroupTaylorImportance()
            elif importance_type == 'hessian':
                self.importance_metric = tp.importance.GroupHessianImportance()
            else:
                self.importance_metric = tp.importance.GroupMagnitudeImportance(p=2)
            
            print(f"Using importance metric: {type(self.importance_metric).__name__}")
            
        except Exception as e:
            print(f"Error setting up torch-pruning components: {e}")
            raise
    
    def create_pruner(self, pruning_ratio=0.3, global_pruning=True, isomorphic=True):
        """
        Create a torch-pruning pruner with specified configuration
        
        Args:
            pruning_ratio: Ratio of channels/dimensions to prune
            global_pruning: Use global importance ranking
            isomorphic: Use isomorphic pruning for better performance
        """
        try:
            # Identify layers to ignore (only final/small layers)
            ignored_layers = []
            unwrapped_parameters = []
            num_heads = {}  # For tracking attention heads
            customized_pruners = {}
            
            for name, module in self.network.named_modules():
                lname = name.lower()
                
                # Ignore final output layers and very small layers
                if any(keyword in lname for keyword in [
                    'conv_last', 'output', 'final', 'conv_after_body'
                ]):
                    ignored_layers.append(module)
                elif hasattr(module, 'weight') and hasattr(module.weight, 'shape'):
                    if len(module.weight.shape) >= 2 and min(module.weight.shape) <= 8:
                        ignored_layers.append(module)
                
                # Handle SwinIR WindowAttention layers
                if 'WindowAttention' in str(type(module)):
                    if hasattr(module, 'num_heads') and hasattr(module, 'qkv'):
                        num_heads[module.qkv] = module.num_heads
                        customized_pruners[type(module)] = SwinIRCustomPruner()
                        print(f"  Found WindowAttention: {name} with {module.num_heads} heads")
                
                # Collect relative_position_bias_table parameters as unwrapped
                if hasattr(module, 'relative_position_bias_table'):
                    unwrapped_parameters.append((module, 'relative_position_bias_table'))
                    
            print(f"Ignoring {len(ignored_layers)} layers from pruning (final/small layers)")
            print(f"Unwrapped parameters (not pruned): {len(unwrapped_parameters)} relative_position_bias_table tensors")
            print(f"Found {len(num_heads)} attention layers with head tracking")
            
            # Create pruner with SwinIR-specific configurations
            pruner_kwargs = {
                'model': self.network,
                'example_inputs': self.example_inputs,
                'importance': self.importance_metric,
                'pruning_ratio': pruning_ratio,
                'ignored_layers': ignored_layers,
                'global_pruning': global_pruning,
                'round_to': 8,
                'unwrapped_parameters': unwrapped_parameters,
                'output_transform': lambda out: out.sum() if isinstance(out, torch.Tensor) else out[0].sum()
            }
            
            # Add SwinIR-specific configurations
            if num_heads:
                pruner_kwargs['num_heads'] = num_heads
            if customized_pruners:
                pruner_kwargs['customized_pruners'] = customized_pruners
            if isomorphic:
                pruner_kwargs['isomorphic'] = isomorphic
            
            self.pruner = tp.pruner.BasePruner(**pruner_kwargs)
            
            print(f"Created pruner with {pruning_ratio:.1%} pruning ratio")
            print(f"Global pruning: {global_pruning}, Isomorphic: {isomorphic}")
            print(f"Custom pruners for {len(customized_pruners)} layer types")
            return True
            
        except Exception as e:
            print(f"Error creating pruner: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def prune_model(self, interactive=False):
        """
        Perform pruning using torch-pruning
        
        Args:
            interactive: Whether to use interactive pruning for fine control
        """
        if self.pruner is None:
            print("Error: Pruner not initialized")
            return False
        
        try:
            print("Starting model pruning...")
            
            if interactive:
                # Interactive pruning for fine control
                for group in self.pruner.step(interactive=True):
                    print(f"Pruning group: {group}")
                    group.prune()
            else:
                # Standard one-shot pruning
                self.pruner.step()
            
            # Update metrics
            self.current_macs, self.current_params = tp.utils.count_ops_and_params(
                self.network, self.example_inputs
            )
            
            # Calculate reduction ratios
            macs_reduction = (self.original_macs - self.current_macs) / self.original_macs
            params_reduction = (self.original_params - self.current_params) / self.original_params
            
            print(f"Pruning completed:")
            print(f"  MACs: {self.original_macs/1e9:.2f}G -> {self.current_macs/1e9:.2f}G ({macs_reduction:.1%} reduction)")
            print(f"  Params: {self.original_params/1e6:.2f}M -> {self.current_params/1e6:.2f}M ({params_reduction:.1%} reduction)")
            
            return True
            
        except Exception as e:
            print(f"Error during pruning: {e}")
            return False
    
    def get_pruning_statistics(self):
        """Get detailed pruning statistics"""
        return {
            'original_macs': self.original_macs,
            'current_macs': self.current_macs,
            'original_params': self.original_params,
            'current_params': self.current_params,
            'macs_reduction': (self.original_macs - self.current_macs) / self.original_macs,
            'params_reduction': (self.original_params - self.current_params) / self.original_params,
            'speedup_estimate': self.original_macs / self.current_macs if self.current_macs > 0 else 1.0
        }
    
    def analyze_model_structure(self):
        """Analyze and print model structure for debugging"""
        print("\nModel Structure Analysis:")
        print("-" * 50)
        
        total_groups = 0
        for group in self.dependency_graph.get_all_groups():
            total_groups += 1
            if total_groups <= 5:  # Show first 5 groups as examples
                print(f"Group {total_groups}:")
                print(group)
                print()
        
        print(f"Total prunable groups: {total_groups}")


class KnowledgeDistillationTrainer:
    """
    Enhanced Knowledge Distillation for fine-tuning pruned models
    """
    
    def __init__(self, teacher_model, student_model, config):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        
        # KD parameters
        self.temperature = config.get('temperature', 4.0)
        self.alpha = config.get('alpha', 0.7)  # Weight for distillation loss
        self.beta = config.get('beta', 0.3)   # Weight for task loss
        
        # Feature distillation
        self.use_feature_distillation = config.get('use_feature_distillation', True)
        self.feature_hooks = []
        
        if self.use_feature_distillation:
            self._register_feature_hooks()
    
    def _register_feature_hooks(self):
        """Register hooks for feature extraction"""
        self.teacher_features = {}
        self.student_features = {}
        
        def get_teacher_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.teacher_features[name] = output.detach()
            return hook
        
        def get_student_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.student_features[name] = output
            return hook
        
        # Register hooks for key layers
        teacher_net = self.teacher_model.netG if hasattr(self.teacher_model, 'netG') else self.teacher_model
        student_net = self.student_model.netG if hasattr(self.student_model, 'netG') else self.student_model
        
        hook_layers = []
        for name, module in teacher_net.named_modules():
            if 'layers' in name and 'blocks' in name and len(name.split('.')) <= 4:
                hook_layers.append(name)
        
        # Select a subset of layers for feature distillation
        selected_layers = hook_layers[::max(1, len(hook_layers)//4)]  # Every 4th layer
        
        for layer_name in selected_layers:
            try:
                teacher_layer = dict(teacher_net.named_modules())[layer_name]
                student_layer = dict(student_net.named_modules())[layer_name]
                
                teacher_hook = teacher_layer.register_forward_hook(get_teacher_hook(layer_name))
                student_hook = student_layer.register_forward_hook(get_student_hook(layer_name))
                
                self.feature_hooks.extend([teacher_hook, student_hook])
                
            except KeyError:
                continue
        
        print(f"Registered feature distillation hooks for {len(selected_layers)} layers")
    
    def compute_distillation_loss(self, student_output, teacher_output, target=None):
        """
        Compute knowledge distillation loss
        
        Args:
            student_output: Output from student model
            teacher_output: Output from teacher model
            target: Ground truth target (optional)
        """
        # For image super-resolution, we use MSE loss between outputs
        if target is not None:
            # Task loss (student vs ground truth)
            task_loss = F.mse_loss(student_output, target)
            # Distillation loss (student vs teacher)
            distillation_loss = F.mse_loss(student_output, teacher_output.detach())
            total_loss = self.beta * task_loss + self.alpha * distillation_loss
        else:
            # Only distillation loss
            total_loss = F.mse_loss(student_output, teacher_output.detach())
        
        # Feature distillation loss
        if self.use_feature_distillation and self.teacher_features and self.student_features:
            feature_loss = self._compute_feature_distillation_loss()
            total_loss += 0.1 * feature_loss  # Small weight for feature loss
        
        return total_loss
    
    def _compute_feature_distillation_loss(self):
        """Compute feature-level distillation loss"""
        feature_loss = 0.0
        count = 0
        
        for layer_name in self.teacher_features:
            if layer_name in self.student_features:
                teacher_feat = self.teacher_features[layer_name]
                student_feat = self.student_features[layer_name]
                
                # Handle size mismatches due to pruning
                if teacher_feat.shape != student_feat.shape:
                    min_channels = min(teacher_feat.shape[1], student_feat.shape[1])
                    teacher_feat = teacher_feat[:, :min_channels]
                    student_feat = student_feat[:, :min_channels]
                
                feature_loss += F.mse_loss(student_feat, teacher_feat)
                count += 1
        
        # Clear feature caches
        self.teacher_features.clear()
        self.student_features.clear()
        
        return feature_loss / max(count, 1)
    
    def cleanup_hooks(self):
        """Remove all registered hooks"""
        for hook in self.feature_hooks:
            hook.remove()
        self.feature_hooks.clear()


class IterativePruningPipeline:
    """
    Iterative pruning pipeline with torch-pruning and knowledge distillation
    """
    
    def __init__(self, model, config, example_inputs):
        self.original_model = copy.deepcopy(model)
        self.current_model = model
        self.config = config
        self.example_inputs = example_inputs
        
        # Pipeline configuration
        self.num_iterations = config.get('num_iterations', 3)
        self.pruning_ratio_per_iter = config.get('pruning_ratio_per_iter', 0.2)
        self.fine_tune_epochs = config.get('fine_tune_epochs', 10)
        
        # Initialize components
        self.pruning_manager = None
        self.kd_trainer = None
        
        # Results tracking
        self.iteration_results = []
        
    def run_complete_pipeline(self, train_loader, test_loader=None):
        """
        Run the complete iterative pruning pipeline
        """
        print("="*70)
        print("STARTING ITERATIVE PRUNING PIPELINE WITH TORCH-PRUNING")
        print("="*70)
        
        try:
            # Initialize pruning manager
            self.pruning_manager = TorchPruningManager(
                self.current_model, 
                self.example_inputs, 
                self.config
            )
            
            # Perform iterative pruning
            for iteration in range(self.num_iterations):
                print(f"\n{'='*50}")
                print(f"ITERATION {iteration + 1}/{self.num_iterations}")
                print(f"{'='*50}")
                
                success = self._run_single_iteration(iteration, train_loader, test_loader)
                if not success:
                    print(f"Iteration {iteration + 1} failed, stopping pipeline")
                    break
            
            # Final evaluation
            final_stats = self.pruning_manager.get_pruning_statistics()
            
            print("\n" + "="*70)
            print("PRUNING PIPELINE COMPLETED")
            print("="*70)
            print(f"Total parameter reduction: {final_stats['params_reduction']:.1%}")
            print(f"Total MACs reduction: {final_stats['macs_reduction']:.1%}")
            print(f"Estimated speedup: {final_stats['speedup_estimate']:.2f}x")
            
            return {
                'success': True,
                'final_model': self.current_model,
                'iterations': self.iteration_results,
                'final_stats': final_stats
            }
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            return {
                'success': False,
                'error': str(e),
                'iterations': self.iteration_results
            }
    
    def _run_single_iteration(self, iteration, train_loader, test_loader):
        """Run a single pruning iteration"""
        try:
            # Step 1: Create pruner for this iteration
            success = self.pruning_manager.create_pruner(
                pruning_ratio=self.pruning_ratio_per_iter,
                global_pruning=True,
                isomorphic=True
            )
            if not success:
                return False
            
            # Step 2: Perform pruning
            success = self.pruning_manager.prune_model(interactive=False)
            if not success:
                return False
            
            # Step 3: Knowledge distillation fine-tuning
            if iteration < self.num_iterations - 1:  # Don't fine-tune on last iteration
                self._fine_tune_with_kd(train_loader)
            
            # Step 4: Evaluate current model
            if test_loader is not None:
                metrics = self._evaluate_model(test_loader)
                self.iteration_results.append({
                    'iteration': iteration + 1,
                    'stats': self.pruning_manager.get_pruning_statistics(),
                    'metrics': metrics
                })
            
            return True
            
        except Exception as e:
            print(f"Error in iteration {iteration + 1}: {e}")
            return False
    
    def _fine_tune_with_kd(self, train_loader):
        """Fine-tune with knowledge distillation"""
        print("Starting knowledge distillation fine-tuning...")
        
        # Initialize KD trainer
        self.kd_trainer = KnowledgeDistillationTrainer(
            teacher_model=self.original_model,
            student_model=self.current_model,
            config=self.config
        )
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.current_model.parameters(),
            lr=self.config.get('kd_lr', 1e-4)
        )
        
        # Fine-tuning loop
        self.current_model.train()
        for epoch in range(self.fine_tune_epochs):
            total_loss = 0.0
            for i, data in enumerate(train_loader):
                if i >= 50:  # Limit fine-tuning batches for efficiency
                    break
                
                # Prepare data
                L = data['L'].cuda() if torch.cuda.is_available() else data['L']
                H = data['H'].cuda() if torch.cuda.is_available() else data['H']
                
                # Forward pass
                optimizer.zero_grad()
                
                with torch.no_grad():
                    teacher_output = self.original_model(L)
                
                student_output = self.current_model(L)
                
                # Compute distillation loss
                loss = self.kd_trainer.compute_distillation_loss(
                    student_output, teacher_output, H
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / min(50, len(train_loader))
            print(f"  KD Epoch {epoch + 1}/{self.fine_tune_epochs}, Loss: {avg_loss:.6f}")
        
        # Cleanup
        self.kd_trainer.cleanup_hooks()
        print("Knowledge distillation fine-tuning completed")
    
    def _evaluate_model(self, test_loader):
        """Evaluate model performance with comprehensive metrics"""
        self.current_model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        count = 0
        inference_times = []
        
        print("Evaluating model performance...")
        
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                if i >= 20:  # Limit evaluation for efficiency
                    break
                
                L = data['L'].cuda() if torch.cuda.is_available() else data['L']
                H = data['H'].cuda() if torch.cuda.is_available() else data['H']
                
                # Measure inference time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                
                E = self.current_model(L)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                
                # Calculate PSNR
                psnr = util.calculate_psnr(E, H, border=0)
                total_psnr += psnr
                count += 1
        
        avg_psnr = total_psnr / count if count > 0 else 0.0
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        
        print(f"Current model PSNR: {avg_psnr:.2f} dB")
        print(f"Average inference time: {avg_inference_time*1000:.2f} ms")
        
        return {
            'psnr': avg_psnr, 
            'ssim': 0.0,  # SSIM calculation can be added if needed
            'inference_time_ms': avg_inference_time * 1000
        }


def create_pruning_config_from_args(args):
    """Create pruning configuration from command line arguments"""
    return {
        'importance_metric': getattr(args, 'importance_metric', 'magnitude'),
        'num_iterations': getattr(args, 'num_iterations', 3),
        'pruning_ratio_per_iter': getattr(args, 'pruning_ratio_per_iter', 0.2),
        'fine_tune_epochs': getattr(args, 'fine_tune_epochs', 10),
        'temperature': getattr(args, 'temperature', 4.0),
        'alpha': getattr(args, 'alpha', 0.7),
        'beta': getattr(args, 'beta', 0.3),
        'use_feature_distillation': getattr(args, 'use_feature_distillation', True),
        'kd_lr': getattr(args, 'kd_lr', 1e-4),
        'global_pruning': getattr(args, 'global_pruning', True),
        'isomorphic': getattr(args, 'isomorphic', True)
    }


def save_pruning_results(results, save_dir):
    """Save comprehensive pruning results"""
    import json
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save statistics
    stats_file = os.path.join(save_dir, 'pruning_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(results['final_stats'], f, indent=2)
    
    # Save iteration results
    if 'iterations' in results:
        iterations_file = os.path.join(save_dir, 'iteration_results.json')
        with open(iterations_file, 'w') as f:
            json.dump(results['iterations'], f, indent=2)
    
    print(f"Pruning results saved to: {save_dir}")


def main():
    """
    Main function to run enhanced SwinIR pruning with torch-pruning
    """
    parser = argparse.ArgumentParser(description="Enhanced SwinIR Pruning with Torch-Pruning")
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    
    # Pruning-specific arguments
    parser.add_argument('--importance_metric', type=str, default='magnitude', 
                       choices=['magnitude', 'taylor', 'hessian'],
                       help='Importance metric for pruning')
    parser.add_argument('--num_iterations', type=int, default=3, 
                       help='Number of pruning iterations')
    parser.add_argument('--pruning_ratio_per_iter', type=float, default=0.2, 
                       help='Pruning ratio per iteration')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, 
                       help='Number of fine-tuning epochs per iteration')
    parser.add_argument('--kd_lr', type=float, default=1e-4, 
                       help='Learning rate for knowledge distillation')
    parser.add_argument('--global_pruning', action='store_true', default=True,
                       help='Use global pruning')
    parser.add_argument('--isomorphic', action='store_true', default=True,
                       help='Use isomorphic pruning')
    parser.add_argument('--save_dir', type=str, default='pruning_results',
                       help='Directory to save pruning results')
    
    args = parser.parse_args()
    
    # Parse options
    opt = option.parse(args.opt, is_train=True)
    
    # Setup distributed training if needed
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    # Random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    utils_logger.set_random_seed(seed)
    
    # Setup logger
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))
    
    # Create datasets
    dataset_type = opt['datasets']['train']['dataset_type']
    train_loader = None
    test_loader = None
    
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set, batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'], 
                                        sampler=train_sampler, num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'], 
                                        drop_last=True, pin_memory=True)
            else:
                train_loader = DataLoader(train_set, batch_size=dataset_opt['dataloader_batch_size'], 
                                        shuffle=dataset_opt['dataloader_shuffle'], num_workers=dataset_opt['dataloader_num_workers'], 
                                        drop_last=True, pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
    
    # Create model
    model = define_Model(opt)
    logger.info(f"Model {model.__class__.__name__} created.")
    
    # Setup pruning configuration from arguments
    pruning_config = create_pruning_config_from_args(args)
    logger.info(f"Pruning configuration: {pruning_config}")
    
    # Create example inputs for torch-pruning
    example_inputs = torch.randn(1, 3, 64, 64)
    if torch.cuda.is_available():
        example_inputs = example_inputs.cuda()
        model.netG.cuda()  # Ensure model is on GPU
    
    # Initialize and run the pruning pipeline
    try:
        pruning_pipeline = IterativePruningPipeline(
            model=model,
            config=pruning_config,
            example_inputs=example_inputs
        )
        
        # Run the complete pipeline
        results = pruning_pipeline.run_complete_pipeline(
            train_loader=train_loader,
            test_loader=test_loader
        )
        
        if results['success']:
            logger.info("Pruning pipeline completed successfully!")
            logger.info(f"Final statistics: {results['final_stats']}")
            
            # Save the pruned model
            save_path = os.path.join(opt['path']['models'], 'pruned_swinir_torch_pruning.pth')
            torch.save(results['final_model'], save_path)
            logger.info(f"Pruned model saved to: {save_path}")
            
            # Save comprehensive results
            save_pruning_results(results, args.save_dir)
            
            # Print final summary
            final_stats = results['final_stats']
            print("\n" + "="*80)
            print("FINAL PRUNING SUMMARY")
            print("="*80)
            print(f"Parameter reduction: {final_stats['params_reduction']:.1%} "
                  f"({final_stats['original_params']/1e6:.2f}M ? {final_stats['current_params']/1e6:.2f}M)")
            print(f"MACs reduction: {final_stats['macs_reduction']:.1%} "
                  f"({final_stats['original_macs']/1e9:.2f}G ? {final_stats['current_macs']/1e9:.2f}G)")
            print(f"Estimated speedup: {final_stats['speedup_estimate']:.2f}x")
            print("="*80)
            
        else:
            logger.error(f"Pruning pipeline failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error during pruning: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
