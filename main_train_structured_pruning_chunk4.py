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

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

# Import from previous chunks
from main_train_structured_pruning_chunk1 import ImportanceMaskModule, AttentionHeadMask, MLPChannelMask
from main_train_structured_pruning_chunk2 import StructuredPruner, count_parameters
from main_train_structured_pruning_chunk3 import KnowledgeDistillationLoss, calculate_psnr, distillation_training_step

'''
# --------------------------------------------
# Structured Pruning for SwinIR - Chunk 4: Iterative Pruning Pipeline
# Multi-stage pruning with intermediate fine-tuning and convergence monitoring
# --------------------------------------------
'''

class IterativePruningScheduler:
    """
    Manages the iterative pruning schedule and convergence monitoring
    """
    def __init__(self, target_ratio=0.4, num_iterations=4, schedule_type='linear'):
        self.target_ratio = target_ratio
        self.num_iterations = num_iterations
        self.schedule_type = schedule_type
        self.schedule = self._create_schedule()
        
        # Convergence monitoring
        self.best_psnr = 0
        self.patience = 3
        self.bad_iterations = 0
        self.min_improvement = 0.1  # Minimum PSNR improvement to consider progress
        
    def _create_schedule(self):
        """Create pruning schedule based on type"""
        if self.schedule_type == 'linear':
            # Linear progression to target ratio
            ratios = [self.target_ratio * (i + 1) / self.num_iterations 
                     for i in range(self.num_iterations)]
        elif self.schedule_type == 'exponential':
            # Exponential progression (more aggressive early pruning)
            ratios = [self.target_ratio * (1 - (0.5 ** (i + 1))) 
                     for i in range(self.num_iterations)]
        elif self.schedule_type == 'conservative':
            # Conservative progression (gradual increase)
            base_ratio = self.target_ratio / self.num_iterations
            ratios = [base_ratio * (i + 1) * 0.8 for i in range(self.num_iterations)]
            ratios[-1] = self.target_ratio  # Ensure final target is reached
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return ratios
    
    def get_iteration_target(self, iteration):
        """Get target pruning ratio for specific iteration"""
        if iteration >= len(self.schedule):
            return self.target_ratio
        return self.schedule[iteration]
    
    def should_continue(self, current_psnr, iteration):
        """Check if pruning should continue based on convergence criteria"""
        if iteration >= self.num_iterations:
            return False, "Maximum iterations reached"
        
        # Check for improvement
        if current_psnr > self.best_psnr + self.min_improvement:
            self.best_psnr = current_psnr
            self.bad_iterations = 0
            return True, "Improvement detected"
        else:
            self.bad_iterations += 1
            
        if self.bad_iterations >= self.patience:
            return False, f"No improvement for {self.patience} iterations"
        
        return True, "Continuing based on patience"

class IterativePruningPipeline:
    """
    Complete iterative pruning pipeline manager
    """
    def __init__(self, model, config=None):
        self.original_model = model
        self.current_model = copy.deepcopy(model)
        
        # Default configuration
        default_config = {
            'target_ratio': 0.4,
            'num_iterations': 4,
            'schedule_type': 'linear',
            'fine_tune_epochs': 5,
            'kd_alpha': 0.7,
            'kd_temperature': 4.0,
            'kd_beta': 0.3,
            'learning_rate': 1e-4,
            'importance_threshold_decay': 0.9,
            'initial_threshold': 0.5
        }
        
        # Merge user config with defaults
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # Initialize components
        self.scheduler = IterativePruningScheduler(
            target_ratio=self.config['target_ratio'],
            num_iterations=self.config['num_iterations'],
            schedule_type=self.config['schedule_type']
        )
        
        self.pruner = StructuredPruner(self.current_model)
        self.kd_criterion = KnowledgeDistillationLoss(
            alpha=self.config['kd_alpha'],
            temperature=self.config['kd_temperature'],
            beta=self.config['kd_beta']
        )
        
        # Results tracking
        self.iteration_results = []
        
    def apply_pruning_iteration(self, iteration, train_loader=None):
        """
        Apply one iteration of the pruning pipeline
        
        Args:
            iteration: Current iteration number (0-based)
            train_loader: Training data loader for fine-tuning
        
        Returns:
            Dictionary with iteration results
        """
        print(f"\n{'='*60}")
        print(f"PRUNING ITERATION {iteration + 1}/{self.config['num_iterations']}")
        print(f"{'='*60}")
        
        iteration_start = time.time()
        
        # Step 1: Generate pruning plan for this iteration
        target_ratio = self.scheduler.get_iteration_target(iteration)
        threshold = self.config['initial_threshold'] * (self.config['importance_threshold_decay'] ** iteration)
        
        print(f"Target ratio: {target_ratio:.1%}")
        print(f"Importance threshold: {threshold:.3f}")
        
        # Update pruner with current model
        self.pruner = StructuredPruner(self.current_model)
        
        # Set some importance scores low (simulated importance learning)
        self._simulate_importance_learning(threshold)
        
        # Generate pruning plan
        pruning_plan = self.pruner.generate_pruning_plan(
            target_ratio=target_ratio,
            threshold=threshold
        )
        
        # Step 2: Apply pruning
        teacher_model = copy.deepcopy(self.current_model)
        student_model = self._create_pruned_model(pruning_plan)
        
        # Calculate actual reduction
        teacher_params = count_parameters(teacher_model)
        student_params = count_parameters(student_model)
        actual_reduction = (teacher_params - student_params) / teacher_params
        
        print(f"Parameter reduction: {actual_reduction:.1%} ({student_params:,} / {teacher_params:,})")
        
        # Step 3: Fine-tune with knowledge distillation
        if train_loader is not None:
            print(f"Fine-tuning with knowledge distillation...")
            fine_tuned_model = self._fine_tune_with_kd(
                student_model, teacher_model, train_loader
            )
        else:
            fine_tuned_model = student_model
            print("Skipping fine-tuning (no training data provided)")
        
        # Step 4: Evaluate performance
        evaluation_results = self._evaluate_model(fine_tuned_model, teacher_model)
        
        # Update current model for next iteration
        self.current_model = fine_tuned_model
        
        # Store results
        iteration_time = time.time() - iteration_start
        iteration_result = {
            'iteration': iteration,
            'target_ratio': target_ratio,
            'actual_reduction': actual_reduction,
            'teacher_params': teacher_params,
            'student_params': student_params,
            'psnr': evaluation_results['student_psnr'],
            'teacher_psnr': evaluation_results['teacher_psnr'],
            'psnr_drop': evaluation_results['psnr_drop'],
            'time': iteration_time,
            'threshold': threshold
        }
        
        self.iteration_results.append(iteration_result)
        
        print(f"Iteration {iteration + 1} completed in {iteration_time:.2f}s")
        print(f"PSNR: {evaluation_results['student_psnr']:.2f}dB (drop: {evaluation_results['psnr_drop']:.2f}dB)")
        
        return iteration_result
    
    def _simulate_importance_learning(self, threshold):
        """Simulate importance learning by setting some scores low"""
        # Set some attention heads to low importance
        for name, mask in self.pruner.head_masks.items():
            num_heads = len(mask.importance_scores)
            # Set progressively more heads to low importance
            low_heads = min(num_heads // 2, num_heads - 1)
            for i in range(low_heads):
                mask.importance_scores.data[i] = threshold * 0.5
        
        # Set some MLP channels to low importance
        for name, mask in self.pruner.channel_masks.items():
            num_channels = len(mask.importance_scores)
            low_channels = num_channels // 3  # Set 1/3 to low importance
            mask.importance_scores.data[:low_channels] = threshold * 0.3
    
    def _create_pruned_model(self, pruning_plan):
        """Create a pruned model based on the pruning plan"""
        # For demonstration, create a model with reduced capacity
        # In practice, this would involve actual weight removal
        
        if pruning_plan['summary']['actual_ratio'] > 0.1:
            # Create smaller model architecture
            original_embed_dim = 96
            original_heads = 4
            
            # Reduce dimensions based on pruning ratio
            reduction_factor = 1 - pruning_plan['summary']['actual_ratio']
            new_embed_dim = max(32, int(original_embed_dim * reduction_factor))
            new_heads = max(1, int(original_heads * reduction_factor))
            
            # Ensure embed_dim is divisible by 4 for pixel shuffle (2x upscaling)
            new_embed_dim = ((new_embed_dim + 3) // 4) * 4
            
            # Ensure heads divides embed_dim evenly
            while new_embed_dim % new_heads != 0 and new_heads > 1:
                new_heads -= 1
            
            # Import the mock model from test_integration
            from test_integration import IntegratedSwinIRModel
            pruned_model = IntegratedSwinIRModel(
                embed_dim=new_embed_dim,
                num_heads=new_heads,
                num_layers=2
            )
            
            print(f"Created pruned model: embed_dim={new_embed_dim}, num_heads={new_heads}")
        else:
            # No significant pruning, return copy
            pruned_model = copy.deepcopy(self.current_model)
        
        return pruned_model
    
    def _fine_tune_with_kd(self, student_model, teacher_model, train_loader):
        """Fine-tune student model with knowledge distillation"""
        student_model.train()
        teacher_model.eval()
        
        optimizer = torch.optim.Adam(
            student_model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        num_epochs = self.config['fine_tune_epochs']
        print(f"  Fine-tuning for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_psnrs = []
            
            for i, train_data in enumerate(train_loader):
                if i >= 10:  # Limit to 10 batches for demo
                    break
                    
                # KD training step
                loss_dict = distillation_training_step(
                    student_model, teacher_model, train_data,
                    self.kd_criterion, optimizer
                )
                
                epoch_losses.append(loss_dict['total_loss'].item())
                epoch_psnrs.append(loss_dict['psnr'])
            
            avg_loss = np.mean(epoch_losses)
            avg_psnr = np.mean(epoch_psnrs)
            
            if epoch % 2 == 0:  # Print every 2nd epoch
                print(f"    Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}dB")
        
        return student_model
    
    def _evaluate_model(self, student_model, teacher_model):
        """Evaluate model performance"""
        student_model.eval()
        teacher_model.eval()
        
        # Create simple test data
        test_data = self._create_test_data(5)
        
        student_psnrs = []
        teacher_psnrs = []
        
        with torch.no_grad():
            for sample in test_data:
                lr_img = sample['L']
                hr_img = sample['H']
                
                # Teacher inference
                teacher_out = teacher_model(lr_img)
                teacher_psnr = calculate_psnr(teacher_out, hr_img)
                teacher_psnrs.append(teacher_psnr)
                
                # Student inference
                student_out = student_model(lr_img)
                student_psnr = calculate_psnr(student_out, hr_img)
                student_psnrs.append(student_psnr)
        
        avg_teacher_psnr = np.mean(teacher_psnrs)
        avg_student_psnr = np.mean(student_psnrs)
        psnr_drop = avg_teacher_psnr - avg_student_psnr
        
        return {
            'teacher_psnr': avg_teacher_psnr,
            'student_psnr': avg_student_psnr,
            'psnr_drop': psnr_drop
        }
    
    def _create_test_data(self, num_samples):
        """Create realistic test data"""
        test_data = []
        for i in range(num_samples):
            # Create structured pattern
            x = torch.linspace(-1, 1, 32)
            y = torch.linspace(-1, 1, 32)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            
            lr_pattern = torch.sin(xx * (2 + i)) * torch.cos(yy * (2 + i))
            lr_pattern = (lr_pattern + 1) / 2
            lr_img = lr_pattern.unsqueeze(0).repeat(1, 3, 1, 1)
            
            hr_pattern = torch.sin(xx * (4 + i)) * torch.cos(yy * (4 + i))
            hr_pattern = (hr_pattern + 1) / 2
            hr_img = F.interpolate(hr_pattern.unsqueeze(0).repeat(1, 3, 1, 1),
                                 size=(64, 64), mode='bicubic', align_corners=False)
            
            test_data.append({'L': lr_img, 'H': hr_img})
        
        return test_data
    
    def run_complete_pipeline(self, train_loader=None):
        """
        Run the complete iterative pruning pipeline
        
        Args:
            train_loader: Optional training data loader
            
        Returns:
            Dictionary with final results
        """
        print("="*70)
        print("ITERATIVE PRUNING PIPELINE")
        print("="*70)
        
        pipeline_start = time.time()
        original_params = count_parameters(self.original_model)
        
        print(f"Original model parameters: {original_params:,}")
        print(f"Target reduction: {self.config['target_ratio']:.1%}")
        print(f"Schedule type: {self.config['schedule_type']}")
        print(f"Number of iterations: {self.config['num_iterations']}")
        
        # Run iterative pruning
        for iteration in range(self.config['num_iterations']):
            iteration_result = self.apply_pruning_iteration(iteration, train_loader)
            
            # Check convergence
            should_continue, reason = self.scheduler.should_continue(
                iteration_result['psnr'], iteration
            )
            
            print(f"Convergence check: {reason}")
            
            if not should_continue:
                print(f"Early stopping at iteration {iteration + 1}")
                break
        
        # Final evaluation
        pipeline_time = time.time() - pipeline_start
        final_params = count_parameters(self.current_model)
        final_reduction = (original_params - final_params) / original_params
        
        final_results = {
            'success': True,
            'original_params': original_params,
            'final_params': final_params,
            'final_reduction': final_reduction,
            'iterations_completed': len(self.iteration_results),
            'total_time': pipeline_time,
            'final_psnr': self.iteration_results[-1]['psnr'] if self.iteration_results else 0,
            'iteration_results': self.iteration_results,
            'final_model': self.current_model
        }
        
        self._print_final_summary(final_results)
        return final_results
    
    def _print_final_summary(self, results):
        """Print final pipeline summary"""
        print(f"\n{'='*70}")
        print("ITERATIVE PRUNING PIPELINE SUMMARY")
        print(f"{'='*70}")
        
        print(f"Original parameters: {results['original_params']:,}")
        print(f"Final parameters: {results['final_params']:,}")
        print(f"Total reduction: {results['final_reduction']:.1%}")
        print(f"Iterations completed: {results['iterations_completed']}")
        print(f"Total time: {results['total_time']:.2f}s")
        print(f"Final PSNR: {results['final_psnr']:.2f}dB")
        
        print(f"\nIteration-by-iteration progress:")
        print(f"{'Iter':<4} {'Reduction':<10} {'PSNR':<8} {'Drop':<8} {'Time':<8}")
        print("-" * 45)
        
        for result in results['iteration_results']:
            print(f"{result['iteration']+1:<4} "
                  f"{result['actual_reduction']:.1%}    "
                  f"{result['psnr']:.2f}dB  "
                  f"{result['psnr_drop']:.2f}dB  "
                  f"{result['time']:.1f}s")
        
        # Success criteria
        success_criteria = {
            'achieved_target_reduction': results['final_reduction'] >= self.config['target_ratio'] * 0.8,
            'maintained_quality': results['final_psnr'] > 10.0,
            'completed_successfully': results['success'],
            'reasonable_time': results['total_time'] < 300  # 5 minutes
        }
        
        print(f"\nSuccess criteria:")
        all_passed = True
        for criterion, passed in success_criteria.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {criterion}: {status}")
            if not passed:
                all_passed = False
        
        print(f"\nOverall result: {'🎉 SUCCESS' if all_passed else '❌ NEEDS IMPROVEMENT'}")

def create_mock_train_loader():
    """Create a mock training data loader for testing"""
    class MockDataset:
        def __init__(self, size=20):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Create realistic training sample
            x = torch.linspace(-1, 1, 32)
            y = torch.linspace(-1, 1, 32)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            
            lr_pattern = torch.sin(xx * (2 + idx * 0.1)) * torch.cos(yy * (2 + idx * 0.1))
            lr_pattern = (lr_pattern + 1) / 2
            lr_img = lr_pattern.unsqueeze(0).repeat(3, 1, 1)
            
            hr_pattern = torch.sin(xx * (4 + idx * 0.1)) * torch.cos(yy * (4 + idx * 0.1))
            hr_pattern = (hr_pattern + 1) / 2
            hr_img = F.interpolate(hr_pattern.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0),
                                 size=(64, 64), mode='bicubic', align_corners=False)[0]
            
            return {'L': lr_img, 'H': hr_img, 'L_path': [f'sample_{idx}.png']}
    
    dataset = MockDataset()
    return DataLoader(dataset, batch_size=1, shuffle=False)

def main(json_path='options/train_swinir_chunk4.json'):
    '''
    # ----------------------------------------
    # Main function for Chunk 4: Iterative Pruning Pipeline
    # ----------------------------------------
    '''
    
    # For demonstration, create a mock model
    from test_integration import IntegratedSwinIRModel
    model = IntegratedSwinIRModel(embed_dim=96, num_heads=4, num_layers=2)
    
    # Configuration for iterative pruning
    config = {
        'target_ratio': 0.4,
        'num_iterations': 4,
        'schedule_type': 'linear',
        'fine_tune_epochs': 3,
        'kd_alpha': 0.7,
        'kd_temperature': 4.0,
        'kd_beta': 0.3,
        'learning_rate': 1e-4,
        'importance_threshold_decay': 0.9,
        'initial_threshold': 0.5
    }
    
    # Create training data loader
    train_loader = create_mock_train_loader()
    
    # Initialize and run pipeline
    pipeline = IterativePruningPipeline(model, config)
    results = pipeline.run_complete_pipeline(train_loader)
    
    return results

if __name__ == '__main__':
    main()
