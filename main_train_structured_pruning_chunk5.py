import os
import time
import torch
import torch.nn as nn
import numpy as np
import psutil
import copy
from collections import defaultdict
from pathlib import Path

# Import from previous chunks
from main_train_structured_pruning_chunk4 import IterativePruningPipeline, create_mock_train_loader
from main_train_structured_pruning_chunk3 import calculate_psnr
from main_train_structured_pruning_chunk2 import count_parameters

'''
# --------------------------------------------
# Structured Pruning for SwinIR - Chunk 5: Comprehensive Evaluation
# Multi-dataset evaluation, hardware profiling, and baseline comparison
# --------------------------------------------
'''

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation framework for pruned models
    """
    def __init__(self, original_model, pruned_model, config=None):
        self.original_model = original_model
        self.pruned_model = pruned_model
        
        # Default evaluation configuration
        default_config = {
            'datasets': ['DIV2K', 'Set5', 'Urban100'],
            'num_samples_per_dataset': 10,
            'warmup_runs': 5,
            'timing_runs': 20,
            'memory_profiling': True,
            'target_reduction': 0.4,
            'max_psnr_drop': 0.5,
            'min_speedup': 1.2,
            'min_memory_reduction': 0.2
        }
        
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        self.results = {}
        
    def create_synthetic_datasets(self):
        """Create synthetic datasets for evaluation"""
        datasets = {}
        
        for dataset_name in self.config['datasets']:
            dataset_samples = []
            num_samples = self.config['num_samples_per_dataset']
            
            for i in range(num_samples):
                if dataset_name == 'DIV2K':
                    # High resolution natural images
                    lr_size, hr_size = (64, 64), (128, 128)
                    complexity = 3.0 + i * 0.1
                elif dataset_name == 'Set5':
                    # Standard test images
                    lr_size, hr_size = (32, 32), (64, 64)
                    complexity = 2.0 + i * 0.1
                elif dataset_name == 'Urban100':
                    # Urban scenes with fine details
                    lr_size, hr_size = (48, 48), (96, 96)
                    complexity = 4.0 + i * 0.2
                else:
                    lr_size, hr_size = (32, 32), (64, 64)
                    complexity = 2.5 + i * 0.1
                
                # Create realistic patterns
                x = torch.linspace(-1, 1, lr_size[0])
                y = torch.linspace(-1, 1, lr_size[1])
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                
                # Generate complex patterns based on dataset characteristics
                lr_pattern = (torch.sin(xx * complexity) * torch.cos(yy * complexity) + 
                             torch.sin(xx * complexity * 1.5) * torch.cos(yy * complexity * 0.7))
                lr_pattern = (lr_pattern + 2) / 4  # Normalize to [0, 1]
                lr_img = lr_pattern.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
                
                # Generate corresponding HR image
                hr_x = torch.linspace(-1, 1, hr_size[0])
                hr_y = torch.linspace(-1, 1, hr_size[1])
                hr_xx, hr_yy = torch.meshgrid(hr_x, hr_y, indexing='ij')
                
                hr_pattern = (torch.sin(hr_xx * complexity) * torch.cos(hr_yy * complexity) + 
                             torch.sin(hr_xx * complexity * 1.5) * torch.cos(hr_yy * complexity * 0.7))
                hr_pattern = (hr_pattern + 2) / 4
                hr_img = hr_pattern.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
                
                dataset_samples.append({
                    'lr': lr_img,
                    'hr': hr_img,
                    'name': f'{dataset_name}_sample_{i:03d}'
                })
            
            datasets[dataset_name] = dataset_samples
            print(f"Created {dataset_name} dataset with {len(dataset_samples)} samples")
        
        return datasets
    
    def evaluate_quality_metrics(self, datasets):
        """Evaluate PSNR and SSIM across datasets"""
        print("\n" + "="*60)
        print("QUALITY METRICS EVALUATION")
        print("="*60)
        
        quality_results = {}
        
        self.original_model.eval()
        self.pruned_model.eval()
        
        with torch.no_grad():
            for dataset_name, samples in datasets.items():
                original_psnrs = []
                pruned_psnrs = []
                
                print(f"\nEvaluating {dataset_name}...")
                
                for sample in samples:
                    lr_img = sample['lr']
                    hr_img = sample['hr']
                    
                    # Original model inference
                    try:
                        original_output = self.original_model(lr_img)
                        original_psnr = calculate_psnr(original_output, hr_img)
                        original_psnrs.append(original_psnr)
                    except Exception as e:
                        print(f"Original model failed: {e}")
                        original_psnrs.append(0.0)
                    
                    # Pruned model inference
                    try:
                        pruned_output = self.pruned_model(lr_img)
                        pruned_psnr = calculate_psnr(pruned_output, hr_img)
                        pruned_psnrs.append(pruned_psnr)
                    except Exception as e:
                        print(f"Pruned model failed: {e}")
                        pruned_psnrs.append(0.0)
                
                # Calculate statistics
                avg_original_psnr = np.mean(original_psnrs)
                avg_pruned_psnr = np.mean(pruned_psnrs)
                psnr_drop = avg_original_psnr - avg_pruned_psnr
                std_original = np.std(original_psnrs)
                std_pruned = np.std(pruned_psnrs)
                
                quality_results[dataset_name] = {
                    'original_psnr': avg_original_psnr,
                    'pruned_psnr': avg_pruned_psnr,
                    'psnr_drop': psnr_drop,
                    'original_std': std_original,
                    'pruned_std': std_pruned,
                    'samples_count': len(samples)
                }
                
                print(f"  Original PSNR: {avg_original_psnr:.2f} ± {std_original:.2f} dB")
                print(f"  Pruned PSNR:   {avg_pruned_psnr:.2f} ± {std_pruned:.2f} dB")
                print(f"  PSNR Drop:     {psnr_drop:.2f} dB")
        
        return quality_results
    
    def profile_inference_performance(self, datasets):
        """Profile inference time and memory usage"""
        print("\n" + "="*60)
        print("HARDWARE PROFILING")
        print("="*60)
        
        performance_results = {}
        
        # Use a representative dataset for profiling
        test_samples = datasets[list(datasets.keys())[0]][:5]  # Use first 5 samples
        
        def measure_inference_time(model, samples, warmup_runs=5, timing_runs=20):
            """Measure inference time with proper warmup"""
            model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup_runs):
                    for sample in samples:
                        _ = model(sample['lr'])
            
            # Actual timing
            times = []
            with torch.no_grad():
                for _ in range(timing_runs):
                    start_time = time.perf_counter()
                    for sample in samples:
                        _ = model(sample['lr'])
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) / len(samples))
            
            return np.mean(times), np.std(times)
        
        def measure_memory_usage(model, sample):
            """Measure peak memory usage during inference"""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                model.eval()
                with torch.no_grad():
                    _ = model(sample['lr'])
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                torch.cuda.empty_cache()
                return peak_memory
            else:
                # CPU memory measurement - estimate based on model parameters
                # Since actual memory profiling is unreliable on CPU, use parameter count as proxy
                total_params = sum(p.numel() for p in model.parameters())
                # Estimate: 4 bytes per parameter + activation memory (rough estimate)
                estimated_memory = (total_params * 4 + total_params * 0.5) / 1024 / 1024  # MB
                return max(estimated_memory, 1.0)  # Minimum 1MB to avoid zero
        
        # Measure inference time
        print("Measuring inference time...")
        original_time, original_time_std = measure_inference_time(self.original_model, test_samples)
        pruned_time, pruned_time_std = measure_inference_time(self.pruned_model, test_samples)
        
        speedup = original_time / pruned_time if pruned_time > 0 else 0
        
        print(f"  Original model: {original_time*1000:.2f} ± {original_time_std*1000:.2f} ms")
        print(f"  Pruned model:   {pruned_time*1000:.2f} ± {pruned_time_std*1000:.2f} ms")
        print(f"  Speedup:        {speedup:.2f}x")
        
        # Measure memory usage
        print("Measuring memory usage...")
        if self.config['memory_profiling'] and test_samples:
            original_memory = measure_memory_usage(self.original_model, test_samples[0])
            pruned_memory = measure_memory_usage(self.pruned_model, test_samples[0])
            
            memory_reduction = (original_memory - pruned_memory) / original_memory if original_memory > 0 else 0
            
            print(f"  Original memory: {original_memory:.1f} MB")
            print(f"  Pruned memory:   {pruned_memory:.1f} MB")
            print(f"  Memory reduction: {memory_reduction:.1%}")
        else:
            original_memory = pruned_memory = memory_reduction = 0
        
        performance_results = {
            'original_inference_time': original_time,
            'pruned_inference_time': pruned_time,
            'speedup': speedup,
            'original_memory': original_memory,
            'pruned_memory': pruned_memory,
            'memory_reduction': memory_reduction
        }
        
        return performance_results
    
    def analyze_model_compression(self):
        """Analyze model compression metrics"""
        print("\n" + "="*60)
        print("MODEL COMPRESSION ANALYSIS")
        print("="*60)
        
        # Parameter count analysis
        original_params = count_parameters(self.original_model)
        pruned_params = count_parameters(self.pruned_model)
        param_reduction = (original_params - pruned_params) / original_params
        
        print(f"Original parameters: {original_params:,}")
        print(f"Pruned parameters:   {pruned_params:,}")
        print(f"Parameter reduction: {param_reduction:.1%}")
        
        # Model size analysis (approximate)
        def estimate_model_size(model):
            """Estimate model size in MB"""
            total_params = sum(p.numel() for p in model.parameters())
            # Assume 4 bytes per parameter (float32)
            size_mb = total_params * 4 / 1024 / 1024
            return size_mb
        
        original_size = estimate_model_size(self.original_model)
        pruned_size = estimate_model_size(self.pruned_model)
        size_reduction = (original_size - pruned_size) / original_size
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"Pruned model size:   {pruned_size:.2f} MB")
        print(f"Size reduction:      {size_reduction:.1%}")
        
        compression_results = {
            'original_params': original_params,
            'pruned_params': pruned_params,
            'param_reduction': param_reduction,
            'original_size_mb': original_size,
            'pruned_size_mb': pruned_size,
            'size_reduction': size_reduction
        }
        
        return compression_results
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation pipeline"""
        print("="*70)
        print("COMPREHENSIVE EVALUATION PIPELINE")
        print("="*70)
        
        evaluation_start = time.time()
        
        # Step 1: Create synthetic datasets
        print("Step 1: Creating synthetic datasets...")
        datasets = self.create_synthetic_datasets()
        
        # Step 2: Evaluate quality metrics
        quality_results = self.evaluate_quality_metrics(datasets)
        
        # Step 3: Profile hardware performance
        performance_results = self.profile_inference_performance(datasets)
        
        # Step 4: Analyze model compression
        compression_results = self.analyze_model_compression()
        
        # Step 5: Generate comprehensive report
        evaluation_time = time.time() - evaluation_start
        final_results = self._generate_final_report(
            quality_results, performance_results, compression_results, evaluation_time
        )
        
        return final_results
    
    def _generate_final_report(self, quality_results, performance_results, compression_results, evaluation_time):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*70)
        
        # Summary statistics
        avg_psnr_drop = np.mean([r['psnr_drop'] for r in quality_results.values()])
        
        # Success criteria evaluation
        success_criteria = {
            'parameter_reduction': {
                'target': f"{self.config['target_reduction']:.1%}",
                'achieved': f"{compression_results['param_reduction']:.1%}",
                'passed': compression_results['param_reduction'] >= self.config['target_reduction']
            },
            'psnr_preservation': {
                'target': f"< {self.config['max_psnr_drop']:.1f} dB drop",
                'achieved': f"{avg_psnr_drop:.2f} dB drop",
                'passed': avg_psnr_drop <= self.config['max_psnr_drop']
            },
            'inference_speedup': {
                'target': f"> {self.config['min_speedup']:.1f}x",
                'achieved': f"{performance_results['speedup']:.2f}x",
                'passed': performance_results['speedup'] >= self.config['min_speedup']
            },
            'memory_reduction': {
                'target': f"> {self.config['min_memory_reduction']:.1%}",
                'achieved': f"{performance_results['memory_reduction']:.1%}",
                'passed': performance_results['memory_reduction'] >= self.config['min_memory_reduction']
            }
        }
        
        print("\nSUCCESS CRITERIA EVALUATION:")
        print("-" * 50)
        all_passed = True
        for criterion, details in success_criteria.items():
            status = "✓ PASS" if details['passed'] else "✗ FAIL"
            print(f"{criterion:20s}: {status} (Target: {details['target']}, Achieved: {details['achieved']})")
            if not details['passed']:
                all_passed = False
        
        print("\nDETAILED RESULTS:")
        print("-" * 50)
        
        # Quality results per dataset
        print("Quality Metrics by Dataset:")
        for dataset_name, results in quality_results.items():
            print(f"  {dataset_name:10s}: Original={results['original_psnr']:.2f}dB, "
                  f"Pruned={results['pruned_psnr']:.2f}dB, Drop={results['psnr_drop']:.2f}dB")
        
        # Performance summary
        print(f"\nPerformance Metrics:")
        print(f"  Inference Time: {performance_results['original_inference_time']*1000:.2f}ms → "
              f"{performance_results['pruned_inference_time']*1000:.2f}ms ({performance_results['speedup']:.2f}x speedup)")
        print(f"  Memory Usage:   {performance_results['original_memory']:.1f}MB → "
              f"{performance_results['pruned_memory']:.1f}MB ({performance_results['memory_reduction']:.1%} reduction)")
        
        # Compression summary
        print(f"\nCompression Metrics:")
        print(f"  Parameters:     {compression_results['original_params']:,} → "
              f"{compression_results['pruned_params']:,} ({compression_results['param_reduction']:.1%} reduction)")
        print(f"  Model Size:     {compression_results['original_size_mb']:.2f}MB → "
              f"{compression_results['pruned_size_mb']:.2f}MB ({compression_results['size_reduction']:.1%} reduction)")
        
        print(f"\nEvaluation Time: {evaluation_time:.2f}s")
        print(f"\nOverall Result: {'🎉 SUCCESS' if all_passed else '❌ NEEDS IMPROVEMENT'}")
        
        # Compile final results
        final_results = {
            'success': all_passed,
            'evaluation_time': evaluation_time,
            'success_criteria': success_criteria,
            'quality_results': quality_results,
            'performance_results': performance_results,
            'compression_results': compression_results,
            'summary': {
                'avg_psnr_drop': avg_psnr_drop,
                'speedup': performance_results['speedup'],
                'param_reduction': compression_results['param_reduction'],
                'memory_reduction': performance_results['memory_reduction']
            }
        }
        
        return final_results

def create_test_models():
    """Create test models for evaluation"""
    from test_integration import IntegratedSwinIRModel
    
    # Create original model
    original_model = IntegratedSwinIRModel(embed_dim=96, num_heads=4, num_layers=2)
    
    # Create pruned model (simulated pruning for testing)
    pruned_model = IntegratedSwinIRModel(embed_dim=64, num_heads=2, num_layers=1)
    
    return original_model, pruned_model

def test_comprehensive_evaluator():
    """Test the comprehensive evaluator"""
    print("Testing Comprehensive Evaluator...")
    
    # Create test models
    original_model, pruned_model = create_test_models()
    
    # Configure evaluation
    eval_config = {
        'datasets': ['DIV2K', 'Set5'],
        'num_samples_per_dataset': 5,
        'warmup_runs': 2,
        'timing_runs': 5,
        'memory_profiling': True,
        'target_reduction': 0.3,
        'max_psnr_drop': 1.0,
        'min_speedup': 1.1,
        'min_memory_reduction': 0.1
    }
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(original_model, pruned_model, eval_config)
    results = evaluator.run_comprehensive_evaluation()
    
    # Validate results
    assert 'success' in results
    assert 'quality_results' in results
    assert 'performance_results' in results
    assert 'compression_results' in results
    
    print("✓ Comprehensive evaluator test passed!")
    return results

def main():
    """
    Main function for Chunk 5: Comprehensive Evaluation
    """
    print("="*70)
    print("CHUNK 5: COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Test with synthetic models first
    print("\n1. Testing with synthetic models...")
    test_results = test_comprehensive_evaluator()
    
    # Demonstrate integration with pruning pipeline
    print("\n2. Integration with pruning pipeline...")
    from test_integration import IntegratedSwinIRModel
    
    # Create and prune a model using the pipeline
    original_model = IntegratedSwinIRModel(embed_dim=64, num_heads=4, num_layers=2)
    
    config = {
        'target_ratio': 0.4,
        'num_iterations': 2,
        'schedule_type': 'linear',
        'fine_tune_epochs': 2
    }
    
    # Run pruning pipeline
    pipeline = IterativePruningPipeline(original_model, config)
    train_loader = create_mock_train_loader()
    pipeline_results = pipeline.run_complete_pipeline(train_loader)
    
    # Evaluate pruned model
    if pipeline_results['success']:
        print("\n3. Evaluating pipeline results...")
        
        eval_config = {
            'datasets': ['DIV2K', 'Set5', 'Urban100'],
            'num_samples_per_dataset': 8,
            'target_reduction': 0.3,
            'max_psnr_drop': 0.8,
            'min_speedup': 1.2,
            'min_memory_reduction': 0.15
        }
        
        evaluator = ComprehensiveEvaluator(
            original_model, 
            pipeline_results['final_model'], 
            eval_config
        )
        
        final_results = evaluator.run_comprehensive_evaluation()
        
        print("\n" + "="*70)
        print("FINAL INTEGRATION TEST RESULTS")
        print("="*70)
        print(f"Pipeline Success: {pipeline_results['success']}")
        print(f"Evaluation Success: {final_results['success']}")
        print(f"Combined Success: {pipeline_results['success'] and final_results['success']}")
        
        return final_results
    else:
        print("Pipeline failed, skipping comprehensive evaluation")
        return None

if __name__ == '__main__':
    main()
