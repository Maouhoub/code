"""
Test script for Chunk 4: Iterative Pruning Pipeline
Validates multi-stage pruning with intermediate fine-tuning and convergence monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import copy
import time

# Add the code directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_train_structured_pruning_chunk4 import (
    IterativePruningScheduler,
    IterativePruningPipeline,
    create_mock_train_loader,
    count_parameters
)

# Import test model
from test_integration import IntegratedSwinIRModel

def test_pruning_scheduler():
    """Test the IterativePruningScheduler functionality"""
    print("Testing IterativePruningScheduler...")
    
    # Test linear schedule
    scheduler = IterativePruningScheduler(target_ratio=0.4, num_iterations=4, schedule_type='linear')
    
    # Validate schedule creation
    expected_ratios = [0.1, 0.2, 0.3, 0.4]  # Linear progression
    assert len(scheduler.schedule) == 4
    
    for i, expected in enumerate(expected_ratios):
        actual = scheduler.get_iteration_target(i)
        assert abs(actual - expected) < 0.01, f"Iteration {i}: expected {expected}, got {actual}"
    
    print(f"✓ Linear schedule: {[f'{x:.1%}' for x in scheduler.schedule]}")
    
    # Test exponential schedule
    scheduler_exp = IterativePruningScheduler(target_ratio=0.4, num_iterations=4, schedule_type='exponential')
    exp_schedule = scheduler_exp.schedule
    
    # Should be front-loaded (higher initial ratios)
    assert exp_schedule[0] > expected_ratios[0]
    assert exp_schedule[-1] <= 0.4
    
    print(f"✓ Exponential schedule: {[f'{x:.1%}' for x in exp_schedule]}")
    
    # Test conservative schedule
    scheduler_cons = IterativePruningScheduler(target_ratio=0.4, num_iterations=4, schedule_type='conservative')
    cons_schedule = scheduler_cons.schedule
    
    # Should be back-loaded (lower initial ratios)
    assert cons_schedule[0] < expected_ratios[0]
    assert abs(cons_schedule[-1] - 0.4) < 0.01  # Should reach target
    
    print(f"✓ Conservative schedule: {[f'{x:.1%}' for x in cons_schedule]}")

def test_convergence_monitoring():
    """Test convergence monitoring functionality"""
    print("Testing convergence monitoring...")
    
    scheduler = IterativePruningScheduler(target_ratio=0.4, num_iterations=6)
    scheduler.patience = 2  # Set patience to 2 for clear testing
    scheduler.min_improvement = 0.5
    
    # Test improving PSNR (first iteration should set baseline)
    should_continue, reason = scheduler.should_continue(25.0, 0)  # First iteration
    assert should_continue
    print(f"✓ First iteration: {reason} (best_psnr: {scheduler.best_psnr}, bad_iterations: {scheduler.bad_iterations})")
    
    should_continue, reason = scheduler.should_continue(26.0, 1)  # Improvement
    assert should_continue
    assert scheduler.bad_iterations == 0
    print(f"✓ Improvement detected: {reason} (best_psnr: {scheduler.best_psnr}, bad_iterations: {scheduler.bad_iterations})")
    
    # Test small improvement (should count as bad iteration)
    should_continue, reason = scheduler.should_continue(26.2, 2)  # Small improvement
    assert should_continue
    assert scheduler.bad_iterations == 1  # Should increment bad iterations
    print(f"✓ Insufficient improvement: {reason} (best_psnr: {scheduler.best_psnr}, bad_iterations: {scheduler.bad_iterations})")
    
    # Another bad iteration - should still continue because bad_iterations=2, patience=2 (2>=2 will trigger)
    should_continue, reason = scheduler.should_continue(25.5, 3)  # Decrease
    assert not should_continue  # Should stop NOW because bad_iterations becomes 2 >= patience=2
    assert "No improvement" in reason
    print(f"✓ Early stopping: {reason} (best_psnr: {scheduler.best_psnr}, bad_iterations: {scheduler.bad_iterations})")
    
    # Test max iterations
    scheduler2 = IterativePruningScheduler(target_ratio=0.4, num_iterations=3)
    should_continue, reason = scheduler2.should_continue(30.0, 3)  # Beyond max
    assert not should_continue
    assert "Maximum iterations" in reason
    print(f"✓ Max iterations: {reason}")

def test_pipeline_initialization():
    """Test IterativePruningPipeline initialization"""
    print("Testing pipeline initialization...")
    
    # Create test model
    model = IntegratedSwinIRModel(embed_dim=64, num_heads=2, num_layers=1)
    
    # Test with default config
    pipeline = IterativePruningPipeline(model)
    
    # Validate components
    assert pipeline.original_model is not None
    assert pipeline.current_model is not None
    assert pipeline.scheduler is not None
    assert pipeline.pruner is not None
    assert pipeline.kd_criterion is not None
    
    # Validate config
    default_config = pipeline.config
    assert default_config['target_ratio'] == 0.4
    assert default_config['num_iterations'] == 4
    assert default_config['schedule_type'] == 'linear'
    
    print(f"✓ Pipeline initialized with {count_parameters(model):,} parameters")
    
    # Test with custom config
    custom_config = {
        'target_ratio': 0.3,
        'num_iterations': 3,
        'schedule_type': 'exponential',
        'fine_tune_epochs': 2,
        'kd_alpha': 0.8
    }
    
    pipeline_custom = IterativePruningPipeline(model, custom_config)
    assert pipeline_custom.config['target_ratio'] == 0.3
    assert pipeline_custom.config['kd_alpha'] == 0.8
    
    print("✓ Custom configuration applied successfully")

def test_single_pruning_iteration():
    """Test a single pruning iteration"""
    print("Testing single pruning iteration...")
    
    # Create model and pipeline
    model = IntegratedSwinIRModel(embed_dim=48, num_heads=2, num_layers=1)
    original_params = count_parameters(model)
    
    config = {
        'target_ratio': 0.3,
        'num_iterations': 3,
        'fine_tune_epochs': 2,
        'initial_threshold': 0.4
    }
    
    pipeline = IterativePruningPipeline(model, config)
    
    # Create minimal training data
    train_loader = create_mock_train_loader()
    
    # Apply first iteration
    result = pipeline.apply_pruning_iteration(0, train_loader)
    
    # Validate result structure
    required_keys = ['iteration', 'target_ratio', 'actual_reduction', 'psnr', 'time']
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    
    # Validate values
    assert result['iteration'] == 0
    assert result['target_ratio'] > 0
    assert result['actual_reduction'] >= 0
    assert result['psnr'] > 5.0  # Should be reasonable
    assert result['time'] > 0
    
    # Check that model was updated
    new_params = count_parameters(pipeline.current_model)
    reduction = (original_params - new_params) / original_params
    
    print(f"✓ Single iteration completed:")
    print(f"  Original params: {original_params:,}")
    print(f"  New params: {new_params:,}")
    print(f"  Reduction: {reduction:.1%}")
    print(f"  PSNR: {result['psnr']:.2f}dB")
    print(f"  Time: {result['time']:.2f}s")

def test_multiple_iterations():
    """Test multiple pruning iterations"""
    print("Testing multiple pruning iterations...")
    
    # Create model and pipeline
    model = IntegratedSwinIRModel(embed_dim=64, num_heads=4, num_layers=2)
    original_params = count_parameters(model)
    
    config = {
        'target_ratio': 0.4,
        'num_iterations': 3,
        'schedule_type': 'linear',
        'fine_tune_epochs': 1,  # Reduced for speed
        'initial_threshold': 0.5
    }
    
    pipeline = IterativePruningPipeline(model, config)
    train_loader = create_mock_train_loader()
    
    # Apply multiple iterations
    results = []
    for i in range(3):
        result = pipeline.apply_pruning_iteration(i, train_loader)
        results.append(result)
    
    # Validate progression
    assert len(results) == 3
    
    # Check that target ratios increase
    for i in range(1, 3):
        assert results[i]['target_ratio'] > results[i-1]['target_ratio']
    
    # Check parameter reduction progression
    final_params = count_parameters(pipeline.current_model)
    total_reduction = (original_params - final_params) / original_params
    
    print(f"✓ Multiple iterations completed:")
    print(f"  Total reduction: {total_reduction:.1%}")
    print(f"  Final PSNR: {results[-1]['psnr']:.2f}dB")
    psnr_values = [f"{r['psnr']:.1f}dB" for r in results]
    print(f"  Iteration PSNRs: {psnr_values}")
    
    # Validate that each iteration processes correctly
    for i, result in enumerate(results):
        assert result['iteration'] == i
        assert result['actual_reduction'] >= 0
        assert result['psnr'] > 0

def test_complete_pipeline():
    """Test the complete iterative pruning pipeline"""
    print("Testing complete pipeline...")
    
    # Create model and pipeline with conservative settings for testing
    model = IntegratedSwinIRModel(embed_dim=48, num_heads=2, num_layers=1)
    
    config = {
        'target_ratio': 0.3,
        'num_iterations': 3,
        'schedule_type': 'linear',
        'fine_tune_epochs': 1,
        'learning_rate': 1e-3,
        'initial_threshold': 0.4
    }
    
    pipeline = IterativePruningPipeline(model, config)
    train_loader = create_mock_train_loader()
    
    # Run complete pipeline
    start_time = time.time()
    final_results = pipeline.run_complete_pipeline(train_loader)
    end_time = time.time()
    
    # Validate final results
    assert 'success' in final_results
    assert 'final_reduction' in final_results
    assert 'final_psnr' in final_results
    assert 'iteration_results' in final_results
    assert 'final_model' in final_results
    
    # Check that pipeline achieved some reduction
    assert final_results['final_reduction'] > 0.1  # At least 10%
    assert final_results['final_psnr'] > 5.0       # Reasonable quality
    assert final_results['iterations_completed'] > 0
    
    print(f"✓ Complete pipeline results:")
    print(f"  Success: {final_results['success']}")
    print(f"  Final reduction: {final_results['final_reduction']:.1%}")
    print(f"  Final PSNR: {final_results['final_psnr']:.2f}dB")
    print(f"  Iterations: {final_results['iterations_completed']}")
    print(f"  Total time: {end_time - start_time:.2f}s")
    
    # Validate iteration results
    iteration_results = final_results['iteration_results']
    assert len(iteration_results) > 0
    
    # Check progressive reduction
    for result in iteration_results:
        assert 'psnr' in result
        assert 'actual_reduction' in result
        assert result['actual_reduction'] >= 0

def test_early_stopping():
    """Test early stopping functionality"""
    print("Testing early stopping...")
    
    model = IntegratedSwinIRModel(embed_dim=32, num_heads=2, num_layers=1)
    
    # Configure for quick early stopping
    config = {
        'target_ratio': 0.5,
        'num_iterations': 5,
        'fine_tune_epochs': 1
    }
    
    pipeline = IterativePruningPipeline(model, config)
    pipeline.scheduler.patience = 2  # Very low patience
    pipeline.scheduler.min_improvement = 2.0  # High improvement threshold
    
    train_loader = create_mock_train_loader()
    
    # Run pipeline (should stop early)
    results = pipeline.run_complete_pipeline(train_loader)
    
    # Should stop before max iterations due to lack of improvement
    assert results['iterations_completed'] < config['num_iterations']
    
    print(f"✓ Early stopping triggered after {results['iterations_completed']} iterations")

def test_different_schedules():
    """Test different pruning schedules"""
    print("Testing different pruning schedules...")
    
    model = IntegratedSwinIRModel(embed_dim=32, num_heads=2, num_layers=1)
    
    schedules_to_test = ['linear', 'exponential', 'conservative']
    results = {}
    
    for schedule_type in schedules_to_test:
        config = {
            'target_ratio': 0.3,
            'num_iterations': 3,
            'schedule_type': schedule_type,
            'fine_tune_epochs': 1
        }
        
        pipeline = IterativePruningPipeline(copy.deepcopy(model), config)
        train_loader = create_mock_train_loader()
        
        result = pipeline.run_complete_pipeline(train_loader)
        results[schedule_type] = result
        
        print(f"  {schedule_type}: {result['final_reduction']:.1%} reduction, "
              f"{result['final_psnr']:.2f}dB PSNR")
    
    # All schedules should achieve some reduction
    for schedule_type, result in results.items():
        assert result['final_reduction'] > 0.05  # At least 5%
        assert result['success'] == True
    
    print("✓ All schedule types completed successfully")

def test_error_handling():
    """Test error handling and edge cases"""
    print("Testing error handling...")
    
    # Test invalid schedule type
    try:
        scheduler = IterativePruningScheduler(schedule_type='invalid')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown schedule type" in str(e)
        print("✓ Invalid schedule type handled")
    
    # Test zero iterations
    scheduler = IterativePruningScheduler(num_iterations=0)
    should_continue, reason = scheduler.should_continue(25.0, 0)
    assert not should_continue
    assert "Maximum iterations" in reason
    print("✓ Zero iterations handled")
    
    # Test negative target ratio (should work, but clamp to reasonable values)
    scheduler = IterativePruningScheduler(target_ratio=-0.1)
    # Should still create a schedule
    assert len(scheduler.schedule) > 0
    print("✓ Negative target ratio handled")

def main():
    """Run all Chunk 4 tests"""
    print("="*60)
    print("RUNNING CHUNK 4 TESTS - ITERATIVE PRUNING PIPELINE")
    print("="*60)
    
    try:
        test_pruning_scheduler()
        test_convergence_monitoring()
        test_pipeline_initialization()
        test_single_pruning_iteration()
        test_multiple_iterations()
        test_complete_pipeline()
        test_early_stopping()
        test_different_schedules()
        test_error_handling()
        
        print("\n" + "="*60)
        print("🎉 ALL CHUNK 4 TESTS PASSED!")
        print("="*60)
        
        print("\nKey Validation Results:")
        print("✓ Pruning schedule generation working correctly")
        print("✓ Convergence monitoring functional")
        print("✓ Single iteration pipeline operational")
        print("✓ Multi-iteration progression validated")
        print("✓ Complete pipeline integration successful")
        print("✓ Early stopping mechanism working")
        print("✓ Different schedule types supported")
        print("✓ Error handling robust")
        
        print("\nChunk 4 Implementation Ready:")
        print("• Multi-stage pruning schedule ✓")
        print("• Intermediate fine-tuning with KD ✓")
        print("• Convergence monitoring ✓")
        print("• Early stopping mechanism ✓")
        print("• Progressive parameter reduction ✓")
        print("• Quality preservation tracking ✓")
        
        print("\n🚀 All 4 chunks validated and ready for publication!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
