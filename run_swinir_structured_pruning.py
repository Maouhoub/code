#!/usr/bin/env python3
"""
Run SwinIR Structured Pruning with Existing Configuration
=========================================================

This script demonstrates how to run the structured pruning program using
your existing SwinIR configuration files.

Usage Examples:
    # Use the structured pruning optimized config (recommended)
    python run_swinir_structured_pruning.py

    # Use your original config
    python run_swinir_structured_pruning.py --config options/swinir/train_swinir_sr_lightweight.json

    # Use any custom config
    python run_swinir_structured_pruning.py --config path/to/your/config.json
"""

import argparse
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run SwinIR Structured Channel Pruning')
    parser.add_argument('--config', type=str, 
                       default='options/swinir/train_swinir_sr_lightweight_structured_pruning.json',
                       help='Path to configuration JSON file')
    parser.add_argument('--test-only', action='store_true',
                       help='Run quick test only (no full training)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(" SwinIR Structured Channel Pruning")
    print("=" * 80)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"? Configuration file not found: {args.config}")
        print("\nAvailable configurations:")
        
        # Look for available configs
        config_dirs = ['options', 'options/swinir']
        for config_dir in config_dirs:
            if os.path.exists(config_dir):
                configs = [f for f in os.listdir(config_dir) if f.endswith('.json')]
                if configs:
                    print(f"  In {config_dir}/:")
                    for config in configs[:5]:  # Show first 5
                        print(f"    - {config}")
                    if len(configs) > 5:
                        print(f"    ... and {len(configs) - 5} more")
        
        return 1
    
    print(f"?? Using configuration: {args.config}")
    
    # Check if enhanced script exists
    enhanced_script = "main_train_psnr_L2_fine_tune_structured_enhanced.py"
    if not os.path.exists(enhanced_script):
        print(f"? Enhanced script not found: {enhanced_script}")
        print("Please ensure the enhanced structured pruning script is in the current directory.")
        return 1
    
    print(f"? Found enhanced script: {enhanced_script}")
    
    if args.test_only:
        # Run quick test
        test_script = "test_structured_pruning.py"
        if os.path.exists(test_script):
            print("\n?? Running quick test...")
            result = subprocess.run([sys.executable, test_script], capture_output=False)
            return result.returncode
        else:
            print(f"? Test script not found: {test_script}")
            return 1
    
    # Show configuration summary
    try:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        print(f"\n?? Configuration Summary:")
        print(f"   Task: {config.get('task', 'unknown')}")
        print(f"   Model: {config.get('netG', {}).get('net_type', 'unknown')}")
        print(f"   Scale Factor: {config.get('scale', 'unknown')}x")
        print(f"   Embed Dim: {config.get('netG', {}).get('embed_dim', 'unknown')}")
        print(f"   Fine-tune Epochs: {config.get('fine_tune', {}).get('L2_ft_epochs', 'unknown')}")
        
        # Show dataset paths
        train_dataset = config.get('datasets', {}).get('train', {})
        if train_dataset:
            print(f"   Training Data HR: {train_dataset.get('dataroot_H', 'not set')}")
            print(f"   Training Data LR: {train_dataset.get('dataroot_L', 'not set')}")
            
        test_dataset = config.get('datasets', {}).get('test', {})
        if test_dataset:
            print(f"   Test Data HR: {test_dataset.get('dataroot_H', 'not set')}")
            print(f"   Test Data LR: {test_dataset.get('dataroot_L', 'not set')}")
            
    except Exception as e:
        print(f"? Could not parse configuration: {e}")
    
    print(f"\n?? Starting structured channel pruning...")
    print(f"   Script: {enhanced_script}")
    print(f"   Config: {args.config}")
    print("   This may take several minutes to hours depending on your dataset size.")
    print("\n" + "=" * 80)
    
    # Run the enhanced script
    cmd = [sys.executable, enhanced_script, '--opt', args.config]
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("? Structured channel pruning completed successfully!")
        
        # Check for results
        models_dir = config.get('path', {}).get('models', '')
        if models_dir and os.path.exists(models_dir):
            results_file = os.path.join(models_dir, 'pruning_results.txt')
            if os.path.exists(results_file):
                print(f"?? Results summary available: {results_file}")
                
                # Show brief results
                try:
                    with open(results_file, 'r') as f:
                        content = f.read()
                        print("\n?? Brief Results:")
                        lines = content.split('\n')
                        for line in lines:
                            if 'Parameters:' in line or 'FLOPs:' in line or 'PSNR' in line:
                                print(f"   {line.strip()}")
                except:
                    pass
        
        print("=" * 80)
        
    else:
        print(f"\n? Structured pruning failed with exit code: {result.returncode}")
        return result.returncode
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)