#!/usr/bin/env python3
"""
Debug script to understand why hook registration fails in the importance collection
"""

import os
import sys
import torch
import torch.nn as nn

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main_train_swinir_structured_pruning_complete import ImportanceMaskManager, IterativePruningPipeline
    from test_importance_collection_simple import SimpleTestableModel
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def debug_hook_registration():
    """Debug the hook registration issue"""
    print("="*70)
    print("DEBUGGING HOOK REGISTRATION")
    print("="*70)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTestableModel()
    model.netG = model.netG.to(device)
    
    # Initialize mask manager
    print("1. Initializing mask manager...")
    mask_manager = ImportanceMaskManager(model)
    mask_manager.initialize_masks()
    
    print(f"   Found {len(mask_manager.attention_masks)} attention masks")
    print(f"   Found {len(mask_manager.channel_masks)} channel masks")
    
    # Debug layer navigation
    print("\n2. Testing layer navigation...")
    network = model.netG if hasattr(model, 'netG') else model
    
    # Test a few layer names
    test_layers = list(mask_manager.attention_masks.keys())[:3]
    
    for layer_name in test_layers:
        print(f"\n   Testing layer: {layer_name}")
        
        # Try to navigate manually
        try:
            current_module = network
            parts = layer_name.split('.')
            print(f"     Navigation path: {parts}")
            
            for i, part in enumerate(parts):
                print(f"       Step {i}: {type(current_module).__name__} -> {part}")
                if hasattr(current_module, part):
                    current_module = getattr(current_module, part)
                    print(f"         ? Found: {type(current_module).__name__}")
                else:
                    print(f"         ? Missing: {part}")
                    print(f"         Available: {[name for name, _ in current_module.named_children()]}")
                    break
            else:
                print(f"     ? Successfully navigated to: {type(current_module).__name__}")
                
        except Exception as e:
            print(f"     ? Navigation failed: {e}")
    
    # Check model structure
    print("\n3. Model structure inspection...")
    print("   Direct model structure:")
    for name, module in network.named_modules():
        if len(name.split('.')) <= 3:  # Limit depth
            print(f"     {name}: {type(module).__name__}")


def test_manual_hook_registration():
    """Test manual hook registration using the same approach as the pipeline"""
    print("\n" + "="*70)
    print("TESTING MANUAL HOOK REGISTRATION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTestableModel()
    model.netG = model.netG.to(device)
    
    # Initialize mask manager
    mask_manager = ImportanceMaskManager(model)
    mask_manager.initialize_masks()
    
    # Get network
    network = model.netG if hasattr(model, 'netG') else model
    
    # Test hook registration like the pipeline does
    print("1. Testing hook registration for attention layers...")
    
    captured_activations = {}
    hooks = []
    hook_count = 0
    
    def create_activation_hook(layer_name, is_attention=True):
        def hook_fn(module, input, output):
            if layer_name not in captured_activations:
                captured_activations[layer_name] = []
            captured_activations[layer_name].append(output.detach().clone())
        return hook_fn
    
    # Try to register hooks for attention layers
    for layer_name in list(mask_manager.attention_masks.keys())[:3]:  # Test first 3
        print(f"   Attempting to register hook for: {layer_name}")
        try:
            # Navigate to the layer (same as pipeline code)
            current_module = network
            parts = layer_name.split('.')
            for part in parts:
                current_module = getattr(current_module, part)
            
            hook = current_module.register_forward_hook(
                create_activation_hook(layer_name, is_attention=True)
            )
            hooks.append(hook)
            hook_count += 1
            print(f"     ? Successfully registered hook #{hook_count}")
            
        except AttributeError as e:
            print(f"     ? Failed to register hook: {e}")
            # Try alternative navigation
            print(f"     Trying alternative navigation...")
            try:
                # Check if we can find it through named_modules
                for name, module in network.named_modules():
                    if name == layer_name:
                        hook = module.register_forward_hook(
                            create_activation_hook(layer_name, is_attention=True)
                        )
                        hooks.append(hook)
                        hook_count += 1
                        print(f"     ? Alternative method succeeded!")
                        break
                else:
                    print(f"     ? Alternative method also failed")
            except Exception as e2:
                print(f"     ? Alternative method failed: {e2}")
    
    print(f"\n2. Total hooks registered: {hook_count}")
    
    if hook_count > 0:
        print("3. Testing forward pass with hooks...")
        with torch.no_grad():
            test_input = torch.randn(2, 3, 64, 64, device=device)
            model.feed_data({'L': test_input, 'H': test_input})
            output = network(test_input)
        
        print(f"   Activations captured: {len(captured_activations)}")
        for name, activations in captured_activations.items():
            print(f"     {name}: {len(activations)} batches, shape={activations[0].shape}")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    return hook_count > 0


def main():
    """Main debug function"""
    print("Hook Registration Debug Tool")
    print("This helps identify why importance collection hook registration fails")
    
    debug_hook_registration()
    success = test_manual_hook_registration()
    
    print("\n" + "="*70)
    print("DEBUG SUMMARY")
    print("="*70)
    
    if success:
        print("? Hook registration CAN work with proper navigation")
        print("?? The issue is likely in the navigation logic in _collect_importance_scores")
        print("?? Solution: Fix the layer navigation or use named_modules() approach")
    else:
        print("? Hook registration fundamentally broken")
        print("?? Need to investigate model structure and layer naming")
    
    return success


if __name__ == "__main__":
    main()
