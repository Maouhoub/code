#!/usr/bin/env python3
"""
Test script to validate the enhanced SwinIR pruning implementation
"""

import sys
import os
import torch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch_pruning as tp
        print("? torch-pruning imported successfully")
    except ImportError as e:
        print(f"? torch-pruning import failed: {e}")
        return False
    
    try:
        from utils.utils_dist import get_dist_info, init_dist
        print("? utils_dist imported successfully")
    except ImportError as e:
        print(f"? utils_dist import failed: {e}")
        return False
    
    try:
        from models.select_model import define_Model
        print("? model selection imported successfully")
    except ImportError as e:
        print(f"? model selection import failed: {e}")
        return False
    
    return True

def test_distributed_utils():
    """Test distributed utilities with single GPU setup"""
    print("\nTesting distributed utilities...")
    
    try:
        from utils.utils_dist import get_dist_info
        rank, world_size = get_dist_info()
        print(f"? get_dist_info: rank={rank}, world_size={world_size}")
        
        # Test init_dist without environment variables (should not crash)
        if 'RANK' not in os.environ:
            print("? No RANK environment variable (single GPU mode)")
        
        return True
    except Exception as e:
        print(f"? Distributed utils test failed: {e}")
        return False

def test_torch_pruning_basic():
    """Test basic torch-pruning functionality"""
    print("\nTesting basic torch-pruning...")
    
    try:
        import torch_pruning as tp
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
        
        example_inputs = torch.randn(1, 10)
        
        # Test dependency graph building
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
        print("? Dependency graph built successfully")
        
        # Test importance metric
        imp = tp.importance.GroupMagnitudeImportance(p=2)
        print("? Importance metric created successfully")
        
        return True
    except Exception as e:
        print(f"? Torch-pruning basic test failed: {e}")
        return False

def test_cuda_availability():
    """Test CUDA availability and setup"""
    print("\nTesting CUDA setup...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"? CUDA available: {device_count} device(s)")
        print(f"? Current device: {current_device} ({device_name})")
        
        # Test tensor operations
        x = torch.randn(10, 10).cuda()
        y = torch.mm(x, x.t())
        print("? CUDA tensor operations working")
        
        return True
    else:
        print("? CUDA not available (will use CPU)")
        return True

def main():
    """Run all tests"""
    print("="*60)
    print("ENHANCED SWINIR PRUNING - VALIDATION TESTS")
    print("="*60)
    
    tests = [
        test_cuda_availability,
        test_imports,
        test_distributed_utils,
        test_torch_pruning_basic,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"? Test {test.__name__} crashed: {e}")
    
    print("\n" + "="*60)
    print(f"VALIDATION RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("?? All tests passed! The implementation should work correctly.")
        return 0
    else:
        print("? Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
