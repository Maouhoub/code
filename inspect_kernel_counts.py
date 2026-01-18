import argparse
import torch
import tensorrt as trt
import numpy as np
import os
import torch_pruning as tp
from torch import nn
from models.select_model import define_Model
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

# Suppress TRT logs except warnings
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class KernelCounter(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.kernel_count = 0
        self.layer_names = []

    def report_layer_time(self, layer_name, ms):
        self.kernel_count += 1
        self.layer_names.append(layer_name)

def export_onnx(model, input_shape, onnx_path):
    print(f"Exporting ONNX to {onnx_path}...")
    dummy = torch.randn(input_shape).cuda()
    torch.onnx.export(model, dummy, onnx_path, 
                      opset_version=13, 
                      input_names=['input'], 
                      output_names=['output'],
                      do_constant_folding=True)

def build_engine(onnx_path):
    print(f"Building TensorRT Engine from {onnx_path}...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Use MemoryPoolType.WORKSPACE for newer TensorRT
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2GB
    except:
        pass 

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    return builder.build_serialized_network(network, config)

def count_kernels(engine_bytes, input_shape):
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    
    # Attach our custom profiler
    profiler = KernelCounter()
    context.profiler = profiler
    
    # IO Setup
    # 0 is Input, 1 is Output for single-input/output models
    input_idx = 0
    output_idx = 1
    
    # Check if indices match names if needed
    if engine.get_tensor_mode(engine.get_tensor_name(0)) != trt.TensorIOMode.INPUT:
        input_idx = 1
        output_idx = 0

    input_name = engine.get_tensor_name(input_idx)
    output_name = engine.get_tensor_name(output_idx)
    
    context.set_input_shape(input_name, input_shape)
    
    # Allocate buffers
    d_input = torch.empty(input_shape, device='cuda')
    out_shape = context.get_tensor_shape(output_name)
    d_output = torch.empty(tuple(out_shape), device='cuda')
    
    context.set_tensor_address(input_name, d_input.data_ptr())
    context.set_tensor_address(output_name, d_output.data_ptr())
    
    # Run Inference (Profiler will capture kernel launches)
    stream = torch.cuda.current_stream().cuda_stream
    context.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()
    
    return profiler.kernel_count

# Copying the pruning function from your training script to replicate exact pruning
def apply_pruning(model, pruning_ratio=0.1):
    print(f"Applying structured pruning (Ratio: {pruning_ratio})...")
    
    # Simplified version of what's in main_train... assuming Torch-Pruning is installed
    example_inputs = torch.randn(1, 3, 64, 64).cuda()
    imp = tp.importance.MagnitudeImportance(p=2)
    
    ignored_layers = []
    
    # Simple strategy: prune Conv2d and Linear
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        root_module_types=[nn.Conv2d, nn.Linear],
        ignored_layers=ignored_layers,
    )
    
    pruner.step()
    return model

def main():
    json_path = 'options/swinir/train_swinir_sr_lightweight_structured_pruning.json'
    
    # Setup Logic
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path)
    parser.add_argument('--launcher', default='pytorch')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    args = parser.parse_args()
    
    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = args.dist
    if opt['dist']: init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    opt = option.dict_to_nonedict(opt)
    
    input_shape = (1, 3, 64, 64)
    
    print("\n" + "="*50)
    print(" KERNEL COUNT PROFILING (Fusion Efficiency)")
    print("="*50)

    # --- Analysis 1: Original Model ---
    print("\n[1/2] Analyzing ORIGINAL Model...")
    model_orig = define_Model(opt) # Fresh load
    model_orig.init_train()
    netG_orig = model_orig.netG.cuda().eval()
    
    export_onnx(netG_orig, input_shape, "temp_orig.onnx")
    orig_engine = build_engine("temp_orig.onnx")
    orig_kernels = count_kernels(orig_engine, input_shape)
    print(f"-> Original Model executes {orig_kernels} kernels per inference.")

    # --- Analysis 2: Pruned Model ---
    print("\n[2/2] Analyzing PRUNED Model...")
    # We must reload a fresh model and apply pruning to it
    model_pruned = define_Model(opt)
    model_pruned.init_train()
    netG_pruned = model_pruned.netG.cuda().eval()
    
    # Apply Pruning (simulating the state of your pruned model)
    # NOTE: You normally load a pruned checkpoint. 
    # If you have a pruned checkpoint, change path below:
    # opt['path']['pretrained_netG'] = 'path_to_pruned.pth'
    # Otherwise, we simulate it here:
    netG_pruned = apply_pruning(netG_pruned, pruning_ratio=0.5) # Use ratio 0.5 or whatever you used
    
    export_onnx(netG_pruned, input_shape, "temp_pruned.onnx")
    pruned_engine = build_engine("temp_pruned.onnx")
    pruned_kernels = count_kernels(pruned_engine, input_shape)
    print(f"-> Pruned Model executes {pruned_kernels} kernels per inference.")
    
    # Cleanup
    if os.path.exists("temp_pruned.onnx"): os.remove("temp_pruned.onnx")
    if os.path.exists("temp_orig.onnx"): os.remove("temp_orig.onnx")

    print("-" * 50)
    diff = pruned_kernels - orig_kernels
    print(f"Difference: {diff:+d} kernels")
    
    if diff > 0:
        print("CONCLUSION: Pruning BROKE layer fusion (More kernels = Less Fusion).")
        print("This indicates TensorRT couldn't fuse layers due to irregular shapes.")
    else:
        print("CONCLUSION: Layer fusion seems intact (Kernel count decrease/same).")
    print("="*50)

if __name__ == '__main__':
    main()
