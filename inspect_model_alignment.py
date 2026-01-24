import argparse
import torch
import numpy as np
import os
import sys
from utils import utils_option as option
from models.select_model import define_Model
from utils.utils_dist import get_dist_info, init_dist

import time
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def benchmark_pytorch(model, input_tensor, num_runs=500, num_warmup=50):
    print(f"\n[PyTorch] Starting benchmark...")
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
            torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    fps = 1000 / avg_ms
    print(f"[PyTorch] Latency: {avg_ms:.4f} ± {std_ms:.4f} ms | {fps:.2f} FPS")
    return avg_ms, std_ms

def export_to_onnx(model, input_tensor, onnx_path):
    print(f"\n[ONNX] Exporting to {onnx_path}...")
    torch.onnx.export(
        model, input_tensor, onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
         dynamo=False  
    )

def build_trt_engine(onnx_path, engine_path):
    print(f"\n[TensorRT] Building FP16 engine...")
    try:
        builder = trt.Builder(TRT_LOGGER)
    except TypeError:
         print("\n[ERROR] TensorRT Builder initialization failed!")
         return None

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print(f"[TensorRT ERROR] Failed to parse ONNX")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    plan = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(plan)
    return plan

def benchmark_trt(engine_buffer, input_tensor, num_runs=500, num_warmup=50):
    print(f"\n[TensorRT] Starting benchmark...")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_buffer)
    context = engine.create_execution_context()

    input_idx = 0
    output_idx = 1
    
    if engine.get_tensor_mode(engine.get_tensor_name(0)) != trt.TensorIOMode.INPUT:
        input_idx = 1
        output_idx = 0
        
    input_name = engine.get_tensor_name(input_idx)
    output_name = engine.get_tensor_name(output_idx)

    context.set_input_shape(input_name, input_tensor.shape)
    
    output_shape = context.get_tensor_shape(output_name)
    output_tensor = torch.empty(tuple(output_shape), device='cuda', dtype=torch.float32)

    context.set_tensor_address(input_name, input_tensor.data_ptr())
    context.set_tensor_address(output_name, output_tensor.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream

    # Warmup
    for _ in range(num_warmup):
        context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    fps = 1000 / avg_ms
    print(f"[TensorRT] Latency: {avg_ms:.4f} ± {std_ms:.4f} ms | {fps:.2f} FPS")
    return avg_ms, std_ms

def inspect_alignment(model):
    print("\n" + "="*60)
    print("       CHANNEL ALIGNMENT ANALYSIS")
    print("="*60)
    print(f"{'Layer Name':<40} | {'Channels':<10} | {'% 8':<5} | {'% 16':<5} | {'% 32':<5}")
    print("-" * 80)
    
    total_layers = 0
    misaligned_8 = 0
    misaligned_16 = 0
    misaligned_32 = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            total_layers += 1
            # Check Output Channels (usually the constraint for subsequent kernels)
            out_c = module.out_channels if isinstance(module, torch.nn.Conv2d) else module.out_features
            
            is_align_8 = (out_c % 8 == 0)
            is_align_16 = (out_c % 16 == 0)
            is_align_32 = (out_c % 32 == 0)
            
            if not is_align_8: misaligned_8 += 1
            if not is_align_16: misaligned_16 += 1
            if not is_align_32: misaligned_32 += 1
            
            # Print only first few or misaligned ones to keep it readable, 
            # or print all if list isn't huge. Let's print misaligned ones.
            print(f"{name[-35:]:<40} | {out_c:<10} | {str(is_align_8):<5} | {str(is_align_16):<5} | {str(is_align_32):<5}")
                
    print("-" * 80)
    print(f"Total Computation Layers: {total_layers}")
    print(f"Layers violating 8-alignment:  {misaligned_8} ({misaligned_8/total_layers*100:.1f}%)")
    print(f"Layers violating 16-alignment: {misaligned_16} ({misaligned_16/total_layers*100:.1f}%)")
    print(f"Layers violating 32-alignment: {misaligned_32} ({misaligned_32/total_layers*100:.1f}%)")
    print("="*60)

def main(json_path='options/swinir/train_swinir_sr_lightweight_structured_pruning.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # Distributed init
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    # Load Model

    root_path = '/content/drive/MyDrive/superresolution/swinir_sr_lightweight_x2/models'

    import glob
    # Get list of all files matching the pattern, including full path
    files = glob.glob(os.path.join(root_path, '*_G.pth'))
    
    # Sort files by modification time descending (newest first)
    files.sort(key=os.path.getmtime, reverse=True)

    print("Found checkpoints (ordered by modified date):")
    for f in files:
        print(" Inspecting" , f)
        if f is not None:
            opt['path']['pretrained_netG'] = f
        
        opt = option.dict_to_nonedict(opt)
        model = define_Model(opt)
        model.init_train()
        
        # Inspect
        inspect_alignment(model.netG)
        
        # --- BENCHMARK LOGIC ---
        device = torch.device('cuda')
        model.netG.to(device)
        model.netG.eval()

        # Dummy Input (Standard 64x64 for SwinIR benchmarks)
        input_size = (1, 3, 64, 64)
        dummy_input = torch.randn(input_size, device=device)

        print(f"Benchmarking Resolution: {input_size}")

        # 2. PyTorch Latency
        pt_time, pt_std = benchmark_pytorch(model.netG, dummy_input)

        # 3. Export ONNX & Build TRT
        temp_name = "temp_" + os.path.splitext(os.path.basename(f))[0]
        onnx_file = f"{temp_name}.onnx"
        trt_file = f"{temp_name}.engine"
        
        export_to_onnx(model.netG, dummy_input, onnx_file)
        trt_engine = build_trt_engine(onnx_file, trt_file)

        # 4. TRT Latency
        trt_time = 0.0
        trt_std = 0.0
        if trt_engine:
            trt_time, trt_std = benchmark_trt(trt_engine, dummy_input)
            
            # Cleanup
            if os.path.exists(onnx_file): os.remove(onnx_file)
            if os.path.exists(trt_file): os.remove(trt_file)

        # Summary
        print("\n" + "="*50)
        print(f"       BENCHMARK SUMMARY FOR {os.path.basename(f)}")
        print("="*50)
        print(f"PyTorch    : {pt_time:.4f} ± {pt_std:.4f} ms  | {1000/pt_time:.2f} FPS")
        if trt_engine and trt_time > 0:
            print(f"TensorRT   : {trt_time:.4f} ± {trt_std:.4f} ms  | {1000/trt_time:.2f} FPS")
            print(f"Speedup    : {pt_time/trt_time:.2f}x")
        print("="*50 + "\n")


if __name__ == '__main__':
    main()
