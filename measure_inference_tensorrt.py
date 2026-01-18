import argparse
import time
import torch
import numpy as np
import os
import tensorrt as trt
from utils import utils_option as option
from models.select_model import define_Model
from utils.utils_dist import get_dist_info, init_dist

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
         print("Likely cause: Driver/Library mismatch (Error Code 35).")
         print("FIX: In Colab, uninstall tensorrt, then run:")
         print("!pip install tensorrt --extra-index-url https://pypi.nvidia.com")
         return None

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 3GB Workspace
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

    # Setup IO buffers
    # 0 is Input, 1 is Output for single-input/output models
    input_idx = 0
    output_idx = 1
    
    # Check if indices match names if needed, but usually 0/1 works for simple export
    if engine.get_tensor_mode(engine.get_tensor_name(0)) != trt.TensorIOMode.INPUT:
        input_idx = 1
        output_idx = 0
        
    input_name = engine.get_tensor_name(input_idx)
    output_name = engine.get_tensor_name(output_idx)

    context.set_input_shape(input_name, input_tensor.shape)
    
    # Allocate output
    output_shape = context.get_tensor_shape(output_name)
    output_tensor = torch.empty(tuple(output_shape), device='cuda', dtype=torch.float32)

    # Set tensor addresses (required for execute_async_v3)
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

def main(json_path='options/swinir/train_swinir_sr_lightweight_structured_pruning.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path)
    parser.add_argument('--bs', type=int, default=1, help='Batch size')
    parser.add_argument('--size', type=int, default=64, help='Input image size')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    args = parser.parse_args()

    # 1. Load Model (Same logic as original script)
    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = args.dist
    if opt['dist']: init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    if init_path_G is not None: opt['path']['pretrained_netG'] = init_path_G
    
    opt = option.dict_to_nonedict(opt)
    model = define_Model(opt)
    model.init_train()
    netG = model.netG
    netG.eval()
    
    device = torch.device('cuda')
    netG.to(device)

    # Dummy Input
    input_size = (args.bs, 3, args.size, args.size)
    dummy_input = torch.randn(input_size, device=device)

    print(f"Benchmarking Resolution: {input_size}")



    # 2. PyTorch Latency
    pt_time, pt_std = benchmark_pytorch(netG, dummy_input)

    # 3. Export ONNX & Build TRT
    onnx_file = "temp_swinir.onnx"
    trt_file = "temp_swinir.engine"
    
    export_to_onnx(netG, dummy_input, onnx_file)
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
    print("       BENCHMARK SUMMARY (Warmup=50, Runs=500)")
    print("="*50)
    print(f"Resolution : {input_size}")
    print(f"PyTorch    : {pt_time:.4f} ± {pt_std:.4f} ms  | {1000/pt_time:.2f} FPS")
    if trt_engine:
        print(f"TensorRT   : {trt_time:.4f} ± {trt_std:.4f} ms  | {1000/trt_time:.2f} FPS")
        print(f"Speedup    : {pt_time/trt_time:.2f}x")
    print("="*50 + "\n")

    
if __name__ == '__main__':
    main()
