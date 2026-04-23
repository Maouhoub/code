import argparse
import time
import torch
import numpy as np
import random
from utils import utils_option as option
from models.select_model import define_Model
from utils.utils_dist import get_dist_info, init_dist

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_pure_compute(model, input_size=(1, 3, 64, 64), device=None, num_warmup=50, num_runs=2000):
    model_network = model.netG if hasattr(model, 'netG') else model
    model_network.eval()
    
    # Create a dummy input tensor ONCE (removes DataLoader overhead)
    dummy_input = torch.randn(input_size).to(device)
    
    # 1. Verification
    params = count_parameters(model_network)
    print(f"\n[Verification] Model Parameters: {params:,}")
    
    # 2. Warm-up
    print(f"Starting warm-up ({num_warmup} runs)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_network(dummy_input)
            if device.type == 'cuda': torch.cuda.synchronize()

    # 3. Measurement
    print(f"Starting measurement ({num_runs} runs)...")
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = model_network(dummy_input)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    avg_time_s = np.mean(times)
    avg_time_ms = avg_time_s * 1000
    variance_time_ms = np.var(times) * (1000 ** 2)
    fps = 1.0 / avg_time_s
    
    print(f"-"*40)
    print(f"Batch Size    : {input_size[0]}")
    print(f"Resolution    : {input_size[2]}x{input_size[3]}")
    print(f"Avg Latency   : {avg_time_ms:.4f} ms")
    print(f"Latency Var   : {variance_time_ms:.4f} ms^2")
    print(f"Avg FPS       : {fps:.2f}")
    print(f"-"*40)

def main(json_path='options/swinir/train_swinir_sr_lightweight_structured_pruning.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path)
    # Default BS=1 (latency), can be increased for throughput
    parser.add_argument('--bs', type=int, default=1, help='Batch size for benchmark')
    # Default resolution 64x64 (patch size), match your use case
    parser.add_argument('--size', type=int, default=64, help='Input image size')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    args = parser.parse_args()
    
    print("Effective Options file used is : ", args.opt)
    
    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = args.dist
    
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        print(f"Benchmarking with Batch Size: {args.bs} | Resolution: {args.size}x{args.size}")

    # ----------------------------------------
    # Update opt to find latest checkpoint (SAME LOGIC AS TRAINING)
    # ----------------------------------------
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    if init_path_G is not None:
        opt['path']['pretrained_netG'] = init_path_G
    if init_path_E is not None:
        opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    if init_path_optimizerG is not None:
        opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    opt = option.dict_to_nonedict(opt)
    
    # ----------------------------------------
    # Initialize Model
    # ----------------------------------------
    model = define_Model(opt)
    model.init_train()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ----------------------------------------
    # Benchmark "Pure Compute"
    # ----------------------------------------
    # Removes DataLoader overhead completely for measuring model speedups
    benchmark_pure_compute(model, input_size=(args.bs, 3, args.size, args.size), device=device)

if __name__ == '__main__':
    main()
