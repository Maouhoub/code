import os.path
import math
import argparse
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

def benchmark_inference(model, test_loader, opt, device=None, num_warmup=50):
    """
    Accurately measures inference time with:
    1. CUDA Synchronization (for correct GPU timing)
    2. High-precision timer (time.perf_counter)
    3. Warm-up phase (to stabilize GPU clocks/allocator)
    4. Statistical reporting (Mean, Std, FPS)
    """
    
    # ----------------------------------------
    # 1. Setup
    # ----------------------------------------
    model_network = model.netG if hasattr(model, 'netG') else model
    model_network.eval()
    
    if device is None:
        try:
            device = next(model_network.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f" BENCHMARKING INFERENCE TIME")
    print(f" Device: {device}")
    print(f" Warm-up runs: {num_warmup}")
    print(f"{'='*60}")

    times = []
    
    # ----------------------------------------
    # 2. Warm-up Phase
    # ----------------------------------------
    # Run dummy inputs to initialize CUDA context, allocators, and caches.
    # This prevents the first few slow runs from skewing results.
    if hasattr(model, 'netG') and device.type == 'cuda':
        print("Starting warm-up...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64).to(device)
            for _ in range(num_warmup):
                model_network(dummy_input)
                torch.cuda.synchronize(device)
        print("Warm-up completed.")

    # ----------------------------------------
    # 3. Measurement Phase
    # ----------------------------------------
    print(f"\nMeasurement started on {len(test_loader)} images...")
    
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            # Load data
            img_name = os.path.basename(test_data['L_path'][0])
            model.feed_data(test_data)
            
            # --- CRITICAL: Synchronize before time start ---
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            
            # Use high-precision counter
            start = time.perf_counter()
            
            # Run Inference
            model.test()
            
            # --- CRITICAL: Synchronize after time end ---
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            
            end = time.perf_counter()
            
            # Record time
            inference_time = end - start
            times.append(inference_time)
            
            print(f"[{i+1}/{len(test_loader)}] {img_name:<20}: {inference_time*1000:.2f} ms")

    # ----------------------------------------
    # 4. Statistics
    # ----------------------------------------
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_fps = 1.0 / avg_time
    
    print(f"\n{'-'*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'-'*60}")
    print(f"Avg Inference Time : {avg_time*1000:.4f} ms")
    print(f"Std Deviation      : {std_time*1000:.4f} ms")
    print(f"Avg FPS            : {avg_fps:.2f} fps")
    print(f"Min / Max Time     : {np.min(times)*1000:.2f} ms / {np.max(times)*1000:.2f} ms")
    print(f"{'='*60}\n")
    
    return avg_time

def main(json_path='options/swinir/train_swinir_sr_lightweight_structured_pruning.json'):

    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    args = parser.parse_args()
    print("Effective Options file used is : ", args.opt)
    
    opt = option.parse(args.opt, is_train=True) # Loading in train mode to get full configs usually, but strictly we are testing
    opt['dist'] = args.dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    # ----------------------------------------
    # Update opt to find latest checkpoint (SAME LOGIC AS TRAINING SCRIPT)
    # ----------------------------------------
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    
    print(f"\nLooking for checkpoints in: {opt['path']['models']}")
    
    if init_path_G is not None:
        print(f"Found Generator checkpoint: {init_path_G}")
        opt['path']['pretrained_netG'] = init_path_G
    else:
        print("! Warning: No Generator checkpoint found in models directory. It might use the random init or config default.")

    if init_path_E is not None:
        opt['path']['pretrained_netE'] = init_path_E

    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # Step--2 (initialize model)
    # ----------------------------------------
    model = define_Model(opt)
    model.init_train() # Initializes networks and loads weights if path provided in opt
    
    # Get device
    device = torch.device('cuda' if opt['gpu_ids'] is not None and torch.cuda.is_available() else 'cpu')
    
    # ----------------------------------------
    # Step--3 (create valid/test dataloader)
    # ----------------------------------------
    # We use the 'train' dataset as requested to measure inference
    if 'datasets' in opt and 'train' in opt['datasets']:
        dataset_opt = opt['datasets']['train']
        print(f"\nProcessing Train Dataset: {dataset_opt['name']}")
        
        # Define Dataset
        train_set = define_Dataset(dataset_opt)
        
        # We use batch_size=1 to benchmark single-image inference latency
        # pin_memory=True for faster host-to-device transfer
        train_loader = DataLoader(train_set, batch_size=1,
                                    shuffle=False, num_workers=1,
                                    drop_last=False, pin_memory=True)
        
        # Run benchmark
        benchmark_inference(model, train_loader, opt, device=device)
    else:
        print("No train dataset found in options.")

if __name__ == '__main__':
    main()
