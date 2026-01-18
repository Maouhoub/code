import argparse
import torch
import numpy as np
import os
import sys
from utils import utils_option as option
from models.select_model import define_Model
from utils.utils_dist import get_dist_info, init_dist

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
            if not is_align_16: 
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
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    if init_path_G is not None:
        opt['path']['pretrained_netG'] = init_path_G
    
    start_epoch = 0
    current_step = 0

    opt = option.dict_to_nonedict(opt)
    model = define_Model(opt)
    model.init_train()
    
    # Inspect
    inspect_alignment(model.netG)

if __name__ == '__main__':
    main()
