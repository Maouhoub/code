"""
Minimal smoke test for torch-pruning integration with SwinIR in this repo.
Creates a SwinIR model via define_Model(), prepares a random example input,
builds TorchPruningManager and runs a single prune step to ensure no exceptions.
"""
import torch
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main_train_swinir_structured_pruning_torchpruning import TorchPruningManager
from models.select_model import define_Model
from utils.utils_option import parse


def run_smoke(json_path='options/swinir/train_swinir_sr_lightweight.json'):
    # load options (fall back to minimal config if file not found)
    try:
        opt = parse(json_path, is_train=True)
    except Exception:
        opt = {'netG': {'net_type': 'swinir', 'upscale': 2, 'in_chans': 3, 'img_size': 64, 'window_size': 8,
                        'depths': [2,2], 'embed_dim': 48, 'num_heads': [4,4], 'mlp_ratio': 2}} 

    model = define_Model(opt)
    model.init_train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create a random example input
    example = torch.randn(1, 3, 64, 64).to(device)

    tp_manager = TorchPruningManager(model, example, config={})
    tp_manager.create_pruner(pruning_ratio=0.15)
    orig_macs, orig_params, new_macs, new_params = tp_manager.prune()

    print(f"SMOKE TEST: MACs {orig_macs/1e9:.4f}G -> {new_macs/1e9:.4f}G")
    print(f"SMOKE TEST: Params {orig_params/1e6:.4f}M -> {new_params/1e6:.4f}M")


if __name__ == '__main__':
    run_smoke()
