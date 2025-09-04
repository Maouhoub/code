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
from models.network_swinir import SwinIR


def run_smoke():
    # Create a lightweight SwinIR instance directly to avoid depending on the
    # repo option parsing (which expects many keys). Keep the model small so
    # pruning is fast in a smoke test.
    model = SwinIR(img_size=64, in_chans=3, embed_dim=48, depths=[2, 2],
                   num_heads=[4, 4], window_size=8, mlp_ratio=2,
                   upsampler='', resi_connection='1conv')
    # ensure model replicates train initialization behavior
    try:
        model.apply(model._init_weights)
    except Exception:
        pass
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
