"""
SwinIR Structured Pruning using Torch-Pruning
-------------------------------------------------

This script is derived from `main_train_swinir_structured_pruning_complete.py` but
replaces the custom importance scoring and surgical mask-based pruning with the
torch-pruning library (DependencyGraph + BasePruner). The goal is to get real
parameter/FLOPs reductions and avoid slower pruned models caused by masking-only
approaches.

It creates a conservative pruning pipeline (default 20% pruning ratio) and offers
an optional lightweight knowledge-distillation fine-tuning to recover quality.

Usage: run similarly to other training scripts in the repo. Defaults to
`options/swinir/train_swinir_sr_lightweight.json`.
"""

import sys
import os
import argparse
import random
import copy
import time
import traceback
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Try to import torch-pruning
try:
    import torch_pruning as tp
except Exception:
    tp = None

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


class SwinIRWindowAttentionPruner(tp.BasePruningFunc if tp is not None else object):
    """Custom pruner for the SwinIR `WindowAttention` used by the project's SwinIR.

    This mirrors the pattern used in the torch-pruning examples for Swin-like
    attention modules: when pruning output channels we remove the corresponding
    slices in QKV and update projection layers.
    """

    def prune_out_channels(self, layer: nn.Module, idxs: list):
        # qkv: Linear with out_features = 3 * embed_dim (concatenated Q,K,V)
        # proj: Linear with in/out = embed_dim
        if hasattr(layer, 'qkv') and hasattr(layer, 'proj'):
            # compute qkv indices
            dim = layer.qkv.in_features
            qkv_idxs = idxs + [i + dim for i in idxs] + [i + 2 * dim for i in idxs]
            tp.prune_linear_out_channels(layer.qkv, qkv_idxs)
            # projection: remove corresponding in-channels and out-channels when safe
            tp.prune_linear_in_channels(layer.proj, idxs)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: list):
        if hasattr(layer, 'qkv'):
            tp.prune_linear_in_channels(layer.qkv, idxs)
        return layer

    def get_out_channels(self, layer):
        if hasattr(layer, 'qkv') and hasattr(layer.qkv, 'weight'):
            return layer.qkv.weight.shape[0]
        return 0

    def get_in_channels(self, layer):
        if hasattr(layer, 'qkv') and hasattr(layer.qkv, 'weight'):
            return layer.qkv.weight.shape[1]
        return 0


class TorchPruningManager:
    """Manager to build DependencyGraph and run BasePruner from torch-pruning."""

    def __init__(self, model, example_inputs, config=None):
        if tp is None:
            raise RuntimeError('torch-pruning is required (pip install torch-pruning)')

        self.model = model
        self.network = model.netG if hasattr(model, 'netG') else model
        self.example_inputs = example_inputs
        self.config = config or {}

        self.dependency_graph = None
        self.pruner = None
        self.original_macs = 0
        self.original_params = 0

        self._setup()

    def _setup(self):
        device = next(self.network.parameters()).device
        if hasattr(self.example_inputs, 'device') and self.example_inputs.device != device:
            self.example_inputs = self.example_inputs.to(device)

        self.dependency_graph = tp.DependencyGraph().build_dependency(self.network, example_inputs=self.example_inputs)
        self.original_macs, self.original_params = tp.utils.count_ops_and_params(self.network, self.example_inputs)

    def create_pruner(self, pruning_ratio=0.2):
        importance = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')
        ignored_layers = []
        num_heads = {}
        customized_pruners = {}

        for name, module in self.network.named_modules():
            # skip common non-prunable layers
            lname = name.lower()
            if any(k in lname for k in ['norm', 'patch_embed', 'upsample', 'output', 'final', 'head']):
                ignored_layers.append(module)
                continue

            # Detect WindowAttention-like modules used in repo's `models/network_swinir.py`:
            if hasattr(module, 'qkv') and hasattr(module, 'proj'):
                n_heads = getattr(module, 'num_heads', None)
                if hasattr(module, 'qkv'):
                    num_heads[module.qkv] = n_heads or 6
                customized_pruners[type(module)] = SwinIRWindowAttentionPruner()

        self.pruner = tp.pruner.BasePruner(
            model=self.network,
            example_inputs=self.example_inputs,
            global_pruning=False,
            importance=importance,
            iterative_steps=1,
            pruning_ratio=pruning_ratio,
            num_heads=num_heads,
            output_transform=lambda out: out.sum() if isinstance(out, torch.Tensor) else out[0].sum(),
            ignored_layers=ignored_layers,
            customized_pruners=customized_pruners,
            root_module_types=(nn.Linear, nn.LayerNorm),
        )
        return True

    def prune(self):
        if self.pruner is None:
            raise RuntimeError('Pruner not created')

        for g in self.pruner.step(interactive=False):
            g.prune()

        # fix attention dims
        for m in self.network.modules():
            if hasattr(m, 'qkv') and hasattr(m, 'proj'):
                if hasattr(m, 'num_heads') and m.num_heads > 0:
                    new_dim = m.qkv.in_features
                    head_dim = new_dim // m.num_heads
                    if hasattr(m, 'head_dim'):
                        m.head_dim = head_dim
                    if hasattr(m, 'scale'):
                        try:
                            m.scale = head_dim ** -0.5
                        except Exception:
                            pass

        cur_macs, cur_params = tp.utils.count_ops_and_params(self.network, self.example_inputs)
        return (self.original_macs, self.original_params, cur_macs, cur_params)


class KDTrainer:
    def __init__(self, teacher, student, config=None):
        self.teacher = teacher
        self.student = student
        self.config = config or {}
        self.alpha = self.config.get('alpha', 0.7)
        self.beta = self.config.get('beta', 0.3)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def compute_loss(self, student_out, teacher_out, target=None):
        distil = F.mse_loss(student_out, teacher_out.detach())
        if target is not None:
            task = F.mse_loss(student_out, target)
            return self.beta * task + self.alpha * distil
        return distil


class PruningPipeline:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}
        self.pruning_ratio = self.config.get('pruning_ratio', 0.2)
        self.fine_tune_epochs = self.config.get('fine_tune_epochs', 10)
        self.use_kd = self.config.get('use_kd', True)

    def run(self, train_loader, test_loader=None):
        # prepare example input
        example = None
        for batch in train_loader:
            if isinstance(batch, dict):
                example = batch.get('L', batch.get('LQ'))
            elif isinstance(batch, (list, tuple)):
                example = batch[0]
            if example is not None:
                example = example[:1]
                break
        if example is None:
            device = next(self.model.parameters()).device
            example = torch.randn(1, 3, 64, 64).to(device)

        tp_manager = TorchPruningManager(self.model, example, self.config)
        tp_manager.create_pruner(self.pruning_ratio)
        orig_macs, orig_params, new_macs, new_params = tp_manager.prune()

        print(f"MACs: {orig_macs/1e9:.3f}G -> {new_macs/1e9:.3f}G")
        print(f"Params: {orig_params/1e6:.3f}M -> {new_params/1e6:.3f}M")

        # KD fine-tune
        if self.use_kd:
            kd = KDTrainer(copy.deepcopy(self.model), self.model, self.config)
            self._fine_tune_kd(kd, train_loader)

        return True

    def _fine_tune_kd(self, kd_trainer, train_loader):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        max_batches = 200
        for epoch in range(min(self.fine_tune_epochs, 5)):
            total_loss = 0.0
            n = 0
            for i, batch in enumerate(train_loader):
                if i >= max_batches:
                    break
                if isinstance(batch, dict):
                    inputs = batch.get('L', batch.get('LQ')).to(next(self.model.parameters()).device)
                    targets = batch.get('H')
                    if targets is not None:
                        targets = targets.to(next(self.model.parameters()).device)
                else:
                    continue

                optimizer.zero_grad()
                student_out = self.model(inputs)
                with torch.no_grad():
                    teacher_out = kd_trainer.teacher(inputs)
                loss = kd_trainer.compute_loss(student_out, teacher_out, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n += 1
            if n:
                print(f"KD Epoch {epoch+1}: avg loss = {total_loss/n:.6f}")


def main(json_path='options/swinir/train_swinir_sr_lightweight.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path)
    parser.add_argument('--launcher', default='pytorch')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    opt = option.dict_to_nonedict(opt)

    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # dataloaders
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set, batch_size=dataset_opt['dataloader_batch_size']//opt['world_size'], shuffle=False, num_workers=dataset_opt['dataloader_num_workers']//opt['world_size'], drop_last=True, pin_memory=True, sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set, batch_size=dataset_opt['dataloader_batch_size'], shuffle=dataset_opt['dataloader_shuffle'], num_workers=dataset_opt['dataloader_num_workers'], drop_last=True, pin_memory=True)
        elif phase.split('_')[0] == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
        else:
            test_loader = None

    model = define_Model(opt)
    model.init_train()

    pruning_config = {
        'pruning_ratio': 0.2,
        'fine_tune_epochs': 10,
        'use_kd': True,
        'alpha': 0.7,
        'beta': 0.3,
    }

    if opt['rank'] == 0:
        pipeline = PruningPipeline(model, pruning_config)
        pipeline.run(train_loader, test_loader)


if __name__ == '__main__':
    main()
