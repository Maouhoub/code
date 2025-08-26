import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch_pruning as tp
from models.network_swinir import SwinIR, SwinTransformerBlock

class SwinIRPruner:
    def __init__(self, model, example_inputs, pruning_ratio=0.5):
        self.model = model
        self.example_inputs = example_inputs
        self.pruning_ratio = pruning_ratio

    def prune_model(self):
        # Define importance criterion
        importance = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")

        # Collect number of attention heads for each layer
        num_heads = {}
        for m in self.model.modules():
            if isinstance(m, SwinTransformerBlock):
                num_heads[m.attn.query] = m.attn.num_heads

        # Define pruner
        pruner = tp.pruner.BasePruner(
            self.model,
            self.example_inputs,
            global_pruning=False,
            importance=importance,
            iterative_steps=1,
            pruning_ratio=self.pruning_ratio,
            num_heads=num_heads,
            output_transform=lambda out: out.sum(),
            root_module_types=(nn.Linear, nn.LayerNorm),
        )

        # Perform pruning
        for group in pruner.step(interactive=True):
            group.prune()

        # Update attention head sizes
        for m in self.model.modules():
            if isinstance(m, SwinTransformerBlock):
                m.attn.attention_head_size = m.attn.query.out_features // m.attn.num_heads
                m.attn.all_head_size = m.attn.query.out_features

        return self.model

if __name__ == "__main__":
    # Example usage
    example_inputs = torch.randn(1, 3, 224, 224)  # Example input tensor
    model = SwinIR(img_size=224, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])

    pruner = SwinIRPruner(model, example_inputs, pruning_ratio=0.5)
    pruned_model = pruner.prune_model()

    print("Pruned model:", pruned_model)
