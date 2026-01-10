import torch

ckpt_path = '/content/drive/MyDrive/TP/5350_G.pth'

x = torch.load(ckpt_path, weights_only=False)

print("Total params:", sum(p.numel() for p in x.parameters()))
print("Trainable params:", sum(p.numel() for p in x.parameters() if p.requires_grad))

print("\n" + "="*80)
print(" LAYER OUTPUT CHANNEL/FEATURE COUNTS (Post-Pruning)")
print("="*80)

print("\n--- Conv2d Layers ---")
for name, module in x.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        print(f"{name:60s} | in: {module.in_channels:4d} | out: {module.out_channels:4d} | kernel: {module.kernel_size}")

print("\n--- Linear Layers (including MLP fc1/fc2) ---")
for name, module in x.named_modules():
    if isinstance(module, torch.nn.Linear):
        layer_type = ""
        if 'mlp.fc1' in name.lower():
            layer_type = " [MLP fc1 - prunable]"
        elif 'mlp.fc2' in name.lower():
            layer_type = " [MLP fc2 - receives input pruning]"
        elif 'attn' in name.lower():
            layer_type = " [Attention - protected]"
        print(f"{name:60s} | in: {module.in_features:4d} | out: {module.out_features:4d}{layer_type}")

print("\n--- Summary of Key Prunable Layers ---")
for name, module in x.named_modules():
    lower_name = name.lower()
    if isinstance(module, torch.nn.Conv2d):
        if any(kw in lower_name for kw in ['conv_first', 'conv_after_body', 'conv_before_upsample', 'conv_up', 'upsample']):
            print(f"[Conv2d] {name:50s} | out_channels: {module.out_channels}")
    elif isinstance(module, torch.nn.Linear):
        if 'mlp.fc1' in lower_name:
            print(f"[MLP fc1] {name:48s} | out_features: {module.out_features}")

# Hardware-friendly alignment check
def check_alignment(value):
    """Check if value is divisible by 8 or 16 (hardware-friendly for tensor cores)"""
    div_8 = "✓" if value % 8 == 0 else "✗"
    div_16 = "✓" if value % 16 == 0 else "✗"
    return div_8, div_16

print("\n" + "="*80)
print(" HARDWARE ALIGNMENT CHECK (Prunable Layers)")
print(" Tensor cores work best with dimensions divisible by 8 or 16")
print("="*80)

print(f"\n{'Layer Name':<55} | {'Dim':>6} | {'÷8':>3} | {'÷16':>4} | Status")
print("-" * 80)

non_aligned_count = 0
total_prunable = 0

for name, module in x.named_modules():
    lower_name = name.lower()
    dim_to_check = None
    dim_type = ""
    
    if isinstance(module, torch.nn.Conv2d):
        if any(kw in lower_name for kw in ['conv_first', 'conv_after_body', 'conv_before_upsample', 'conv_up', 'upsample']):
            dim_to_check = module.out_channels
            dim_type = "out_ch"
    elif isinstance(module, torch.nn.Linear):
        if 'mlp.fc1' in lower_name:
            dim_to_check = module.out_features
            dim_type = "out_ft"
    
    if dim_to_check is not None:
        total_prunable += 1
        div_8, div_16 = check_alignment(dim_to_check)
        
        if dim_to_check % 8 == 0:
            status = "✓ HW-friendly (÷8)" if dim_to_check % 16 != 0 else "✓ Optimal (÷16)"
        else:
            status = "⚠ NOT aligned"
            non_aligned_count += 1
        
        print(f"{name:<55} | {dim_to_check:>6} | {div_8:>3} | {div_16:>4} | {status}")

print("\n" + "-" * 80)
print(f"Summary: {total_prunable - non_aligned_count}/{total_prunable} prunable layers are hardware-friendly (divisible by 8)")
if non_aligned_count > 0:
    print(f"⚠ Warning: {non_aligned_count} layer(s) have non-aligned dimensions - may reduce GPU efficiency")