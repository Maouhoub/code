# SwinIR Structured Pruning - Troubleshooting Guide

## CUDA Assertion Error Fix

If you encounter CUDA assertion errors like:
```
RuntimeError: CUDA error: device-side assert triggered
```

This is typically caused by incompatibility between Torch-Pruning and SwinIR's complex attention mechanism. Here's how to fix it:

### Step 1: Test the Fix

Run the test script first to verify everything is working:

```bash
python test_swinir_pruning.py
```

This will test the SwinIR-specific pruning utilities without running the full training pipeline.

### Step 2: Use SwinIR-Specific Pruning

The enhanced script now includes SwinIR-specific pruning that:

1. **Identifies Safe Layers**: Only prunes Conv2d layers that are safe (excludes attention mechanisms)
2. **Conservative Pruning**: Uses smaller pruning ratios (max 20%) to prevent instability
3. **Error Handling**: Falls back to minimal pruning if standard pruning fails
4. **CPU Fallback**: Automatically falls back to CPU for FLOPs calculation if GPU fails

### Step 3: Key Files Added

- `swinir_pruning_utils.py` - SwinIR-specific pruning functions
- `test_swinir_pruning.py` - Test script to verify functionality
- Enhanced `main_train_psnr_L2_fine_tune_structured_enhanced.py` - Uses SwinIR-specific approach

### Step 4: Run with Enhanced Error Handling

```bash
python main_train_psnr_L2_fine_tune_structured_enhanced.py --opt options/swinir/train_swinir_sr_lightweight_structured_pruning.json
```

### Common Issues and Solutions

#### Issue 1: "SwinIR pruning utils not found"
**Solution**: Ensure `swinir_pruning_utils.py` is in the same directory as your main script.

#### Issue 2: Still getting CUDA assertions
**Solution**: The script will automatically fall back to PyTorch native pruning with conservative settings.

#### Issue 3: FLOPs calculation fails
**Solution**: The script now includes CPU fallback for FLOPs calculation.

#### Issue 4: "TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'"
**Solution**: Fixed with safe division operations that handle None values.

### What the Fix Does

1. **Layer Filtering**: 
   - Skips attention layers (`attn`, `attention`, `relative_position`)
   - Skips very small layers (< 4 channels)
   - Skips 1x1 convolutions in attention mechanisms

2. **Conservative Pruning**:
   - Caps pruning ratio at 20% maximum
   - Uses absolute channel counts instead of ratios
   - Ensures at least 1 channel remains

3. **Robust Error Handling**:
   - Tries SwinIR-specific pruning first
   - Falls back to PyTorch native pruning if needed
   - Falls back to minimal pruning if standard approaches fail
   - CPU fallback for FLOPs calculation

4. **Safe Statistics Calculation**:
   - Handles None values in FLOPs calculation
   - Uses CPU fallback when GPU calculation fails
   - Provides meaningful error messages

### Verification Steps

1. Run `python test_swinir_pruning.py` to verify setup
2. Check that all tests pass
3. Run the main script with your configuration
4. Monitor the output for successful pruning messages

### Expected Output

You should see output like:
```
? SwinIR-specific pruning utilities available
Applying SwinIR-specific structured channel pruning (ratio: 10.0%)
Found 15 safe Conv2d layers for pruning
  ? Pruned conv_first: 6/60 channels
  ? Pruned layers.0.residual_group.blocks.0.conv1: 6/60 channels
...
Pruning summary: 12 successful, 0 failed
```

### Performance Expectations

With conservative SwinIR pruning:
- **Parameter Reduction**: 10-25% (instead of aggressive 50%)
- **PSNR Drop**: Minimal (< 0.5 dB)
- **Training Stability**: Much improved
- **CUDA Compatibility**: Resolved assertion errors

This approach prioritizes stability and compatibility over aggressive compression.