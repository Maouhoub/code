# SwinIR-Light Structured Pruning Implementation

This implementation provides comprehensive structured pruning for SwinIR-Light models with the goal of making them more efficient while maintaining accuracy.

## Features

### ?? Core Functionality
- **Structured Channel Pruning**: Remove entire channels based on L1-norm importance
- **Torch-Pruning Integration**: Advanced dependency-aware pruning using the torch-pruning library
- **Progressive Pruning**: Gradual pruning with increasing ratios (20%, 30%, 40%)
- **Automatic Fine-tuning**: Re-train the model after each pruning iteration

### ?? Comprehensive Metrics
- **PSNR/SSIM Measurement**: Track image quality before and after pruning
- **FLOP Counting**: Measure computational complexity using fvcore or thop
- **Parameter Counting**: Track model size reduction
- **Inference Time**: Measure actual speed improvements
- **Sparsity Analysis**: Monitor pruning effectiveness

### ?? Success Criteria
- **Accuracy Preservation**: PSNR drop ? 0.2 dB
- **Efficiency Gains**: Significant reduction in parameters, FLOPs, and inference time
- **Quality Metrics**: Minimal SSIM degradation

## Installation

### Required Packages
```bash
pip install torch-pruning fvcore thop
```

Or install from the requirements file:
```bash
pip install -r requirements_pruning.txt
```

## Usage

### Running the Structured Pruning
```bash
python main_train_psnr_L2_fine_tune_structured.py --opt options/train_msrresnet_psnr.json
```

### Testing the Implementation
```bash
python test_pruning.py
```

## Implementation Details

### 1. Pruning Strategy
- **L1-Norm Importance**: Channels with smaller L1-norms are considered less important
- **Structured Pruning**: Remove entire channels rather than individual weights
- **Progressive Approach**: Start with 20% pruning, increase gradually
- **Dependency Awareness**: Use torch-pruning for complex architectures

### 2. Evaluation Pipeline
```python
# Before pruning
original_metrics = evaluate_model(model)

# Apply pruning
pruned_model = apply_structured_pruning(model, ratio=0.3)

# Fine-tune
fine_tune_model(pruned_model, dataset)

# After pruning
final_metrics = evaluate_model(pruned_model)

# Compare results
compare_metrics(original_metrics, final_metrics)
```

### 3. Key Functions

#### Model Evaluation
- `evaluate_model_metrics()`: Compute PSNR/SSIM on test set
- `count_parameters()`: Count total and trainable parameters
- `count_flops()`: Measure computational complexity
- `measure_inference_time()`: Benchmark actual speed

#### Pruning Implementation
- `apply_structured_pruning_torch_pruning()`: Advanced pruning with torch-pruning
- `apply_basic_structured_pruning()`: Fallback PyTorch pruning
- `create_pruning_importance_dict()`: L1-norm based importance scoring

#### Analysis and Reporting
- `print_model_summary()`: Comprehensive model statistics
- Results saved to JSON for detailed analysis
- Visual comparison tables for before/after metrics

## Expected Results

### Efficiency Gains
- **Parameter Reduction**: 20-40% fewer parameters
- **FLOP Reduction**: 20-40% computational savings
- **Speed Improvement**: 1.2-2.0x faster inference
- **Memory Savings**: Proportional to parameter reduction

### Quality Preservation
- **PSNR Drop**: ? 0.2 dB (target threshold)
- **SSIM Preservation**: Minimal degradation
- **Visual Quality**: Imperceptible differences

## Output Structure

### Results Directory
```
results/
??? pruning_results.json          # Detailed metrics comparison
??? iteration_1/                  # Sample images from each iteration
?   ??? sample1_iter1.png
?   ??? ...
??? iteration_2/
??? ...
```

### Console Output
- Real-time pruning progress
- Iteration-by-iteration metrics
- Final comparison table
- Success criteria evaluation

## Advanced Features

### Torch-Pruning Integration
- Dependency graph analysis
- Proper handling of skip connections
- BatchNorm layer adjustment
- Advanced pruning strategies

### Fallback Mechanisms
- Automatic fallback to PyTorch pruning if torch-pruning fails
- Multiple FLOP counting libraries support
- Robust error handling

### Customization Options
- Adjustable pruning ratios
- Configurable success thresholds
- Multiple importance criteria
- Fine-tuning epoch control

## Troubleshooting

### Common Issues
1. **Import Errors**: Install required packages with `pip install torch-pruning fvcore thop`
2. **CUDA Memory**: Reduce batch size or image resolution
3. **Pruning Failures**: Check model architecture compatibility
4. **FLOP Counting**: Ensure input shapes match model expectations

### Performance Tips
- Use GPU acceleration for faster evaluation
- Limit test set size for quicker iterations
- Save intermediate models for recovery
- Monitor memory usage during pruning

## References

- [Torch-Pruning](https://github.com/VainF/Torch-Pruning)
- [SwinIR Paper](https://arxiv.org/abs/2108.10257)
- [Structured Pruning Methods](https://arxiv.org/abs/1608.08710)