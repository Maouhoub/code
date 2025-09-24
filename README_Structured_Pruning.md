# Structured Channel Pruning for SwinIR-Light

This enhanced training script applies structured channel pruning to SwinIR-Light models for image super-resolution while maintaining high image quality.

## ?? Features

- **Structured Channel Pruning**: Uses both Torch-Pruning and PyTorch native structured pruning
- **Progressive Pruning**: Gradual pruning (5-10% per iteration) with fine-tuning after each step
- **Comprehensive Metrics**: Measures FLOPs, parameters, PSNR, SSIM, and inference time
- **Colab-Ready**: Includes all necessary installs and clear documentation
- **Results Comparison**: Detailed before/after comparison table

## ?? Installation (Colab)

```bash
# Install required packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install opencv-python
!pip install scikit-image
!pip install tensorboard
!pip install ptflops

# Install Torch-Pruning for advanced structured pruning
!git clone https://github.com/VainF/Torch-Pruning.git
!cd Torch-Pruning && pip install -e .
```

## ?? Usage

### Basic Usage
```python
# Use the enhanced script with SwinIR configuration
python main_train_psnr_L2_fine_tune_structured_enhanced.py --opt options/train_swinir_sr_structured_pruning.json
```

### Configuration

The script uses the configuration file `options/train_swinir_sr_structured_pruning.json` which includes:

- **Model**: SwinIR-Light architecture optimized for efficiency
- **Pruning**: Progressive structured channel pruning (50% total reduction in 5 steps)
- **Fine-tuning**: 10 epochs of fine-tuning after each pruning step
- **Threshold**: Stop pruning if PSNR drops below 34.45 dB

### Key Configuration Parameters

```json
{
  "netG": {
    "net_type": "swinir",
    "upscale": 2,
    "embed_dim": 60,          // Lightweight configuration
    "depths": [6, 6, 6, 6],   // Reduced depth for efficiency
    "num_heads": [6, 6, 6, 6]
  },
  "fine_tune": {
    "L2_ft_epochs": 10        // Fine-tuning epochs after pruning
  }
}
```

## ?? Expected Results

The script produces a comprehensive comparison table:

```
====================================================================================================
 STRUCTURED CHANNEL PRUNING RESULTS COMPARISON
====================================================================================================
Metric                   Baseline             Pruned               Change               Change %   
----------------------------------------------------------------------------------------------------
Parameters               1.50M                0.75M                -0.75M                  -50.00%
FLOPs                    12.30G               6.15G                -6.15G                  -50.00%
PSNR (dB)                36.2500              35.8000              -0.4500                  -1.24%
SSIM                     0.9200               0.9100               -0.0100                  -1.09%
Inference Time (s)       0.0250               0.0150               -0.0100                 -40.00%
====================================================================================================
```

## ?? Key Features Explained

### 1. **Structured Channel Pruning**
- Uses L1-norm importance metric for channel selection
- Prunes entire channels (structured) vs individual weights (unstructured)
- Maintains model structure for hardware acceleration

### 2. **Progressive Approach**
- Gradual pruning prevents catastrophic performance loss
- Fine-tuning after each step recovers performance
- Configurable pruning steps and ratios

### 3. **Comprehensive Evaluation**
- **FLOPs**: Computational complexity measurement
- **Parameters**: Model size reduction
- **PSNR/SSIM**: Image quality metrics
- **Inference Time**: Speed improvement measurement

### 4. **Fallback Support**
- Automatically falls back to PyTorch native pruning if Torch-Pruning unavailable
- Robust error handling for different environments

## ?? Output Structure

```
superresolution/swinir_sr_structured_pruning/
??? models/
?   ??? [timestamp]_G.pth              # Final pruned model
?   ??? pruning_results.txt            # Detailed results summary
??? images/
?   ??? [image_name]_baseline/         # Baseline results
?   ??? [image_name]_pruned_iter_[N]/  # Results for each pruning iteration
??? train.log                          # Training logs
```

## ?? Customization

### Adjust Pruning Parameters
```python
# In main function
total_pruning_ratio = 0.3      # Target 30% pruning instead of 50%
pruning_steps = 3              # 3 steps instead of 5
target_psnr_threshold = 35.0   # Higher quality threshold
```

### Change Model Architecture
```python
# Use different SwinIR configuration in JSON
"embed_dim": 96,               # Larger model
"depths": [6, 6, 6, 6, 6, 6],  # More layers
```

## ?? Running in Colab

1. **Upload your dataset** to Colab or use mounted Google Drive
2. **Clone the repository** and upload the enhanced script
3. **Run the installation commands** from the script header
4. **Execute the script** with your desired configuration
5. **Download results** including the pruned model and comparison metrics

## ?? Performance Tips

1. **Start with smaller datasets** for faster experimentation
2. **Monitor PSNR threshold** to prevent over-pruning
3. **Adjust fine-tuning epochs** based on dataset size
4. **Use GPU** for faster training and evaluation

## ?? Integration with Existing Code

The enhanced script maintains compatibility with the original KAIR framework:
- Same data loading pipeline
- Compatible model saving/loading
- Consistent logging and evaluation
- Minimal changes to existing workflow

## ?? References

- **Torch-Pruning**: [https://github.com/VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning)
- **SwinIR**: [https://github.com/JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
- **KAIR**: [https://github.com/cszn/KAIR](https://github.com/cszn/KAIR)