import argparse
import os
import torch
import numpy as np
import time
from collections import OrderedDict

# Import local modules
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from models.select_model import define_Model
from data.select_dataset import define_Dataset
from torch.utils.data import DataLoader

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, data_range=255, multichannel=True, channel_axis=2)
    except ImportError:
        # Fallback or simplified calculation if needed, but usually skimage is available
        print("Warning: scikit-image not available for SSIM")
        return 0.0

def evaluate_model_on_dataset(model, dataset_opt, opt):
    """
    Evaluates the model on a specific dataset definition.
    """
    # 1. Create Dataset and Dataloader
    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
    
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_time = 0.0
    idx = 0
    border = opt['scale'] 
    
    # Ensure model is in eval mode
    model_network = model.netG if hasattr(model, 'netG') else model
    model_network.eval()
    
    print(f"Processing {dataset_opt['name']} ({len(test_set)} images)...")
    
    # Warmup
    if idx == 0 and len(test_set) > 0:
         pass 

    with torch.no_grad():
        for test_data in test_loader:
            idx += 1
            image_name_ext = os.path.basename(test_data['L_path'][0])
            
            # 2. Inference
            start = time.time()
            model.feed_data(test_data)
            model.test() 
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()
            avg_time += (end - start)
            
            # 3. Get Visuals
            visuals = model.current_visuals()
            E_img = util.tensor2uint(visuals['E']) # Estimated (Model Output)
            H_img = util.tensor2uint(visuals['H']) # High Res (Ground Truth)
            
            # 4. Calculate Metrics
            # Note: Standard SR benchmark usually calculates PSNR on Y-channel of YCbCr.
            # calculate_psnr in BasicSR/KAIR usually handles RGB/Y conversion or uses raw RGB if specified.
            # Here we follow your training script's approach.
            current_psnr = util.calculate_psnr(E_img, H_img, border=border)
            current_ssim = calculate_ssim(E_img, H_img)
            
            avg_psnr += current_psnr
            avg_ssim += current_ssim
            
    if idx == 0:
        return 0.0, 0.0, 0.0

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_time = avg_time / idx
    
    return avg_psnr, avg_ssim, avg_time


def main(json_path='options/swinir/prod.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # ----------------------------------------
    # 1. Load Options
    # ----------------------------------------
    if not os.path.exists(args.opt):
         print(f"Options file not found: {args.opt}")
         return

    print("Reading options from:", args.opt)
    opt = option.parse(args.opt, is_train=True) # Use True to ensure safe parsing of all fields
    opt['dist'] = False
    opt['rank'] = 0
    
    # ----------------------------------------
    # 2. Find Latest Checkpoint (Auto-resume logic)
    # ----------------------------------------
    print("Looking for pretrained models...")
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    if init_path_G is not None:
        print(f"Found checkpoint: {init_path_G}")
        opt['path']['pretrained_netG'] = init_path_G
    else:
        print(f"No checkpoint found in {opt['path']['models']}. Loading from JSON config: {opt['path']['pretrained_netG']}")

    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # 3. Model Initialization
    # ----------------------------------------
    model = define_Model(opt)
    model.init_train() # This loads the weights defined in opt['path']['pretrained_netG']
    
    # ----------------------------------------
    # 4. Define Test Sets
    # ----------------------------------------
    # Try to load from JSON first

        # Fallback to hardcoded if not in JSON (matches prompt)
    benchmark_datasets = [
        {
        "name": "BSDS100",
        "dataset_type": "sr",
        "dataroot_H": "/content/TEST_SETS/BSDS100/HR",
        "dataroot_L": "/content/TEST_SETS/BSDS100/x2"
        },
        {
        "name": "Set14",
        "dataset_type": "sr",
        "dataroot_H": "/content/TEST_SETS/Set14/HR",
        "dataroot_L": "/content/TEST_SETS/Set14/x2"
        },
        {
        "name": "Set5",
        "dataset_type": "sr",
        "dataroot_H": "/content/TEST_SETS/Set5/HR",
        "dataroot_L": "/content/TEST_SETS/Set5/x2"
        },
        {
        "name": "manga109",
        "dataset_type": "sr",
        "dataroot_H": "/content/TEST_SETS/manga109/HR",
        "dataroot_L": "/content/TEST_SETS/manga109/x2"
        },
        {
        "name": "urban100",
        "dataset_type": "sr",
        "dataroot_H": "/content/TEST_SETS/urban100/HR",
        "dataroot_L": "/content/TEST_SETS/urban100/x2"
        }
    ]

    # ----------------------------------------
    # 5. Run Evaluation Loop
    # ----------------------------------------
    print("\n" + "="*80)
    print(f" BENCHMARK EVALUATION (Scale: x{opt['scale']})")
    print("="*80)
    print(f"{'Dataset':<15} {'PSNR (dB)':<15} {'SSIM':<15} {'Time (s/img)':<15}")
    print("-"*60)
    
    for ds_opt in benchmark_datasets:
        # Validate paths exist before trying to load
        if 'dataroot_H' in ds_opt and not os.path.exists(ds_opt['dataroot_H']):
             # Try to guess or skip
             # In colab environments, paths might not exist locally if running locally
             # Warning only
             pass

        ds_opt['n_channels'] = opt['n_channels']
        ds_opt['scale'] = opt['scale']
        
        try:
            psnr, ssim, inf_time = evaluate_model_on_dataset(model, ds_opt, opt)
            print(f"{ds_opt['name']:<15} {psnr:<15.4f} {ssim:<15.4f} {inf_time:<15.4f}")
        except Exception as e:
            print(f"{ds_opt['name']:<15} [ERROR: {str(e)}]")

    print("="*80)

if __name__ == '__main__':
    main()
