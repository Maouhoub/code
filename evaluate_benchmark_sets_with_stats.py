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

# FLOPs counting (prefer ptflops; fall back to local implementation)
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except Exception:
    PTFLOPS_AVAILABLE = False

try:
    from utils.utils_modelsummary import get_model_flops
    LOCAL_FLOPS_AVAILABLE = True
except Exception:
    LOCAL_FLOPS_AVAILABLE = False


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def calculate_model_stats(model, input_shape=(3, 64, 64)):
    """Calculate parameters and FLOPs with robust fallback handling."""
    model.eval()

    total_params, trainable_params = count_parameters(model)
    stats = {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'flops': 0,
    }

    try:
        if PTFLOPS_AVAILABLE:
            try:
                # ptflops can fail on some CUDA ops; mirror training script behavior
                flops, _params = get_model_complexity_info(
                    model,
                    input_shape,
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False,
                )
                stats['flops'] = int(flops) if flops is not None else 0
            except Exception as cuda_e:
                print(f"GPU FLOPs calculation failed: {cuda_e}")
                try:
                    print("Attempting CPU FLOPs calculation...")
                    original_device = next(model.parameters()).device
                    model_cpu = model.cpu()
                    flops, _params = get_model_complexity_info(
                        model_cpu,
                        input_shape,
                        as_strings=False,
                        print_per_layer_stat=False,
                        verbose=False,
                    )
                    stats['flops'] = int(flops) if flops is not None else 0
                    model.to(original_device)
                except Exception as cpu_e:
                    print(f"CPU FLOPs calculation also failed: {cpu_e}")
                    stats['flops'] = 0
        elif LOCAL_FLOPS_AVAILABLE:
            flops = get_model_flops(model, input_shape, print_per_layer_stat=False)
            stats['flops'] = int(flops) if flops is not None else 0
        else:
            stats['flops'] = 0
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs: {e}")
        stats['flops'] = 0

    if stats['flops'] is None:
        stats['flops'] = 0

    return stats


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, data_range=255, multichannel=True, channel_axis=2)
    except ImportError:
        # Fallback or simplified calculation if needed, but usually skimage is available
        print("Warning: scikit-image not available for SSIM")
        return 0.0


def _sync():
    """Block until all queued CUDA work has finished, so timings are not truncated."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_model_on_dataset(model, dataset_opt, opt, repeats=1, warmup=3):
    """
    Evaluates the model on a specific dataset definition.

    PSNR/SSIM are deterministic for a fixed checkpoint, so they are computed once
    on the first timed pass. Per-image inference time is measured over `repeats`
    independent passes so that a mean and standard deviation can be reported.

    The timed region covers feed_data + test, matching the original measurement
    definition. A warmup pass is run first and is excluded from all statistics, so
    CUDA context setup and cuDNN autotuning do not inflate the first pass.

    Returns (psnr, ssim, mean_time, std_time, per_repeat_times).
    """
    # 1. Create Dataset and Dataloader (built once and reused across repeats)
    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

    border = opt['scale']

    # Ensure model is in eval mode
    model_network = model.netG if hasattr(model, 'netG') else model
    model_network.eval()

    if len(test_set) == 0:
        return 0.0, 0.0, 0.0, 0.0, []

    print(f"Processing {dataset_opt['name']} ({len(test_set)} images, {repeats} timed pass(es))...")

    avg_psnr = 0.0
    avg_ssim = 0.0
    per_repeat_times = []

    with torch.no_grad():
        # Warmup: excluded from timing, absorbs CUDA init and cuDNN autotuning cost.
        if warmup > 0:
            for w_idx, test_data in enumerate(test_loader):
                if w_idx >= warmup:
                    break
                model.feed_data(test_data)
                model.test()
            _sync()

        for repeat in range(repeats):
            total_time = 0.0
            idx = 0
            collect_metrics = (repeat == 0)

            for test_data in test_loader:
                idx += 1

                # 2. Inference (timed)
                _sync()
                start = time.perf_counter()
                model.feed_data(test_data)
                model.test()
                _sync()
                end = time.perf_counter()
                total_time += (end - start)

                # 3. Metrics, computed once only (identical on every repeat)
                if collect_metrics:
                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])  # Estimated (Model Output)
                    H_img = util.tensor2uint(visuals['H'])  # High Res (Ground Truth)

                    avg_psnr += util.calculate_psnr(E_img, H_img, border=border)
                    avg_ssim += calculate_ssim(E_img, H_img)

            if idx == 0:
                return 0.0, 0.0, 0.0, 0.0, []

            if collect_metrics:
                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx

            repeat_time = total_time / idx
            per_repeat_times.append(repeat_time)
            print(f"  pass {repeat + 1}/{repeats}: {repeat_time:.4f} s/img")

    mean_time = float(np.mean(per_repeat_times))
    # Sample standard deviation (ddof=1); undefined for a single pass, reported as 0.
    std_time = float(np.std(per_repeat_times, ddof=1)) if len(per_repeat_times) > 1 else 0.0

    return avg_psnr, avg_ssim, mean_time, std_time, per_repeat_times



def main(json_path='options/swinir/prod.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=1,
                        help='Number of timed passes per dataset. Use 5 to report mean +/- std.')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Untimed warmup images per dataset, excluded from all statistics.')
    parser.add_argument('--label', type=str, default='',
                        help='Label printed in the final summary, e.g. Taylor / LAMP / Original.')
    parser.add_argument('--include-div2k', action='store_true',
                        help='Also evaluate the DIV2K validation partition, i.e. the split the '
                             'pruning loop uses for its termination check. Off by default because '
                             'these are 2K images and slow to evaluate.')
    args = parser.parse_args()

    # ----------------------------------------
    # 1. Load Options
    # ----------------------------------------
    if not os.path.exists(args.opt):
        print(f"Options file not found: {args.opt}")
        return

    print("Reading options from:", args.opt)
    opt = option.parse(args.opt, is_train=True)  # Use True to ensure safe parsing of all fields
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
        print(
            f"No checkpoint found in {opt['path']['models']}. Loading from JSON config: {opt['path']['pretrained_netG']}"
        )

    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # 3. Model Initialization
    # ----------------------------------------
    model = define_Model(opt)
    model.init_train()  # This loads the weights defined in opt['path']['pretrained_netG']

    # ----------------------------------------
    # 3.1 Model Params/FLOPs (added)
    # ----------------------------------------
    netG = model.netG if hasattr(model, 'netG') else model
    # Use a representative LR input shape derived from training H_size and scale.
    try:
        h_size = opt['datasets']['train'].get('H_size', 128) if 'train' in opt['datasets'] else 128
    except Exception:
        h_size = 128
    scale = int(opt.get('scale', 2) or 2)
    l_size = max(1, int(h_size // scale))
    input_shape = (3, l_size, l_size)

    stats = calculate_model_stats(netG, input_shape=input_shape)
    print("\n" + "=" * 80)
    print(" MODEL STATS")
    print("=" * 80)
    print(f"Input shape (for FLOPs): {input_shape}")
    print(f"Parameters (total)     : {stats['total_params']:,}")
    print(f"Parameters (trainable) : {stats['trainable_params']:,}")
    if stats['flops'] > 0:
        print(f"FLOPs (approx)         : {stats['flops']:,}")
    else:
        print("FLOPs (approx)         : N/A (ptflops/local flops not available)")
    print("=" * 80)

    # ----------------------------------------
    # 4. Define Test Sets
    # ----------------------------------------
    # Fallback to hardcoded list
    benchmark_datasets = [
        {
            "name": "BSDS100",
            "dataset_type": "sr",
            "dataroot_H": "/content/TEST_SETS/BSDS100/HR",
            "dataroot_L": "/content/TEST_SETS/BSDS100/x2",
        },
        {
            "name": "Set14",
            "dataset_type": "sr",
            "dataroot_H": "/content/TEST_SETS/Set14/HR",
            "dataroot_L": "/content/TEST_SETS/Set14/x2",
        },
        {
            "name": "Set5",
            "dataset_type": "sr",
            "dataroot_H": "/content/TEST_SETS/Set5/HR",
            "dataroot_L": "/content/TEST_SETS/Set5/x2",
        },
        {
            "name": "manga109",
            "dataset_type": "sr",
            "dataroot_H": "/content/TEST_SETS/manga109/HR",
            "dataroot_L": "/content/TEST_SETS/manga109/x2",
        },
        {
            "name": "urban100",
            "dataset_type": "sr",
            "dataroot_H": "/content/TEST_SETS/urban100/HR",
            "dataroot_L": "/content/TEST_SETS/urban100/x2",
        },
    ]

    if args.include_div2k:
        # Same partition the pruning loop evaluates for its termination check, so the
        # benchmark result can be compared directly against the value in the training log.
        benchmark_datasets.insert(0, {
            "name": "DIV2K",
            "dataset_type": "sr",
            "dataroot_H": "/content/div2k-dataset-for-super-resolution/Dataset/DIV2K_valid_HR",
            "dataroot_L": "/content/div2k-dataset-for-super-resolution/Dataset/DIV2K_valid_LR_bicubic/X2",
        })

    # ----------------------------------------
    # 5. Run Evaluation Loop
    # ----------------------------------------
    print("\n" + "=" * 80)
    print(f" BENCHMARK EVALUATION (Scale: x{opt['scale']})")
    print("=" * 80)
    print(f"Timed passes per dataset: {args.repeats} (warmup: {args.warmup} images, excluded)")
    print(f"{'Dataset':<15} {'PSNR (dB)':<15} {'SSIM':<15} {'Time (s/img)':<15}")
    print("-" * 60)

    collected = []

    for ds_opt in benchmark_datasets:
        # Validate paths exist before trying to load
        print("Evaluating on dataset", ds_opt)
        if 'dataroot_H' in ds_opt and not os.path.exists(ds_opt['dataroot_H']):
            print("Passing !")
            pass

        ds_opt['n_channels'] = opt['n_channels']
        ds_opt['scale'] = opt['scale']
        # Fix for KeyError: 'H_size' - Required by DatasetSR init
        ds_opt['H_size'] = opt['datasets']['train'].get('H_size', 128) if 'train' in opt['datasets'] else 128
        ds_opt['phase'] = 'test'

        try:
            psnr, ssim, inf_time, std_time, per_repeat = evaluate_model_on_dataset(
                model, ds_opt, opt, repeats=args.repeats, warmup=args.warmup
            )
            print(f"{ds_opt['name']:<15} {psnr:<15.4f} {ssim:<15.4f} {inf_time:<15.4f}")
            collected.append({
                'name': ds_opt['name'],
                'psnr': psnr,
                'ssim': ssim,
                'time': inf_time,
                'std': std_time,
                'per_repeat': per_repeat,
            })
        except Exception as e:
            print(f"{ds_opt['name']:<15} [ERROR: {str(e)}]")

    print("=" * 80)

    # ----------------------------------------
    # 6. Final Summary (paste-ready)
    # ----------------------------------------
    label = args.label or 'Model'
    print("\n" + "=" * 80)
    print(f" FINAL BENCHMARK SUMMARY [{label}]")
    print("=" * 80)
    print(f"Parameters: {stats['total_params']:,}")
    print(f"FLOPs: {stats['flops']:,}")
    print(f"Timed passes: {args.repeats} (warmup {args.warmup} images/dataset, excluded)")
    print()
    print("| Dataset  | PSNR (dB) | SSIM   | Time (s/img)      |")
    print("|----------|-----------|--------|-------------------|")
    for r in collected:
        time_cell = f"{r['time']:.4f} +/- {r['std']:.4f}"
        print(f"| {r['name']:<8} | {r['psnr']:<9.4f} | {r['ssim']:<6.4f} | {time_cell:<17} |")
    print()
    print("Per-pass times (s/img):")
    for r in collected:
        print(f"  {r['name']:<10} " + ", ".join(f"{t:.4f}" for t in r['per_repeat']))
    print("=" * 80)


if __name__ == '__main__':
    main()
