import os.path
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
import torch

from utils import utils_logger
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

from main_train_psnr_L2_fine_tune_structured_enhanced import calculate_model_stats, evaluate_model


def build_test_loaders(opt):
    test_opt = opt['datasets'].get('test')
    if not test_opt:
        raise ValueError("The option file must define datasets.test for benchmark evaluation.")

    if isinstance(test_opt, dict):
        test_opt = [test_opt]

    loaders = []
    for dataset_cfg in test_opt:
        test_set = define_Dataset(dataset_cfg)
        test_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False,
            pin_memory=True,
        )
        loaders.append((dataset_cfg['name'], test_loader))
    return loaders


def print_benchmark_summary(stats, benchmark_results):
    print("\n" + "=" * 80)
    print(" FINAL BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Parameters: {stats['total_params']:,}")
    print(f"FLOPs: {stats['flops']:,}")

    for dataset_name, metrics in benchmark_results.items():
        print(
            f"{dataset_name:<12} | "
            f"PSNR: {metrics['psnr']:.4f} dB | "
            f"SSIM: {metrics['ssim']:.4f} | "
            f"Time: {metrics['inference_time']:.4f} s"
        )


def main(json_path='options/swinir/prod.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    parser.add_argument('--max_images', type=int, default=None, help='Override max images per dataset.')
    parser.add_argument('--save_images', action='store_true', help='Save selected output images during evaluation.')
    parser.add_argument('--log_per_image', action='store_true', help='Print per-image evaluation lines.')
    args = parser.parse_args()

    print("Effective Options file used is : ", args.opt)
    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = args.dist

    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    print("iterations : ", init_iter_optimizerG, init_path_optimizerG)

    if init_path_G is not None:
        opt['path']['pretrained_netG'] = init_path_G
    if init_path_E is not None:
        opt['path']['pretrained_netE'] = init_path_E
    if init_path_optimizerG is not None:
        opt['path']['pretrained_optimizerG'] = init_path_optimizerG

    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    if opt['rank'] == 0:
        logger_name = 'benchmark_eval'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))

    opt = option.dict_to_nonedict(opt)

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = define_Model(opt)
    model.init_train()

    test_loaders = build_test_loaders(opt)
    input_shape = (3, 64, 64)
    device = next(model.parameters()).device
    final_network = model.netG if hasattr(model, 'netG') else model
    final_stats = calculate_model_stats(final_network, input_shape, device)

    structured_opt = opt.get('structured_pruning', {}) or {}
    max_images = args.max_images
    if max_images is None:
        max_images = int(structured_opt.get('full_eval_images', 100))

    benchmark_results = {}
    for dataset_name, test_loader in test_loaders:
        psnr, ssim, inference_time = evaluate_model(
            model,
            test_loader,
            opt,
            current_step,
            suffix=f"benchmark_{dataset_name}",
            max_images=max_images,
            save_images=args.save_images,
            log_per_image=args.log_per_image,
        )
        benchmark_results[dataset_name] = {
            'psnr': psnr,
            'ssim': ssim,
            'inference_time': inference_time,
        }

    if opt['rank'] == 0:
        print_benchmark_summary(final_stats, benchmark_results)


if __name__ == '__main__':
    main()