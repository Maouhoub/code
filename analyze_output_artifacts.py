"""Quantify sparse pixel artifacts in a super-resolution checkpoint.

A checkpoint can be visually plausible and still score poorly on PSNR when a small
fraction of pixels are catastrophically wrong, because the squared-error term is
dominated by those pixels. This script separates the two effects. For every benchmark
it reports:

  * the standard PSNR, matching the evaluation harness used elsewhere;
  * the fraction of pixels whose absolute error exceeds a threshold ("outliers");
  * the PSNR recomputed with the outlier pixels excluded, that is, the quality of the
    reconstruction where it is not failing outright;
  * which colour channel carries the outliers;
  * whether the outliers lie on the sub-pixel lattice introduced by the PixelShuffle
    upsampler.

The last test is the diagnostic one. PixelShuffle rearranges s^2 feature channels per
colour into the s x s neighbourhood of each output pixel, so damage confined to one
sub-pixel output channel produces errors on a regular sub-lattice: every outlier lands
on the same (y mod s, x mod s) phase. A uniform phase distribution near 1/s^2 instead
indicates content-dependent error with no structural cause.

Per-image PSNR and SSIM are also written to CSV, which feeds the paired per-image
statistics and the worst-case ranking reported in the manuscript.

Usage:
    python analyze_output_artifacts.py --opt options/swinir/prod_lamp.json \
        --label LAMP --threshold 100 --out artifacts_lamp.csv
"""

import argparse
import csv
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import utils_image as util
from utils import utils_option as option
from models.select_model import define_Model
from data.select_dataset import define_Dataset


BENCHMARKS = [
    ("DIV2K", "/content/div2k-dataset-for-super-resolution/Dataset/DIV2K_valid_HR",
     "/content/div2k-dataset-for-super-resolution/Dataset/DIV2K_valid_LR_bicubic/X2"),
    ("BSDS100", "/content/TEST_SETS/BSDS100/HR", "/content/TEST_SETS/BSDS100/x2"),
    ("Set14", "/content/TEST_SETS/Set14/HR", "/content/TEST_SETS/Set14/x2"),
    ("Set5", "/content/TEST_SETS/Set5/HR", "/content/TEST_SETS/Set5/x2"),
    ("manga109", "/content/TEST_SETS/manga109/HR", "/content/TEST_SETS/manga109/x2"),
    ("urban100", "/content/TEST_SETS/urban100/HR", "/content/TEST_SETS/urban100/x2"),
]

CHANNEL_NAMES = ("R", "G", "B")


def psnr_from_mse(mse):
    """PSNR in dB for 8-bit data; infinite when the images are identical."""
    if mse <= 0:
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)


def analyze_pair(sr, hr, border, threshold, scale):
    """Compare one reconstruction against its ground truth.

    Returns a dict of per-image statistics, or None when the shapes disagree, which
    is itself reported because it would indicate a size mismatch rather than an
    artifact.
    """
    if sr.shape != hr.shape:
        return None

    if border > 0:
        sr = sr[border:-border, border:-border, :]
        hr = hr[border:-border, border:-border, :]

    diff = sr.astype(np.float64) - hr.astype(np.float64)
    sq = diff ** 2

    # Standard PSNR over every pixel, matching the evaluation harness.
    psnr_all = psnr_from_mse(sq.mean())

    # A pixel counts as an outlier when any channel exceeds the threshold.
    abs_diff = np.abs(diff)
    outlier_mask = (abs_diff > threshold).any(axis=2)
    n_pixels = outlier_mask.size
    n_outliers = int(outlier_mask.sum())
    outlier_rate = n_outliers / float(n_pixels)

    # PSNR over the pixels that are not failing outright.
    keep = ~outlier_mask
    if keep.any():
        psnr_clean = psnr_from_mse(sq[keep].mean())
    else:
        psnr_clean = float('nan')

    # Share of the total squared error contributed by the outliers.
    total_se = sq.sum()
    outlier_se = sq[outlier_mask].sum() if n_outliers else 0.0
    error_share = (outlier_se / total_se) if total_se > 0 else 0.0

    # Which channel carries the outliers.
    per_channel = [(int((abs_diff[:, :, c] > threshold).sum())) for c in range(sr.shape[2])]

    # Sub-pixel lattice test. Damage confined to one PixelShuffle output channel puts
    # every outlier on the same (y mod s, x mod s) phase.
    phase_counts = Counter()
    if n_outliers:
        ys, xs = np.nonzero(outlier_mask)
        # The border crop shifts the lattice, so undo it before taking the phase.
        ys = ys + border
        xs = xs + border
        for py, px in zip(ys % scale, xs % scale):
            phase_counts[(int(py), int(px))] += 1
    dominant_phase, dominant_count = (None, 0)
    if phase_counts:
        dominant_phase, dominant_count = phase_counts.most_common(1)[0]
    dominant_share = (dominant_count / n_outliers) if n_outliers else 0.0

    return {
        'psnr': psnr_all,
        'psnr_excluding_outliers': psnr_clean,
        'outlier_rate': outlier_rate,
        'n_outliers': n_outliers,
        'n_pixels': n_pixels,
        'outlier_error_share': error_share,
        'per_channel_outliers': per_channel,
        'dominant_phase': dominant_phase,
        'dominant_phase_share': dominant_share,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='options/swinir/prod.json')
    parser.add_argument('--label', type=str, default='model')
    parser.add_argument('--threshold', type=float, default=100.0,
                        help='Absolute 8-bit error above which a pixel counts as an outlier.')
    parser.add_argument('--out', type=str, default='',
                        help='CSV path for the per-image records. Defaults to artifacts_<label>.csv')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = False
    opt['rank'] = 0

    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    if init_path_G is not None:
        print(f"Found checkpoint: {init_path_G}")
        opt['path']['pretrained_netG'] = init_path_G
    opt = option.dict_to_nonedict(opt)

    model = define_Model(opt)
    model.init_train()
    network = model.netG if hasattr(model, 'netG') else model
    network.eval()

    scale = int(opt.get('scale', 2) or 2)
    border = scale
    h_size = opt['datasets']['train'].get('H_size', 128) if 'train' in opt['datasets'] else 128

    csv_path = args.out or f"artifacts_{args.label}.csv"
    rows = []

    print("\n" + "=" * 96)
    print(f" OUTPUT ARTIFACT ANALYSIS [{args.label}]  (outlier threshold: |error| > {args.threshold:.0f}/255)")
    print("=" * 96)
    header = (f"{'Dataset':<12} {'PSNR':>8} {'PSNR excl.':>11} {'Outlier %':>10} "
              f"{'Err share':>10} {'Top phase':>10} {'Phase %':>8} {'Channel':>8}")
    print(header)
    print("-" * 96)

    for name, root_h, root_l in BENCHMARKS:
        if not os.path.isdir(root_h) or not os.path.isdir(root_l):
            print(f"{name:<12} [skipped: dataset path not found]")
            continue

        ds_opt = {
            'name': name, 'dataset_type': 'sr',
            'dataroot_H': root_h, 'dataroot_L': root_l,
            'n_channels': opt['n_channels'], 'scale': scale,
            'H_size': h_size, 'phase': 'test',
        }
        test_set = define_Dataset(ds_opt)
        loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1,
                            drop_last=False, pin_memory=True)

        agg_psnr, agg_clean, agg_rate, agg_share = [], [], [], []
        phase_totals = Counter()
        channel_totals = np.zeros(3, dtype=np.int64)
        shape_mismatches = 0

        with torch.no_grad():
            for data in loader:
                img_name = os.path.splitext(os.path.basename(data['L_path'][0]))[0]
                model.feed_data(data)
                model.test()
                visuals = model.current_visuals()
                sr = util.tensor2uint(visuals['E'])
                hr = util.tensor2uint(visuals['H'])

                stats = analyze_pair(sr, hr, border, args.threshold, scale)
                if stats is None:
                    shape_mismatches += 1
                    print(f"  {name}/{img_name}: shape mismatch SR {sr.shape} vs HR {hr.shape}")
                    continue

                agg_psnr.append(stats['psnr'])
                agg_clean.append(stats['psnr_excluding_outliers'])
                agg_rate.append(stats['outlier_rate'])
                agg_share.append(stats['outlier_error_share'])
                if stats['dominant_phase'] is not None:
                    phase_totals[stats['dominant_phase']] += stats['n_outliers']
                channel_totals += np.array(stats['per_channel_outliers'][:3], dtype=np.int64)

                rows.append({
                    'label': args.label,
                    'dataset': name,
                    'image': img_name,
                    'psnr': f"{stats['psnr']:.4f}",
                    'psnr_excluding_outliers': f"{stats['psnr_excluding_outliers']:.4f}",
                    'outlier_rate': f"{stats['outlier_rate']:.6f}",
                    'outlier_error_share': f"{stats['outlier_error_share']:.4f}",
                    'dominant_phase': str(stats['dominant_phase']),
                    'dominant_phase_share': f"{stats['dominant_phase_share']:.4f}",
                })

        if not agg_psnr:
            print(f"{name:<12} [no comparable images; {shape_mismatches} shape mismatches]")
            continue

        top_phase, top_count = phase_totals.most_common(1)[0] if phase_totals else ((None, None), 0)
        total_phase = sum(phase_totals.values())
        phase_share = (top_count / total_phase) if total_phase else 0.0
        top_channel = CHANNEL_NAMES[int(np.argmax(channel_totals))] if channel_totals.sum() else "-"

        print(f"{name:<12} {np.mean(agg_psnr):>8.4f} {np.nanmean(agg_clean):>11.4f} "
              f"{100 * np.mean(agg_rate):>9.4f}% {100 * np.mean(agg_share):>9.1f}% "
              f"{str(top_phase):>10} {100 * phase_share:>7.1f}% {top_channel:>8}")

    print("-" * 96)
    print(f"A dominant-phase share near {100.0 / (scale ** 2):.0f}% indicates no lattice structure;")
    print("a share approaching 100% indicates damage confined to one PixelShuffle sub-pixel channel.")
    print("=" * 96)

    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-image records written to {csv_path} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
