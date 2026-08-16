"""Per-image PSNR comparison between two checkpoints, with paired statistics.

Fills Table 9 (paired significance analysis) and Table 12 (worst-case images) of the
revision. PSNR uses the same definition as the rest of the paper:
util.calculate_psnr(E, H, border=scale) in the RGB domain, so the per-image values
average to the dataset figures already reported in Table 4.

The two checkpoints are loaded with independent `with_definition` settings, because
the released baseline is a plain state_dict while a pruned checkpoint is a pickled
module. `find_last_checkpoint` is never called, so neither model can be silently
substituted by whatever happens to sit in the models directory.

Usage:
    python paired_per_image_analysis.py --opt options/swinir/prod.json \
        --original /path/to/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth \
        --pruned   /path/to/18300_G.pth \
        --csv per_image_psnr.csv
"""

import argparse
import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import utils_image as util
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model


# Datasets reported in Table 4, in the order used there. DIV2K is the validation
# partition used inside the pruning loop and is not part of the benchmark suite.
BENCHMARK_ORDER = ['BSDS100', 'Set14', 'Set5', 'Manga109', 'Urban100']


def _canonical(name):
    return name.strip().lower()


def build_test_datasets(opt):
    """Return [(display_name, dataset_opt)] for the benchmark sets in the option file."""
    entries = opt['datasets'].get('test') or []
    if isinstance(entries, dict):
        entries = [entries]

    wanted = {_canonical(n): n for n in BENCHMARK_ORDER}
    found = {}
    for cfg in entries:
        key = _canonical(cfg['name'])
        if key not in wanted:
            continue                      # skips DIV2K and anything unrecognised
        cfg = copy.deepcopy(cfg)
        cfg['n_channels'] = opt['n_channels']
        cfg['scale'] = opt['scale']
        cfg['H_size'] = opt['datasets']['train'].get('H_size', 128)
        cfg['phase'] = 'test'
        found[key] = (wanted[key], cfg)

    ordered = [found[_canonical(n)] for n in BENCHMARK_ORDER if _canonical(n) in found]
    missing = [n for n in BENCHMARK_ORDER if _canonical(n) not in found]
    if missing:
        print('Warning: not defined in the option file, skipping: {}'.format(', '.join(missing)))
    return ordered


def load_model(opt, checkpoint_path, with_definition):
    """Load one checkpoint. opt is mutated locally, never shared between calls."""
    local = copy.deepcopy(opt)
    local['path']['pretrained_netG'] = checkpoint_path
    local['path']['pretrained_netE'] = None
    local['train']['with_definition'] = with_definition
    local['train']['G_optimizer_reuse'] = False
    local['dist'] = False
    local['rank'] = 0
    local = option.dict_to_nonedict(local)

    model = define_Model(local)
    model.init_train()
    network = model.netG if hasattr(model, 'netG') else model
    network.eval()
    return model, local


def per_image_psnr(model, opt, dataset_cfg):
    """Return an ordered dict-like list of (image_name, psnr) for one dataset."""
    test_set = define_Dataset(dataset_cfg)
    loader = DataLoader(test_set, batch_size=1, shuffle=False,
                        num_workers=1, drop_last=False, pin_memory=True)
    border = opt['scale']

    results = []
    with torch.no_grad():
        for test_data in loader:
            name = os.path.splitext(os.path.basename(test_data['L_path'][0]))[0]
            model.feed_data(test_data)
            model.test()
            visuals = model.current_visuals()
            E_img = util.tensor2uint(visuals['E'])
            H_img = util.tensor2uint(visuals['H'])
            results.append((name, util.calculate_psnr(E_img, H_img, border=border)))
    return results


def evaluate_checkpoint(base_opt, checkpoint_path, with_definition, datasets, label):
    print('\n{}\n Evaluating {}: {}\n{}'.format('=' * 70, label, checkpoint_path, '=' * 70))
    model, opt = load_model(base_opt, checkpoint_path, with_definition)

    scores = {}
    for display_name, cfg in datasets:
        values = per_image_psnr(model, opt, cfg)
        scores[display_name] = values
        mean = float(np.mean([v for _, v in values])) if values else float('nan')
        print('  {:<10} {:>4} images   mean PSNR {:.4f} dB'.format(display_name, len(values), mean))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def paired_statistics(deltas):
    """Return n, mean, 95% CI half-width, paired t p-value, Wilcoxon p-value."""
    from scipy import stats

    deltas = np.asarray(deltas, dtype=np.float64)
    n = deltas.size
    mean = float(np.mean(deltas))

    if n < 2:
        return n, mean, float('nan'), float('nan'), float('nan')

    sem = float(stats.sem(deltas))
    half = float(stats.t.ppf(0.975, n - 1) * sem)

    # Paired t-test against a zero mean difference is equivalent to a one-sample
    # t-test on the differences, which is what is computed here.
    t_p = float(stats.ttest_1samp(deltas, 0.0).pvalue)

    try:
        w_p = float(stats.wilcoxon(deltas).pvalue)
    except ValueError:
        # Raised when every difference is exactly zero.
        w_p = float('nan')

    return n, mean, half, t_p, w_p


def format_p(value):
    if not np.isfinite(value):
        return 'n/a'
    if value < 1e-4:
        return '$<10^{-4}$'
    return '{:.4f}'.format(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='options/swinir/prod.json',
                        help='Option file supplying the dataset definitions and netG spec.')
    parser.add_argument('--original', type=str, required=True,
                        help='Checkpoint of the unpruned baseline model.')
    parser.add_argument('--pruned', type=str, required=True,
                        help='Checkpoint of the pruned model to compare against it.')
    parser.add_argument('--original-with-definition', action='store_true',
                        help='Set if the baseline checkpoint is a pickled module rather than a state_dict.')
    parser.add_argument('--pruned-no-definition', action='store_true',
                        help='Set if the pruned checkpoint is a plain state_dict rather than a pickled module.')
    parser.add_argument('--csv', type=str, default='per_image_psnr.csv')
    parser.add_argument('--worst', type=int, default=5,
                        help='Number of worst-case images to report for Table 12.')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = False
    opt['rank'] = 0

    datasets = build_test_datasets(opt)
    if not datasets:
        raise SystemExit('No benchmark datasets found in {}'.format(args.opt))

    original = evaluate_checkpoint(opt, args.original, args.original_with_definition,
                                   datasets, 'ORIGINAL')
    pruned = evaluate_checkpoint(opt, args.pruned, not args.pruned_no_definition,
                                 datasets, 'PRUNED')

    # ------------------------------------------------------------------
    # Pair the two records image by image and write the raw values out.
    # ------------------------------------------------------------------
    rows = []
    for display_name, _ in datasets:
        a, b = original[display_name], pruned[display_name]
        if len(a) != len(b):
            print('  Skipping {}: {} original vs {} pruned images'.format(display_name, len(a), len(b)))
            continue
        for (name_a, psnr_a), (name_b, psnr_b) in zip(a, b):
            if name_a != name_b:
                print('  Skipping {}: image order differs ({} vs {})'.format(display_name, name_a, name_b))
                rows = [r for r in rows if r[0] != display_name]
                break
            rows.append((display_name, name_a, psnr_a, psnr_b, psnr_b - psnr_a))

    with open(args.csv, 'w', encoding='utf-8') as handle:
        handle.write('dataset,image,psnr_original,psnr_pruned,delta_psnr\n')
        for dataset, name, psnr_a, psnr_b, delta in rows:
            handle.write('{},{},{:.6f},{:.6f},{:.6f}\n'.format(dataset, name, psnr_a, psnr_b, delta))
    print('\nPer-image values written to {} ({} images)'.format(args.csv, len(rows)))

    # ------------------------------------------------------------------
    # Table 9: paired per-image statistics, one row per dataset.
    # ------------------------------------------------------------------
    print('\n' + '=' * 70)
    print(' TABLE 9 -- paired per-image analysis')
    print('=' * 70)
    print('{:<10} {:>4} {:>12} {:>22} {:>14} {:>14}'.format(
        'Dataset', 'n', 'Mean dPSNR', '95% CI', 'p (paired t)', 'p (Wilcoxon)'))

    latex_rows = []
    for display_name, _ in datasets:
        deltas = [r[4] for r in rows if r[0] == display_name]
        if not deltas:
            continue
        n, mean, half, t_p, w_p = paired_statistics(deltas)
        ci = '[{:+.4f}, {:+.4f}]'.format(mean - half, mean + half)
        print('{:<10} {:>4} {:>12.4f} {:>22} {:>14} {:>14}'.format(
            display_name, n, mean, ci, format_p(t_p), format_p(w_p)))
        latex_rows.append('{} & {} & ${:+.4f}$ & $[{:+.4f}, {:+.4f}]$ & {} & {} \\\\'.format(
            display_name, n, mean, mean - half, mean + half, format_p(t_p), format_p(w_p)))

    print('\nLaTeX rows for Table 9:')
    for row in latex_rows:
        print(row)

    # ------------------------------------------------------------------
    # Table 12: the images that lose the most quality, across all datasets.
    # ------------------------------------------------------------------
    worst = sorted(rows, key=lambda r: r[4])[:args.worst]
    print('\n' + '=' * 70)
    print(' TABLE 12 -- {} worst-case images'.format(args.worst))
    print('=' * 70)
    print('\nLaTeX rows for Table 12:')
    for dataset, name, psnr_a, psnr_b, delta in worst:
        safe_name = name.replace('_', r'\_')
        print('{} & {} & {:.4f} & {:.4f} & ${:+.4f}$ \\\\'.format(
            safe_name, dataset, psnr_a, psnr_b, delta))


if __name__ == '__main__':
    main()
