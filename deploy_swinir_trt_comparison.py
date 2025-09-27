#!/usr/bin/env python3
"""SwinIR ONNX -> TensorRT benchmarking pipeline for baseline vs pruned models."""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda  # type: ignore
import pycuda.autoinit  # noqa: F401  # automatically init CUDA context
from tabulate import tabulate

from main_train_psnr_L2_fine_tune_structured_enhanced import (  # type: ignore
    calculate_model_stats,
    remove_pruning_masks,
    calculate_ssim,
)
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_network import define_G
from torch.utils.data import DataLoader
from torch.nn import Module
import utils.utils_image as util

# TensorRT logger (global to reuse between build/runtime)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


@dataclass
class ModelBenchmarkResult:
    name: str
    params: int
    flops: float
    onnx_path: str
    engine_path: str
    onnx_size_bytes: int
    engine_size_bytes: int
    latency_ms: float
    throughput_fps: float
    psnr: Optional[float] = None
    ssim: Optional[float] = None

    @property
    def total_size_bytes(self) -> int:
        return self.onnx_size_bytes + self.engine_size_bytes

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['total_size_bytes'] = self.total_size_bytes
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and pruned SwinIR models via TensorRT deployment.")
    parser.add_argument('--opt', type=str, required=True, help='Path to SwinIR options JSON (same used for training).')
    parser.add_argument('--baseline-weights', type=str, required=True, help='Path to baseline .pth weights (state dict).')
    parser.add_argument('--pruned-weights', type=str, required=True, help='Path to pruned .pth weights (state dict).')
    parser.add_argument('--output-dir', type=str, default='./deployment_artifacts', help='Directory to store ONNX/engine files and logs.')
    parser.add_argument('--patch-size', type=int, default=64, help='Input spatial size (assumes square patches).')
    parser.add_argument('--warmup-iters', type=int, default=30, help='Number of warmup iterations for TensorRT timing.')
    parser.add_argument('--benchmark-iters', type=int, default=200, help='Number of timed iterations for TensorRT timing.')
    parser.add_argument('--workspace-gb', type=float, default=4.0, help='TensorRT builder workspace size in GB.')
    parser.add_argument('--quality-max-images', type=int, default=20, help='Max images for PSNR/SSIM evaluation (0 disables).')
    parser.add_argument('--save-json', action='store_true', help='Save raw metrics as JSON next to comparison table.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for PyTorch export/evaluation (TensorRT requires CUDA).')
    return parser.parse_args()


def load_swinir_network(opt_path: str, weights_path: str, device: torch.device) -> Tuple[Module, Dict]:
    """Instantiate SwinIR network from options JSON and load weights."""
    opt = option.parse(opt_path, is_train=False)
    opt = option.dict_to_nonedict(opt)

    net = define_G(opt)
    net.to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = None
    candidate_keys = ['params', 'params_ema', 'state_dict', 'model', 'net']
    if isinstance(checkpoint, dict):
        for key in candidate_keys:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    if state_dict is None:
        if isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise RuntimeError(f"Unsupported checkpoint format in {weights_path}")

    # Remove potential DataParallel prefixes
    cleaned_state = {}
    for k, v in state_dict.items():
        cleaned_state[k.replace('module.', '')] = v

    missing, unexpected = net.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"Warning: missing keys when loading {weights_path}: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading {weights_path}: {unexpected}")

    net = remove_pruning_masks(net)
    net.eval()
    return net, opt


def export_to_onnx(model: Module, onnx_path: str, input_shape: Tuple[int, int, int, int]) -> None:
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    dummy_input = torch.randn(*input_shape, device=next(model.parameters()).device)
    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=13,
            do_constant_folding=True,
        )
    print(f"Saved ONNX model: {onnx_path}")


def build_trt_engine(onnx_path: str, engine_path: str, workspace_gb: float, enable_fp16: bool = True) -> None:
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(flags) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        with open(onnx_path, 'rb') as f:
            onnx_bytes = f.read()
        if not parser.parse(onnx_bytes):
            print("TensorRT ONNX parsing errors:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        config.max_workspace_size = int(workspace_gb * (1024 ** 3))
        if enable_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_shape = tuple(input_tensor.shape)
        if len(input_shape) != 4:
            raise ValueError(f"Unexpected network input shape {input_shape}, expected NCHW")
        min_shape = tuple(max(dim, 1) for dim in input_shape)
        profile.set_shape(input_tensor.name, min_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"Saved TensorRT engine: {engine_path}")


def profile_trt_engine(engine_path: str, input_shape: Tuple[int, int, int, int],
                       warmup_iters: int, benchmark_iters: int) -> Tuple[float, float]:
    with open(engine_path, 'rb') as f:
        engine_bytes = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")

    context = engine.create_execution_context()
    context.set_binding_shape(0, input_shape)

    stream = cuda.Stream()

    bindings = [0] * engine.num_bindings
    inputs = []
    outputs = []

    for binding_idx in range(engine.num_bindings):
        dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
        shape = context.get_binding_shape(binding_idx)
        size = math.prod(shape)
        host_mem = np.random.random(size).astype(dtype) if engine.binding_is_input(binding_idx) else np.empty(size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings[binding_idx] = int(device_mem)
        if engine.binding_is_input(binding_idx):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    # Warmup
    for _ in range(warmup_iters):
        for host_mem, device_mem in inputs:
            cuda.memcpy_htod_async(device_mem, host_mem, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for host_mem, device_mem in outputs:
            cuda.memcpy_dtoh_async(host_mem, device_mem, stream)
        stream.synchronize()

    latencies = []
    for _ in range(benchmark_iters):
        for host_mem, device_mem in inputs:
            cuda.memcpy_htod_async(device_mem, host_mem, stream)
        start = time.perf_counter()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for host_mem, device_mem in outputs:
            cuda.memcpy_dtoh_async(host_mem, device_mem, stream)
        stream.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    avg_latency_ms = float(np.mean(latencies))
    throughput = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else float('inf')

    for _, device_mem in inputs + outputs:
        device_mem.free()

    return avg_latency_ms, throughput


def compute_model_stats(model: Module, input_shape_hw: Tuple[int, int], device: torch.device) -> Tuple[int, float]:
    stats = calculate_model_stats(model, input_shape=(3, *input_shape_hw), device=str(device))
    params = stats.get('total_params', 0)
    flops = stats.get('flops', 0)
    return params, flops


def maybe_evaluate_quality(model: Module, opt: Dict, device: torch.device, max_images: int) -> Tuple[Optional[float], Optional[float]]:
    if max_images == 0 or 'datasets' not in opt or 'test' not in opt['datasets']:
        return None, None

    dataset_opt = opt['datasets']['test']
    dataset = define_Dataset(dataset_opt)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0
    border = dataset_opt.get('scale', opt.get('scale', 1))

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if 'H' not in batch:
                break
            lr = batch['L'].to(device)
            hr = batch['H'].to(device)
            sr = model(lr)
            sr_img = util.tensor2uint(sr[0].cpu())
            hr_img = util.tensor2uint(hr[0].cpu())
            psnr = util.calculate_psnr(sr_img, hr_img, border=border)
            ssim = calculate_ssim(sr_img, hr_img)
            psnr_sum += psnr
            ssim_sum += ssim
            count += 1
            if 0 < max_images <= count:
                break

    if count == 0:
        return None, None
    return psnr_sum / count, ssim_sum / count


def format_bytes(num_bytes: int) -> str:
    if num_bytes == 0:
        return '0 B'
    sign = '-' if num_bytes < 0 else ''
    num_bytes = abs(num_bytes)
    units = ['B', 'KB', 'MB', 'GB']
    idx = int(min(len(units) - 1, math.floor(math.log(num_bytes, 1024))))
    value = num_bytes / (1024 ** idx)
    return f"{sign}{value:.2f} {units[idx]}"


def format_flops(flops: float) -> str:
    if flops == 0:
        return '0'
    sign = '-' if flops < 0 else ''
    flops = abs(flops)
    units = ['FLOPs', 'KFLOPs', 'MFLOPs', 'GFLOPs', 'TFLOPs']
    idx = int(min(len(units) - 1, math.floor(math.log(flops, 1000))))
    value = flops / (1000 ** idx)
    return f"{sign}{value:.2f} {units[idx]}"


def format_count(num: float) -> str:
    if num == 0:
        return '0'
    sign = '-' if num < 0 else ''
    abs_num = abs(num)
    if abs_num >= 1e6:
        return f"{sign}{abs_num / 1e6:.2f} M"
    if abs_num >= 1e3:
        return f"{sign}{abs_num / 1e3:.2f} K"
    return f"{sign}{abs_num:.0f}"


def summarize_results(baseline: ModelBenchmarkResult, pruned: ModelBenchmarkResult) -> str:
    def compute_change(base_val: float, pruned_val: float) -> Tuple[float, float]:
        delta = pruned_val - base_val
        pct = (delta / base_val * 100.0) if base_val else float('nan')
        return delta, pct

    rows = []
    metrics = [
        ('Parameters', baseline.params, pruned.params, format_count, True),
        ('FLOPs', baseline.flops, pruned.flops, format_flops, False),
        ('ONNX Size', baseline.onnx_size_bytes, pruned.onnx_size_bytes, format_bytes, True),
        ('TensorRT Size', baseline.engine_size_bytes, pruned.engine_size_bytes, format_bytes, True),
        ('Total Deployment Size', baseline.total_size_bytes, pruned.total_size_bytes, format_bytes, True),
        ('Latency (ms)', baseline.latency_ms, pruned.latency_ms, lambda x: f"{x:.2f} ms", True),
        ('Throughput (FPS)', baseline.throughput_fps, pruned.throughput_fps, lambda x: f"{x:.2f}", False),
    ]

    if baseline.psnr is not None and pruned.psnr is not None:
        metrics.append(('PSNR (dB)', baseline.psnr, pruned.psnr, lambda x: f"{x:.3f}", False))
    if baseline.ssim is not None and pruned.ssim is not None:
        metrics.append(('SSIM', baseline.ssim, pruned.ssim, lambda x: f"{x:.4f}", False))

    for name, base_val, pruned_val, fmt_fn, lower_is_better in metrics:
        delta, pct = compute_change(base_val, pruned_val)
        if lower_is_better:
            pct = -pct  # display reduction as positive percentage when pruned is smaller
            delta_display = -delta
        else:
            delta_display = delta
        rows.append([
            name,
            fmt_fn(base_val),
            fmt_fn(pruned_val),
            fmt_fn(delta_display),
            f"{pct:+.2f}%",
        ])

    headers = ['Metric', 'Baseline', 'Pruned', 'Change', 'Change %']
    return tabulate(rows, headers=headers, tablefmt='github')


def benchmark_model(name: str, weights_path: str, opt_path: str, args: argparse.Namespace) -> ModelBenchmarkResult:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model, opt = load_swinir_network(opt_path, weights_path, device)

    params, flops = compute_model_stats(model, (args.patch_size, args.patch_size), device)
    model.to(device)

    dummy_input = torch.randn(1, 3, args.patch_size, args.patch_size, device=device)
    with torch.no_grad():
        _ = model(dummy_input)

    model_dir = os.path.join(args.output_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    export_to_onnx(model, onnx_path, dummy_input.shape)

    engine_path = os.path.join(model_dir, f"{name}.trt")
    build_trt_engine(onnx_path, engine_path, workspace_gb=args.workspace_gb, enable_fp16=True)

    latency_ms, throughput_fps = profile_trt_engine(
        engine_path,
        input_shape=tuple(dummy_input.shape),
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )

    psnr, ssim = maybe_evaluate_quality(model, opt, device, args.quality_max_images)

    return ModelBenchmarkResult(
        name=name,
        params=int(params),
        flops=float(flops),
        onnx_path=onnx_path,
        engine_path=engine_path,
        onnx_size_bytes=os.path.getsize(onnx_path),
        engine_size_bytes=os.path.getsize(engine_path),
        latency_ms=latency_ms,
        throughput_fps=throughput_fps,
        psnr=psnr,
        ssim=ssim,
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for TensorRT benchmarking.")
    if args.device == 'cpu':
        raise ValueError("TensorRT benchmarking requires --device cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    baseline_result = benchmark_model('baseline', args.baseline_weights, args.opt, args)
    pruned_result = benchmark_model('pruned', args.pruned_weights, args.opt, args)

    summary_table = summarize_results(baseline_result, pruned_result)
    print('\n=== SwinIR TensorRT Deployment Comparison ===')
    print(summary_table)

    if args.save_json:
        summary_path = os.path.join(args.output_dir, 'trt_comparison.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'baseline': baseline_result.to_dict(),
                'pruned': pruned_result.to_dict(),
            }, f, indent=2)
        print(f"Saved raw metrics to {summary_path}")


if __name__ == '__main__':
    main()
