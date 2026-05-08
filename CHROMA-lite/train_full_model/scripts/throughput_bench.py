#!/usr/bin/env python3
"""
Throughput Benchmark — Full Model Training on L40S vs A100 80GB NVLink

Measures training throughput with synthetic data to isolate compute
performance from I/O. Benchmarks across batch sizes with optional
gradient accumulation comparison.

Primary question: At the same effective batch size, does the A100 80GB's
higher memory bandwidth translate to meaningfully faster step times?

Secondary question: Does the A100 80GB's ability to run larger physical
batch sizes (no grad accumulation) give a throughput advantage?

Metrics captured per configuration:
    - Samples/second and optimizer steps/second
    - Wall-clock time per optimizer step
    - Time breakdown: forward vs backward vs optimizer.step
    - Peak VRAM usage (torch.cuda.max_memory_allocated)

Usage:
    # Auto-detect GPU and use default batch sizes:
    python train_full_model/scripts/throughput_bench.py

    # Specify GPU type explicitly:
    python train_full_model/scripts/throughput_bench.py --gpu-type l40s
    python train_full_model/scripts/throughput_bench.py --gpu-type a100_80gb

    # Custom batch sizes, quick run:
    python train_full_model/scripts/throughput_bench.py --batch-sizes 1,2,4 --measure-steps 5

    # Disable gradient checkpointing (will OOM at lower BS):
    python train_full_model/scripts/throughput_bench.py --no-grad-checkpoint
"""

import sys
import gc
import json
import argparse
import time
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from train_full_model.scripts.training import log
from train_full_model.src.model import FullModelConfig, FullModel


# ============================================================================
# Constants
# ============================================================================

DEFAULT_BATCH_SIZES_L40S = [1, 2, 4, 8, 12, 16]
DEFAULT_BATCH_SIZES_A100_80GB = [1, 2, 4, 8, 12, 16, 20, 24, 32]
EFFECTIVE_BATCH_TARGET = 32


# ============================================================================
# Synthetic data
# ============================================================================

def generate_synthetic_batch(
    batch_size: int,
    max_text_len: int,
    vocab_size: int,
    structure_dim: int,
    llm_vocab_size: int,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """
    Generate one synthetic training batch on device.

    Returns dict matching FullModel.forward() signature:
        input_ids, attention_mask, cached_base_logits,
        n_steps, structure_matrices, target_tokens
    """
    g = torch.Generator(device='cpu')
    g.manual_seed(seed + batch_size)

    # n_steps per example: uniform in [3, 6]
    n_steps = torch.randint(3, 7, (batch_size,), generator=g, dtype=torch.long)
    total_steps = n_steps.sum().item()

    input_ids = torch.randint(
        1, llm_vocab_size, (batch_size, max_text_len),
        generator=g, dtype=torch.long)
    attention_mask = torch.ones(
        batch_size, max_text_len, dtype=torch.long)
    base_logits = torch.randn(
        total_steps, vocab_size, generator=g, dtype=torch.float32)
    target_tokens = torch.randint(
        0, vocab_size, (total_steps,), generator=g, dtype=torch.long)
    structure_matrices = torch.randn(
        total_steps, structure_dim, generator=g, dtype=torch.float32)

    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'cached_base_logits': base_logits.to(device),
        'n_steps': n_steps.to(device),
        'structure_matrices': structure_matrices.to(device),
        'target_tokens': target_tokens.to(device),
    }


# ============================================================================
# Benchmark dataclasses
# ============================================================================

@dataclasses.dataclass
class BenchConfig:
    """One benchmark configuration to measure."""
    batch_size: int
    grad_accum_steps: int
    label: str


@dataclasses.dataclass
class BenchResult:
    """Results from one benchmark configuration."""
    batch_size: int
    grad_accum_steps: int
    effective_batch: int
    label: str
    status: str

    # Throughput
    samples_per_sec: float
    steps_per_sec: float

    # Timing breakdown (seconds, averaged over measure_steps)
    wall_time_per_step: float
    forward_time: float
    backward_time: float
    optimizer_time: float

    # Memory
    peak_vram_mb: float

    # Raw
    total_wall_time: float
    n_measured_steps: int


# ============================================================================
# Core benchmark
# ============================================================================

def benchmark_config(
    model: FullModel,
    bench_cfg: BenchConfig,
    device: torch.device,
    llm_vocab_size: int,
    warmup_steps: int = 5,
    measure_steps: int = 20,
) -> BenchResult:
    """
    Benchmark one (batch_size, grad_accum_steps) configuration.

    Generates synthetic data, runs warmup + measured optimizer steps,
    and returns timing/memory metrics. Catches CUDA OOM gracefully.
    """
    bs = bench_cfg.batch_size
    accum = bench_cfg.grad_accum_steps
    effective = bs * accum

    def _make_oom_result():
        return BenchResult(
            batch_size=bs, grad_accum_steps=accum,
            effective_batch=effective, label=bench_cfg.label,
            status="OOM", samples_per_sec=0, steps_per_sec=0,
            wall_time_per_step=0, forward_time=0, backward_time=0,
            optimizer_time=0, peak_vram_mb=0, total_wall_time=0,
            n_measured_steps=0,
        )

    try:
        # Generate synthetic batch
        batch = generate_synthetic_batch(
            batch_size=bs,
            max_text_len=model.config.max_text_len,
            vocab_size=model.config.vocab_size,
            structure_dim=200,  # NUM_MATERIALS * MAX_LAYERS = 25 * 8
            llm_vocab_size=llm_vocab_size,
            device=device,
        )

        # Fresh optimizer
        param_groups = [
            {
                'params': [p for p in model.llm.parameters() if p.requires_grad],
                'lr': 7e-4,
                'weight_decay': 0.01,
            },
            {
                'params': list(model.constraint_mlp.parameters()),
                'lr': 7e-4,
                'weight_decay': 0.01,
            },
            {
                'params': list(model.mixing_mlp.parameters()),
                'lr': 7e-4,
                'weight_decay': 0.01,
            },
        ]
        optimizer = AdamW(param_groups)

        # Collect trainable params for grad clipping
        trainable_params = (
            list(model.constraint_mlp.parameters()) +
            list(model.mixing_mlp.parameters()) +
            [p for p in model.llm.parameters() if p.requires_grad]
        )

        # Set model modes — use train() so gradient checkpointing activates
        # (transformers checks self.training before using checkpoint)
        model.constraint_mlp.train()
        model.mixing_mlp.train()
        model.llm.train()

        # ---- Warmup ----
        for _ in range(warmup_steps):
            for _ in range(accum):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        cached_base_logits=batch['cached_base_logits'],
                        n_steps=batch['n_steps'],
                        structure_matrices=batch['structure_matrices'],
                        target_tokens=batch['target_tokens'],
                    )
                    loss = outputs['loss'] / accum
                loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # ---- Measurement ----
        torch.cuda.reset_peak_memory_stats()

        forward_times = []
        backward_times = []
        optimizer_times = []

        t_total_start = time.perf_counter()

        for _ in range(measure_steps):
            step_fwd = 0.0
            step_bwd = 0.0

            for _ in range(accum):
                # Forward
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        cached_base_logits=batch['cached_base_logits'],
                        n_steps=batch['n_steps'],
                        structure_matrices=batch['structure_matrices'],
                        target_tokens=batch['target_tokens'],
                    )
                    loss = outputs['loss'] / accum
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                step_fwd += (t1 - t0)

                # Backward
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                step_bwd += (t1 - t0)

            # Optimizer step
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            forward_times.append(step_fwd)
            backward_times.append(step_bwd)
            optimizer_times.append(t1 - t0)

        torch.cuda.synchronize()
        t_total_end = time.perf_counter()

        # ---- Compute metrics ----
        total_wall = t_total_end - t_total_start
        wall_per_step = total_wall / measure_steps
        samples_per_sec = (effective * measure_steps) / total_wall
        steps_per_sec = measure_steps / total_wall
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        avg_fwd = sum(forward_times) / len(forward_times)
        avg_bwd = sum(backward_times) / len(backward_times)
        avg_opt = sum(optimizer_times) / len(optimizer_times)

        # Cleanup
        del optimizer, batch
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gc.collect()

        return BenchResult(
            batch_size=bs, grad_accum_steps=accum,
            effective_batch=effective, label=bench_cfg.label,
            status="ok",
            samples_per_sec=round(samples_per_sec, 2),
            steps_per_sec=round(steps_per_sec, 4),
            wall_time_per_step=round(wall_per_step, 4),
            forward_time=round(avg_fwd, 4),
            backward_time=round(avg_bwd, 4),
            optimizer_time=round(avg_opt, 4),
            peak_vram_mb=round(peak_vram_mb, 1),
            total_wall_time=round(total_wall, 2),
            n_measured_steps=measure_steps,
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log(f"    OOM at {bench_cfg.label}")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()
            return _make_oom_result()
        raise


# ============================================================================
# GPU info
# ============================================================================

def get_gpu_info() -> dict:
    """Gather GPU metadata for output JSON."""
    if not torch.cuda.is_available():
        return {"name": "N/A", "total_memory_mb": 0}

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_mb": round(props.total_memory / (1024 ** 2), 1),
        "compute_capability": f"{props.major}.{props.minor}",
        "cuda_version": torch.version.cuda or "N/A",
        "pytorch_version": torch.__version__,
    }


def detect_gpu_type() -> str:
    """Auto-detect GPU type from device name."""
    if not torch.cuda.is_available():
        return "unknown"
    name = torch.cuda.get_device_name(0).lower()
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if "l40s" in name:
        return "l40s"
    if "a100" in name and total_mem_gb > 48:
        return "a100_80gb"
    if "a100" in name:
        return "a100_40gb"
    return "unknown"


# ============================================================================
# Main benchmark runner
# ============================================================================

def build_bench_configs(batch_sizes: List[int]) -> List[BenchConfig]:
    """Build list of (batch_size, grad_accum) configs to benchmark."""
    configs = []
    for bs in batch_sizes:
        # Direct: no accumulation
        configs.append(BenchConfig(
            batch_size=bs, grad_accum_steps=1,
            label=f"bs{bs}_accum1"))

        # Accumulated: target effective batch = EFFECTIVE_BATCH_TARGET
        accum = max(1, EFFECTIVE_BATCH_TARGET // bs)
        if accum > 1:
            configs.append(BenchConfig(
                batch_size=bs, grad_accum_steps=accum,
                label=f"bs{bs}_accum{accum}"))

    return configs


def run_benchmark(args) -> dict:
    """Run the full benchmark suite and return results dict."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        log("[ERROR] No CUDA GPU available. Exiting.")
        sys.exit(1)

    # Resolve GPU type and batch sizes
    gpu_type = args.gpu_type
    if gpu_type == 'auto':
        gpu_type = detect_gpu_type()
        log(f"[INFO] Auto-detected GPU type: {gpu_type}")

    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    elif gpu_type == 'a100_80gb':
        batch_sizes = DEFAULT_BATCH_SIZES_A100_80GB
    else:
        batch_sizes = DEFAULT_BATCH_SIZES_L40S

    gpu_info = get_gpu_info()
    log(f"[INFO] GPU: {gpu_info['name']} ({gpu_info['total_memory_mb']:.0f} MB)")
    log(f"[INFO] Batch sizes to test: {batch_sizes}")

    # ================================================================
    # Build model
    # ================================================================

    config = FullModelConfig(
        encoder_name=args.encoder,
        llm_hidden_dim=args.llm_hidden_dim,
        d_model=args.d_model,
        constraint_layers=args.constraint_layers,
        mixing_layers=args.mixing_layers,
        dropout=0.1,
        max_text_len=args.max_text_len,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
        vocab_size=args.vocab_size,
        learning_rate=7e-4,
        batch_size=1,  # placeholder
        epochs=1,
    )

    log(f"\n[INFO] Loading model: {config.encoder_name}")
    model = FullModel(config)
    model.load_llm(device)

    # Enable gradient checkpointing
    grad_ckpt = not args.no_grad_checkpoint
    if grad_ckpt:
        if hasattr(model.llm, 'base_model'):
            model.llm.base_model.model.model.gradient_checkpointing_enable()
            log("[INFO] Gradient checkpointing: ENABLED")
        else:
            log("[WARN] Could not enable gradient checkpointing (no base_model)")
            grad_ckpt = False
    else:
        log("[INFO] Gradient checkpointing: DISABLED")

    model.constraint_mlp.to(device)
    model.mixing_mlp.to(device)

    # Read tokenizer vocab size for synthetic data
    llm_vocab_size = model.tokenizer.vocab_size
    log(f"[INFO] LLM vocab size: {llm_vocab_size}")

    # Parameter counts
    n_lora = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
    n_constraint = sum(p.numel() for p in model.constraint_mlp.parameters())
    n_mixing = sum(p.numel() for p in model.mixing_mlp.parameters())
    n_total = n_lora + n_constraint + n_mixing
    n_frozen = sum(p.numel() for p in model.llm.parameters() if not p.requires_grad)

    log(f"[INFO] Trainable: {n_total:,} (LoRA={n_lora:,}, "
        f"ConstraintMLP={n_constraint:,}, MixingMLP={n_mixing:,})")
    log(f"[INFO] Frozen LLM base: {n_frozen:,}")

    # ================================================================
    # Build configs and run
    # ================================================================

    bench_configs = build_bench_configs(batch_sizes)
    log(f"\n[INFO] Configurations to benchmark: {len(bench_configs)}")
    log(f"[INFO] Warmup steps: {args.warmup_steps}, Measure steps: {args.measure_steps}")
    log(f"\n{'='*70}")

    results = []
    for i, bcfg in enumerate(bench_configs):
        log(f"\n--- [{i+1}/{len(bench_configs)}] {bcfg.label} "
            f"(effective_batch={bcfg.batch_size * bcfg.grad_accum_steps}) ---")

        result = benchmark_config(
            model=model,
            bench_cfg=bcfg,
            device=device,
            llm_vocab_size=llm_vocab_size,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )
        results.append(result)

        if result.status == "ok":
            log(f"    {result.samples_per_sec:.1f} samples/s | "
                f"{result.wall_time_per_step:.3f}s/step | "
                f"fwd={result.forward_time:.3f}s bwd={result.backward_time:.3f}s "
                f"opt={result.optimizer_time:.3f}s | "
                f"VRAM={result.peak_vram_mb:.0f} MB")

    # ================================================================
    # Summary
    # ================================================================

    ok_results = [r for r in results if r.status == "ok"]
    oom_results = [r for r in results if r.status == "OOM"]

    summary = {}
    if ok_results:
        peak = max(ok_results, key=lambda r: r.samples_per_sec)
        max_bs = max(r.batch_size for r in ok_results)
        summary = {
            "max_batch_size_without_oom": max_bs,
            "peak_throughput_samples_per_sec": peak.samples_per_sec,
            "peak_throughput_config": peak.label,
            "oom_configs": [r.label for r in oom_results],
        }

    log(f"\n{'='*70}")
    log(f"BENCHMARK SUMMARY")
    log(f"{'='*70}")
    if ok_results:
        log(f"  Peak throughput: {summary['peak_throughput_samples_per_sec']:.1f} "
            f"samples/s ({summary['peak_throughput_config']})")
        log(f"  Max BS without OOM: {summary['max_batch_size_without_oom']}")
    if oom_results:
        log(f"  OOM configs: {', '.join(r.label for r in oom_results)}")

    # Direct comparison table
    log(f"\n{'Label':<20} {'Status':>6} {'Samp/s':>8} {'s/step':>8} "
        f"{'Fwd':>7} {'Bwd':>7} {'Opt':>7} {'VRAM MB':>9}")
    log(f"{'-'*75}")
    for r in results:
        if r.status == "ok":
            log(f"{r.label:<20} {'ok':>6} {r.samples_per_sec:>8.1f} "
                f"{r.wall_time_per_step:>8.3f} {r.forward_time:>7.3f} "
                f"{r.backward_time:>7.3f} {r.optimizer_time:>7.3f} "
                f"{r.peak_vram_mb:>9.0f}")
        else:
            log(f"{r.label:<20} {'OOM':>6} {'--':>8} {'--':>8} "
                f"{'--':>7} {'--':>7} {'--':>7} {'--':>9}")
    log(f"{'='*70}")

    # ================================================================
    # Compile output
    # ================================================================

    output = {
        "benchmark": "throughput_bench",
        "timestamp": datetime.now().isoformat(),
        "gpu_type": gpu_type,
        "gpu_info": gpu_info,
        "model_config": {
            "encoder_name": config.encoder_name,
            "llm_hidden_dim": config.llm_hidden_dim,
            "d_model": config.d_model,
            "constraint_layers": config.constraint_layers,
            "mixing_layers": config.mixing_layers,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_targets": config.lora_targets,
            "max_text_len": config.max_text_len,
            "vocab_size": config.vocab_size,
            "gradient_checkpointing": grad_ckpt,
        },
        "benchmark_params": {
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "effective_batch_target": EFFECTIVE_BATCH_TARGET,
        },
        "trainable_params": {
            "lora": n_lora,
            "constraint_mlp": n_constraint,
            "mixing_mlp": n_mixing,
            "total": n_total,
            "frozen_llm": n_frozen,
        },
        "results": [dataclasses.asdict(r) for r in results],
        "summary": summary,
    }

    return output


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Throughput benchmark for full model training')

    # GPU
    parser.add_argument('--gpu-type', type=str, default='auto',
        choices=['l40s', 'a100_80gb', 'auto'],
        help='GPU type — determines default batch sizes (default: auto-detect)')
    parser.add_argument('--batch-sizes', type=str, default=None,
        help='Override batch sizes, comma-separated (e.g., "1,2,4,8")')

    # Architecture (scaled-up defaults for 8B benchmark)
    parser.add_argument('--encoder', type=str,
        default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--llm-hidden-dim', type=int, default=4096)
    parser.add_argument('--d-model', type=int, default=1024)
    parser.add_argument('--constraint-layers', type=int, default=4)
    parser.add_argument('--mixing-layers', type=int, default=10)
    parser.add_argument('--max-text-len', type=int, default=756)
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-targets', type=str, default='q_proj,v_proj')
    parser.add_argument('--vocab-size', type=int, default=1002)

    # Benchmark params
    parser.add_argument('--warmup-steps', type=int, default=5,
        help='Warmup optimizer steps before measurement (default: 5)')
    parser.add_argument('--measure-steps', type=int, default=20,
        help='Optimizer steps to measure (default: 20)')
    parser.add_argument('--no-grad-checkpoint', action='store_true',
        help='Disable gradient checkpointing (default: enabled)')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
        help='Output directory (default: train_full_model/outputs/bench/)')

    return parser.parse_args()


def main():
    args = parse_args()

    output = run_benchmark(args)

    # Write JSON results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _repo_root / 'train_full_model' / 'outputs' / 'bench'

    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_type = output['gpu_type']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'bench_{gpu_type}_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    log(f"\n[INFO] Results saved to {output_file}")


if __name__ == '__main__':
    main()
