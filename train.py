"""
train.py – Mixtral-8x7B fine-tuning with DeepSpeed ZeRO

Demonstrates:
  • ZeRO Stage-2/3 optimizer/gradient sharding across N GPUs
  • QLoRA (4-bit base weights + BF16 LoRA adapters) so the 87 GB model
    actually fits on a 2–4 × A100 80 GB node
  • PyTorch Profiler for inter-GPU communication tracing
  • W&B logging of VRAM, throughput, and MFU

Experiments (pass different --config and --run_name):
  Exp A – overlap_comm on vs off:
    deepspeed --num_gpus 4 train/train.py --config configs/zero2.json          --run_name "zero2_overlap_on"
    deepspeed --num_gpus 4 train/train.py --config configs/zero2_no_overlap.json --run_name "zero2_overlap_off"

  Exp B – ZeRO-2 vs ZeRO-3 throughput:
    deepspeed --num_gpus 4 train/train.py --config configs/zero2.json  --run_name "zero2_4gpu"
    deepspeed --num_gpus 4 train/train.py --config configs/zero3.json  --run_name "zero3_4gpu"

  Exp C – scaling efficiency (2 vs 4 GPUs, same ZeRO-2):
    deepspeed --num_gpus 2 train/train.py --config configs/zero2.json  --run_name "zero2_2gpu"
    deepspeed --num_gpus 4 train/train.py --config configs/zero2.json  --run_name "zero2_4gpu"

Usage via SLURM:
    sbatch slurm/run_experiments.sh
"""

import argparse
import math
import os
import time

import deepspeed
import torch
import wandb
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    default_data_collator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--dataset_path", type=str, default="data/tokenized")
    parser.add_argument("--config", type=str, default="configs/zero2.json",
                        help="DeepSpeed config JSON")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--profile", action="store_true",
                        help="Run PyTorch Profiler for the first 15 steps")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name, e.g. 'zero2_overlap_on'. "
                             "Defaults to the config filename stem.")
    parser.add_argument("--wandb_project", type=str, default="mixtral-zero2-showcase")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Set automatically by DeepSpeed launcher")
    # DeepSpeed adds its own args; we parse only known ones to avoid conflicts
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def is_main() -> bool:
    return get_rank() == 0


def compute_mfu(tokens_per_sec: float, model_param_count: int, gpu_tflops: float) -> float:
    """
    Model FLOPs Utilization (MFU) – fraction of peak hardware FLOPs actually used.
    Approximation: ~6 * N * tokens/s  (forward + backward ≈ 3x forward, 2 passes)
    Ref: PaLM paper (Chowdhery et al. 2022)
    """
    achieved_tflops = 6 * model_param_count * tokens_per_sec / 1e12
    return achieved_tflops / gpu_tflops


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(args):
    """
    Load Mixtral in 4-bit (QLoRA) to fit on fewer GPUs, then attach LoRA adapters.
    The base weights stay frozen in INT4; only the small LoRA matrices are trained in BF16.

    VRAM breakdown (per GPU, 4 GPUs):
      Base model (4-bit):  ~24 GB  (87 GB / 4 GPUs, roughly)
      LoRA adapters BF16:  ~0.5 GB
      Activations:         ~2–4 GB
      ZeRO-2 optimizer:    sharded across GPUs  ← key benefit
    """
    print(f"[Rank {get_rank()}] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,   # nested quantization → saves ~0.4 GB
        bnb_4bit_quant_type="nf4",        # NormalFloat4 is best for LLM weights
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"[Rank {get_rank()}] Loading model {args.model_name} in 4-bit ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": args.local_rank},  # each rank gets exactly its own GPU
        trust_remote_code=True,
    )

    # Prepare for QLoRA: casts LayerNorm to BF16, enables gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA targets the Q/K/V/O projections in every attention layer
    # and the gate/up/down projections in every MoE expert FFN
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "w1", "w2", "w3",                          # MoE expert FFN (Mixtral naming)
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    if is_main():
        model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # DeepSpeed initializes the distributed process group
    deepspeed.init_distributed()
    world_size = torch.distributed.get_world_size()
    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", 0))
    args.local_rank = local_rank
    torch.cuda.set_device(local_rank)

    # Derive a clean run name from the config filename if not explicitly set
    import json as _json
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    run_name = args.run_name or f"{config_stem}_{world_size}gpu"

    # Read zero stage from the actual config file for accurate logging
    with open(args.config) as f:
        ds_cfg = _json.load(f)
    zero_stage = ds_cfg.get("zero_optimization", {}).get("stage", 0)
    overlap_comm = ds_cfg.get("zero_optimization", {}).get("overlap_comm", False)

    if is_main():
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.model_name,
                "zero_stage": zero_stage,
                "overlap_comm": overlap_comm,
                "lora_rank": args.lora_rank,
                "max_steps": args.max_steps,
                "world_size": world_size,
                "seq_len": args.seq_len,
                "config_file": args.config,
            },
        )

    # -----------------------------------------------------------------------
    # Dataset & DataLoader
    # -----------------------------------------------------------------------
    print(f"[Rank {get_rank()}] Loading dataset from {args.dataset_path} ...")
    dataset = load_from_disk(args.dataset_path)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=get_rank(),
        shuffle=True,
        drop_last=True,
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=2,          # micro-batch per GPU; accumulation in ds_config
        collate_fn=default_data_collator,
        num_workers=4,
        pin_memory=True,
    )

    # -----------------------------------------------------------------------
    # Model + DeepSpeed engine
    # -----------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(args)

    # DeepSpeed wraps the model, optimizer, and LR scheduler in one engine.
    # ZeRO-2 shards optimizer states AND gradients across all ranks.
    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=args.config,
    )

    # -----------------------------------------------------------------------
    # Optional: PyTorch Profiler (first 15 steps)
    # Produces a Chrome Trace JSON → committed to profiling/traces/
    # -----------------------------------------------------------------------
    profiler_ctx = None
    if args.profile and is_main():
        os.makedirs("profiling/traces", exist_ok=True)
        profiler_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=2, active=10, repeat=1),
            on_trace_ready=tensorboard_trace_handler("profiling/traces"),
            record_shapes=True,
            with_stack=False,
        )
        profiler_ctx.__enter__()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    engine.train()
    step = 0
    data_iter = iter(dataloader)

    # A100 80GB peak BF16 TFLOPs (used for MFU calculation)
    A100_PEAK_TFLOPS = 312.0
    # Trainable parameter count (LoRA only, not 4-bit base)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[Rank {get_rank()}] Starting training for {args.max_steps} steps ...")

    t0 = time.time()
    tokens_seen = 0

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(engine.device)
        attention_mask = batch["attention_mask"].to(engine.device)

        # Causal LM: labels = input_ids shifted by 1 (HuggingFace handles shift internally)
        labels = input_ids.clone()
        # Mask padding tokens in the loss
        labels[attention_mask == 0] = -100

        # --- Forward pass (pure compute) ---
        torch.cuda.synchronize()
        t_fwd_start = time.time()
        outputs = engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        torch.cuda.synchronize()
        t_fwd_ms = (time.time() - t_fwd_start) * 1000

        # --- Backward + ZeRO comm (AllReduce / AllGather) ---
        # With overlap_comm=True:  AllReduce runs alongside backward → bwd_ms ≈ pure compute
        # With overlap_comm=False: AllReduce is sequential → bwd_ms = compute + exposed latency
        # Comparing bwd_ms across Exp A configs directly shows the overlap benefit.
        torch.cuda.synchronize()
        t_bwd_start = time.time()
        engine.backward(loss)
        engine.step()
        torch.cuda.synchronize()
        t_bwd_ms = (time.time() - t_bwd_start) * 1000

        step_time = (t_fwd_ms + t_bwd_ms) / 1000
        batch_tokens = input_ids.numel() * world_size
        tokens_seen += batch_tokens
        tokens_per_sec = batch_tokens / step_time

        if profiler_ctx is not None:
            profiler_ctx.step()

        # Logging on main rank every 10 steps
        if is_main() and step % 10 == 0:
            elapsed = time.time() - t0
            mfu = compute_mfu(tokens_per_sec, trainable_params, A100_PEAK_TFLOPS)
            vram_gb = torch.cuda.max_memory_allocated(local_rank) / 1e9

            print(
                f"Step {step:4d}/{args.max_steps} | "
                f"loss={loss.item():.4f} | "
                f"{tokens_per_sec:,.0f} tok/s | "
                f"fwd={t_fwd_ms:.0f}ms bwd={t_bwd_ms:.0f}ms | "
                f"VRAM={vram_gb:.1f} GB | "
                f"MFU={mfu*100:.1f}%"
            )

            wandb.log({
                "train/loss": loss.item(),
                "perf/tokens_per_sec": tokens_per_sec,
                "perf/mfu_percent": mfu * 100,
                "perf/vram_gb": vram_gb,
                "perf/fwd_ms": t_fwd_ms,
                "perf/bwd_ms": t_bwd_ms,      # Exp A: overlap on vs off
                "perf/step_ms": t_fwd_ms + t_bwd_ms,
                "perf/elapsed_sec": elapsed,
                "train/step": step,
            })

        step += 1

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    if profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)

    if is_main():
        total_time = time.time() - t0
        print(f"\nDone! {args.max_steps} steps in {total_time/60:.1f} min")
        print(f"Total tokens processed: {tokens_seen:,}")
        wandb.finish()

    # Save LoRA adapters (small, only a few hundred MB)
    if is_main():
        engine.module.save_pretrained("checkpoints/lora_adapters")
        tokenizer.save_pretrained("checkpoints/lora_adapters")
        print("LoRA adapters saved to checkpoints/lora_adapters")


if __name__ == "__main__":
    main()
