"""
convert_checkpoint.py
─────────────────────
Convert mistralai/Mixtral-8x7B-v0.1 from HuggingFace format into
Megatron-Core distributed checkpoints for the two benchmark experiments.

Usage
─────
Experiment 1  (TP=2, EP=1, PP=1):
    python convert_checkpoint.py \
        --hf_model  mistralai/Mixtral-8x7B-v0.1 \
        --output    <workspace>/checkpoints/mixtral_tp2_ep1_pp1 \
        --tp 2 --ep 1 --pp 1

Experiment 2  (TP=2, EP=4, expert_tp=4, PP=1):
    python convert_checkpoint.py \
        --hf_model  mistralai/Mixtral-8x7B-v0.1 \
        --output    <workspace>/checkpoints/mixtral_tp2_ep4_etp4_pp1 \
        --tp 2 --ep 4 --expert_tp 4 --pp 1

Requirements
────────────
This script calls Megatron-Core's official HF→Megatron conversion tool
(tools/checkpoint/convert.py in the Megatron-LM repo).
Set MEGATRON_PATH to your Megatron-LM checkout.

    export MEGATRON_PATH=/path/to/Megatron-LM
    pip install transformers sentencepiece

Background
──────────
Megatron stores a separate checkpoint shard per (tp_rank, pp_rank) pair.
Converting at the right TP/PP/EP split is mandatory — you cannot change
tensor-parallel degree at load time without re-converting.

For EP, Megatron-Core places expert weight shards inside the TP shards;
the conversion tool handles expert partitioning automatically when
--num-expert-parallel and --num-experts flags are given.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_model",   type=str,
                   default="mistralai/Mixtral-8x7B-v0.1",
                   help="HF model name or local path")
    p.add_argument("--hf_cache",   type=str, default=None)
    p.add_argument("--output",     type=str, required=True,
                   help="Directory to write the Megatron checkpoint")
    p.add_argument("--tp",         type=int, default=2,
                   help="Tensor parallel size")
    p.add_argument("--ep",         type=int, default=1,
                   help="Expert parallel size")
    p.add_argument("--expert_tp",  type=int, default=1,
                   help="Expert tensor parallel size")
    p.add_argument("--pp",         type=int, default=1,
                   help="Pipeline parallel size")
    p.add_argument("--megatron",   type=str,
                   default=os.environ.get("MEGATRON_PATH", ""),
                   help="Path to Megatron-LM repo root")
    p.add_argument("--dtype",      type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    p.add_argument("--num_experts", type=int, default=8,
                   help="Number of experts in Mixtral (8)")
    p.add_argument("--dry_run",    action="store_true",
                   help="Print the command without running it")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.megatron:
        print("ERROR: set --megatron or export MEGATRON_PATH=/path/to/Megatron-LM")
        sys.exit(1)

    megatron = Path(args.megatron)
    convert_script = megatron / "tools" / "checkpoint" / "convert.py"
    if not convert_script.exists():
        print(f"ERROR: conversion script not found at {convert_script}")
        print("Make sure you are pointing at a recent Megatron-LM checkout")
        print("(commit >= mid-2024 for MoE/EP support)")
        sys.exit(1)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── Build conversion command ──────────────────────────────────────────────
    # Megatron's convert.py uses --model-type GPT for causal LM.
    # For Mixtral we need the MoE-capable path; ensure your Megatron-LM
    # version supports --num-experts and --moe-grouped-gemm.
    cmd = [
        sys.executable, str(convert_script),
        "--model-type",              "GPT",
        "--loader",                  "llama_mistral",   # HF Mixtral loader
        "--saver",                   "megatron",
        "--load-dir",                args.hf_model,
        "--save-dir",                str(out),
        "--tokenizer-model",         args.hf_model,
        "--megatron-path",           str(megatron),
        # Parallelism targets
        "--target-tensor-parallel-size",    str(args.tp),
        "--target-pipeline-parallel-size",  str(args.pp),
        "--target-expert-parallel-size",    str(args.ep),
        # MoE architecture
        "--num-experts",             str(args.num_experts),
        # Dtype
        f"--{args.dtype}",
        # Activation function used by Mixtral FFN experts
        "--swiglu",
        # Grouped GEMM for expert compute (faster on H100)
        "--moe-grouped-gemm",
    ]

    # expert_tp is passed as an env var that Megatron's saver reads
    env = os.environ.copy()
    env["EXPERT_TENSOR_PARALLEL_SIZE"] = str(args.expert_tp)
    if args.hf_cache:
        env["TRANSFORMERS_CACHE"] = args.hf_cache

    print("=" * 70)
    print(f"Converting {args.hf_model}")
    print(f"  TP={args.tp}  EP={args.ep}  expert_TP={args.expert_tp}  PP={args.pp}")
    print(f"  Output: {out}")
    print("=" * 70)
    print("Command:")
    print("  " + " \\\n    ".join(cmd))
    print()

    if args.dry_run:
        print("[dry-run] Not executing.")
        return

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"ERROR: conversion exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nCheckpoint written to: {out}")
    print("Verify with:")
    print(f"  ls -lh {out}/")


if __name__ == "__main__":
    main()
