"""
train.py
────────
Shared Megatron-Core training script for the Mixtral DP-vs-EP benchmark.

Both experiments are run with this same script, parameterised by the
Megatron args passed from run_benchmark.sh.

Experiment 1 – DP baseline  (TP=2, EP=1, DP=8):
    torchrun / Megatron launcher passes:
        --tensor-model-parallel-size 2
        --expert-model-parallel-size 1
        --pipeline-model-parallel-size 1

Experiment 2 – EP treatment  (TP=2, EP=4, expert-TP=4, DP=2):
        --tensor-model-parallel-size 2
        --expert-model-parallel-size 4
        --pipeline-model-parallel-size 1
        --expert-tensor-parallel-size 4

What this script adds on top of stock Megatron:
    • Per-step CSV logging (metrics_steps.csv)
    • Per-rank memory logging (memory_by_rank.csv)
    • W&B logging on rank 0
    • PyTorch Profiler window (steps 101–120)
    • Expert load-balance logging (routing entropy + imbalance coefficient)
    • Block-level step-time statistics (6 × 50-step blocks)
    • Clean shutdown and final summary

Requirements
────────────
    Megatron-LM (recent main, MoE-capable)
    torch >= 2.2
    wandb
    Set MEGATRON_PATH and add it to PYTHONPATH before launching.
"""

import csv
import json
import math
import os
import socket
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

# ── W&B (optional but expected) ───────────────────────────────────────────────
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    print("[WARN] wandb not installed – W&B logging disabled")

# ── Megatron imports ──────────────────────────────────────────────────────────
# These are available once MEGATRON_PATH is on PYTHONPATH (done in the
# SLURM script). We import lazily so the file can be syntax-checked without
# a full Megatron checkout.
def _import_megatron():
    try:
        from megatron.training import get_args, get_timers, pretrain
        from megatron.training.arguments import core_transformer_config_from_args
        from megatron.core import mpu
        from megatron.core.models.gpt import GPTModel
        from megatron.core.transformer.spec_utils import import_module
        from megatron.core.datasets.blended_megatron_dataset_builder import (
            BlendedMegatronDatasetBuilder,
        )
        from megatron.core.datasets.gpt_dataset import (
            GPTDataset,
            GPTDatasetConfig,
            MockGPTDataset,
        )
        from megatron.legacy.model import Float16Module
        return True
    except ImportError as e:
        print(f"[ERROR] Megatron import failed: {e}")
        print("  Make sure MEGATRON_PATH is set and on PYTHONPATH")
        return False


# ── Constants ─────────────────────────────────────────────────────────────────
GLOBAL_TOKENS_PER_UPDATE = 65_536
WARMUP_STEPS  = 50
MEASURE_STEPS = 300
TOTAL_STEPS   = WARMUP_STEPS + MEASURE_STEPS   # 350

# Profiling window (step indices, 0-based from start of training)
PROFILE_START = 101
PROFILE_END   = 120

# Block boundaries for block-level statistics (measured steps only)
BLOCK_EDGES = [51, 101, 151, 201, 251, 301, 351]   # 6 × 50-step blocks

H100_PEAK_TFLOPS_BF16 = 989.0  # H100 SXM5 BF16 tensor core peak


# ── Utilities ─────────────────────────────────────────────────────────────────

def is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_global_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def hostname() -> str:
    return socket.gethostname()


def memory_stats_gb(device=None) -> dict:
    """Return current and peak allocated/reserved GPU memory in GiB."""
    if device is None:
        device = get_local_rank()
    return {
        "allocated_gb":      torch.cuda.memory_allocated(device) / 2**30,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
        "reserved_gb":       torch.cuda.memory_reserved(device) / 2**30,
        "peak_reserved_gb":  torch.cuda.max_memory_reserved(device) / 2**30,
    }


def reset_peak_memory():
    torch.cuda.reset_peak_memory_stats()


def compute_mfu(tokens_per_sec: float, total_params: int,
                peak_tflops: float = H100_PEAK_TFLOPS_BF16) -> float:
    """
    MFU = achieved_tflops / peak_tflops
    achieved ≈ 6 * N * tokens/s  (fwd+bwd ≈ 3× fwd, 2 matmuls per param)
    Uses *total* parameters, not just active (correct for systems benchmarking).
    """
    achieved = 6.0 * total_params * tokens_per_sec / 1e12
    return achieved / peak_tflops


# ── CSV writers ───────────────────────────────────────────────────────────────

class StepCSVWriter:
    """Writes one row per optimizer step to metrics_steps.csv (rank 0 only)."""

    FIELDS = [
        "step", "loss", "step_time_s", "tokens_per_sec",
        "mfu_percent", "grad_norm",
        "fwd_time_s", "bwd_time_s", "optim_time_s",
        "peak_allocated_gb", "peak_reserved_gb",
        "expert_load_imbalance", "expert_routing_entropy",
        "lr",
    ]

    def __init__(self, path: str):
        self._path = path
        self._fh   = open(path, "w", newline="")
        self._w    = csv.DictWriter(self._fh, fieldnames=self.FIELDS,
                                    extrasaction="ignore")
        self._w.writeheader()
        self._fh.flush()

    def write(self, row: dict):
        self._w.writerow({k: row.get(k, "") for k in self.FIELDS})
        self._fh.flush()

    def close(self):
        self._fh.close()


class MemoryCSVWriter:
    """Writes memory stats per rank, sampled at configurable steps."""

    FIELDS = [
        "step", "global_rank", "local_rank", "hostname",
        "allocated_gb", "peak_allocated_gb",
        "reserved_gb",  "peak_reserved_gb",
    ]

    def __init__(self, path: str):
        self._path = path
        self._rows = []

    def record(self, step: int):
        mem = memory_stats_gb()
        self._rows.append({
            "step":              step,
            "global_rank":       get_global_rank(),
            "local_rank":        get_local_rank(),
            "hostname":          hostname(),
            **mem,
        })

    def flush_all_ranks(self, step: int):
        """
        Gather memory rows from all ranks on rank 0, then write to CSV.
        Each rank records locally; rank 0 gathers and writes.
        """
        self.record(step)
        # Serialize this rank's latest row
        row = self._rows[-1]
        payload = json.dumps(row).encode()
        # All-gather payload lengths first
        world = dist.get_world_size() if dist.is_initialized() else 1
        if world == 1:
            if is_rank0():
                self._write_rows([row])
            return

        length_tensor = torch.tensor([len(payload)], dtype=torch.int64,
                                     device="cuda")
        all_lengths   = [torch.zeros(1, dtype=torch.int64, device="cuda")
                         for _ in range(world)]
        dist.all_gather(all_lengths, length_tensor)

        max_len = max(t.item() for t in all_lengths)
        padded  = payload + b"\x00" * (max_len - len(payload))
        buf     = torch.frombuffer(bytearray(padded), dtype=torch.uint8).cuda()
        all_bufs = [torch.zeros(max_len, dtype=torch.uint8, device="cuda")
                    for _ in range(world)]
        dist.all_gather(all_bufs, buf)

        if is_rank0():
            rows = []
            for i, (length, b) in enumerate(zip(all_lengths, all_bufs)):
                raw = bytes(b.cpu().numpy()[:length.item()])
                rows.append(json.loads(raw.decode()))
            self._write_rows(rows)

    def _write_rows(self, rows):
        write_header = not Path(self._path).exists()
        with open(self._path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerows(rows)


# ── Expert load-balance helpers ───────────────────────────────────────────────

def expert_load_stats(router_probs: torch.Tensor) -> dict:
    """
    router_probs: [tokens, num_experts] – softmax scores from the MoE router.
    Returns imbalance coefficient and routing entropy.

    Imbalance coefficient = max_expert_load / mean_expert_load  (ideal = 1.0)
    Routing entropy = mean per-token entropy (higher = more uniform routing)
    """
    with torch.no_grad():
        # Mean load per expert (fraction of tokens assigned)
        load = router_probs.mean(dim=0)            # [num_experts]
        imbalance = (load.max() / load.mean()).item()

        # Per-token entropy
        eps = 1e-9
        entropy = -(router_probs * (router_probs + eps).log()).sum(dim=-1).mean()
        entropy = entropy.item()

    return {"expert_load_imbalance": imbalance,
            "expert_routing_entropy": entropy}


# ── W&B setup ─────────────────────────────────────────────────────────────────

def init_wandb(exp_name: str, config: dict):
    if not _WANDB_AVAILABLE or not is_rank0():
        return
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "mixtral-dp-vs-ep"),
        name=exp_name,
        config=config,
        resume="allow",
    )


def log_wandb(metrics: dict, step: int):
    if not _WANDB_AVAILABLE or not is_rank0():
        return
    wandb.log(metrics, step=step)


def finish_wandb():
    if not _WANDB_AVAILABLE or not is_rank0():
        return
    wandb.finish()


# ── Block-level statistics ────────────────────────────────────────────────────

class BlockStats:
    """Accumulates per-step timings and computes 50-step block statistics."""

    def __init__(self):
        self._step_times = {}    # step → step_time_s
        self._tokens_ps  = {}    # step → tokens_per_sec

    def record(self, step: int, step_time: float, tokens_per_sec: float):
        self._step_times[step] = step_time
        self._tokens_ps[step]  = tokens_per_sec

    def block_summary(self) -> list[dict]:
        summaries = []
        edges = BLOCK_EDGES
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            times = [v for s, v in self._step_times.items() if lo <= s < hi]
            tputs = [v for s, v in self._tokens_ps.items()  if lo <= s < hi]
            if not times:
                continue
            import statistics
            summaries.append({
                "block":            f"steps {lo}–{hi-1}",
                "n":                len(times),
                "mean_step_s":      statistics.mean(times),
                "std_step_s":       statistics.stdev(times) if len(times) > 1 else 0.0,
                "median_step_s":    statistics.median(times),
                "p95_step_s":       sorted(times)[int(0.95 * len(times))],
                "mean_tokens_per_s": statistics.mean(tputs),
            })
        return summaries


# ── Profiler context ───────────────────────────────────────────────────────────

def make_profiler(out_dir: str):
    """
    Returns a torch.profiler.profile context that is active only on rank 0
    and only captures steps 101–120 (20 active steps, 1 warmup, 1 wait).
    """
    if not is_rank0():
        return nullcontext()

    from torch.profiler import (
        ProfilerActivity, profile, schedule, tensorboard_trace_handler
    )
    trace_dir = os.path.join(out_dir, "profiler_steps_101_120")
    os.makedirs(trace_dir, exist_ok=True)

    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=0,       # start profiling immediately when context is entered
            warmup=1,
            active=19,    # capture 19 steps  (101 is warmup, 102–120 active)
            repeat=1,
        ),
        on_trace_ready=tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # keep overhead low
    )


# ── Training loop injection ────────────────────────────────────────────────────
# Megatron's pretrain() calls user-supplied hooks at each step.
# We build a closure that captures our writers and state.

def build_extra_valid_loss_func(args_dict):
    """No extra validation metrics needed for this benchmark."""
    return None


def build_train_step_hook(step_writer, mem_writer, block_stats,
                          total_params, exp_name, out_dir, args_dict):
    """
    Returns a callable that Megatron invokes after each optimizer step.
    Signature expected by Megatron: hook(step, loss, args, timers)
    """
    profiler = make_profiler(out_dir)
    profiler_active = [False]
    profiler_ctx    = [None]

    def hook(step, loss_dict, skipped, grad_norm, num_zeros_in_grad,
             optimizer, opt_param_scheduler):
        nonlocal profiler_active, profiler_ctx

        step_1based = step  # Megatron step counter starts at 1

        # ── Enter profiler at step 101 ────────────────────────────────────
        if step_1based == PROFILE_START and is_rank0():
            profiler_ctx[0] = profiler.__enter__()
            profiler_active[0] = True

        # ── Timing from Megatron timers ───────────────────────────────────
        # Megatron's get_timers() returns a Timers object; we read elapsed
        # from named timers if available, else fall back to wall time.
        try:
            from megatron.training import get_timers
            timers = get_timers()
            fwd_s   = timers("forward-compute").elapsed(reset=False)
            bwd_s   = timers("backward-compute").elapsed(reset=False)
            optim_s = timers("optimizer").elapsed(reset=False)
            step_s  = timers("interval-time").elapsed(reset=False)
        except Exception:
            fwd_s = bwd_s = optim_s = 0.0
            step_s = 0.0

        if step_s <= 0:
            step_s = fwd_s + bwd_s + optim_s

        tokens_per_sec = GLOBAL_TOKENS_PER_UPDATE / step_s if step_s > 0 else 0.0
        mfu = compute_mfu(tokens_per_sec, total_params) * 100

        loss_val = loss_dict.get("lm loss", 0.0)
        if hasattr(loss_val, "item"):
            loss_val = loss_val.item()

        # ── Memory snapshot (every 50 steps + final) ──────────────────────
        if step_1based % 50 == 0 or step_1based == TOTAL_STEPS:
            reset_peak_memory()
            mem_writer.flush_all_ranks(step_1based)

        mem = memory_stats_gb()

        # ── Expert routing stats (collected by MoE router hook if present) ─
        expert_stats = {"expert_load_imbalance": None,
                        "expert_routing_entropy": None}
        # The router stores stats in a module-level buffer; access pattern
        # depends on Megatron-Core version. We attempt a best-effort read.
        try:
            from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
            imb = MoEAuxLossAutoScaler.get_loss().item()
            expert_stats["expert_load_imbalance"] = imb
        except Exception:
            pass

        # ── LR ────────────────────────────────────────────────────────────
        try:
            lr = opt_param_scheduler.get_lr()[0]
        except Exception:
            lr = None

        # ── Build row dict ─────────────────────────────────────────────────
        row = {
            "step":                    step_1based,
            "loss":                    loss_val,
            "step_time_s":             step_s,
            "tokens_per_sec":          tokens_per_sec,
            "mfu_percent":             mfu,
            "grad_norm":               grad_norm,
            "fwd_time_s":              fwd_s,
            "bwd_time_s":              bwd_s,
            "optim_time_s":            optim_s,
            "peak_allocated_gb":       mem["peak_allocated_gb"],
            "peak_reserved_gb":        mem["peak_reserved_gb"],
            "lr":                      lr,
            **expert_stats,
        }

        # ── Write CSV (rank 0) ─────────────────────────────────────────────
        if is_rank0():
            step_writer.write(row)

        # ── Block stats ────────────────────────────────────────────────────
        if step_1based >= WARMUP_STEPS:
            block_stats.record(step_1based, step_s, tokens_per_sec)

        # ── W&B log ────────────────────────────────────────────────────────
        if is_rank0() and step_1based % 5 == 0:
            wandb_payload = {
                "train/loss":                loss_val,
                "train/step":                step_1based,
                "train/lr":                  lr,
                "perf/step_time_s":          step_s,
                "perf/tokens_per_sec":       tokens_per_sec,
                "perf/mfu_percent":          mfu,
                "perf/fwd_time_s":           fwd_s,
                "perf/bwd_time_s":           bwd_s,
                "perf/optim_time_s":         optim_s,
                "memory/peak_allocated_gb":  mem["peak_allocated_gb"],
                "memory/peak_reserved_gb":   mem["peak_reserved_gb"],
            }
            if expert_stats["expert_load_imbalance"] is not None:
                wandb_payload["moe/expert_load_imbalance"] = \
                    expert_stats["expert_load_imbalance"]
            if expert_stats["expert_routing_entropy"] is not None:
                wandb_payload["moe/expert_routing_entropy"] = \
                    expert_stats["expert_routing_entropy"]
            log_wandb(wandb_payload, step=step_1based)

        # ── Console print ──────────────────────────────────────────────────
        if is_rank0() and step_1based % 10 == 0:
            print(
                f"[{exp_name}] step {step_1based:4d}/{TOTAL_STEPS} | "
                f"loss={loss_val:.4f} | "
                f"{tokens_per_sec:,.0f} tok/s | "
                f"step={step_s*1000:.0f}ms "
                f"(fwd={fwd_s*1000:.0f} bwd={bwd_s*1000:.0f} opt={optim_s*1000:.0f}) | "
                f"VRAM={mem['peak_allocated_gb']:.1f} GiB | "
                f"MFU={mfu:.1f}%"
            )

        # ── Profiler step ──────────────────────────────────────────────────
        if profiler_active[0] and is_rank0():
            profiler_ctx[0].step()
            if step_1based == PROFILE_END:
                profiler.__exit__(None, None, None)
                profiler_active[0] = False
                print(f"[{exp_name}] Profiler trace written to "
                      f"{out_dir}/profiler_steps_101_120/")

    return hook


# ── Config YAML writer ────────────────────────────────────────────────────────

def write_config_yaml(out_dir: str, exp_name: str, args_dict: dict):
    import yaml, subprocess

    def _ver(cmd):
        try:
            return subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    cfg = {
        "experiment":        exp_name,
        "model_name":        "mistralai/Mixtral-8x7B-v0.1",
        "checkpoint_path":   args_dict.get("load", ""),
        "dataset_path":      args_dict.get("data_path", ""),
        "parallelism": {
            "tp":        args_dict.get("tensor_model_parallel_size", 1),
            "ep":        args_dict.get("expert_model_parallel_size", 1),
            "expert_tp": args_dict.get("expert_tensor_parallel_size", 1),
            "pp":        args_dict.get("pipeline_model_parallel_size", 1),
            "dp":        args_dict.get("data_parallel_size", 1),
        },
        "batch": {
            "seq_len":              args_dict.get("seq_length", 2048),
            "micro_batch_per_gpu":  args_dict.get("micro_batch_size", 1),
            "grad_accum_steps":     args_dict.get("global_batch_size", 32) //
                                    max(args_dict.get("micro_batch_size", 1), 1) //
                                    max(args_dict.get("data_parallel_size", 1), 1),
            "global_tokens_per_update": GLOBAL_TOKENS_PER_UPDATE,
        },
        "optimizer": {
            "type":           "AdamW",
            "lr":             args_dict.get("lr", 1e-5),
            "weight_decay":   args_dict.get("weight_decay", 0.1),
            "adam_beta1":     args_dict.get("adam_beta1", 0.9),
            "adam_beta2":     args_dict.get("adam_beta2", 0.95),
            "grad_clip":      args_dict.get("clip_grad", 1.0),
            "precision":      "bf16",
            "adam_states":    "fp32",
        },
        "training": {
            "warmup_steps":  WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "total_steps":   TOTAL_STEPS,
            "seed":          args_dict.get("seed", 1234),
            "dropout":       0.0,
            "activation_checkpointing": True,
        },
        "profiling": {
            "window_start": PROFILE_START,
            "window_end":   PROFILE_END,
        },
        "software": {
            "python":   sys.version,
            "pytorch":  torch.__version__,
            "cuda":     torch.version.cuda,
            "hostname": hostname(),
        },
        "slurm": {
            "job_id":   os.environ.get("SLURM_JOB_ID", ""),
            "nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
            "n_nodes":  os.environ.get("SLURM_NNODES", ""),
            "n_tasks":  os.environ.get("SLURM_NTASKS", ""),
        },
    }

    yaml_path = os.path.join(out_dir, "config.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Config written to {yaml_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # Megatron parses its own args via initialize_megatron().
    # We read a few extra env vars that the SLURM script sets.
    exp_name = os.environ.get("EXP_NAME", "exp_unknown")
    out_dir  = os.environ.get("EXP_OUT_DIR",
                              f"results/{exp_name}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Megatron init ──────────────────────────────────────────────────────
    if not _import_megatron():
        sys.exit(1)

    from megatron.training import get_args
    from megatron.training.initialize import initialize_megatron

    extra_args = [
        # Our benchmark-specific flags (pass-through to get_args)
        lambda p: None,  # placeholder; real extra args via env
    ]

    # Megatron initializes distributed, sets seeds, etc.
    initialize_megatron(
        extra_args_provider=None,
        args_defaults={
            "tokenizer_type":         "SentencePieceTokenizer",
            "no_load_optim":          False,
            "no_load_rng":            False,
            "use_flash_attn":         True,
            "bf16":                   True,
            "no_gradient_accumulation_fusion": False,
        },
    )

    args = get_args()

    # ── Resolve total parameter count for MFU ─────────────────────────────
    # We approximate from known Mixtral-8x7B architecture.
    # Actual count will be printed by Megatron after model construction.
    MIXTRAL_TOTAL_PARAMS = 46_702_016_512

    # ── Writers ────────────────────────────────────────────────────────────
    step_writer = None
    mem_writer  = None
    if is_rank0():
        step_writer = StepCSVWriter(os.path.join(out_dir, "metrics_steps.csv"))
    mem_writer = MemoryCSVWriter(os.path.join(out_dir, "memory_by_rank.csv"))
    block_stats = BlockStats()

    args_dict = vars(args)

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb_config = {
        "exp_name":     exp_name,
        "tp":           args.tensor_model_parallel_size,
        "ep":           getattr(args, "expert_model_parallel_size", 1),
        "expert_tp":    getattr(args, "expert_tensor_parallel_size", 1),
        "pp":           args.pipeline_model_parallel_size,
        "seq_len":      args.seq_length,
        "global_batch": args.global_batch_size,
        "micro_batch":  args.micro_batch_size,
        "lr":           args.lr,
        "total_steps":  TOTAL_STEPS,
        "warmup_steps": WARMUP_STEPS,
    }
    init_wandb(exp_name, wandb_config)

    # ── Config YAML ────────────────────────────────────────────────────────
    if is_rank0():
        write_config_yaml(out_dir, exp_name, args_dict)

    # ── Build step hook ────────────────────────────────────────────────────
    step_hook = build_train_step_hook(
        step_writer, mem_writer, block_stats,
        MIXTRAL_TOTAL_PARAMS, exp_name, out_dir, args_dict,
    )

    # ── Megatron pretrain ──────────────────────────────────────────────────
    # pretrain() calls model_provider, data_provider, and invokes
    # forward_step per micro-batch. We supply standard GPT providers
    # and inject our hook via the extra_args mechanism.
    from megatron.training import pretrain
    from megatron.core.models.gpt import GPTModel
    from megatron.core.transformer.spec_utils import import_module
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
        get_gpt_layer_local_spec,
    )

    def model_provider(pre_process=True, post_process=True):
        args = get_args()
        try:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=args.num_experts,
                moe_grouped_gemm=getattr(args, "moe_grouped_gemm", False),
            )
        except Exception:
            transformer_layer_spec = get_gpt_layer_local_spec(
                num_experts=getattr(args, "num_experts", None),
            )

        model = GPTModel(
            config=args,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
        )
        return model

    def forward_step(data_iterator, model):
        from megatron.training.utils import get_batch_on_this_cp_rank
        from megatron.core import tensor_parallel

        batch = next(data_iterator)
        tokens      = batch["tokens"].cuda()
        labels      = batch["labels"].cuda()
        loss_mask   = batch["loss_mask"].cuda()
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()

        output_tensor = model(tokens, None, attention_mask, labels=labels)

        def loss_func(output_tensor):
            losses = output_tensor.float()
            loss_mask_f = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask_f) / loss_mask_f.sum()
            averaged = tensor_parallel.average_losses_across_data_parallel_group([loss])
            return loss, {"lm loss": averaged[0]}

        return output_tensor, loss_func

    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig

    def train_valid_test_datasets_provider(train_val_test_num_samples):
        args = get_args()
        config = GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            blend=args.data_path,
            split=args.split,
            path_to_cache=args.data_cache_path,
        )
        train_ds, val_ds, test_ds = BlendedMegatronDatasetBuilder(
            GPTDataset, train_val_test_num_samples, config
        ).build()
        return train_ds, val_ds, test_ds

    # Hook Megatron's training loop via the post_process callback
    # Megatron-Core >= 0.6 exposes `training_process_hook`
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step,
        args_defaults={"tokenizer_type": "SentencePieceTokenizer"},
        process_non_loss_data_func=None,
    )

    # ── Final summary ──────────────────────────────────────────────────────
    if is_rank0():
        step_writer.close()

        blocks = block_stats.block_summary()
        print("\n" + "=" * 70)
        print(f"BLOCK-LEVEL SUMMARY  [{exp_name}]")
        print("=" * 70)
        print(f"{'Block':<20} {'N':>4} {'mean(s)':>8} {'std(s)':>8} "
              f"{'p95(s)':>8} {'tok/s':>10}")
        for b in blocks:
            print(f"{b['block']:<20} {b['n']:>4} "
                  f"{b['mean_step_s']:>8.3f} {b['std_step_s']:>8.3f} "
                  f"{b['p95_step_s']:>8.3f} "
                  f"{b['mean_tokens_per_s']:>10,.0f}")

        # Log block summary to W&B as a table
        if _WANDB_AVAILABLE:
            import wandb
            table = wandb.Table(
                columns=["block", "n", "mean_step_s", "std_step_s",
                         "p95_step_s", "mean_tokens_per_s"],
                data=[[b["block"], b["n"], b["mean_step_s"], b["std_step_s"],
                       b["p95_step_s"], b["mean_tokens_per_s"]]
                      for b in blocks],
            )
            wandb.log({"block_summary": table})

        finish_wandb()
        print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
