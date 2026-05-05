#!/usr/bin/env bash
# run_experiments.sh
# ──────────────────
# SLURM job script for the Mixtral DP-vs-EP benchmark on bwUniCluster 3.0.
#
# Allocation: 4 nodes × 4 H100 SXM5 GPUs = 16 GPUs total
# Both experiments run sequentially in a single allocation.
#
# One-time setup before submitting:
#   1. Edit the USER CONFIGURATION block below
#   2. Run `wandb login` on the login node (stores key in ~/.netrc)
#   3. sbatch run_experiments.sh

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=mixtral-dp-vs-ep
#SBATCH --partition=gpu_h100          # adjust to your bwUniCluster partition
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4           # one task per GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_benchmark.log
#SBATCH --error=logs/%j_benchmark.err
#SBATCH --exclusive                   # no node sharing; deterministic timing

# ── Fail fast ─────────────────────────────────────────────────────────────────
set -euo pipefail

# ══ USER CONFIGURATION — edit these ══════════════════════════════════════════
WORKSPACE=/scratch/$USER/distributed-llm-training   # large files on scratch
CODE_DIR=~/projects/distributed-llm-training        # code stays in home
MEGATRON_PATH=~/megatron-lm
HF_HOME=/scratch/$USER/hf_cache
WANDB_PROJECT=mixtral-dp-vs-ep
CONDA_ENV=mixtral-benchmark
# ═════════════════════════════════════════════════════════════════════════════

CKPT_DIR="${WORKSPACE}/checkpoints"
DATA_PREFIX="${WORKSPACE}/datasets/fineweb_edu_mixtral_100M/fineweb_edu_mixtral_100M_text_document"
RESULTS_DIR="${WORKSPACE}/results"
LOG_DIR="${CODE_DIR}/logs"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

# ── Software environment ──────────────────────────────────────────────────────
module purge
module load devel/cuda/12.1
module load compiler/gnu/13.2
module load mpi/openmpi/4.1

conda activate "${CONDA_ENV}"

export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}"
export HF_HOME
export WANDB_PROJECT
# W&B API key is read automatically from ~/.netrc (set once via `wandb login`)

# ── Distributed setup ─────────────────────────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_ADDR

NNODES="${SLURM_NNODES}"
GPUS_PER_NODE=4
WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))   # 16

# ── NCCL tuning for InfiniBand (bwUniCluster 3.0) ────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=ib0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ── Common Megatron model flags (Mixtral 8x7B architecture) ──────────────────
COMMON_MODEL_ARGS=(
    # Architecture
    --num-layers 32
    --hidden-size 4096
    --num-attention-heads 32
    --num-key-value-heads 8           # GQA
    --ffn-hidden-size 14336
    --max-position-embeddings 32768
    --rotary-base 1000000
    --swiglu
    --normalization RMSNorm
    --disable-bias-linear
    --num-experts 8
    --moe-router-topk 2
    --moe-grouped-gemm                # fused expert GEMM on H100
    --pipeline-model-parallel-size 1
    # Sequence
    --seq-length 2048
    # Precision
    --bf16
    # Tokenizer
    --tokenizer-type SentencePieceTokenizer
    --tokenizer-model "${HF_HOME}/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/latest/tokenizer.model"
    # Dataset
    --data-path "${DATA_PREFIX}"
    --split 990,10,0
    # Optimizer
    --optimizer adam
    --lr 1e-5
    --min-lr 1e-6
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --use-distributed-optimizer     # ZeRO-2 style: shard grad + opt states
    --overlap-grad-reduce           # overlap grad reduce with bwd
    --overlap-param-gather          # overlap param all-gather
    # LR schedule
    --lr-decay-style cosine
    --lr-warmup-iters 50
    --train-iters 350
    # Activation checkpointing
    --recompute-activations
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    # Attention
    --use-flash-attn
    # Logging & output
    --log-interval 10
    --no-save-optim                  # no checkpoint saving during benchmark
    --no-save-rng
    --eval-iters 0                   # no evaluation during benchmark
    --eval-interval 999999
    # Reproducibility
    --seed 1234
    --data-cache-path "${WORKSPACE}/.data_cache"
)

# ── Launch helper ─────────────────────────────────────────────────────────────
launch_experiment() {
    local exp_name="$1"
    local out_dir="$2"
    local ckpt_subdir="$3"
    shift 3
    local extra_args=("$@")

    local master_port
    master_port=$(( 29500 + RANDOM % 1000 ))

    mkdir -p "${out_dir}"

    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Launching: ${exp_name}"
    echo "  Output:    ${out_dir}"
    echo "  Port:      ${master_port}"
    echo "════════════════════════════════════════════════════════════════════"

    EXP_NAME="${exp_name}" \
    EXP_OUT_DIR="${out_dir}" \
    srun \
        --nodes="${NNODES}" \
        --ntasks-per-node="${GPUS_PER_NODE}" \
        --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
        bash -c "
            torchrun \
                --nproc_per_node=${GPUS_PER_NODE} \
                --nnodes=${NNODES} \
                --node_rank=\${SLURM_NODEID} \
                --master_addr=${MASTER_ADDR} \
                --master_port=${master_port} \
                ${CODE_DIR}/train.py \
                    ${COMMON_MODEL_ARGS[*]} \
                    ${extra_args[*]} \
                    --load ${CKPT_DIR}/${ckpt_subdir} \
                2>&1 | tee ${out_dir}/train.log
        "

    echo "  Done: ${exp_name}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — DP baseline  (TP=2, EP=1, DP=8, PP=1)
#
# Batch math:
#   global_tokens_per_update = 65,536
#   seq_len = 2048  →  global_batch_size = 32 sequences
#   DP=8, micro_batch=1  →  grad_accum = 32 / (8 × 1) = 4
# ─────────────────────────────────────────────────────────────────────────────
launch_experiment \
    "exp1_tp2_ep1_dp8" \
    "${RESULTS_DIR}/exp1_tp2_ep1_dp8" \
    "tp2_ep1_pp1" \
    --tensor-model-parallel-size   2 \
    --expert-model-parallel-size   1 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size             1 \
    --global-batch-size            32

sleep 30

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — EP treatment  (TP=2, EP=4, expert_TP=4, DP=2, PP=1)
#
# Batch math:
#   global_tokens_per_update = 65,536
#   seq_len = 2048  →  global_batch_size = 32 sequences
#   DP=2, micro_batch=1  →  grad_accum = 32 / (2 × 1) = 16
# ─────────────────────────────────────────────────────────────────────────────
launch_experiment \
    "exp2_tp2_ep4_etp4_dp2" \
    "${RESULTS_DIR}/exp2_tp2_ep4_etp4_dp2" \
    "tp2_ep4_etp4_pp1" \
    --tensor-model-parallel-size   2 \
    --expert-model-parallel-size   4 \
    --pipeline-model-parallel-size 1 \
    --expert-tensor-parallel-size  4 \
    --micro-batch-size             1 \
    --global-batch-size            32 \
    --moe-token-dispatcher-type    alltoall

# ─────────────────────────────────────────────────────────────────────────────
# Collect results
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Collecting results..."
echo "════════════════════════════════════════════════════════════════════"

python "${CODE_DIR}/collect_results.py" \
    --exp1_dir  "${RESULTS_DIR}/exp1_tp2_ep1_dp8" \
    --exp2_dir  "${RESULTS_DIR}/exp2_tp2_ep4_etp4_dp2" \
    --exp1_name "exp1_tp2_ep1_dp8" \
    --exp2_name "exp2_tp2_ep4_etp4_dp2" \
    --out_dir   "${RESULTS_DIR}" \
    --wandb_project "${WANDB_PROJECT}"

# ─────────────────────────────────────────────────────────────────────────────
# Archive results to home directory
# Scratch is not backed up and files are purged after inactivity.
# We copy only the lightweight outputs (CSVs, logs, configs) — not checkpoints
# or profiler traces, which are too large for home quota.
# ─────────────────────────────────────────────────────────────────────────────
ARCHIVE_DIR=~/results/mixtral-dp-vs-ep/${SLURM_JOB_ID}
mkdir -p "${ARCHIVE_DIR}"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Archiving results to home: ${ARCHIVE_DIR}"
echo "════════════════════════════════════════════════════════════════════"

rsync -av \
    --exclude="ckpt/" \
    --exclude="profiler_steps_101_120/" \
    "${RESULTS_DIR}/" "${ARCHIVE_DIR}/"

# Copy the SLURM log files too
cp "${LOG_DIR}/${SLURM_JOB_ID}_benchmark.log" "${ARCHIVE_DIR}/" 2>/dev/null || true
cp "${LOG_DIR}/${SLURM_JOB_ID}_benchmark.err" "${ARCHIVE_DIR}/" 2>/dev/null || true

echo ""
echo "  Archived files:"
find "${ARCHIVE_DIR}" -type f | sort | sed 's/^/    /'
echo ""
echo "  Scratch (large, will be purged): ${RESULTS_DIR}"
echo "  Home archive (backed up)        : ${ARCHIVE_DIR}"
echo ""
echo "  Profiler traces and checkpoints remain on scratch only."
echo "  Copy manually before scratch purge if needed."
