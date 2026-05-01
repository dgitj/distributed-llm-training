#!/bin/bash
#SBATCH --job-name=mixtral-benchmarks
#SBATCH --partition=gpu_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00        # ~75 min per run × 5 runs = ~6h with buffer
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load devel/cuda/12.1
module load compiler/gnu/12
source activate mixtral-showcase

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=$HOME/mixtral-zero2-showcase
cd $PROJECT_DIR
mkdir -p logs checkpoints profiling/traces

# Prepare dataset once
if [ ! -d "data/tokenized" ]; then
  python data/prepare_dataset.py --output_dir data/tokenized --max_samples 5000 --seq_len 512
fi

# Helper: run one experiment
run_exp() {
  local CONFIG=$1
  local RUN_NAME=$2
  local NUM_GPUS=${3:-4}
  echo ""
  echo "========================================"
  echo "Starting: $RUN_NAME  (config=$CONFIG, gpus=$NUM_GPUS)"
  echo "========================================"
  deepspeed \
    --num_gpus $NUM_GPUS \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train/train.py \
      --config $CONFIG \
      --run_name $RUN_NAME \
      --max_steps 300 \
      --seq_len 512 \
      --profile \
      --wandb_project "mixtral-benchmarks-bwunicluster"
  echo "Finished: $RUN_NAME"
  # Cool-down between runs to avoid NCCL port conflicts
  sleep 30
}

# -----------------------------------------------------------------------
# Experiment A: overlap_comm ON vs OFF  (same ZeRO-2, 4 GPUs)
# What we learn: how much comm latency is hidden by overlapping with backward
# Key metric:    perf/bwd_ms  (lower = better hiding)
# -----------------------------------------------------------------------
run_exp configs/zero2.json            "expA_overlap_ON"   4
run_exp configs/zero2_no_overlap.json "expA_overlap_OFF"  4

# -----------------------------------------------------------------------
# Experiment B: ZeRO-2 vs ZeRO-3  (overlap_comm=True, 4 GPUs)
# What we learn: cost of weight sharding (AllGather before every forward)
# Key metrics:   perf/vram_gb (ZeRO-3 wins) vs perf/tokens_per_sec (ZeRO-2 wins)
# -----------------------------------------------------------------------
run_exp configs/zero2.json "expB_ZeRO2_4gpu" 4
run_exp configs/zero3.json "expB_ZeRO3_4gpu" 4

# -----------------------------------------------------------------------
# Experiment C: scaling efficiency – 2 GPU vs 4 GPU (ZeRO-2)
# What we learn: parallel efficiency = actual_speedup / ideal_speedup
# Key metric:    perf/tokens_per_sec  (should be ~1.7-1.9x for 2→4 GPUs)
# -----------------------------------------------------------------------
run_exp configs/zero2.json "expC_ZeRO2_2gpu" 2
# expC 4gpu result is the same run as expB ZeRO2 4gpu – reuse from W&B

echo ""
echo "All experiments done. Check W&B for results."
echo "Run:  python benchmarks/collect_results.py  to generate the summary table."
