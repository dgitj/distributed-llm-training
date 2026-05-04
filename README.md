# A Scaling & Parallelism Benchmark Framework

This framework is designed to profile and analyze the performance trade-offs of distributed training strategies for Large Language Models. Using **Mixtral-8x7B** (a Mixture-of-Experts model) as a baseline, this project benchmarks various sharding and parallelization dimensions on HPC infrastructure.

## 🎯 Project Objectives
The core intent of this framework is to identify experimentally identify key trade-offs between scaling strategies, e.g., memory efficiency and training throughput. 

**Key focus areas:**
*   **Memory Efficiency:** Quantifying VRAM savings from ZeRO-1, ZeRO-2, and ZeRO-3.
*   **Communication Overhead:** Analyzing the impact of `overlap_comm` and NCCL collective primitives (All-Reduce vs. All-Gather) on step latency.
*   **Hardware Utilization:** Measuring **Model FLOPs Utilization (MFU)** across different GPU counts to identify scaling bottlenecks.

## 🛠 Technical Stack
*   **Model:** `Mixtral-8x7B-v0.1` (approx. 47B parameters).
*   **Training Method:** **QLoRA** (4-bit base weights + 16-bit LoRA adapters) to enable large-scale model training on limited GPU nodes (e.g., 2–4x A100 80GB).
*   **Parallelism Suite:** DeepSpeed ZeRO (Stages 1–3) with integrated PyTorch Profiling.
*   **Dataset:** `WikiText-103` (subset) / `TinyStories` for reproducible, high-throughput training cycles.

## 📊 Extended Benchmark Dimensions
The full framework investigates the synergistic effects of 3D-parallelism and quantization to overcome the memory wall.

### 1. ZeRO Sharding Depth (Data Parallelism)
*   **ZeRO-1 vs. ZeRO-2:** Analyzing redundancy minimization in optimizer states and gradients.
*   **ZeRO-3 (Full Sharding):** Benchmarking VRAM reduction during parameter partitioning against the increase in `All-Gather` operations.

### 2. Intra-Layer Parallelism (Tensor Parallelism - TP)
*   **Attention-Head Splitting:** Measuring latency improvements via parallel computation across NVLink.
*   **Communication vs. Compute:** Identifying the crossover point where tensor communication outweighs compute gains.

### 3. Vertical Partitioning (Pipeline Parallelism - PP)
*   **Pipeline Bubbles:** Benchmarking micro-batch sizes to minimize idle time between pipeline stages.
*   **Inter-Node Bandwidth:** Analyzing throughput during activation transfers across physical server nodes.

### 4. Expert Parallelism (EP) - MoE Specific
*   **Expert Sharding:** Distributing the 8 experts across different GPUs.
*   **All-to-All Latency:** Measuring network load during token routing to specific experts.

### 5. Precision & Quantization Trade-offs
*   **BF16 vs. INT4 (QLoRA):** Quantifying dequantization overhead versus the gain in batch size capacity.

## 🚀 Key Features
*   **Telemetry:** Real-time logging of VRAM, throughput (tokens/sec), and MFU via Weights & Biases.
*   **Low-Level Profiling:** Automated PyTorch Profiler traces to visualize NCCL kernels and "Pipeline Bubbles".
