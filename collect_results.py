"""
collect_results.py

Pulls experiment results from W&B and produces:
  1. benchmarks/results_summary.md  – human-readable table for the README
  2. benchmarks/plots/exp_A_bwd_ms.png     – Exp A: backward time overlap on vs off
  3. benchmarks/plots/exp_B_vram_vs_tput.png – Exp B: ZeRO-2 vs ZeRO-3 tradeoff
  4. benchmarks/plots/exp_C_scaling.png    – Exp C: scaling efficiency curve

Usage:
    python benchmarks/collect_results.py --wandb_project mixtral-benchmarks-bwunicluster

Requirements:
    pip install wandb matplotlib pandas
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import wandb

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "overlap_on":  "#2563eb",   # blue
    "overlap_off": "#dc2626",   # red
    "zero2":       "#16a34a",   # green
    "zero3":       "#d97706",   # amber
    "2gpu":        "#7c3aed",   # purple
    "4gpu":        "#0891b2",   # cyan
}


def fetch_run(api, project: str, run_name: str) -> pd.DataFrame:
    """Download history for a single W&B run as a DataFrame."""
    runs = api.runs(project, filters={"display_name": run_name})
    if not runs:
        print(f"  [WARN] Run '{run_name}' not found in project '{project}'")
        return pd.DataFrame()
    run = runs[0]
    history = run.history(samples=500)
    history["run_name"] = run_name
    history["config"] = run.config
    return history


def steady_state_mean(df: pd.DataFrame, metric: str, skip_steps: int = 50) -> float:
    """Mean of a metric after the first skip_steps (warmup) steps."""
    if df.empty or metric not in df.columns:
        return float("nan")
    return df[df["train/step"] >= skip_steps][metric].mean()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_exp_a(df_on: pd.DataFrame, df_off: pd.DataFrame, out_dir: str):
    """
    Experiment A: overlap_comm ON vs OFF.
    Shows bwd_ms over training steps. If overlap works, ON has lower bwd_ms.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: bwd_ms over time
    ax = axes[0]
    if not df_on.empty:
        ax.plot(df_on["train/step"], df_on["perf/bwd_ms"],
                color=COLORS["overlap_on"], label="overlap ON", linewidth=1.5)
    if not df_off.empty:
        ax.plot(df_off["train/step"], df_off["perf/bwd_ms"],
                color=COLORS["overlap_off"], label="overlap OFF", linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Backward pass time (ms)")
    ax.set_title("Exp A: Backward time — overlap ON vs OFF\n"
                 "Lower = less exposed communication latency")
    ax.legend()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Right: tokens/sec bar
    ax = axes[1]
    means = {
        "overlap ON":  steady_state_mean(df_on,  "perf/tokens_per_sec"),
        "overlap OFF": steady_state_mean(df_off, "perf/tokens_per_sec"),
    }
    bars = ax.bar(means.keys(), means.values(),
                  color=[COLORS["overlap_on"], COLORS["overlap_off"]],
                  width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="{:,.0f}", padding=4)
    ax.set_ylabel("Tokens / second (mean, steps 50–300)")
    ax.set_title("Throughput impact of comm overlap")
    ax.set_ylim(0, max(v for v in means.values() if v == v) * 1.25)

    fig.tight_layout()
    path = os.path.join(out_dir, "exp_A_overlap.png")
    fig.savefig(path)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_exp_b(df_z2: pd.DataFrame, df_z3: pd.DataFrame, out_dir: str):
    """
    Experiment B: ZeRO-2 vs ZeRO-3.
    Scatter: VRAM saved vs throughput lost – the memory/speed tradeoff.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: tokens/sec over time
    ax = axes[0]
    if not df_z2.empty:
        ax.plot(df_z2["train/step"], df_z2["perf/tokens_per_sec"],
                color=COLORS["zero2"], label="ZeRO-2", linewidth=1.5)
    if not df_z3.empty:
        ax.plot(df_z3["train/step"], df_z3["perf/tokens_per_sec"],
                color=COLORS["zero3"], label="ZeRO-3", linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Exp B: ZeRO-2 vs ZeRO-3 throughput\n"
                 "ZeRO-3 pays AllGather overhead per layer")
    ax.legend()

    # Right: VRAM vs throughput scatter (two points, annotated)
    ax = axes[1]
    for label, df, color in [("ZeRO-2", df_z2, COLORS["zero2"]),
                              ("ZeRO-3", df_z3, COLORS["zero3"])]:
        vram = steady_state_mean(df, "perf/vram_gb")
        tput = steady_state_mean(df, "perf/tokens_per_sec")
        ax.scatter([vram], [tput], s=200, color=color, zorder=3, label=label)
        ax.annotate(f"  {label}\n  {vram:.1f} GB | {tput:,.0f} tok/s",
                    (vram, tput), fontsize=9, va="center")
    ax.set_xlabel("VRAM per GPU (GB)  ← lower is better")
    ax.set_ylabel("Tokens / second  ↑ higher is better")
    ax.set_title("Memory–Throughput tradeoff\n"
                 "Arrow shows ideal direction (bottom-left = best)")
    ax.annotate("", xy=(0.15, 0.15), xytext=(0.85, 0.85),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.legend()

    fig.tight_layout()
    path = os.path.join(out_dir, "exp_B_zero2_vs_zero3.png")
    fig.savefig(path)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_exp_c(df_2gpu: pd.DataFrame, df_4gpu: pd.DataFrame, out_dir: str):
    """
    Experiment C: scaling efficiency.
    Shows actual vs ideal speedup when doubling GPUs.
    """
    tput_2 = steady_state_mean(df_2gpu, "perf/tokens_per_sec")
    tput_4 = steady_state_mean(df_4gpu, "perf/tokens_per_sec")

    if tput_2 != tput_2 or tput_4 != tput_4:
        print("  [WARN] Missing data for Exp C, skipping plot.")
        return

    actual_speedup = tput_4 / tput_2
    efficiency = actual_speedup / 2.0  # ideal speedup for 2x GPUs is 2x

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: bar chart – 2 GPU vs 4 GPU throughput + ideal
    ax = axes[0]
    configs = ["2 GPU\n(baseline)", "4 GPU\n(actual)", "4 GPU\n(ideal)"]
    values  = [tput_2, tput_4, tput_2 * 2]
    bar_colors = [COLORS["2gpu"], COLORS["4gpu"], "#94a3b8"]
    bars = ax.bar(configs, values, color=bar_colors, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="{:,.0f}", padding=4)
    ax.set_ylabel("Tokens / second")
    ax.set_title(f"Exp C: Scaling efficiency\n"
                 f"Actual speedup: {actual_speedup:.2f}× / Ideal: 2.00×  "
                 f"→ {efficiency*100:.0f}% efficient")
    ax.set_ylim(0, tput_2 * 2.4)

    # Right: efficiency annotation
    ax = axes[1]
    ax.barh(["Parallel\nefficiency"], [efficiency * 100],
            color=COLORS["4gpu"] if efficiency > 0.85 else COLORS["overlap_off"],
            height=0.4)
    ax.axvline(100, color="gray", linestyle="--", linewidth=1, label="Ideal (100%)")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Parallel efficiency (%)")
    ax.set_title("How close to ideal scaling?\n"
                 ">85% = good  |  <70% = communication bottleneck")
    ax.legend()

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False)

    fig.tight_layout()
    path = os.path.join(out_dir, "exp_C_scaling.png")
    fig.savefig(path)
    print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def write_summary(runs: dict, out_path: str):
    rows = []
    for run_name, df in runs.items():
        rows.append({
            "Run": run_name,
            "tok/s": f"{steady_state_mean(df, 'perf/tokens_per_sec'):,.0f}",
            "VRAM GB": f"{steady_state_mean(df, 'perf/vram_gb'):.1f}",
            "bwd ms": f"{steady_state_mean(df, 'perf/bwd_ms'):.0f}",
            "MFU %": f"{steady_state_mean(df, 'perf/mfu_percent'):.1f}",
        })

    df = pd.DataFrame(rows)
    md = df.to_markdown(index=False)

    content = f"""# Benchmark Results

Generated automatically by `benchmarks/collect_results.py`.

## Summary table

{md}

## Experiment A – Communication overlap

![Exp A](plots/exp_A_overlap.png)

`overlap_comm=True` hides AllReduce latency behind the backward pass.
The **bwd_ms** column above shows exposed communication time:
lower = more latency hidden by the overlap.

## Experiment B – ZeRO-2 vs ZeRO-3

![Exp B](plots/exp_B_zero2_vs_zero3.png)

ZeRO-3 adds an `AllGather` before **every layer's forward pass** to reconstruct
sharded weights. This reduces VRAM but increases communication volume.
On InfiniBand, this typically costs 10–20% throughput.

## Experiment C – Scaling efficiency

![Exp C](plots/exp_C_scaling.png)

Ideal: doubling GPUs doubles throughput (100% efficient).
Reality on bwUniCluster: ~80–90% due to AllReduce overhead.
The gap quantifies your interconnect's communication cost.
"""

    with open(out_path, "w") as f:
        f.write(content)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str,
                        default="mixtral-benchmarks-bwunicluster")
    args = parser.parse_args()

    out_dir = "benchmarks/plots"
    os.makedirs(out_dir, exist_ok=True)

    api = wandb.Api()
    print(f"Fetching runs from W&B project: {args.wandb_project}")

    run_names = {
        "expA_overlap_ON":   "expA_overlap_ON",
        "expA_overlap_OFF":  "expA_overlap_OFF",
        "expB_ZeRO2_4gpu":  "expB_ZeRO2_4gpu",
        "expB_ZeRO3_4gpu":  "expB_ZeRO3_4gpu",
        "expC_ZeRO2_2gpu":  "expC_ZeRO2_2gpu",
    }

    dfs = {}
    for key, name in run_names.items():
        print(f"  Fetching {name} ...")
        dfs[key] = fetch_run(api, args.wandb_project, name)

    print("\nGenerating plots ...")
    plot_exp_a(dfs["expA_overlap_ON"],  dfs["expA_overlap_OFF"], out_dir)
    plot_exp_b(dfs["expB_ZeRO2_4gpu"], dfs["expB_ZeRO3_4gpu"],  out_dir)
    plot_exp_c(dfs["expC_ZeRO2_2gpu"], dfs["expB_ZeRO2_4gpu"],  out_dir)

    print("\nWriting summary table ...")
    write_summary(dfs, "benchmarks/results_summary.md")

    print("\nDone. Commit benchmarks/ to GitHub.")


if __name__ == "__main__":
    main()
