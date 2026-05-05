"""
collect_results.py
──────────────────
Read per-step CSVs from both benchmark experiments, compute summary
statistics, print a comparison table, and log everything to W&B.

Usage (called automatically by run_benchmark.sh, or manually):

    python collect_results.py \
        --exp1_dir  results/exp1_tp2_ep1_dp8 \
        --exp2_dir  results/exp2_tp2_ep4_etp4_dp2 \
        --exp1_name exp1_tp2_ep1_dp8 \
        --exp2_name exp2_tp2_ep4_etp4_dp2 \
        --out_dir   results \
        --wandb_project mixtral-dp-vs-ep
"""

import argparse
import csv
import math
import os
import statistics
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp1_dir",       type=str, required=True)
    p.add_argument("--exp2_dir",       type=str, required=True)
    p.add_argument("--exp1_name",      type=str, default="exp1_dp_baseline")
    p.add_argument("--exp2_name",      type=str, default="exp2_ep_treatment")
    p.add_argument("--out_dir",        type=str, default="results")
    p.add_argument("--wandb_project",  type=str, default="mixtral-dp-vs-ep")
    p.add_argument("--warmup_steps",   type=int, default=50,
                   help="Steps to exclude from statistics")
    return p.parse_args()


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_metrics(csv_path: str, warmup_steps: int) -> list[dict]:
    """Load metrics_steps.csv, skip warmup, return measured steps only."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            if step <= warmup_steps:
                continue
            # Cast numeric fields
            for key in row:
                if row[key] in ("", "None"):
                    row[key] = None
                else:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def load_memory(csv_path: str) -> list[dict]:
    """Load memory_by_rank.csv."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["allocated_gb", "peak_allocated_gb",
                        "reserved_gb", "peak_reserved_gb", "step"]:
                try:
                    row[key] = float(row[key])
                except (ValueError, KeyError):
                    pass
            rows.append(row)
    return rows


# ── Statistics helpers ────────────────────────────────────────────────────────

def percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def safe_mean(vals):
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    return statistics.mean(clean) if clean else float("nan")


def safe_median(vals):
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    return statistics.median(clean) if clean else float("nan")


def safe_stdev(vals):
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    return statistics.stdev(clean) if len(clean) > 1 else float("nan")


def summarise(rows: list[dict], field: str) -> dict:
    vals = [r[field] for r in rows if r.get(field) is not None]
    return {
        "mean":   safe_mean(vals),
        "median": safe_median(vals),
        "std":    safe_stdev(vals),
        "p95":    percentile(vals, 95),
        "min":    min(vals) if vals else float("nan"),
        "max":    max(vals) if vals else float("nan"),
    }


def block_stats(rows: list[dict], field: str,
                edges: list[int]) -> list[dict]:
    """Compute per-block statistics."""
    blocks = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        block_rows = [r for r in rows if lo <= int(r["step"]) < hi]
        vals = [r[field] for r in block_rows if r.get(field) is not None]
        blocks.append({
            "block":  f"steps {lo}–{hi-1}",
            "n":      len(vals),
            "mean":   safe_mean(vals),
            "std":    safe_stdev(vals),
            "median": safe_median(vals),
            "p95":    percentile(vals, 95),
        })
    return blocks


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt(val, fmt_str=".3f"):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return format(val, fmt_str)


def pct_diff(a, b):
    """Percentage difference of b relative to a: (b-a)/a * 100."""
    if a is None or b is None or math.isnan(a) or math.isnan(b) or a == 0:
        return float("nan")
    return (b - a) / abs(a) * 100


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    exp1_steps_path  = os.path.join(args.exp1_dir, "metrics_steps.csv")
    exp2_steps_path  = os.path.join(args.exp2_dir, "metrics_steps.csv")
    exp1_memory_path = os.path.join(args.exp1_dir, "memory_by_rank.csv")
    exp2_memory_path = os.path.join(args.exp2_dir, "memory_by_rank.csv")

    for path in [exp1_steps_path, exp2_steps_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing: {path}")

    print(f"Loading {args.exp1_name} ...")
    exp1_rows = load_metrics(exp1_steps_path, args.warmup_steps)
    print(f"  {len(exp1_rows)} measured steps")

    print(f"Loading {args.exp2_name} ...")
    exp2_rows = load_metrics(exp2_steps_path, args.warmup_steps)
    print(f"  {len(exp2_rows)} measured steps")

    # ── Per-field summaries ───────────────────────────────────────────────────
    fields_of_interest = [
        ("step_time_s",            "Step time (s)"),
        ("tokens_per_sec",         "Throughput (tok/s)"),
        ("mfu_percent",            "MFU (%)"),
        ("fwd_time_s",             "Fwd time (s)"),
        ("bwd_time_s",             "Bwd time (s)"),
        ("optim_time_s",           "Optim time (s)"),
        ("grad_norm",              "Grad norm"),
        ("expert_load_imbalance",  "Expert load imbalance"),
        ("expert_routing_entropy", "Expert routing entropy"),
    ]

    summaries_exp1 = {f: summarise(exp1_rows, f) for f, _ in fields_of_interest}
    summaries_exp2 = {f: summarise(exp2_rows, f) for f, _ in fields_of_interest}

    # ── Memory summaries ──────────────────────────────────────────────────────
    mem1 = load_memory(exp1_memory_path) if Path(exp1_memory_path).exists() else []
    mem2 = load_memory(exp2_memory_path) if Path(exp2_memory_path).exists() else []

    def peak_vram_per_rank(mem_rows):
        """Max peak_allocated_gb across all ranks and steps."""
        vals = [r["peak_allocated_gb"] for r in mem_rows
                if isinstance(r.get("peak_allocated_gb"), float)]
        return max(vals) if vals else float("nan")

    peak_vram_exp1 = peak_vram_per_rank(mem1)
    peak_vram_exp2 = peak_vram_per_rank(mem2)

    # ── Block-level step-time statistics ──────────────────────────────────────
    BLOCK_EDGES = [51, 101, 151, 201, 251, 301, 351]

    blocks_exp1 = block_stats(exp1_rows, "step_time_s", BLOCK_EDGES)
    blocks_exp2 = block_stats(exp2_rows, "step_time_s", BLOCK_EDGES)

    # ── Print comparison table ────────────────────────────────────────────────
    COL_W = 18
    def row_line(label, v1, v2, fmt_str=".3f"):
        diff = pct_diff(v1, v2)
        diff_str = f"{diff:+.1f}%" if not math.isnan(diff) else "N/A"
        return (f"  {label:<30} {fmt(v1, fmt_str):>{COL_W}} "
                f"{fmt(v2, fmt_str):>{COL_W}}   {diff_str}")

    sep = "─" * 80
    print("\n" + "═" * 80)
    print(f"  BENCHMARK RESULTS: DP vs EP — Mixtral 8x7B")
    print("═" * 80)
    print(f"  {'Metric':<30} {args.exp1_name:>{COL_W}} "
          f"{args.exp2_name:>{COL_W}}   Δ (ep vs dp)")
    print(sep)

    for field, label in fields_of_interest:
        s1 = summaries_exp1[field]
        s2 = summaries_exp2[field]
        if math.isnan(s1["mean"]) and math.isnan(s2["mean"]):
            continue
        fmt_str = ".0f" if field == "tokens_per_sec" else ".3f"
        print(row_line(f"{label} [mean]",   s1["mean"],   s2["mean"],   fmt_str))
        print(row_line(f"{label} [median]", s1["median"], s2["median"], fmt_str))
        print(row_line(f"{label} [p95]",    s1["p95"],    s2["p95"],    fmt_str))
        print(row_line(f"{label} [std]",    s1["std"],    s2["std"],    fmt_str))
        print()

    print(sep)
    print(row_line("Peak VRAM / rank [GiB]", peak_vram_exp1, peak_vram_exp2, ".2f"))
    print()

    # ── Block-level table ─────────────────────────────────────────────────────
    print(sep)
    print(f"  BLOCK-LEVEL STEP TIME (s)")
    print(f"  {'Block':<20} {'exp1 mean':>10} {'exp1 p95':>10} "
          f"{'exp2 mean':>10} {'exp2 p95':>10}   Δ mean")
    print(sep)
    for b1, b2 in zip(blocks_exp1, blocks_exp2):
        diff = pct_diff(b1["mean"], b2["mean"])
        diff_str = f"{diff:+.1f}%" if not math.isnan(diff) else "N/A"
        print(f"  {b1['block']:<20} "
              f"{fmt(b1['mean']):>10} {fmt(b1['p95']):>10} "
              f"{fmt(b2['mean']):>10} {fmt(b2['p95']):>10}   {diff_str}")

    # ── Loss sanity check ─────────────────────────────────────────────────────
    print()
    print(sep)
    print("  LOSS SANITY")
    loss1_first = exp1_rows[0]["loss"]  if exp1_rows else None
    loss1_last  = exp1_rows[-1]["loss"] if exp1_rows else None
    loss2_first = exp2_rows[0]["loss"]  if exp2_rows else None
    loss2_last  = exp2_rows[-1]["loss"] if exp2_rows else None
    print(f"  {args.exp1_name}: first={fmt(loss1_first)}  last={fmt(loss1_last)}")
    print(f"  {args.exp2_name}: first={fmt(loss2_first)}  last={fmt(loss2_last)}")
    print()
    print("═" * 80)

    # ── Write summary CSV ─────────────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, "comparison_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "stat",
                         args.exp1_name, args.exp2_name, "delta_pct"])
        for field, label in fields_of_interest:
            for stat in ["mean", "median", "p95", "std"]:
                v1 = summaries_exp1[field][stat]
                v2 = summaries_exp2[field][stat]
                diff = pct_diff(v1, v2)
                writer.writerow([label, stat,
                                 "" if math.isnan(v1) else v1,
                                 "" if math.isnan(v2) else v2,
                                 "" if math.isnan(diff) else diff])
        writer.writerow(["Peak VRAM/rank (GiB)", "max",
                         peak_vram_exp1, peak_vram_exp2,
                         pct_diff(peak_vram_exp1, peak_vram_exp2)])
    print(f"Summary CSV written to: {summary_path}")

    # ── W&B summary table ─────────────────────────────────────────────────────
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name="comparison_summary",
            job_type="analysis",
        )

        # Summary scalars
        for field, label in fields_of_interest:
            for stat in ["mean", "median", "p95", "std"]:
                v1 = summaries_exp1[field][stat]
                v2 = summaries_exp2[field][stat]
                if not math.isnan(v1):
                    wandb.summary[f"exp1/{label}/{stat}"] = v1
                if not math.isnan(v2):
                    wandb.summary[f"exp2/{label}/{stat}"] = v2
                diff = pct_diff(v1, v2)
                if not math.isnan(diff):
                    wandb.summary[f"delta_pct/{label}/{stat}"] = diff

        wandb.summary["exp1/peak_vram_gb"] = peak_vram_exp1
        wandb.summary["exp2/peak_vram_gb"] = peak_vram_exp2

        # Block-level table
        block_table = wandb.Table(
            columns=["experiment", "block", "n", "mean_step_s",
                     "std_step_s", "median_step_s", "p95_step_s"],
        )
        for b in blocks_exp1:
            block_table.add_data(args.exp1_name, b["block"], b["n"],
                                 b["mean"], b["std"], b["median"], b["p95"])
        for b in blocks_exp2:
            block_table.add_data(args.exp2_name, b["block"], b["n"],
                                 b["mean"], b["std"], b["median"], b["p95"])
        wandb.log({"block_step_time_table": block_table})

        # Overlay throughput curves (sampled every 5 steps)
        for rows, exp_name in [(exp1_rows, args.exp1_name),
                                (exp2_rows, args.exp2_name)]:
            for r in rows:
                step = int(r["step"])
                if step % 5 != 0:
                    continue
                payload = {
                    f"{exp_name}/tokens_per_sec": r.get("tokens_per_sec"),
                    f"{exp_name}/step_time_s":    r.get("step_time_s"),
                    f"{exp_name}/loss":           r.get("loss"),
                    f"{exp_name}/mfu_percent":    r.get("mfu_percent"),
                }
                payload = {k: v for k, v in payload.items() if v is not None}
                wandb.log(payload, step=step)

        wandb.finish()
        print(f"W&B summary logged to project: {args.wandb_project}")

    except ImportError:
        print("[WARN] wandb not installed — skipping W&B summary upload")
    except Exception as e:
        print(f"[WARN] W&B logging failed: {e}")


if __name__ == "__main__":
    main()
