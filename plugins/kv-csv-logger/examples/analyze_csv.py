"""Analyze a kv_csv output file: prefix-cache hit rate, latency, and a plot.

Usage:
    python examples/analyze_csv.py /tmp/kv_run.0.csv [out.png]

Reads the per-step CSV written by the kv_csv plugin, prints summary metrics, and
saves a PNG with KV-cache usage / running-queue depth / cumulative prefix hit
rate over time.
"""

import sys

import matplotlib

matplotlib.use("Agg")  # headless: render to file, no display needed
import matplotlib.pyplot as plt
import pandas as pd


def main(path: str, out_png: str) -> None:
    df = pd.read_csv(path)
    # Numeric coercion; blank cells (absent stats that step) become NaN -> 0.
    df = df.apply(pd.to_numeric, errors="coerce")

    # Seconds since the first row, for a readable time axis.
    t0 = df["wall_time"].iloc[0]
    df["t"] = df["wall_time"] - t0

    steps = len(df)
    duration = df["t"].iloc[-1]

    # --- Prefix cache hit rate (sum hits / sum queries over the whole run) ---
    q = df["prefix_queries"].fillna(0).sum()
    h = df["prefix_hits"].fillna(0).sum()
    hit_rate = (h / q) if q else 0.0

    # --- Latency: aggregate the per-step count/sum pairs ---
    ttft_n = df["ttft_count"].fillna(0).sum()
    ttft_s = df["ttft_sum"].fillna(0).sum()
    itl_n = df["itl_count"].fillna(0).sum()
    itl_s = df["itl_sum"].fillna(0).sum()
    avg_ttft = (ttft_s / ttft_n) if ttft_n else float("nan")
    avg_itl = (itl_s / itl_n) if itl_n else float("nan")

    # --- Throughput / occupancy ---
    gen_tokens = df["num_generation_tokens"].fillna(0).sum()
    gen_tps = (gen_tokens / duration) if duration else float("nan")
    peak_kv = df["kv_cache_usage"].fillna(0).max()
    peak_running = df["num_running_reqs"].fillna(0).max()
    finished = df["num_finished_reqs"].fillna(0).sum()
    preempted = df["num_preempted_reqs"].fillna(0).sum()

    print(f"file:                {path}")
    print(f"steps (rows):        {steps}")
    print(f"wall duration:       {duration:.3f} s")
    print(f"prefix hit rate:     {hit_rate:.1%}  ({int(h)}/{int(q)} tokens)")
    print(f"avg TTFT:            {avg_ttft * 1e3:.1f} ms  (n={int(ttft_n)})")
    print(f"avg inter-token lat: {avg_itl * 1e3:.2f} ms  (n={int(itl_n)})")
    print(f"generation tokens:   {int(gen_tokens)}  (~{gen_tps:.1f} tok/s)")
    print(f"peak KV usage:       {peak_kv:.4f}")
    print(f"peak running reqs:   {int(peak_running)}")
    print(f"finished / preempt:  {int(finished)} / {int(preempted)}")

    # --- Plot: KV usage + running depth (left axis), cum hit rate (right) ---
    df["cum_hit_rate"] = (
        df["prefix_hits"].fillna(0).cumsum()
        / df["prefix_queries"].fillna(0).cumsum().replace(0, pd.NA)
    )

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(df["t"], df["kv_cache_usage"], color="tab:red", label="KV usage")
    ax1.plot(
        df["t"],
        df["num_running_reqs"] / max(peak_running, 1),
        color="tab:blue",
        alpha=0.6,
        label="running reqs (norm.)",
    )
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("KV usage / normalized running")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(
        df["t"], df["cum_hit_rate"], color="tab:green", label="cum prefix hit rate"
    )
    ax2.set_ylabel("cumulative prefix hit rate")
    ax2.set_ylim(0, 1.05)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"plot saved:          {out_png}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kv_run.0.csv"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "kv_cache_analysis.png"
    main(path, out_png)
