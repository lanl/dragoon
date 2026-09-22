#!/usr/bin/env python3
"""
Plotting utilities for `dragoon bench cpu-agent` sweep results.

This module loads a directory of JSON metrics files produced by running
`dragoon bench cpu-agent` across a range of parameter values (either
--iterations at fixed --workers, or --workers at fixed --iterations,
e.g. via `scripts/cpu_scaling_sweep.sh` or
`scripts/cpu_worker_scaling_sweep.sh`), and produces plots that visualize
how the benchmark's derived metrics scale with the swept parameter:

- iterations_per_second       (throughput)
- effective_cpu_cores         (average logical cpus effectively consumed)
- worker_cpu_efficiency_percent
- pool_overhead_percent       (ProcessPoolExecutor spawn/teardown overhead
                                as a share of tool wall_seconds)
- agent_overhead_percent      (LLM/agent-framework latency as a share of
                                agent_wall_seconds)
- wall_seconds vs cpu_seconds (tool execution time breakdown)

Each input JSON file is expected to be a single run's metrics dict (as
written by `dragoon bench cpu-agent --output <file>.json`), containing at
least: iterations, workers, wall_seconds, cpu_seconds,
iterations_per_second, effective_cpu_cores, worker_cpu_efficiency_percent,
pool_overhead_percent, agent_wall_seconds, agent_overhead_percent.

Usage (CLI), for an iteration-count sweep (fixed workers):

    python3 -m dragoon.utils.plot_cpu_sweep \\
        --input-dir scripts/results/cpu_iter_sweep \\
        --output-dir scripts/results/cpu_iter_sweep/plots \\
        --x-key iterations

Usage (CLI), for a worker-count sweep (fixed iterations):

    python3 -m dragoon.utils.plot_cpu_sweep \\
        --input-dir scripts/results/cpu_worker_sweep \\
        --output-dir scripts/results/cpu_worker_sweep/plots \\
        --x-key workers

Usage (import):

    from dragoon.utils.plot_cpu_sweep import load_sweep_results, plot_all

    runs = load_sweep_results("scripts/results/cpu_worker_sweep", x_key="workers")
    plot_all(runs, output_dir="scripts/results/cpu_worker_sweep/plots", x_key="workers")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

# use a non-interactive backend so this works headlessly on HPC nodes
# without a display, e.g. over SSH with no X forwarding
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_sweep_results(
    input_dir: str | Path,
    x_key: str = "iterations",
) -> list[dict]:
    """load every *.json file in input_dir as a run's metrics dict.

    args:
        input_dir: directory containing one JSON file per benchmark run
            (as produced by `dragoon bench cpu-agent --output <file>.json`)
        x_key: metric field to sort runs by, e.g. "iterations" for an
            iteration-count sweep, or "workers" for a worker-count sweep
    returns:
        list of metrics dicts, sorted by ascending value of x_key
    """
    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"input_dir does not exist or is not a directory: {input_dir}"
        )

    runs = []
    for json_path in sorted(input_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # a single run's metrics is a dict; --repeat > 1 runs write a list
        # of dicts to a single file, so normalize both shapes to a list
        if isinstance(data, list):
            runs.extend(data)
        else:
            runs.append(data)

    if not runs:
        raise ValueError(f"no JSON metrics files found in {input_dir}")

    # sort by x_key so line plots draw left-to-right in increasing
    # workload/concurrency order regardless of filesystem glob order
    runs.sort(key=lambda run: run.get(x_key, 0))

    return runs


def _extract_series(runs: Iterable[dict], key: str) -> list:
    """pull out a metric series from a list of run dicts, defaulting
    missing values to None so plots can skip gaps rather than error out"""
    return [run.get(key) for run in runs]


def _use_log_x(x_values: list) -> bool:
    """use a log x-axis only when values span multiple orders of
    magnitude (e.g. an iterations sweep); a small linear worker sweep
    (e.g. 1..32) reads better on a linear axis"""
    numeric = [v for v in x_values if v is not None and v > 0]
    if len(numeric) < 2:
        return False
    return (max(numeric) / min(numeric)) >= 10


def plot_throughput(
    runs: list[dict],
    output_path: str | Path,
    x_key: str = "iterations",
) -> None:
    """plot iterations_per_second vs x_key"""
    x_values = _extract_series(runs, x_key)
    throughput = _extract_series(runs, "iterations_per_second")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, throughput, marker="o")
    if _use_log_x(x_values):
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_key)
    ax.set_ylabel("iterations_per_second")
    ax.set_title(f"CPU benchmark throughput vs {x_key}")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_effective_cpu_cores(
    runs: list[dict],
    output_path: str | Path,
    x_key: str = "iterations",
) -> None:
    """plot effective_cpu_cores vs x_key.

    when x_key == "workers", draws a diagonal y=x reference line
    representing ideal linear core scaling (effective_cpu_cores ==
    workers). otherwise (e.g. an iterations sweep at fixed workers),
    draws a horizontal reference line at the fixed worker count.
    """
    x_values = _extract_series(runs, x_key)
    effective_cores = _extract_series(runs, "effective_cpu_cores")
    workers = _extract_series(runs, "workers")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        x_values,
        effective_cores,
        marker="o",
        label="effective_cpu_cores",
    )

    if x_key == "workers":
        numeric_x = [v for v in x_values if v is not None]
        if numeric_x:
            ax.plot(
                numeric_x,
                numeric_x,
                color="gray",
                linestyle="--",
                label="ideal (effective_cpu_cores == workers)",
            )
    elif workers and workers[0] is not None:
        # assume workers is constant across the sweep (as produced by
        # cpu_scaling_sweep.sh); if it varies, just use the first value
        ax.axhline(
            workers[0],
            color="gray",
            linestyle="--",
            label=f"workers={workers[0]} (theoretical ceiling)",
        )

    if _use_log_x(x_values):
        ax.set_xscale("log")
    ax.set_xlabel(x_key)
    ax.set_ylabel("effective_cpu_cores")
    ax.set_title(f"Effective CPU cores consumed vs {x_key}")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_efficiency_and_overhead(
    runs: list[dict],
    output_path: str | Path,
    x_key: str = "iterations",
) -> None:
    """plot worker_cpu_efficiency_percent and pool_overhead_percent vs
    x_key on the same axes"""
    x_values = _extract_series(runs, x_key)
    efficiency = _extract_series(runs, "worker_cpu_efficiency_percent")
    pool_overhead = _extract_series(runs, "pool_overhead_percent")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        x_values,
        efficiency,
        marker="o",
        label="worker_cpu_efficiency_percent",
    )
    ax.plot(
        x_values,
        pool_overhead,
        marker="s",
        label="pool_overhead_percent",
    )

    if _use_log_x(x_values):
        ax.set_xscale("log")
    ax.set_xlabel(x_key)
    ax.set_ylabel("percent")
    ax.set_title(f"Worker CPU efficiency vs process-pool overhead ({x_key} sweep)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_agent_overhead(
    runs: list[dict],
    output_path: str | Path,
    x_key: str = "iterations",
) -> None:
    """plot agent_overhead_percent vs x_key, showing how much of
    agent_wall_seconds is LLM/agent-framework latency vs tool execution"""
    x_values = _extract_series(runs, x_key)
    agent_overhead = _extract_series(runs, "agent_overhead_percent")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, agent_overhead, marker="o", color="tab:red")

    if _use_log_x(x_values):
        ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel(x_key)
    ax.set_ylabel("agent_overhead_percent")
    ax.set_title(
        f"LLM/agent-framework overhead as a share of total run time ({x_key} sweep)"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_wall_vs_cpu_time(
    runs: list[dict],
    output_path: str | Path,
    x_key: str = "iterations",
) -> None:
    """plot tool wall_seconds and cpu_seconds vs x_key (log y-axis),
    showing when compute time overtakes fixed overhead"""
    x_values = _extract_series(runs, x_key)
    wall_seconds = _extract_series(runs, "wall_seconds")
    cpu_seconds = _extract_series(runs, "cpu_seconds")
    pool_overhead_seconds = _extract_series(runs, "pool_overhead_seconds")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, wall_seconds, marker="o", label="wall_seconds")
    ax.plot(x_values, cpu_seconds, marker="s", label="cpu_seconds")
    if any(v is not None for v in pool_overhead_seconds):
        ax.plot(
            x_values,
            pool_overhead_seconds,
            marker="^",
            label="pool_overhead_seconds",
        )

    if _use_log_x(x_values):
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_key)
    ax.set_ylabel("seconds")
    ax.set_title(f"Tool wall-clock vs CPU time vs pool overhead ({x_key} sweep)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_speedup(
    runs: list[dict],
    output_path: str | Path,
    x_key: str = "workers",
) -> None:
    """plot parallel speedup vs x_key (typically "workers"), i.e.
    iterations_per_second at each point divided by the throughput of the
    smallest x_key value in the sweep (usually workers=1). Includes an
    ideal linear-speedup reference line (speedup == workers / min_workers)
    to visualize how far the benchmark falls short of perfect scaling as
    concurrency increases -- this is the classic strong-scaling plot for
    a worker-count sweep at fixed iterations.
    """
    x_values = _extract_series(runs, x_key)
    throughput = _extract_series(runs, "iterations_per_second")

    valid = [
        (x, t)
        for x, t in zip(x_values, throughput)
        if x is not None and t is not None
    ]
    if not valid:
        return

    baseline_x, baseline_throughput = valid[0]
    if not baseline_throughput:
        return

    speedup = [
        t / baseline_throughput if baseline_throughput else None
        for _, t in valid
    ]
    xs = [x for x, _ in valid]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, speedup, marker="o", label="measured speedup")

    if x_key == "workers":
        ideal = [x / baseline_x for x in xs]
        ax.plot(
            xs,
            ideal,
            color="gray",
            linestyle="--",
            label="ideal linear speedup",
        )

    ax.set_xlabel(x_key)
    ax.set_ylabel(f"speedup (relative to {x_key}={baseline_x})")
    ax.set_title(f"Parallel speedup vs {x_key}")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_concurrency_scaling(
    runs: list[dict],
    output_path: str | Path,
    x_key: str = "concurrency",
) -> None:
    """plot aggregate iterations_per_second vs concurrency.

    includes an ideal linear-scaling reference line (throughput scaling
    in proportion to concurrency, relative to the smallest concurrency in
    the sweep) so the saturation/knee point -- where adding more agents
    stops improving aggregate throughput -- is easy to see.
    """
    x_values = _extract_series(runs, x_key)
    throughput = _extract_series(runs, "iterations_per_second")

    valid = [
        (x, t)
        for x, t in zip(x_values, throughput)
        if x is not None and t is not None
    ]
    if not valid:
        return

    xs = [x for x, _ in valid]
    ys = [t for _, t in valid]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, ys, marker="o", label="aggregate iterations_per_second")

    # ideal linear scaling relative to the smallest concurrency point
    baseline_x, baseline_throughput = valid[0]
    if baseline_throughput and baseline_x:
        ideal = [
            baseline_throughput * (x / baseline_x)
            for x in xs
        ]
        ax.plot(
            xs,
            ideal,
            color="gray",
            linestyle="--",
            label="ideal linear scaling",
        )

    ax.set_xlabel(x_key)
    ax.set_ylabel("aggregate iterations_per_second")
    ax.set_title(f"Aggregate agent throughput vs {x_key}")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_latency_vs_concurrency(
    runs: list[dict],
    output_path: str | Path,
    x_key: str = "concurrency",
) -> None:
    """plot mean and p95 per-agent latency (agent_wall_seconds) vs
    concurrency, so the rise in per-agent latency as the node saturates is
    visible alongside the throughput curve"""
    x_values = _extract_series(runs, x_key)
    mean_latency = _extract_series(runs, "agent_latency_mean_seconds")
    p95_latency = _extract_series(runs, "agent_latency_p95_seconds")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        x_values,
        mean_latency,
        marker="o",
        label="agent_latency_mean_seconds",
    )
    ax.plot(
        x_values,
        p95_latency,
        marker="s",
        label="agent_latency_p95_seconds",
    )

    ax.set_xlabel(x_key)
    ax.set_ylabel("seconds")
    ax.set_title(f"Per-agent latency vs {x_key}")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_all(
    runs: list[dict],
    output_dir: str | Path,
    x_key: str = "iterations",
) -> list[Path]:
    """generate every sweep plot in this module and save into output_dir.

    args:
        runs: list of run metrics dicts (see load_sweep_results)
        output_dir: directory to write PNG plot files into (created if
            it does not exist)
        x_key: metric field swept across runs, e.g. "iterations",
            "workers", or "concurrency" -- used as the x-axis for every
            plot
    returns:
        list of paths to the generated plot files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = []

    # a concurrency sweep (produced by run_concurrent_agents across
    # increasing --concurrency) carries aggregate throughput and per-agent
    # latency fields, but not the per-run worker-oriented fields
    # (effective_cpu_cores, worker_cpu_efficiency_percent, pool/agent
    # overhead) that the other sweeps plot. plotting those worker-oriented
    # curves against an all-None series would crash the log-scaled axes, so
    # a concurrency sweep gets its own dedicated plot set instead
    if x_key == "concurrency":
        concurrency_scaling_path = output_dir / "concurrency_scaling.png"
        plot_concurrency_scaling(
            runs,
            concurrency_scaling_path,
            x_key=x_key,
        )
        generated_paths.append(concurrency_scaling_path)

        latency_path = output_dir / "latency_vs_concurrency.png"
        plot_latency_vs_concurrency(
            runs,
            latency_path,
            x_key=x_key,
        )
        generated_paths.append(latency_path)

        return generated_paths

    # every other sweep (iterations or workers) plots the worker-oriented
    # per-run metrics produced by run_agent_benchmark / run_cpu_benchmark
    plot_specs = [
        ("throughput.png", plot_throughput),
        ("effective_cpu_cores.png", plot_effective_cpu_cores),
        ("efficiency_and_overhead.png", plot_efficiency_and_overhead),
        ("agent_overhead.png", plot_agent_overhead),
        ("wall_vs_cpu_time.png", plot_wall_vs_cpu_time),
    ]

    for filename, plot_fn in plot_specs:
        output_path = output_dir / filename
        plot_fn(runs, output_path, x_key=x_key)
        generated_paths.append(output_path)

    # the speedup plot is most meaningful for a worker-count sweep, but
    # is still generated (relative to the smallest x_key value) for an
    # iterations sweep for completeness
    speedup_path = output_dir / "speedup.png"
    plot_speedup(runs, speedup_path, x_key=x_key)
    generated_paths.append(speedup_path)

    return generated_paths


def plot_cpu_timeseries(metrics: dict, output_path: str | Path) -> bool:
    """plot session-wide cpu utilization over time for a single run.

    reads metrics["cpu_utilization_timeseries"] (as produced by the
    CPUSampler in dragoon.utils.cpu_sampler and attached by
    run_agent_benchmark / run_concurrent_agents) and draws total cpu
    utilization vs elapsed time. on nodes with a modest core count the
    per-core series are overlaid faintly; on high-core-count nodes a
    shaded min/max band across cores is drawn instead so the figure stays
    readable.

    args:
        metrics: a single run's metrics dict containing a
            "cpu_utilization_timeseries" entry
        output_path: PNG file path to write the plot to
    returns:
        True if a plot was written, False if there was nothing to plot
    """
    timeseries = metrics.get("cpu_utilization_timeseries")
    if not timeseries:
        return False

    samples = timeseries.get("samples", [])
    if not samples:
        return False

    # elapsed time (x) and total utilization (y) for the main line
    elapsed = [sample.get("elapsed_seconds") for sample in samples]
    total = [sample.get("total_cpu_percent") for sample in samples]

    # only overlay individual per-core lines when the core count is small
    # enough to stay legible; otherwise summarize the cores as a band
    logical_cores = timeseries.get("logical_cores") or 0
    per_core_overlay_limit = 16

    fig, ax = plt.subplots(figsize=(10, 6))

    per_core_series = [
        sample.get("per_core_cpu_percent")
        for sample in samples
    ]
    have_per_core = all(
        isinstance(values, list) and values
        for values in per_core_series
    )

    if have_per_core and 0 < logical_cores <= per_core_overlay_limit:
        # overlay each logical cpu's utilization as a thin faint line
        for core_index in range(logical_cores):
            core_values = [
                values[core_index]
                if core_index < len(values)
                else None
                for values in per_core_series
            ]
            ax.plot(
                elapsed,
                core_values,
                color="tab:gray",
                alpha=0.3,
                linewidth=0.8,
            )
    elif have_per_core and logical_cores > per_core_overlay_limit:
        # shade the min-to-max utilization across all cores at each sample
        min_core = [min(values) for values in per_core_series]
        max_core = [max(values) for values in per_core_series]
        ax.fill_between(
            elapsed,
            min_core,
            max_core,
            color="tab:gray",
            alpha=0.25,
            label="per-core min-max range",
        )

    # draw the total (all-core average) utilization on top
    ax.plot(
        elapsed,
        total,
        color="tab:blue",
        linewidth=2.0,
        label="total_cpu_percent",
    )

    ax.set_xlabel("elapsed_seconds")
    ax.set_ylabel("cpu_utilization_percent")
    ax.set_ylim(0, 105)
    ax.set_title(
        "CPU utilization over the session "
        f"(iterations={metrics.get('iterations')}, "
        f"workers={metrics.get('workers')})"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return True


def plot_single_run(metrics: dict, output_dir: str | Path) -> list[Path]:
    """generate figures for a single `dragoon bench cpu-agent` run.

    Unlike `plot_all`, which compares many runs across a sweep of
    --iterations or --workers values, this visualizes one run's metrics
    dict (as returned by `run_agent_benchmark()`), breaking down where
    the total `agent_wall_seconds` was spent and how work was
    distributed across workers.

    args:
        metrics: a single run's metrics dict (as returned by
            run_agent_benchmark / run_cpu_benchmark)
        output_dir: directory to write PNG plot files into (created if
            it does not exist)
    returns:
        list of paths to the generated plot files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = []

    # --- time breakdown pie: agent overhead vs pool overhead vs compute ---
    agent_overhead_seconds = metrics.get("agent_overhead_seconds")
    pool_overhead_seconds = metrics.get("pool_overhead_seconds", 0.0) or 0.0
    wall_seconds = metrics.get("wall_seconds", 0.0) or 0.0
    worker_compute_seconds = max(wall_seconds - pool_overhead_seconds, 0.0)

    if agent_overhead_seconds is not None:
        labels = []
        sizes = []
        if agent_overhead_seconds > 0:
            labels.append("agent_overhead_seconds\n(LLM/agent latency)")
            sizes.append(agent_overhead_seconds)
        if pool_overhead_seconds > 0:
            labels.append("pool_overhead_seconds\n(process-pool spawn)")
            sizes.append(pool_overhead_seconds)
        if worker_compute_seconds > 0:
            labels.append("worker_compute_seconds\n(actual cpu workload)")
            sizes.append(worker_compute_seconds)

        if sizes:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.set_title(
                "Time breakdown of agent_wall_seconds "
                f"(iterations={metrics.get('iterations')}, "
                f"workers={metrics.get('workers')})"
            )
            fig.tight_layout()
            time_breakdown_path = output_dir / "time_breakdown.png"
            fig.savefig(time_breakdown_path, dpi=150)
            plt.close(fig)
            generated_paths.append(time_breakdown_path)

    # --- per-worker wall vs cpu seconds -----------------------------------
    worker_metrics = metrics.get("worker_metrics", [])
    if worker_metrics:
        pids = [str(w.get("pid")) for w in worker_metrics]
        worker_wall = [w.get("wall_seconds") for w in worker_metrics]
        worker_cpu = [w.get("cpu_seconds") for w in worker_metrics]

        x_positions = range(len(pids))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(pids) * 0.6), 6))
        ax.bar(
            [x - bar_width / 2 for x in x_positions],
            worker_wall,
            width=bar_width,
            label="wall_seconds",
        )
        ax.bar(
            [x + bar_width / 2 for x in x_positions],
            worker_cpu,
            width=bar_width,
            label="cpu_seconds",
        )
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(pids, rotation=45, ha="right")
        ax.set_xlabel("worker pid")
        ax.set_ylabel("seconds")
        ax.set_title("Per-worker wall_seconds vs cpu_seconds")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        per_worker_time_path = output_dir / "per_worker_time.png"
        fig.savefig(per_worker_time_path, dpi=150)
        plt.close(fig)
        generated_paths.append(per_worker_time_path)

        # --- per-worker cpu utilization percent -----------------------------
        worker_util = [w.get("cpu_utilization_percent") for w in worker_metrics]

        fig, ax = plt.subplots(figsize=(max(8, len(pids) * 0.6), 6))
        ax.bar(x_positions, worker_util, color="tab:green")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(pids, rotation=45, ha="right")
        ax.set_xlabel("worker pid")
        ax.set_ylabel("cpu_utilization_percent")
        ax.set_ylim(0, 105)
        ax.set_title("Per-worker CPU utilization")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        per_worker_util_path = output_dir / "per_worker_utilization.png"
        fig.savefig(per_worker_util_path, dpi=150)
        plt.close(fig)
        generated_paths.append(per_worker_util_path)

    # --- session-wide cpu utilization time series -------------------------
    # draw the whole-session cpu utilization plot when sampling data is
    # present (added by run_agent_benchmark when --no-cpu-sampling is not
    # set); this is the time-series view of total/per-core utilization
    cpu_timeseries_path = output_dir / "cpu_utilization_timeseries.png"
    if plot_cpu_timeseries(metrics, cpu_timeseries_path):
        generated_paths.append(cpu_timeseries_path)

    return generated_paths


def plot_concurrent_run(metrics: dict, output_dir: str | Path) -> list[Path]:
    """generate aggregate figures for a single concurrent multi-agent run.

    Visualizes one aggregate metrics dict (as returned by
    `run_concurrent_agents()`), covering the node-wide cpu utilization time
    series across all concurrent agents and the distribution of per-agent
    latencies.

    args:
        metrics: an aggregate concurrent-run metrics dict (as returned by
            run_concurrent_agents)
        output_dir: directory to write PNG plot files into (created if
            it does not exist)
    returns:
        list of paths to the generated plot files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = []

    # --- node-wide cpu utilization time series ----------------------------
    # the sampler in run_concurrent_agents spans every concurrent agent, so
    # this shows aggregate node load over the whole stress test
    cpu_timeseries_path = output_dir / "cpu_utilization_timeseries.png"
    if plot_cpu_timeseries(metrics, cpu_timeseries_path):
        generated_paths.append(cpu_timeseries_path)

    # --- per-agent latency distribution -----------------------------------
    # one bar per agent showing its full workflow latency, with reference
    # lines for the mean and p95 so outliers/tail latency are obvious
    agent_metrics = metrics.get("agent_metrics", [])
    agent_latencies = [
        agent.get("agent_wall_seconds")
        for agent in agent_metrics
        if agent.get("agent_wall_seconds") is not None
    ]

    if agent_latencies:
        x_positions = range(len(agent_latencies))

        fig, ax = plt.subplots(
            figsize=(max(8, len(agent_latencies) * 0.4), 6)
        )
        ax.bar(x_positions, agent_latencies, color="tab:purple")

        # overlay the mean and p95 latency reference lines when available
        mean_latency = metrics.get("agent_latency_mean_seconds")
        if mean_latency is not None:
            ax.axhline(
                mean_latency,
                color="tab:orange",
                linestyle="--",
                label=f"mean={mean_latency:.2f}s",
            )
        p95_latency = metrics.get("agent_latency_p95_seconds")
        if p95_latency is not None:
            ax.axhline(
                p95_latency,
                color="tab:red",
                linestyle="--",
                label=f"p95={p95_latency:.2f}s",
            )

        ax.set_xlabel("agent index")
        ax.set_ylabel("agent_wall_seconds")
        ax.set_title(
            "Per-agent latency across concurrent agents "
            f"(concurrency={metrics.get('concurrency')})"
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend()

        fig.tight_layout()
        latency_path = output_dir / "agent_latency_distribution.png"
        fig.savefig(latency_path, dpi=150)
        plt.close(fig)
        generated_paths.append(latency_path)

    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot dragoon cpu-agent benchmark sweep results (JSON files "
            "in a directory, one per run at increasing --iterations or "
            "--workers)."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Directory containing benchmark JSON files, e.g. "
            "scripts/results/cpu_iter_sweep"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write PNG plots into "
            "(default: <input-dir>/plots)"
        ),
    )
    parser.add_argument(
        "--x-key",
        default="iterations",
        choices=["iterations", "workers", "concurrency"],
        help=(
            "Metric field that was swept across runs, used as the x-axis "
            "for every plot (default: iterations)"
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"

    runs = load_sweep_results(input_dir, x_key=args.x_key)
    generated_paths = plot_all(runs, output_dir, x_key=args.x_key)

    print(f"Loaded {len(runs)} run(s) from {input_dir}")
    for path in generated_paths:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()

