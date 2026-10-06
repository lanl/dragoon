from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

#from dragoon.benchmark.bench_remote_endpoint_cli import register_subparser as register_remote_bench
#from dragoon.benchmark.bench_litellm_cli import register_subparser as register_litellm_bench


def bench(args: argparse.Namespace) -> None:
    if hasattr(args, "func"):
        args.func(args)
        return
    raise SystemExit("No bench subcommand selected.")


def bench_cpu_agent(args: argparse.Namespace) -> None:
    """Handle the `dragoon bench cpu-agent` command."""
    # Load environment variables from a .env file so that DRGN_MODEL,
    # DRGN_BASE_URL and DRGN_API_KEY can be configured without exporting
    # them manually. Defaults to ./.env if --env-file is not provided.
    env_path = Path(args.env_file) if args.env_file else Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        print(f"Warning: env file not found: {env_path}; using DRGN_* defaults", file=sys.stderr)

    # import cpu_agent only after .env has been loaded, since cpu_agent
    # reads DRGN_MODEL/DRGN_BASE_URL/DRGN_API_KEY and builds the ChatOpenAI
    # client at module import time
    from dragoon.agents.cpu_agent import (
        run_agent_benchmark,
        run_concurrent_agents,
    )

    results = []
    for run_index in range(args.repeat):
        try:
            # run several agents at once when concurrency > 1 to stress
            # test agent performance at scale on a single node; otherwise
            # run the single-agent benchmark path unchanged
            if args.concurrency > 1:
                metrics = run_concurrent_agents(
                    iterations=args.iterations,
                    workers=args.workers,
                    concurrency=args.concurrency,
                    sample_cpu=not args.no_cpu_sampling,
                    sample_interval=args.sample_interval,
                )
            else:
                metrics = run_agent_benchmark(
                    iterations=args.iterations,
                    workers=args.workers,
                    sample_cpu=not args.no_cpu_sampling,
                    sample_interval=args.sample_interval,
                )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        # stamp the requested concurrency onto every run's metrics so a
        # concurrency sweep can plot the single-agent (concurrency=1) path
        # -- which runs through run_agent_benchmark and has no concurrency
        # field of its own -- on the same x-axis as the multi-agent runs
        metrics.setdefault("concurrency", args.concurrency)

        metrics["run_index"] = run_index
        results.append(metrics)

        print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                results if args.repeat != 1 else results[0],
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"\n Results written to {output_path}")


def register_cpu_agent_bench(subparsers: argparse._SubParsersAction) -> None:
    cpu_agent_parser = subparsers.add_parser(
        "cpu-agent",
        help="Run the LangChain CPU benchmark agent",
    )
    cpu_agent_parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="Total number of CPU loop iterations to perform",
    )
    cpu_agent_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of CPU worker processes to use (default: 1)",
    )
    cpu_agent_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Number of LangChain agents to run simultaneously to stress "
            "test agent performance at scale on a single node (default: 1)"
        ),
    )
    cpu_agent_parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.25,
        help=(
            "Seconds between successive CPU utilization samples for the "
            "session-wide time series (default: 0.25)"
        ),
    )
    cpu_agent_parser.add_argument(
        "--no-cpu-sampling",
        action="store_true",
        help="Disable session-wide CPU utilization sampling and its time-series plot",
    )
    cpu_agent_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the benchmark run (default: 1)",
    )
    cpu_agent_parser.add_argument(
        "--output",
        "-o",
        help="Path to save benchmark metrics as JSON",
    )
    cpu_agent_parser.add_argument(
        "--env-file",
        help="Path to a .env file with DRGN_MODEL, DRGN_BASE_URL, DRGN_API_KEY (default: ./.env)",
    )
    cpu_agent_parser.set_defaults(func=bench_cpu_agent)


def bench_memory_agent(args: argparse.Namespace) -> None:
    """Handle the `dragoon bench memory-agent` command."""
    # Load environment variables from a .env file so that DRGN_MODEL,
    # DRGN_BASE_URL and DRGN_API_KEY can be configured without exporting
    # them manually. Defaults to ./.env if --env-file is not provided.
    env_path = Path(args.env_file) if args.env_file else Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        print(f"Warning: env file not found: {env_path}; using DRGN_* defaults", file=sys.stderr)

    # import memory_agent only after .env has been loaded, since it reads
    # DRGN_MODEL/DRGN_BASE_URL/DRGN_API_KEY and builds the ChatOpenAI
    # client at module import time
    from dragoon.agents.memory_agent import run_agent_benchmark

    results = []
    for run_index in range(args.repeat):
        try:
            metrics = run_agent_benchmark(
                array_size_mb=args.array_size_mb,
                passes=args.passes,
                workers=args.workers,
                generate_figures=not args.no_figures,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        metrics["run_index"] = run_index
        results.append(metrics)

        print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                results if args.repeat != 1 else results[0],
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"\n✓ Results written to {output_path}")


def register_memory_agent_bench(subparsers: argparse._SubParsersAction) -> None:
    memory_agent_parser = subparsers.add_parser(
        "memory-agent",
        help="Run the LangChain memory-bandwidth benchmark agent",
    )
    memory_agent_parser.add_argument(
        "--array-size-mb",
        type=int,
        required=True,
        help="Size of EACH array in MiB, per worker (source and destination)",
    )
    memory_agent_parser.add_argument(
        "--passes",
        type=int,
        required=True,
        help="Number of full-array copies each worker performs",
    )
    memory_agent_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent memory-stream worker processes (default: 1)",
    )
    memory_agent_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the benchmark run (default: 1)",
    )
    memory_agent_parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Disable automatic per-run figure generation",
    )
    memory_agent_parser.add_argument(
        "--output",
        "-o",
        help="Path to save benchmark metrics as JSON",
    )
    memory_agent_parser.add_argument(
        "--env-file",
        help="Path to a .env file with DRGN_MODEL, DRGN_BASE_URL, DRGN_API_KEY (default: ./.env)",
    )
    memory_agent_parser.set_defaults(func=bench_memory_agent)


def main() -> None:
    parser = argparse.ArgumentParser(prog="dragoon")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench_parser = subparsers.add_parser("bench", help="Run benchmarks")
    bench_subparsers = bench_parser.add_subparsers(dest="bench_command", required=True)

    #register_remote_bench(bench_subparsers) # WIP
    #register_litellm_bench(bench_subparsers) # WIP
    register_cpu_agent_bench(bench_subparsers)
    #register_memory_agent_bench(bench_subparsers) # WIP

    args = parser.parse_args()

    try:
        if hasattr(args, "func"):
            args.func(args)
        else:
            parser.print_help()
            raise SystemExit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()