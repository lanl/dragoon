"""Model-agnostic TensorRT-LLM deployment and benchmarking.

TensorRT-LLM resolves the model architecture from the model's Hugging Face
configuration. A model must be supported by the installed TensorRT-LLM release.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml


class DeploymentError(RuntimeError):
    """Raised when a TensorRT-LLM deployment cannot be initialized."""


class BenchmarkConfigError(ValueError):
    """Raised when a Dragoon trtllm-bench scenario file is invalid."""


@dataclass(slots=True)
class DeploymentConfig:
    model: str
    tokenizer: str | None = None
    engine_dir: str | None = None
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    revision: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False
    seed: int = 0

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.engine_dir is not None:
            engine = Path(self.engine_dir)
            if not engine.exists():
                raise FileNotFoundError(f"engine directory does not exist: {engine}")
        model_path = Path(self.model)
        if model_path.is_absolute() or self.model.startswith("./"):
            if not model_path.exists():
                raise FileNotFoundError(f"model path does not exist: {model_path}")
        if self.tensor_parallel_size < 1 or self.pipeline_parallel_size < 1:
            raise ValueError("parallel sizes must be positive")


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 128
    min_new_tokens: int = 1
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    repetition_penalty: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    skip_special_tokens: bool = True

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.min_new_tokens < 0 or self.min_new_tokens > self.max_new_tokens:
            raise ValueError("min_new_tokens must be between 0 and max_new_tokens")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 0 or not 0 < self.top_p <= 1:
            raise ValueError("top_k must be non-negative and top_p must be in (0, 1]")


@dataclass(slots=True)
class GenerationResult:
    prompt: str
    text: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float | None = None
    raw_output: Any = field(default=None, repr=False)


@dataclass(slots=True)
class BenchmarkResult:
    model: str
    batch_size: int
    warmup_runs: int
    benchmark_runs: int
    latency_seconds: list[float]
    mean_latency_seconds: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    p99_latency_seconds: float
    prompt_tokens: int
    output_tokens: int
    requests_per_second: float
    output_tokens_per_second: float
    environment: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


@dataclass(slots=True)
class BenchmarkScenario:
    """One native ``trtllm-bench`` invocation."""

    name: str
    command: str
    options: dict[str, Any]


@dataclass(slots=True)
class BenchmarkScenarioResult:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkFile:
    """A config file containing one model and multiple benchmark scenarios."""

    model: str
    scenarios: list[BenchmarkScenario]
    model_path: str | None = None
    workspace: str = "/tmp"
    revision: str | None = None


def _resolve_config_path(value: Any, config_dir: Path) -> Any:
    """Resolve path-valued benchmark options relative to the config file."""
    if not isinstance(value, str):
        return value
    if not value or value.startswith("/") or value.startswith("~"):
        return value
    return str((config_dir / value).resolve())


def load_benchmark_file(path: str | os.PathLike[str]) -> BenchmarkFile:
    """Load a YAML or JSON Dragoon benchmark scenario file.

    The keys in each scenario's ``options`` mapping are native
    ``trtllm-bench`` option names, without the leading ``--``. For example,
    ``num_requests: 100`` becomes ``--num_requests 100``.
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"benchmark config does not exist: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise BenchmarkConfigError(f"invalid YAML/JSON benchmark config: {exc}") from exc
    if not isinstance(data, dict):
        raise BenchmarkConfigError("benchmark config must contain a mapping at its top level")
    model = data.get("model")
    scenarios = data.get("scenarios")
    if not isinstance(model, str) or not model.strip():
        raise BenchmarkConfigError("benchmark config requires a non-empty 'model'")
    if not isinstance(scenarios, list) or not scenarios:
        raise BenchmarkConfigError("benchmark config requires a non-empty 'scenarios' list")

    parsed: list[BenchmarkScenario] = []
    for index, raw in enumerate(scenarios):
        if not isinstance(raw, dict):
            raise BenchmarkConfigError(f"scenario {index} must be a mapping")
        name = raw.get("name", f"scenario-{index + 1}")
        command = raw.get("command", "throughput")
        options = raw.get("options", {})
        if not isinstance(name, str) or not name.strip():
            raise BenchmarkConfigError(f"scenario {index} has an invalid name")
        if command not in {"throughput", "latency"}:
            raise BenchmarkConfigError(
                f"scenario {name!r} command must be 'throughput' or 'latency'"
            )
        if not isinstance(options, dict):
            raise BenchmarkConfigError(f"scenario {name!r} options must be a mapping")
        normalized: dict[str, Any] = {}
        for option, value in options.items():
            if not isinstance(option, str) or not option.strip():
                raise BenchmarkConfigError(f"scenario {name!r} contains an invalid option")
            key = option.removeprefix("--").replace("-", "_")
            if key in {"model", "model_path", "workspace", "revision"}:
                raise BenchmarkConfigError(
                    f"scenario {name!r} cannot override top-level option {key!r}"
                )
            if key in {"dataset", "sampler_options", "medusa_choices", "custom_module_dirs", "report_json", "iteration_log", "output_json", "request_json", "config", "extra_llm_api_options"}:
                if isinstance(value, list):
                    value = [_resolve_config_path(item, config_path.parent) for item in value]
                else:
                    value = _resolve_config_path(value, config_path.parent)
            normalized[key] = value
        parsed.append(BenchmarkScenario(name=name, command=command, options=normalized))

    model_path = data.get("model_path")
    workspace = data.get("workspace", "/tmp")
    if model_path is not None:
        model_path = _resolve_config_path(model_path, config_path.parent)
    workspace = _resolve_config_path(workspace, config_path.parent)
    return BenchmarkFile(
        model=model,
        model_path=model_path,
        workspace=workspace,
        revision=data.get("revision"),
        scenarios=parsed,
    )


def _trtllm_bench_executable() -> str:
    candidates = [Path(sys.prefix) / "bin" / "trtllm-bench"]
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidates.append(Path(virtual_env) / "bin" / "trtllm-bench")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    executable = shutil.which("trtllm-bench")
    if executable:
        return executable
    raise DeploymentError(
        "trtllm-bench was not found in the active Python environment; "
        "install TensorRT-LLM and activate its venv"
    )


def _option_arguments(options: dict[str, Any]) -> list[str]:
    arguments: list[str] = []
    for key, value in options.items():
        option = f"--{key.replace('-', '_')}"
        if value is None or value is False:
            # False flags are omitted. Native negative flags such as
            # --disable_chunked_context can be supplied explicitly as a key.
            continue
        if value is True:
            arguments.append(option)
        elif isinstance(value, (list, tuple)):
            for item in value:
                arguments.extend((option, str(item)))
        else:
            arguments.extend((option, str(value)))
    return arguments


def build_trtllm_bench_command(
    benchmark: BenchmarkFile, scenario: BenchmarkScenario
) -> list[str]:
    """Build, but do not execute, one ``trtllm-bench`` command."""
    command = [_trtllm_bench_executable(), "--model", benchmark.model]
    if benchmark.model_path is not None:
        command.extend(("--model_path", benchmark.model_path))
    if benchmark.workspace is not None:
        command.extend(("--workspace", benchmark.workspace))
    if benchmark.revision is not None:
        command.extend(("--revision", benchmark.revision))
    command.append(scenario.command)
    command.extend(_option_arguments(scenario.options))
    return command


def run_benchmark_file(
    path: str | os.PathLike[str], *, check: bool = True
) -> list[BenchmarkScenarioResult]:
    """Execute all scenarios through ``trtllm-bench`` with live output."""
    benchmark = load_benchmark_file(path)
    results: list[BenchmarkScenarioResult] = []
    for scenario in benchmark.scenarios:
        command = build_trtllm_bench_command(benchmark, scenario)
        print(f"[dragoon] Starting scenario: {scenario.name}", flush=True)
        print(f"[dragoon] Command: {shlex.join(command)}", flush=True)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines: list[str] = []
        try:
            assert process.stdout is not None
            for line in process.stdout:
                output_lines.append(line)
                print(f"[{scenario.name}] {line}", end="", flush=True)
            returncode = process.wait()
        except KeyboardInterrupt:
            print(
                f"\n[dragoon] Interrupt received; stopping scenario {scenario.name!r}...",
                flush=True,
            )
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise DeploymentError(f"benchmark scenario {scenario.name!r} was interrupted")
        result = BenchmarkScenarioResult(
            name=scenario.name,
            command=command,
            returncode=returncode,
            stdout="".join(output_lines),
            stderr="",
        )
        results.append(result)
        if returncode == 0:
            print(f"[dragoon] Completed scenario: {scenario.name}", flush=True)
        if check and returncode != 0:
            raise DeploymentError(
                f"trtllm-bench scenario {scenario.name!r} failed with exit code "
                f"{returncode}\n{''.join(output_lines).strip()}"
            )
    return results


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile over no values")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _normalize_prompts(prompts: str | Sequence[str]) -> list[str]:
    normalized = [prompts] if isinstance(prompts, str) else list(prompts)
    if not normalized or any(not isinstance(prompt, str) for prompt in normalized):
        raise ValueError("prompts must contain at least one string")
    return normalized


class TensorRTLLMDeployment:
    """Reusable lifecycle wrapper around TensorRT-LLM's high-level LLM API."""

    def __init__(self, config: DeploymentConfig):
        config.validate()
        self.config = config
        self._llm: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def validate_environment(self) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise DeploymentError("PyTorch is required for TensorRT-LLM deployment") from exc

        info: dict[str, Any] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_count": int(torch.cuda.device_count()),
            "gpus": [],
        }
        if info["cuda_available"]:
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(info["gpu_count"])]
        required_gpus = self.config.tensor_parallel_size * self.config.pipeline_parallel_size
        if not info["cuda_available"]:
            raise DeploymentError(
                "CUDA is unavailable. Run Dragoon inside a GPU allocation and verify "
                "the CUDA library paths before loading TensorRT-LLM."
            )
        if info["gpu_count"] < required_gpus:
            raise DeploymentError(
                f"requested {required_gpus} GPUs but only {info['gpu_count']} are visible"
            )
        return info

    def load(self) -> "TensorRTLLMDeployment":
        if self.loaded:
            return self
        self.validate_environment()
        try:
            from tensorrt_llm import LLM
        except Exception as exc:
            raise DeploymentError(
                "TensorRT-LLM could not be imported. Load OpenMPI and expose the "
                "CUDA/TensorRT shared libraries before running Dragoon."
            ) from exc

        kwargs: dict[str, Any] = {
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "dtype": self.config.dtype,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.tokenizer is not None:
            kwargs["tokenizer"] = self.config.tokenizer
        if self.config.revision is not None:
            kwargs["revision"] = self.config.revision
        if self.config.tokenizer_revision is not None:
            kwargs["tokenizer_revision"] = self.config.tokenizer_revision
        try:
            # TensorRT-LLM's high-level API accepts a model ID/path as its
            # positional model argument. A prebuilt engine directory is also
            # supplied through that same argument, rather than an
            # ``engine_dir`` keyword (which is not part of the 1.2 API).
            model = self.config.engine_dir or self.config.model
            self._llm = LLM(model, **kwargs)
        except Exception as exc:
            raise DeploymentError(
                f"TensorRT-LLM could not load model {self.config.model!r}. "
                "The model architecture must be supported by the installed release."
            ) from exc
        return self

    def generate(
        self,
        prompts: str | Sequence[str],
        generation: GenerationConfig | None = None,
    ) -> list[GenerationResult]:
        if not self.loaded:
            raise DeploymentError("deployment is not loaded; call load() first")
        prompts_list = _normalize_prompts(prompts)
        generation = generation or GenerationConfig()
        generation.validate()
        try:
            from tensorrt_llm import SamplingParams

            params = SamplingParams(
                max_tokens=generation.max_new_tokens,
                min_tokens=generation.min_new_tokens,
                temperature=generation.temperature,
                top_k=generation.top_k,
                top_p=generation.top_p,
                repetition_penalty=generation.repetition_penalty,
                stop=generation.stop,
                seed=generation.seed if generation.seed is not None else self.config.seed,
                skip_special_tokens=generation.skip_special_tokens,
            )
            outputs = self._llm.generate(prompts_list, sampling_params=params, use_tqdm=False)
        except Exception as exc:
            raise DeploymentError("TensorRT-LLM generation failed") from exc

        if not isinstance(outputs, list):
            outputs = [outputs]
        results: list[GenerationResult] = []
        for prompt, request in zip(prompts_list, outputs):
            completion = request.outputs[0]
            prompt_ids = getattr(request, "prompt_token_ids", None)
            output_ids = getattr(completion, "token_ids", None)
            results.append(
                GenerationResult(
                    prompt=prompt,
                    text=getattr(completion, "text", str(completion)),
                    prompt_tokens=len(prompt_ids) if prompt_ids is not None else None,
                    output_tokens=len(output_ids) if output_ids is not None else None,
                    total_tokens=(len(prompt_ids) + len(output_ids))
                    if prompt_ids is not None and output_ids is not None
                    else None,
                    raw_output=request,
                )
            )
        return results

    def benchmark(
        self,
        prompts: str | Sequence[str],
        generation: GenerationConfig | None = None,
        *,
        warmup_runs: int = 2,
        benchmark_runs: int = 10,
    ) -> BenchmarkResult:
        if warmup_runs < 0 or benchmark_runs < 1:
            raise ValueError("warmup_runs must be non-negative and benchmark_runs positive")
        prompts_list = _normalize_prompts(prompts)
        generation = generation or GenerationConfig()
        generation.validate()
        if not self.loaded:
            self.load()
        for _ in range(warmup_runs):
            self.generate(prompts_list, generation)

        try:
            import torch
        except ImportError as exc:
            raise DeploymentError("PyTorch is required for benchmark timing") from exc
        latencies: list[float] = []
        measured_outputs: list[GenerationResult] = []
        for _ in range(benchmark_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            measured_outputs = self.generate(prompts_list, generation)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

        prompt_tokens = sum(r.prompt_tokens or 0 for r in measured_outputs)
        output_tokens = sum(r.output_tokens or 0 for r in measured_outputs)
        total_time = sum(latencies)
        return BenchmarkResult(
            model=self.config.model,
            batch_size=len(prompts_list),
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            latency_seconds=latencies,
            mean_latency_seconds=statistics.mean(latencies),
            p50_latency_seconds=_percentile(latencies, 50),
            p95_latency_seconds=_percentile(latencies, 95),
            p99_latency_seconds=_percentile(latencies, 99),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            requests_per_second=(benchmark_runs * len(prompts_list)) / total_time,
            output_tokens_per_second=(output_tokens * benchmark_runs) / total_time,
            environment=self.validate_environment(),
        )

    def unload(self) -> None:
        llm = self._llm
        self._llm = None
        if llm is not None and hasattr(llm, "shutdown"):
            llm.shutdown()

    def __enter__(self) -> "TensorRTLLMDeployment":
        return self.load()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.unload()


def benchmark_model(
    config: DeploymentConfig,
    prompts: str | Sequence[str],
    generation: GenerationConfig | None = None,
    *,
    warmup_runs: int = 2,
    benchmark_runs: int = 10,
) -> BenchmarkResult:
    with TensorRTLLMDeployment(config) as deployment:
        return deployment.benchmark(
            prompts,
            generation,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )