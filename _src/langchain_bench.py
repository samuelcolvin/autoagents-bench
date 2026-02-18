import asyncio
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import yaml
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .metrics import PerformanceMonitor
from .timing import (
    LlmTimingRecorder,
    TimingRecorder,
    llm_request_id,
    patch_openai_timing,
    unpatch_openai_timing,
)
from .trip_tool import (
    TripDataEmptyError,
    TripDataNotFoundError,
    compute_average_trip_duration_minutes,
    format_average_trip_duration,
)

SUCCESS_ABS_TOLERANCE: float = 0.01
_TOOL_TIMINGS = TimingRecorder()
_LLM_TIMINGS = LlmTimingRecorder()


@dataclass
class BenchmarkConfig:
    total_requests: int
    concurrency: int
    model: str
    prompt_template: str


@dataclass
class TimingBreakdown:
    total: float
    queue_wait: float
    call: float
    status: bool


@dataclass
class BenchmarkResult:
    name: str
    total_requests: int
    concurrency: int
    total_duration: float
    throughput_rps: float
    average_latency_ms: float
    p95_latency_ms: float
    average_queue_ms: float
    p95_queue_ms: float
    average_call_ms: float
    p95_call_ms: float
    average_processing_ms: float
    p95_processing_ms: float
    average_tool_ms: float
    p95_tool_ms: float
    average_framework_overhead_ms: float
    p95_framework_overhead_ms: float
    average_llm_total_ms: float
    p95_llm_total_ms: float
    average_framework_overhead_corrected_ms: float
    p95_framework_overhead_corrected_ms: float
    total_success: int
    total_failure: int
    cpu_usage_percent: float
    memory_peak_mb: float
    p99_latency_ms: float
    cold_start_ms: float
    determinism_rate: float

    def print(self) -> None:
        print(f"--- {self.name} ---")
        print(f"requests      : {self.total_requests}")
        print(f"concurrency   : {self.concurrency}")
        print(f"cold start    : {self.cold_start_ms:.2f} ms")
        print(f"total time    : {self.total_duration:.3f} s")
        print(f"throughput    : {self.throughput_rps:.2f} req/s")
        print(f"avg latency   : {self.average_latency_ms:.2f} ms")
        print(f"p95 latency   : {self.p95_latency_ms:.2f} ms")
        print(f"p99 latency   : {self.p99_latency_ms:.2f} ms")
        print(
            f"queue wait    : avg {self.average_queue_ms:.2f} ms | p95 {self.p95_queue_ms:.2f} ms"
        )
        print(
            f"llm latency   : avg {self.average_call_ms:.2f} ms | p95 {self.p95_call_ms:.2f} ms"
        )
        print(
            f"framework ovh : avg {self.average_processing_ms:.2f} ms | p95 {self.p95_processing_ms:.2f} ms"
        )
        print(
            f"tool exec     : avg {self.average_tool_ms:.2f} ms | p95 {self.p95_tool_ms:.2f} ms"
        )
        print(
            "framework ovh : avg {:.2f} ms | p95 {:.2f} ms (tool+llm subtracted)".format(
                self.average_framework_overhead_ms, self.p95_framework_overhead_ms
            )
        )
        print(
            f"llm total     : avg {self.average_llm_total_ms:.2f} ms | p95 {self.p95_llm_total_ms:.2f} ms"
        )
        print(
            "framework ovh : avg {:.2f} ms | p95 {:.2f} ms (corrected)".format(
                self.average_framework_overhead_corrected_ms,
                self.p95_framework_overhead_corrected_ms,
            )
        )
        print(f"cpu usage    : {self.cpu_usage_percent:.2f} %")
        print(f"peak memory  : {self.memory_peak_mb:.2f} MB")
        print(f"determinism  : {self.determinism_rate * 100.0:.2f} %")
        print(f"success count  : {self.total_success}")
        print(f"failure count  : {self.total_failure}")


def persist_result(result: BenchmarkResult, output_path: Path) -> None:
    payload = asdict(result)
    payload["total_duration"] = float(payload.get("total_duration", 0.0))

    data = {}
    if output_path.exists():
        try:
            loaded = json.loads(output_path.read_text())
            if isinstance(loaded, dict):
                data = loaded
        except json.JSONDecodeError:
            pass

    data[result.name] = payload
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True))


def trip_data_average_duration_tool(_: str | None = None) -> str:
    """Summarise the average TLC trip duration from the parquet dataset."""
    started = time.perf_counter()
    try:
        return format_average_trip_duration()
    except (TripDataNotFoundError, TripDataEmptyError) as exc:
        return f"Unable to compute trip duration: {exc}"
    except Exception as exc:  # pragma: no cover - defensive guard
        return f"Unexpected error while computing trip duration: {exc}"
    finally:
        _TOOL_TIMINGS.record(time.perf_counter() - started)


def percentile(data: List[float], perc: float) -> float:
    if not data:
        return 0.0
    percentage = max(0.0, min(1.0, perc))
    rank = max(0, min(len(data) - 1, math.ceil(percentage * len(data)) - 1))
    return data[rank]


def expected_value() -> float:
    aggregate = compute_average_trip_duration_minutes()
    return round(float(aggregate.average_minutes), 2)


def build_llm_only_prompt(request_id: int, expected: float) -> str:
    return (
        f"Request {request_id}. The average trip duration in minutes is {expected:.2f}. "
        f'Return JSON only in the format {{"value": {expected:.2f}}}.'
    )


def is_successful_result(value: float, expected: float) -> bool:
    return math.isclose(value, expected, abs_tol=SUCCESS_ABS_TOLERANCE)


class FloatResponse(BaseModel):
    value: float = Field(description="The numerical result as a float")


def extract_value(text: str) -> Optional[float]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "value" in parsed:
            return float(parsed["value"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def apply_overhead_metrics(
    tool_result: BenchmarkResult, llm_result: BenchmarkResult
) -> None:
    tool_result.average_framework_overhead_ms = max(
        0.0,
        tool_result.average_call_ms
        - llm_result.average_call_ms
        - tool_result.average_tool_ms,
    )
    tool_result.p95_framework_overhead_ms = max(
        0.0,
        tool_result.p95_call_ms - llm_result.p95_call_ms - tool_result.p95_tool_ms,
    )
    tool_result.average_framework_overhead_corrected_ms = max(
        0.0,
        tool_result.average_call_ms
        - tool_result.average_llm_total_ms
        - tool_result.average_tool_ms,
    )
    tool_result.p95_framework_overhead_corrected_ms = max(
        0.0,
        tool_result.p95_call_ms
        - tool_result.p95_llm_total_ms
        - tool_result.p95_tool_ms,
    )


async def run_langchain_benchmark(
    config: BenchmarkConfig, mode: str, expected: float
) -> BenchmarkResult:
    setup_start = time.perf_counter()
    if config.concurrency < 1:
        raise ValueError("concurrency must be greater than zero")
    if config.total_requests < 1:
        raise ValueError("total_requests must be greater than zero")
    if mode == "tool":
        _TOOL_TIMINGS.reset()
        _LLM_TIMINGS.reset()
        patch_openai_timing(_LLM_TIMINGS)

    print(
        f"Preparing LangChain benchmark ({mode}): {config.total_requests} requests with concurrency {config.concurrency}"
    )

    llm = ChatOpenAI(model=config.model)
    structured_llm = llm.with_structured_output(FloatResponse)

    if mode == "tool":
        tool = Tool.from_function(
            func=trip_data_average_duration_tool,
            name="trip_data_average_duration",
            description="Compute the average trip duration in minutes from the TLC dataset.",
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        agent = create_tool_calling_agent(llm=llm, tools=[tool], prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=False)

        async def run_call(prompt: str) -> Optional[float]:
            result = await agent_executor.ainvoke({"input": prompt})
            output = result.get("output") if isinstance(result, dict) else str(result)
            return extract_value(output)

    elif mode == "llm":

        async def run_call(prompt: str) -> Optional[float]:
            response = await structured_llm.ainvoke(prompt)
            return float(response.value)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    breakdowns: List[TimingBreakdown] = []
    sem = asyncio.Semaphore(config.concurrency)
    monitor = PerformanceMonitor()
    monitor.start()

    async def worker(request_id: int) -> None:
        if mode == "tool":
            prompt = (
                f"{config.prompt_template.format(i=request_id)}\n"
                'Return JSON only in the format {"value": <number>}.'
            )
        else:
            prompt = build_llm_only_prompt(request_id, expected)
        submitted = time.perf_counter()
        async with sem:
            dequeued = time.perf_counter()
            queue_wait = dequeued - submitted
            call_started = time.perf_counter()

            llm_request_id.set(request_id)
            value = await run_call(prompt)
            status = value is not None and is_successful_result(value, expected)

            call_duration = time.perf_counter() - call_started
            total_duration = time.perf_counter() - submitted

        breakdowns.append(
            TimingBreakdown(
                total=total_duration,
                queue_wait=queue_wait,
                call=call_duration,
                status=status,
            )
        )

    cold_start_ms = (time.perf_counter() - setup_start) * 1_000.0
    overall_started = time.perf_counter()
    tasks = [asyncio.create_task(worker(i)) for i in range(config.total_requests)]
    await asyncio.gather(*tasks)
    total_duration = time.perf_counter() - overall_started
    perf = monitor.stop()

    if mode == "tool":
        unpatch_openai_timing()

    totals = sorted(b.total for b in breakdowns)
    queue_waits = sorted(b.queue_wait for b in breakdowns)
    call_latencies = sorted(b.call for b in breakdowns)
    processing_latencies = sorted(max(0.0, b.total - b.call) for b in breakdowns)

    divisor = len(breakdowns) or 1
    avg_latency_ms = (sum(totals) / divisor) * 1_000.0
    avg_queue_ms = (sum(queue_waits) / divisor) * 1_000.0
    avg_call_ms = (sum(call_latencies) / divisor) * 1_000.0
    avg_processing_ms = (sum(processing_latencies) / divisor) * 1_000.0

    throughput_rps = len(breakdowns) / max(total_duration, 1e-9)
    success_count = sum(1 for b in breakdowns if b.status)
    failure_count = len(breakdowns) - success_count
    determinism_rate = success_count / max(1, len(breakdowns))
    tool_durations = sorted(_TOOL_TIMINGS.snapshot()) if mode == "tool" else []
    tool_divisor = len(tool_durations) or 1
    avg_tool_ms = (
        (sum(tool_durations) / tool_divisor) * 1_000.0 if tool_durations else 0.0
    )
    p95_tool_ms = percentile(tool_durations, 0.95) * 1_000.0 if tool_durations else 0.0

    if mode == "tool":
        llm_totals = sorted(_LLM_TIMINGS.snapshot_request_totals())
        llm_divisor = len(llm_totals) or 1
        avg_llm_total_ms = (
            (sum(llm_totals) / llm_divisor) * 1_000.0 if llm_totals else 0.0
        )
        p95_llm_total_ms = percentile(llm_totals, 0.95) * 1_000.0 if llm_totals else 0.0
    else:
        avg_llm_total_ms = avg_call_ms
        p95_llm_total_ms = percentile(call_latencies, 0.95) * 1_000.0

    return BenchmarkResult(
        name="LangChain",
        total_requests=len(breakdowns),
        concurrency=config.concurrency,
        total_duration=total_duration,
        throughput_rps=throughput_rps,
        average_latency_ms=avg_latency_ms,
        p95_latency_ms=percentile(totals, 0.95) * 1_000.0,
        p99_latency_ms=percentile(totals, 0.99) * 1_000.0,
        cold_start_ms=cold_start_ms,
        average_queue_ms=avg_queue_ms,
        p95_queue_ms=percentile(queue_waits, 0.95) * 1_000.0,
        average_call_ms=avg_call_ms,
        p95_call_ms=percentile(call_latencies, 0.95) * 1_000.0,
        average_processing_ms=avg_processing_ms,
        p95_processing_ms=percentile(processing_latencies, 0.95) * 1_000.0,
        average_tool_ms=avg_tool_ms,
        p95_tool_ms=p95_tool_ms,
        average_framework_overhead_ms=0.0,
        p95_framework_overhead_ms=0.0,
        average_llm_total_ms=avg_llm_total_ms,
        p95_llm_total_ms=p95_llm_total_ms,
        average_framework_overhead_corrected_ms=0.0,
        p95_framework_overhead_corrected_ms=0.0,
        total_success=success_count,
        total_failure=failure_count,
        cpu_usage_percent=perf.cpu_usage_percent,
        memory_peak_mb=perf.memory_peak_mb,
        determinism_rate=determinism_rate,
    )


def load_config() -> BenchmarkConfig:
    config_path = Path(os.getenv("BENCH_CONFIG", "benchmark.yaml"))
    try:
        raw = config_path.read_text()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Benchmark config not found at {config_path}") from exc

    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("Benchmark config must be a YAML mapping")

    try:
        total_requests = int(data["total_requests"])
        concurrency = int(data["concurrency"])
        model = str(data.get("model", "gpt-4o-mini"))
        prompt_template = str(
            data.get(
                "prompt_template",
                "Calculate the average trip duration in minutes using the available tool.",
            )
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc

    return BenchmarkConfig(
        total_requests=total_requests,
        concurrency=concurrency,
        model=model,
        prompt_template=prompt_template,
    )


async def run_langchain() -> None:
    config = load_config()
    expected = expected_value()

    llm_result = await run_langchain_benchmark(config, mode="llm", expected=expected)
    print("\n=== LangChain LLM Results ===")
    llm_result.print()
    persist_result(llm_result, Path("benchmark_results_llm.json"))

    tool_result = await run_langchain_benchmark(config, mode="tool", expected=expected)
    apply_overhead_metrics(tool_result, llm_result)
    print("\n=== LangChain Tool Results ===")
    tool_result.print()
    persist_result(tool_result, Path("benchmark_results_tool.json"))
