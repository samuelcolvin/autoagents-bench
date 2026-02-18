import argparse
import asyncio
from pathlib import Path

import _src.graphbit_bench as graphbit_mod
import _src.langchain_bench as langchain_mod
import _src.langgraph_bench as langgraph_mod
import _src.llamaindex_bench as llamaindex_mod
import _src.pydantic_ai_bench as pydantic_ai_mod

_BENCHMARKS = {
    "graphbit": (graphbit_mod, graphbit_mod.run_graphbit_benchmark, "GraphBit"),
    "langchain": (langchain_mod, langchain_mod.run_langchain_benchmark, "LangChain"),
    "langgraph": (langgraph_mod, langgraph_mod.run_langgraph_benchmark, "LangGraph"),
    "pydantic_ai": (
        pydantic_ai_mod,
        pydantic_ai_mod.run_pydantic_ai_benchmark,
        "PydanticAI",
    ),
    "llamaindex": (
        llamaindex_mod,
        llamaindex_mod.run_llamaindex_benchmark,
        "LlamaIndex",
    ),
}


async def run_single(name: str, mode: str) -> None:
    mod, runner, label = _BENCHMARKS[name]
    config = mod.load_config()
    expected = mod.expected_value()

    llm_result = None
    if mode in ("llm", "both"):
        llm_result = await runner(config, mode="llm", expected=expected)
        print(f"\n=== {label} LLM Results ===")
        llm_result.print()
        mod.persist_result(llm_result, Path("benchmark_results_llm.json"))

    if mode in ("tool", "both"):
        tool_result = await runner(config, mode="tool", expected=expected)
        if llm_result is not None:
            mod.apply_overhead_metrics(tool_result, llm_result)
        print(f"\n=== {label} Tool Results ===")
        tool_result.print()
        mod.persist_result(tool_result, Path("benchmark_results_tool.json"))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent framework benchmarks")
    parser.add_argument(
        "benchmarks",
        nargs="*",
        choices=list(_BENCHMARKS.keys()),
        metavar="BENCHMARK",
        help=(
            f"One or more benchmarks to run: {', '.join(_BENCHMARKS)}. "
            "Runs all if not specified."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["llm", "tool", "both"],
        default="both",
        help="Benchmark mode to run (default: both)",
    )
    args = parser.parse_args()

    selected = args.benchmarks if args.benchmarks else list(_BENCHMARKS.keys())

    for name in selected:
        await run_single(name, args.mode)


if __name__ == "__main__":
    asyncio.run(main())
