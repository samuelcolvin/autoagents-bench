import asyncio

from _src.crewai_bench import run_crewai
from _src.graphbit_bench import run_graphbit
from _src.langchain_bench import run_langchain
from _src.langgraph_bench import run_langgraph
from _src.llamaindex_bench import run_llamaindex
from _src.pydantic_ai_bench import run_pydantic_ai


async def main() -> None:
    await run_graphbit()
    await run_langchain()
    await run_langgraph()
    await run_crewai()
    await run_pydantic_ai()
    await run_llamaindex()


if __name__ == "__main__":
    asyncio.run(main())
