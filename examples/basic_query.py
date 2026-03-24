"""Basic one-shot query example.

Runs a single prompt through the Codex agent and prints the response.

Usage:
    export CODEX_API_KEY="sk-..."  # or OPENAI_API_KEY
    python examples/basic_query.py
"""

import asyncio

from codex_agent_sdk import (
    AgentMessageItem,
    CodexAgentOptions,
    CommandExecutionItem,
    ItemCompletedEvent,
    SandboxMode,
    query,
)


async def main() -> None:
    options = CodexAgentOptions(
        sandbox=SandboxMode.READ_ONLY,
    )

    async for event in query("What files are in the current directory?", options=options):
        if isinstance(event, ItemCompletedEvent):
            item = event.item
            if isinstance(item, CommandExecutionItem):
                print(f"[command] {item.command}")
                print(item.aggregated_output)
            elif isinstance(item, AgentMessageItem):
                print(f"\n{item.text}")


if __name__ == "__main__":
    asyncio.run(main())
