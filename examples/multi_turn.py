"""Multi-turn conversation example using CodexSDKClient.

Demonstrates how the client automatically resumes the thread across turns.

Usage:
    export CODEX_API_KEY="sk-..."
    python examples/multi_turn.py
"""

import asyncio

from codex_agent_sdk import (
    CodexAgentOptions,
    CodexSDKClient,
    SandboxMode,
)


async def main() -> None:
    options = CodexAgentOptions(
        sandbox=SandboxMode.WORKSPACE_WRITE,
        full_auto=True,
    )

    async with CodexSDKClient(options) as client:
        # Turn 1
        result = await client.run("Create a file called hello.txt with the content 'Hello, World!'")
        print(f"Turn 1: {result.final_response}")
        print(f"Thread ID: {client.thread_id}")

        # Turn 2 — automatically resumes the thread
        result = await client.run("Now read the file you just created and tell me its contents")
        print(f"\nTurn 2: {result.final_response}")

        # Print usage
        if result.usage:
            print(f"\nTokens — in: {result.usage.input_tokens}, out: {result.usage.output_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
