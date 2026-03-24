"""Structured output example using an output schema.

Demonstrates how to get structured JSON output from the Codex agent.

Usage:
    export CODEX_API_KEY="sk-..."
    python examples/structured_output.py
"""

import asyncio
import json
import tempfile
from pathlib import Path

from codex_agent_sdk import (
    AgentMessageItem,
    CodexAgentOptions,
    CodexSDKClient,
    ItemCompletedEvent,
    SandboxMode,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "description"],
            },
        },
    },
    "required": ["files"],
}


async def main() -> None:
    # Write schema to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SCHEMA, f)
        schema_path = f.name

    try:
        options = CodexAgentOptions(
            sandbox=SandboxMode.READ_ONLY,
            output_schema_file=schema_path,
        )

        client = CodexSDKClient(options)
        result = await client.run("List the files in the current directory with descriptions")

        if result.final_response:
            parsed = json.loads(result.final_response)
            for file_info in parsed.get("files", []):
                print(f"  {file_info['name']}: {file_info['description']}")
    finally:
        Path(schema_path).unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
