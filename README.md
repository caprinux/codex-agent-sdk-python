# Codex Agent SDK (Python)

Python SDK for the [OpenAI Codex CLI](https://github.com/openai/codex) agent. Wraps the Codex CLI as a programmable Python library — spawn coding agent sessions, stream events, and build multi-turn workflows.

Follows the same architectural pattern as [claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python).

## Installation

```bash
pip install git+https://github.com/caprinux/codex-agent-sdk-python.git
```

**Prerequisites:** The [Codex CLI](https://github.com/openai/codex) must be installed and available on your `$PATH`:

```bash
npm install -g @openai/codex
```

## Quick Start

### One-shot query

```python
import asyncio
from codex_agent_sdk import query, ItemCompletedEvent, AgentMessageItem

async def main():
    async for event in query("What files are in this directory?"):
        if isinstance(event, ItemCompletedEvent):
            if isinstance(event.item, AgentMessageItem):
                print(event.item.text)

asyncio.run(main())
```

### Multi-turn conversation

```python
import asyncio
from codex_agent_sdk import CodexSDKClient, CodexAgentOptions, SandboxMode

async def main():
    options = CodexAgentOptions(
        sandbox=SandboxMode.WORKSPACE_WRITE,
        full_auto=True,
    )

    async with CodexSDKClient(options) as client:
        result = await client.run("Create a hello.py file")
        print(result.final_response)

        result = await client.run("Now add type hints to it")
        print(result.final_response)

asyncio.run(main())
```

### Streaming events

```python
import asyncio
from codex_agent_sdk import (
    CodexSDKClient, CodexAgentOptions, SandboxMode,
    ItemStartedEvent, ItemCompletedEvent,
    CommandExecutionItem, AgentMessageItem,
)

async def main():
    client = CodexSDKClient(CodexAgentOptions(sandbox=SandboxMode.READ_ONLY))

    async for event in client.run_streamed("Explain this codebase"):
        match event:
            case ItemStartedEvent(item=CommandExecutionItem(command=cmd)):
                print(f"Running: {cmd}")
            case ItemCompletedEvent(item=AgentMessageItem(text=text)):
                print(text)

asyncio.run(main())
```

## API Reference

### `query(prompt, *, options=None, resume_thread_id=None)`

One-shot async generator. Spawns a Codex CLI process, sends the prompt, and yields `ThreadEvent` objects.

### `CodexSDKClient(options=None)`

Multi-turn client. Automatically captures the `thread_id` and resumes the session on subsequent calls.

| Method | Description |
|---|---|
| `await client.run(prompt)` | Execute one turn, return a `TurnResult` |
| `async for event in client.run_streamed(prompt)` | Execute one turn, yield events |
| `client.thread_id` | The current session's thread ID |

### `CodexAgentOptions`

| Field | Type | Description |
|---|---|---|
| `model` | `str \| None` | Model to use (e.g. `"o4-mini"`) |
| `sandbox` | `SandboxMode \| None` | `READ_ONLY`, `WORKSPACE_WRITE`, or `FULL_ACCESS` |
| `approval_policy` | `ApprovalPolicy \| None` | `NEVER`, `ON_REQUEST`, `ON_FAILURE`, `UNTRUSTED` |
| `full_auto` | `bool` | Shorthand for workspace-write + on-request approval |
| `cwd` | `str \| None` | Working directory for the agent |
| `api_key` | `str \| None` | OpenAI API key (or set `CODEX_API_KEY` / `OPENAI_API_KEY`) |
| `base_url` | `str \| None` | Custom API base URL |
| `reasoning_effort` | `ReasoningEffort \| None` | `MINIMAL` through `XHIGH` |
| `web_search` | `WebSearch \| None` | `DISABLED`, `CACHED`, or `LIVE` |
| `images` | `list[str]` | Image file paths to attach |
| `output_schema_file` | `str \| None` | JSON Schema file for structured output |
| `ephemeral` | `bool` | Don't persist the session |
| `cli_path` | `str \| None` | Explicit path to the `codex` binary |
| `env` | `dict \| None` | Extra environment variables for the CLI process |

### Event Types

| Event | Description |
|---|---|
| `ThreadStartedEvent` | Carries `thread_id` for session resumption |
| `TurnStartedEvent` | Agent begins processing |
| `ItemStartedEvent` | New item (command, message, etc.) begins |
| `ItemUpdatedEvent` | Item state updated |
| `ItemCompletedEvent` | Item reached terminal state |
| `TurnCompletedEvent` | Turn finished; carries `Usage` |
| `TurnFailedEvent` | Turn failed; carries error message |
| `StreamErrorEvent` | Unrecoverable stream error |

### Item Types

| Item | Key Fields |
|---|---|
| `AgentMessageItem` | `text` |
| `ReasoningItem` | `text` |
| `CommandExecutionItem` | `command`, `aggregated_output`, `exit_code`, `status` |
| `FileChangeItem` | `changes` (list of `FileChange`), `status` |
| `McpToolCallItem` | `server`, `tool`, `arguments`, `result`, `error`, `status` |
| `WebSearchItem` | `query` |
| `TodoListItem` | `items` (list of `TodoItem`) |
| `ErrorItem` | `message` |

## Architecture

```
codex_agent_sdk/
├── __init__.py              # Public API exports
├── query.py                 # query() one-shot async generator
├── client.py                # CodexSDKClient multi-turn client
├── types.py                 # All type definitions (events, items, options)
├── _errors.py               # Error hierarchy
├── _version.py              # Package version
└── _internal/
    ├── client.py            # InternalClient (bridge to transport)
    ├── query.py             # Query (turn lifecycle manager)
    ├── message_parser.py    # JSONL → typed events
    └── transport/
        ├── _base.py         # Transport ABC
        └── subprocess_cli.py # SubprocessCLITransport
```

The SDK spawns `codex exec --experimental-json` as a subprocess, writes the prompt to stdin, and reads JSONL events from stdout. Each turn is a separate process; multi-turn conversations use `codex exec resume <thread_id>`.

## License

MIT
