"""Error hierarchy for the Codex Agent SDK."""

from __future__ import annotations


class CodexSDKError(Exception):
    """Base exception for all Codex SDK errors."""


class CLINotFoundError(CodexSDKError):
    """The ``codex`` CLI binary could not be found."""

    def __init__(self, searched_paths: list[str] | None = None) -> None:
        paths = ", ".join(searched_paths) if searched_paths else "(none)"
        super().__init__(
            f"Could not find the codex CLI binary. Searched: {paths}. "
            "Install it via: npm install -g @openai/codex"
        )
        self.searched_paths = searched_paths


class CLIConnectionError(CodexSDKError):
    """Failed to communicate with the ``codex`` CLI process."""


class ProcessError(CodexSDKError):
    """The ``codex`` CLI process exited with a non-zero status."""

    def __init__(self, exit_code: int, stderr: str = "") -> None:
        msg = f"codex process exited with code {exit_code}"
        if stderr:
            msg += f": {stderr}"
        super().__init__(msg)
        self.exit_code = exit_code
        self.stderr = stderr


class CLIJSONDecodeError(CodexSDKError):
    """Failed to decode a JSONL line from the CLI."""

    def __init__(self, line: str, original_error: Exception) -> None:
        super().__init__(f"Failed to decode JSONL: {line!r} ({original_error})")
        self.line = line
        self.original_error = original_error
