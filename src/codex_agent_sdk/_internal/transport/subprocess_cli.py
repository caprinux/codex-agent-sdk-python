"""Subprocess-based transport that spawns ``codex exec --experimental-json``."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from typing import Any, AsyncIterator, Callable, Optional

import anyio
import anyio.abc
from anyio.streams.text import TextReceiveStream

from codex_agent_sdk._errors import (
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
)
from codex_agent_sdk._internal.transport._base import Transport
from codex_agent_sdk.types import (
    CodexAgentOptions,
    ImageInput,
    TextInput,
    UserInput,
)


def _find_cli(explicit_path: Optional[str] = None) -> str:
    """Locate the ``codex`` CLI binary.

    Resolution order:
    1. Explicit path from :pyattr:`CodexAgentOptions.cli_path`
    2. ``CODEX_CLI_PATH`` environment variable
    3. ``which codex`` (i.e. on ``$PATH``)
    4. Well-known install locations
    """
    candidates: list[str] = []

    if explicit_path:
        candidates.append(explicit_path)

    env_path = os.environ.get("CODEX_CLI_PATH")
    if env_path:
        candidates.append(env_path)

    which_path = shutil.which("codex")
    if which_path:
        candidates.append(which_path)

    # Common npm global install locations
    home = os.path.expanduser("~")
    candidates.extend(
        [
            os.path.join(home, ".npm-global", "bin", "codex"),
            os.path.join(home, ".local", "bin", "codex"),
            "/usr/local/bin/codex",
        ]
    )

    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    raise CLINotFoundError(searched_paths=candidates)


class SubprocessCLITransport(Transport):
    """Spawns the Codex CLI as a child process and speaks JSONL over stdio."""

    def __init__(
        self,
        options: CodexAgentOptions,
        prompt: UserInput,
        *,
        resume_thread_id: Optional[str] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._options = options
        self._prompt = prompt
        self._resume_thread_id = resume_thread_id
        self._on_stderr = on_stderr
        self._process: Optional[anyio.abc.Process] = None

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _build_command(self) -> list[str]:
        cli = _find_cli(self._options.cli_path)
        cmd: list[str] = [cli, "exec", "--experimental-json"]

        opts = self._options

        if opts.model:
            cmd.extend(["--model", opts.model])

        if opts.sandbox:
            cmd.extend(["--sandbox", opts.sandbox.value])

        if opts.cwd:
            cmd.extend(["-C", opts.cwd])

        for d in opts.additional_writable_dirs:
            cmd.extend(["--add-dir", d])

        if opts.full_auto:
            cmd.append("--full-auto")

        if opts.skip_git_repo_check:
            cmd.append("--skip-git-repo-check")

        if opts.ephemeral:
            cmd.append("--ephemeral")

        if opts.profile:
            cmd.extend(["--profile", opts.profile])

        # Images from options
        for img in opts.images:
            cmd.extend(["--image", img])

        # Images from structured input
        if not isinstance(self._prompt, str):
            for part in self._prompt:
                if isinstance(part, ImageInput):
                    cmd.extend(["--image", part.path])

        # Output schema (file path or inline JSON string written to a temp file)
        if opts.output_schema_file:
            cmd.extend(["--output-schema", opts.output_schema_file])
        elif opts.output_schema:
            # Inline schema string — write to a temp file that the caller
            # is responsible for cleaning up.
            import tempfile

            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="codex_schema_"
            )
            tmp.write(opts.output_schema)
            tmp.close()
            self._schema_tmp_path = tmp.name
            cmd.extend(["--output-schema", tmp.name])

        # Config overrides (--config key=value)
        config = dict(opts.config_overrides)
        if opts.base_url:
            config["openai_base_url"] = opts.base_url
        if opts.reasoning_effort:
            config["model_reasoning_effort"] = f'"{opts.reasoning_effort.value}"'
        if opts.web_search:
            config["web_search"] = f'"{opts.web_search.value}"'
        if opts.approval_policy:
            config["approval_policy"] = f'"{opts.approval_policy.value}"'
        for k, v in config.items():
            cmd.extend(["--config", f"{k}={v}"])

        # Resume subcommand
        if self._resume_thread_id:
            cmd.extend(["resume", self._resume_thread_id])

        return cmd

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self._options.env:
            env.update(self._options.env)
        if self._options.api_key:
            env["CODEX_API_KEY"] = self._options.api_key
        env["CODEX_INTERNAL_ORIGINATOR_OVERRIDE"] = "codex_sdk_py"
        return env

    # ------------------------------------------------------------------
    # Transport interface
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        cmd = self._build_command()
        env = self._build_env()

        self._process = await anyio.open_process(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

    async def write(self, data: str) -> None:
        if self._process is None or self._process.stdin is None:
            raise CLIConnectionError("Transport is not connected")
        await self._process.stdin.send(data.encode())

    async def end_input(self) -> None:
        if self._process is not None and self._process.stdin is not None:
            await self._process.stdin.aclose()

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:  # type: ignore[override]
        if self._process is None or self._process.stdout is None:
            raise CLIConnectionError("Transport is not connected")

        buffer = ""
        async for chunk in TextReceiveStream(self._process.stdout):
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CLIJSONDecodeError(line, exc) from exc

        # Process remaining buffer
        if buffer.strip():
            try:
                yield json.loads(buffer.strip())
            except json.JSONDecodeError:
                pass  # Partial line at EOF — discard

    async def close(self) -> None:
        if self._process is None:
            return

        # Collect stderr with a timeout to avoid hanging if the process
        # never closes its stderr fd.
        stderr_text = ""
        if self._process.stderr is not None:
            try:
                with anyio.fail_after(5):
                    async for chunk in TextReceiveStream(self._process.stderr):
                        stderr_text += chunk
                        if self._on_stderr:
                            self._on_stderr(chunk)
            except (TimeoutError, anyio.ClosedResourceError):
                pass

        # Give the process a moment to exit gracefully
        try:
            with anyio.fail_after(5):
                await self._process.wait()
        except TimeoutError:
            try:
                self._process.terminate()
                with anyio.fail_after(3):
                    await self._process.wait()
            except (TimeoutError, ProcessLookupError):
                try:
                    self._process.kill()
                    await self._process.wait()
                except ProcessLookupError:
                    pass

        # Clean up temp schema file if we created one
        schema_tmp = getattr(self, "_schema_tmp_path", None)
        if schema_tmp:
            try:
                os.unlink(schema_tmp)
            except OSError:
                pass

        exit_code = self._process.returncode
        if exit_code is not None and exit_code not in (0, -signal.SIGTERM):
            raise ProcessError(exit_code, stderr_text)

    def is_ready(self) -> bool:
        return self._process is not None and self._process.returncode is None
