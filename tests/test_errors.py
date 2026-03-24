"""Tests for the error hierarchy."""

import json

from codex_agent_sdk._errors import (
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    CodexSDKError,
    ProcessError,
)


class TestErrorHierarchy:
    def test_base_class(self) -> None:
        err = CodexSDKError("test error")
        assert str(err) == "test error"
        assert isinstance(err, Exception)

    def test_all_errors_inherit_from_base(self) -> None:
        assert issubclass(CLINotFoundError, CodexSDKError)
        assert issubclass(CLIConnectionError, CodexSDKError)
        assert issubclass(ProcessError, CodexSDKError)
        assert issubclass(CLIJSONDecodeError, CodexSDKError)


class TestCLINotFoundError:
    def test_with_paths(self) -> None:
        err = CLINotFoundError(searched_paths=["/usr/bin/codex", "/usr/local/bin/codex"])
        assert "/usr/bin/codex" in str(err)
        assert "/usr/local/bin/codex" in str(err)
        assert "npm install -g @openai/codex" in str(err)
        assert err.searched_paths == ["/usr/bin/codex", "/usr/local/bin/codex"]

    def test_without_paths(self) -> None:
        err = CLINotFoundError()
        assert "(none)" in str(err)
        assert err.searched_paths is None


class TestCLIConnectionError:
    def test_message(self) -> None:
        err = CLIConnectionError("not connected")
        assert str(err) == "not connected"


class TestProcessError:
    def test_with_stderr(self) -> None:
        err = ProcessError(1, "segfault")
        assert err.exit_code == 1
        assert err.stderr == "segfault"
        assert "code 1" in str(err)
        assert "segfault" in str(err)

    def test_without_stderr(self) -> None:
        err = ProcessError(2)
        assert err.exit_code == 2
        assert err.stderr == ""
        assert "code 2" in str(err)


class TestCLIJSONDecodeError:
    def test_preserves_line_and_original(self) -> None:
        original = json.JSONDecodeError("bad", "doc", 0)
        err = CLIJSONDecodeError("not json", original)
        assert err.line == "not json"
        assert err.original_error is original
        assert "not json" in str(err)
