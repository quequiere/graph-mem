"""Unit tests for the UserPromptSubmit hook."""

import json
import subprocess
import sys

import pytest

from graph_mem.hooks.user_prompt import main


def test_spawns_worker_for_valid_message(monkeypatch):
    """main() spawns _ingest_worker for a valid message."""
    spawned = []

    monkeypatch.setattr("sys.stdin", _make_stdin({"prompt": "I prefer TDD for all projects", "cwd": "/tmp"}))
    monkeypatch.setattr(
        "graph_mem.hooks.user_prompt.subprocess.Popen",
        lambda args, **kw: spawned.append(args),
    )

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0

    assert len(spawned) == 1
    assert "graph_mem.hooks._ingest_worker" in spawned[0][2]


def test_skips_short_messages(monkeypatch):
    """main() exits without spawning for messages under 10 chars."""
    spawned = []

    monkeypatch.setattr("sys.stdin", _make_stdin({"prompt": "ok", "cwd": "/tmp"}))
    monkeypatch.setattr(
        "graph_mem.hooks.user_prompt.subprocess.Popen",
        lambda args, **kw: spawned.append(args),
    )

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert len(spawned) == 0


def test_skips_empty_prompt(monkeypatch):
    """main() exits without spawning when prompt is empty."""
    spawned = []

    monkeypatch.setattr("sys.stdin", _make_stdin({"prompt": "", "cwd": "/tmp"}))
    monkeypatch.setattr(
        "graph_mem.hooks.user_prompt.subprocess.Popen",
        lambda args, **kw: spawned.append(args),
    )

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert len(spawned) == 0


def test_skips_missing_prompt(monkeypatch):
    """main() exits without spawning when prompt key is missing."""
    spawned = []

    monkeypatch.setattr("sys.stdin", _make_stdin({"cwd": "/tmp"}))
    monkeypatch.setattr(
        "graph_mem.hooks.user_prompt.subprocess.Popen",
        lambda args, **kw: spawned.append(args),
    )

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert len(spawned) == 0


def test_truncates_long_messages(monkeypatch):
    """main() truncates messages to 8000 chars for argv safety."""
    spawned = []

    long_msg = "x" * 10000
    monkeypatch.setattr("sys.stdin", _make_stdin({"prompt": long_msg, "cwd": "/tmp"}))
    monkeypatch.setattr(
        "graph_mem.hooks.user_prompt.subprocess.Popen",
        lambda args, **kw: spawned.append(args),
    )

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0

    assert len(spawned) == 1
    # Last arg is the truncated message
    assert len(spawned[0][-1]) == 8000


def test_skips_empty_stdin(monkeypatch):
    """main() exits gracefully when stdin is empty."""
    spawned = []

    monkeypatch.setattr("sys.stdin", _make_stdin_raw(""))
    monkeypatch.setattr(
        "graph_mem.hooks.user_prompt.subprocess.Popen",
        lambda args, **kw: spawned.append(args),
    )

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert len(spawned) == 0


# --- Helpers ---

class _FakeStdin:
    """Fake stdin that returns preset data."""
    def __init__(self, data: str):
        self._data = data

    def read(self):
        return self._data

    def isatty(self):
        return False


def _make_stdin(hook_input: dict) -> _FakeStdin:
    return _FakeStdin(json.dumps(hook_input))


def _make_stdin_raw(data: str) -> _FakeStdin:
    return _FakeStdin(data)
