"""Unit tests for the message classifier."""

import subprocess

import pytest

from graph_mem.hooks._classifier import classify_message


def test_classify_user(monkeypatch):
    """classify_message returns USER for personal info."""
    monkeypatch.setattr(
        "graph_mem.hooks._classifier.subprocess.run",
        lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0, stdout="USER"),
    )
    assert classify_message("I prefer dark mode and vim keybindings") == "USER"


def test_classify_project(monkeypatch):
    """classify_message returns PROJECT for project decisions."""
    monkeypatch.setattr(
        "graph_mem.hooks._classifier.subprocess.run",
        lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0, stdout="PROJECT"),
    )
    assert classify_message("We decided to use PostgreSQL for the backend") == "PROJECT"


def test_classify_skip(monkeypatch):
    """classify_message returns SKIP for routine messages."""
    monkeypatch.setattr(
        "graph_mem.hooks._classifier.subprocess.run",
        lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0, stdout="SKIP"),
    )
    assert classify_message("fix the bug in line 42") == "SKIP"


def test_classify_extracts_token_from_verbose_response(monkeypatch):
    """classify_message extracts the token even if the response has extra text."""
    monkeypatch.setattr(
        "graph_mem.hooks._classifier.subprocess.run",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="I think this is a USER preference"
        ),
    )
    assert classify_message("I always use TDD") == "USER"


def test_classify_returns_none_on_failure(monkeypatch):
    """classify_message returns None when Claude CLI fails."""
    monkeypatch.setattr(
        "graph_mem.hooks._classifier.subprocess.run",
        lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=1, stdout=""),
    )
    assert classify_message("some message") is None


def test_classify_returns_none_on_unknown_response(monkeypatch):
    """classify_message returns None when response doesn't contain a known token."""
    monkeypatch.setattr(
        "graph_mem.hooks._classifier.subprocess.run",
        lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0, stdout="UNKNOWN"),
    )
    assert classify_message("some message") is None


def test_classify_returns_none_on_timeout(monkeypatch):
    """classify_message returns None when Claude CLI times out."""
    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=15)

    monkeypatch.setattr("graph_mem.hooks._classifier.subprocess.run", mock_run)
    assert classify_message("some message") is None


def test_classify_returns_none_on_missing_cli(monkeypatch):
    """classify_message returns None when Claude CLI is not installed."""
    def mock_run(*args, **kwargs):
        raise FileNotFoundError("claude not found")

    monkeypatch.setattr("graph_mem.hooks._classifier.subprocess.run", mock_run)
    assert classify_message("some message") is None


def test_classify_truncates_long_messages(monkeypatch):
    """classify_message truncates messages to 2000 chars for the prompt."""
    captured_args = []

    def mock_run(args, **kwargs):
        captured_args.append(args)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="SKIP")

    monkeypatch.setattr("graph_mem.hooks._classifier.subprocess.run", mock_run)

    long_message = "a" * 5000
    classify_message(long_message)

    # The prompt passed to claude CLI should contain at most 2000 chars of the message
    prompt = captured_args[0][-1]  # last arg is the prompt
    assert "a" * 2000 in prompt
    assert "a" * 2001 not in prompt
