"""Integration test fixtures: Docker stack + MCP client session.

By default, Graphiti connects to the host's local Ollama instance
(via host.docker.internal:11434). To use a dockerized Ollama instead,
run: docker compose --profile ollama up -d
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

COMPOSE_FILE = str(Path(__file__).parent / "docker-compose.yml")
PROJECT_NAME = "graphmem-integration"


def _get_host_port(service: str, container_port: int) -> int | None:
    """Get the host-mapped port via docker inspect.

    Uses docker inspect rather than `docker compose port` because Docker Compose
    v2.5.1 has a naming bug where it looks for `service_1` instead of `service-1`.
    """
    result = subprocess.run(
        ["docker", "inspect", f"{PROJECT_NAME}-{service}-1"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    data = json.loads(result.stdout)
    if not data:
        return None
    ports = data[0].get("NetworkSettings", {}).get("Ports", {})
    bindings = ports.get(f"{container_port}/tcp", [])
    if bindings:
        return int(bindings[0]["HostPort"])
    return None


@pytest.fixture(scope="session")
def docker_compose_file():
    return COMPOSE_FILE


@pytest.fixture(scope="session")
def docker_compose_project_name():
    return PROJECT_NAME


@pytest.fixture(scope="session")
def docker_setup():
    """Start containers in background; graphiti_url handles readiness polling."""
    return ["up --build -d"]


@pytest.fixture(scope="session")
def docker_cleanup():
    # Preserve the ollama_data volume — models are ~4GB, re-downloading is expensive.
    return ["down"]


@pytest.fixture(scope="session")
def graphiti_url(docker_services):
    """Wait for Graphiti to become healthy and return its base URL.

    `docker_services` is injected to trigger pytest-docker's docker_setup.
    We then poll via docker inspect + HTTP instead of docker_services.port_for()
    because Docker Compose v2.5.1 has a v1/v2 container naming incompatibility.
    """
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        port = _get_host_port("graphiti", 8000)
        if port:
            # Use 127.0.0.1 explicitly — on Windows, 'localhost' may resolve to
            # IPv6 ::1 but Docker only listens on IPv4 0.0.0.0.
            url = f"http://127.0.0.1:{port}"
            try:
                result = subprocess.run(
                    ["curl", "-sf", "--max-time", "5", f"{url}/healthcheck"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and "healthy" in result.stdout:
                    return url
            except Exception:
                pass
        time.sleep(5)
    raise RuntimeError("Graphiti did not become responsive within 300s")


def _make_server_params(graphiti_url: str):
    """Build StdioServerParameters for the graph-mem MCP server."""
    from mcp import StdioServerParameters

    env = {
        "GRAPHITI_URL": graphiti_url,
        "PATH": os.environ.get("PATH", ""),
    }
    if sys.platform == "win32":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        env["PATHEXT"] = os.environ.get("PATHEXT", "")
    if os.environ.get("VIRTUAL_ENV"):
        env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]

    return StdioServerParameters(command="graph-mem", env=env)


@pytest.fixture()
async def mcp_session(graphiti_url):
    """Launch graph-mem MCP server as a stdio subprocess and yield a ClientSession.

    Function-scoped so each test gets a fresh session on its own event loop,
    avoiding anyio cancel-scope cross-task errors that occur with session scope.
    """
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    server_params = _make_server_params(graphiti_url)
    ctx = stdio_client(server_params)

    try:
        read_stream, write_stream = await ctx.__aenter__()
    except Exception as e:
        raise RuntimeError(f"Failed to start graph-mem MCP server: {e}") from e

    session_ctx = ClientSession(read_stream, write_stream)
    try:
        session = await session_ctx.__aenter__()
        await session.initialize()
        yield session
    finally:
        # Suppress the anyio cancel-scope teardown error: it's a known
        # pytest-asyncio + anyio incompatibility that doesn't affect test results.
        try:
            await session_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            await ctx.__aexit__(None, None, None)
        except Exception:
            pass


@pytest.fixture(autouse=True)
async def _clean(mcp_session):
    """Reset test groups before each test (Neo4j data persists across MCP sessions)."""
    try:
        await asyncio.wait_for(
            mcp_session.call_tool(
                "reset_memory",
                arguments={"group_ids": ["user_profile", "project_test-project"]},
            ),
            timeout=60,
        )
    except Exception:
        pass  # ignore if groups don't exist yet
