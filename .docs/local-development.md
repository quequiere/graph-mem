# Local development

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker & Docker Compose.

## 1. Clone and start Graphiti + Neo4j

```bash
git clone https://github.com/quequiere/graph-mem && cd graph-mem
cp .env.example .env
# Edit .env — set OPENAI_API_KEY (OpenRouter, Ollama, or any OpenAI-compatible provider)
docker compose -f docker-compose.dev.yml up -d --build
```

Verify Graphiti is healthy:

```bash
docker compose ps
curl http://127.0.0.1:8000/healthcheck
```

## 2. Run tests

```bash
# Unit tests (no Docker needed)
uv run pytest tests/ -v

# Integration tests (requires Docker running)
uv run pytest tests/integration/ -v -m integration
```

## 3. Configure Claude Code MCP server

Register graph-mem as an MCP server using `uv run` so changes are picked up immediately without reinstalling:

```bash
claude mcp add graph-mem -s user -- uv run --directory /path/to/graph-mem graph-mem
```

## 4. Configure Claude Code hooks

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|clear|compact",
        "hooks": [
          {
            "type": "command",
            "command": "uv run --directory /path/to/graph-mem graph-mem-session-start",
            "timeout": 10
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "uv run --directory /path/to/graph-mem graph-mem-session-end"
          }
        ]
      }
    ]
  }
}
```

Replace `/path/to/graph-mem` with your clone's absolute path.

## 5. Configure VS Code MCP server (optional)

Add to `.vscode/mcp.json` in any project where you want graph-mem available:

```json
{
  "servers": {
    "graph-mem": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/graph-mem", "graph-mem"],
      "type": "stdio"
    }
  }
}
```

## 6. Interactive debugging with MCP Inspector

Test MCP tools interactively without Claude Code:

```bash
npx @modelcontextprotocol/inspector uv run --directory /path/to/graph-mem graph-mem
```

This opens a web UI where you can call `save_memory`, `search_memory`, etc. and inspect responses.
