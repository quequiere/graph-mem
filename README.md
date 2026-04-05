# graph-mem

Persistent, intelligent memory for AI coding assistants powered by [Graphiti](https://github.com/getzep/graphiti) knowledge graphs.

## What is this?

graph-mem is an MCP server that helps your AI coding assistant learn about you over time. It captures your development patterns, project context, and preferences into a knowledge graph, then automatically resurfaces relevant context in future sessions.

Unlike flat memory stores, graph-mem leverages Graphiti's temporally-aware knowledge graph to understand **relationships** between concepts - connecting your projects, tools, preferences, and workflows into a rich, queryable network.

Works with any MCP-compatible client (Claude Code, Cursor, etc.). Claude Code users get additional automation via lifecycle hooks.

## Features

> **Work in progress** - this project is in early design phase.

### Planned

- **Automatic context injection** - Relevant memories loaded at session start based on your current project
- **Smart memory capture** - Learns when to store new information (session summaries, preferences, project insights)
- **Project onboarding** - Understands the essence of a project on first encounter, without indexing code
- **Developer profile** - Builds a persistent understanding of your habits, expertise, and preferences
- **Multi-project support** - Navigate between client projects with full context switching
- **Temporal awareness** - Tracks how your knowledge and projects evolve over time
- **Cross-machine** - Same memory accessible from any workstation

## Architecture

```
Your machine                          Docker (local or remote)
┌──────────────────────┐              ┌──────────────────────┐
│ Claude Code / Cursor  │              │  Graphiti REST API   │
│         │             │              │         │            │
│    stdio │             │    HTTP      │    graphiti_core     │
│         ▼             │ ──────────►  │         │            │
│  graph-mem MCP server │              │       Neo4j          │
└──────────────────────┘              └──────────────────────┘
```

- **graph-mem** runs locally as an MCP server (stdio transport)
- **Graphiti** runs in Docker, exposing a REST API
- Communication is standard HTTP - no JSON-RPC, works in enterprise networks

Built on top of:

- **[Graphiti](https://github.com/getzep/graphiti)** - Temporally-aware knowledge graph framework
- **[FastMCP](https://github.com/jlowin/fastmcp)** - Python MCP server framework
- **Neo4j** - Graph database backend

## Getting Started

> Coming soon - project is in implementation phase.

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for Graphiti + Neo4j)
- An OpenAI API key (used by Graphiti for entity extraction and embeddings)

### Quick Start (planned)

1. Start Graphiti:

   ```bash
   docker compose up -d
   ```

2. Install graph-mem:

   ```bash
   pip install graph-mem
   ```

3. Add to your MCP client config:

   ```json
   {
     "mcpServers": {
       "graph-mem": {
         "command": "uvx",
         "args": ["graph-mem"],
         "env": {
           "GRAPHITI_URL": "http://localhost:8000"
         }
       }
     }
   }
   ```

## Contributing

Contributions are welcome! Please see the open issues for areas where help is needed.

## License

TBD
