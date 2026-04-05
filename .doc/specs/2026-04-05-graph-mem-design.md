# graph-mem Design Spec

> Date: 2026-04-05
> Status: Validated

## 1. Overview

graph-mem is a Claude Code plugin that provides persistent, intelligent memory for developers using Graphiti's temporally-aware knowledge graph. It learns about the developer over time - their preferences, expertise, projects, colleagues, and workflows - and automatically resurfaces relevant context in future sessions.

### Goals

- Build a rich developer profile across sessions and machines
- Maintain per-project context (team, stack, decisions, blockers)
- Automatically inject relevant context at session start
- Detect new projects and offer onboarding
- Work with any CLI that supports MCP (Claude Code, Cursor, etc.)
- Claude Code hooks for automatic capture; skill for guided usage

### Non-Goals

- No code indexing or source code storage
- No privacy/tag system (future consideration)
- No intermediate worker service - keep it simple

## 2. Architecture

### Distribution Model

```
[Machine du dev]
  Claude Code / Cursor / any MCP client
      |
      └── stdio --> [graph-mem MCP Server (local, installed via pip/uvx)]
                        |
                        └── REST (httpx) --> [Graphiti REST API (Docker)]
                                                 └── [Neo4j]
```

**graph-mem MCP server (local):**
- Installed locally via `pip install graph-mem` or `uvx`
- Launched by the MCP client via **stdio** transport (standard MCP local)
- Exposes all our tools (custom + passthrough)
- Calls Graphiti via **standard REST/HTTP** (not MCP JSON-RPC)
- Config: `GRAPHITI_URL` (e.g., `http://localhost:8000`) + optional credentials

**Graphiti REST server (remote or local):**
- Deployed via official Docker image `zepai/graphiti:latest` + Neo4j
- `docker compose up` for local dev, or hosted on cloud/VPS for remote access
- Exposes REST API with Swagger docs at `/docs`
- No JSON-RPC, no MCP protocol - standard HTTP that works in enterprise environments

This separation means:
- Enterprise networks that block JSON-RPC can use graph-mem (pure REST over HTTP)
- The dev only provides `GRAPHITI_URL` to connect
- Graphiti can be shared across machines (same URL from any workstation)

### Graphiti REST API Endpoints Used

| Endpoint | Method | Purpose |
|---|---|---|
| `/messages` | POST | Add episodes (memories) to the graph |
| `/search` | POST | Search for facts (relationships) by query |
| `/get-memory` | POST | Search memories using conversational context |
| `/entity-node` | POST | Create an entity node directly |
| `/group/{group_id}` | DELETE | Delete all data for a group |
| `/healthcheck` | GET | Health check |

### Key Decisions

- **Language**: Python
- **Transport**: Our MCP server runs locally via stdio; calls Graphiti via REST/HTTP
- **No MCP-to-MCP**: We don't use MCP client SDK to call Graphiti - pure HTTP with httpx
- **No graphiti_core dependency**: We only depend on Graphiti's REST API, not its Python library
- **Multi-CLI compatible**: MCP tools usable by any client; hooks are Claude Code specific
- **Embeddings**: Handled by Graphiti server-side - we send text, Graphiti handles the rest

### Project Structure

```
graph-mem/
├── src/
│   └── graph_mem/
│       ├── server.py              # Our MCP server (FastMCP, stdio)
│       ├── client.py              # HTTP client for Graphiti REST API (httpx)
│       ├── tools/
│       │   ├── context.py         # get_context
│       │   ├── onboard.py         # onboard_project, check_project
│       │   ├── memory.py          # save_session, save_memory
│       │   ├── profile.py         # get_profile
│       │   ├── reminders.py       # add_reminder, get_reminders
│       │   └── passthrough.py     # Re-exposed Graphiti tools
│       ├── hooks/
│       │   ├── session_start.py   # SessionStart hook for Claude Code
│       │   └── session_end.py     # SessionEnd (Stop) hook for Claude Code
│       ├── utils/
│       │   └── project_id.py      # Git remote URL -> project identifier
│       └── config.py
├── skills/
│   └── graph-mem/
│       └── SKILL.md               # Skill for guiding CLI agents
├── pyproject.toml
└── README.md
```

## 3. Group ID Strategy

Hybrid approach with two group ID scopes:

### `user_profile` (global, persists across all projects and machines)

Contains the developer's identity, preferences, expertise, habits, and a lightweight reference to all active projects.

### `project_{identifier}` (per-project, detailed context)

Contains project-specific information: team, stack, decisions, sessions, blockers.

### Project Identifier

Derived from git remote origin URL, normalized:
1. If remote exists: normalize URL (e.g., `github.com/user/repo` - strip protocol, `.git` suffix)
2. If no remote: fallback to directory name with a warning (not portable across machines)

This ensures the same project is recognized across different machines when Graphiti is deployed remotely.

### Query Strategy

At session start, query both:
1. `user_profile` for global context (who is the dev, preferences, active projects)
2. `project_{id}` for current project specifics (team, stack, recent sessions, blockers)

Results are merged into a single context injection.

## 4. Entity Types

### user_profile entities

| Entity Type | Description | Example |
|---|---|---|
| **Developer** | The user themselves, identity and global expertise | "quequ - senior TypeScript, learning Rust" |
| **Project** | Lightweight reference to projects the dev works on | "graph-mem - Claude Code memory plugin" |
| **Technology** | Languages, frameworks, tools mastered globally | "Python", "Neo4j", "Docker" |
| **Preference** | General work preferences | "prefers pnpm over npm", "dark mode always" |
| **Workflow** | Recurring work patterns and habits | "uses TDD", "reviews PRs before merge" |
| **Principle** | Personal technical convictions | "YAGNI first", "no mocks in integration tests" |
| **Learning** | What the dev is learning or wants to learn | "learning Rust", "interested in WebAssembly" |
| **Interest** | Passions beyond current work | "passionate about knowledge graphs", "follows AI news" |
| **Schedule** | Work rhythm, availability | "morning person, available 9h-18h" |
| **Environment** | Machines, OS, dev configurations | "PC Windows 11", "MacBook Pro M3" |
| **Achievement** | Personal milestones, career accomplishments | "first open source project published" |
| **Frustration** | Recurring global frictions | "Docker slow on Windows", "npm install too slow" |
| **Reminder** | Things to not forget | "renew OpenAI API key before April 15" |
| **Organization** | Companies, teams the dev is part of | "Freelance", "Acme Corp client" |

### project_{id} entities

| Entity Type | Description | Example |
|---|---|---|
| **Project** | The project in detail - objectives, description, state | "graph-mem - memory plugin using Graphiti knowledge graph" |
| **Colleague** | People involved in the project | "Marc - tech lead", "Sarah - PM client side" |
| **Organization** | Company, team, client tied to the project | "Acme Corp - client", "Team Platform" |
| **Technology** | Project-specific tech stack | "React", "Node.js", "PostgreSQL" |
| **Decision** | Technical decisions made | "chose FastMCP, no intermediate worker" |
| **Goal** | Current objectives | "implement hooks system", "ship v1 by end of month" |
| **Blocker** | Problems, tech debt, issues | "SQL query performance issue", "flaky CI" |
| **Session** | Work session summaries | "2026-04-05: designed entity types and MCP tools" |
| **Reminder** | Project-specific reminders | "update dependencies before release" |
| **Frustration** | Project-specific frictions | "slow build times", "unclear API docs" |
| **Achievement** | Project milestones | "v1 deployed to production", "first user onboarded" |
| **Question** | Open questions about the project | "should we switch from Neo4j to FalkorDB?" |
| **Context** | Situational context | "deadline Friday", "POC phase" |
| **Mindset** | Current work mode on this project | "exploration mode", "delivery mode" |

### How entity extraction works

When text is sent to Graphiti via `add_memory`, Graphiti's LLM automatically:
1. Identifies entities in the text and maps them to configured entity types
2. Extracts relationships (facts) between entities
3. Deduplicates against existing entities in the graph
4. Stores everything with temporal metadata

Example: "Julie is the tech lead on Project Atlas, they use FastAPI and have latency issues with MongoDB"

Produces:
```
[Colleague: Julie] --role_in--> [Project: Atlas] (fact: "tech lead")
[Project: Atlas] --uses--> [Technology: FastAPI]
[Project: Atlas] --uses--> [Technology: MongoDB]
[Blocker: MongoDB latency] --affects--> [Project: Atlas]
```

## 5. MCP Tools

### Custom tools (our business logic)

#### `get_context`
Retrieves the full context for the current session. Queries `user_profile` + `project_{id}`.

| Param | Type | Description |
|---|---|---|
| `project_path` | str, optional | Path to project root. Defaults to cwd. |

Returns: Merged context (developer profile + project specifics + active reminders).

#### `onboard_project`
Analyzes a project and stores its essence in Graphiti. Reads README, manifests (package.json, pyproject.toml, Cargo.toml, etc.), and directory structure summary.

| Param | Type | Description |
|---|---|---|
| `project_path` | str, optional | Path to project root. Defaults to cwd. |
| `description` | str, optional | Developer's own description of the project (overrides/supplements auto-analysis). |

Stores results in both `project_{id}` (detailed) and `user_profile` (lightweight reference).

#### `save_session`
Sends a session summary to Graphiti for entity/relation extraction.

| Param | Type | Description |
|---|---|---|
| `summary` | str | Session summary text. |
| `project_path` | str, optional | Path to project root. Defaults to cwd. |

Stores in `project_{id}`. Also updates `user_profile` if new cross-project info is detected.

#### `save_memory`
Stores a specific piece of information at any time. For when the dev says "remember that..."

| Param | Type | Description |
|---|---|---|
| `content` | str | The information to store. |
| `group_id` | str, optional | Target group. If omitted, the skill instructs the agent to choose: personal info (preferences, expertise, habits) goes to `user_profile`, project-specific info goes to `project_{id}` derived from cwd. The agent must always provide a group_id explicitly. |

#### `get_profile`
Retrieves the complete developer profile (preferences, expertise, active projects, principles).

No parameters. Queries `user_profile`.

#### `check_project`
Checks if a project is known in the graph. Used by SessionStart hook to detect new projects.

| Param | Type | Description |
|---|---|---|
| `project_path` | str, optional | Path to project root. Defaults to cwd. |

Returns: Whether the project is known + basic info if it is.

#### `add_reminder`
Creates a reminder (Reminder entity).

| Param | Type | Description |
|---|---|---|
| `content` | str | What to remember. |
| `group_id` | str, optional | `user_profile` for personal, `project_{id}` for project-specific. |

#### `get_reminders`
Lists active reminders.

| Param | Type | Description |
|---|---|---|
| `group_id` | str, optional | Filter by group. If omitted, returns all reminders. |

### Graphiti passthrough tools (re-interfaced)

| Our name | Original Graphiti tool | Description |
|---|---|---|
| `add_raw_memory` | `add_memory` | Low-level: add an episode directly to the graph |
| `search_entities` | `search_nodes` | Search for entities by semantic query, with group_id and entity_type filters |
| `search_facts` | `search_memory_facts` | Search for relationships between entities |
| `reset_memory` | `clear_graph` | Purge all data for given group_ids. Dangerous, use with caution. |
| `status` | `get_status` | Check if Graphiti is running and healthy |

### Removed Graphiti tools (too low-level)

- `get_entity_edge` - Requires UUID, no practical CLI use case
- `get_episodes` - Covered by Session entities
- `delete_entity_edge` - Requires UUID, dangerous
- `delete_episode` - Requires UUID, dangerous

## 6. Claude Code Hooks

### SessionStart

Executed when a Claude Code session begins.

```
1. Detect project from cwd (git remote URL → project identifier)
2. check_project() → is this project known?
   - If unknown → inject message: "New project detected. Use onboard_project to set it up."
3. get_context() → query user_profile + project_{id}
   - Inject: developer preferences, principles, expertise
   - Inject: project context (stack, team, recent decisions, blockers)
4. get_reminders() → inject active reminders
```

Output is injected into the agent's context via `hookSpecificOutput.additionalContext`.

### SessionEnd (Stop hook)

Executed when a Claude Code session ends.

```
1. The Claude Code LLM generates a session summary (built into the hook prompt)
2. save_session(summary) → sends to Graphiti for extraction
   - Graphiti extracts: new entities, new relations, updated facts
   - Stored in project_{id}
3. If cross-project info detected → also update user_profile
```

## 7. Skill (SKILL.md)

A distributable skill that guides CLI agents in their usage of graph-mem.

### When to store memory

**STORE:**
- Developer expresses a preference ("I prefer X", "I don't like Y")
- Developer shares personal info (expertise, role, habits)
- Developer briefs a project (stack, team, objectives, problems)
- Developer makes a technical decision
- Developer says "remember that...", "remind me..."
- End of session: always summarize

**DO NOT STORE:**
- Source code
- Secrets, tokens, passwords
- Ephemeral details ("run npm install")
- Info derivable from code (file structure, etc.)

### How to choose group_id

- Info about the developer themselves (preference, expertise, habit) -> `user_profile`
- Info about a specific project -> `project_{id}`
- When in doubt -> `user_profile`

### How to use injected context

At session start, context is injected containing:
- Developer profile (adapt tone, technicality level)
- Current project context (know what was done, decisions made)
- Active reminders (mention them naturally)
- Developer principles (respect them in suggestions)

## 8. Data Flow Scenarios

### Scenario 1: First session on a new project

```
1. SessionStart hook fires
2. → check_project(cwd) → project unknown
3. → get_context() → injects dev profile (preferences, expertise)
4. → Message to agent: "New project detected. Use onboard_project to set it up."
5. Dev says "yes, it's a GraphQL API for Acme Corp, stack is Node/Postgres"
6. → onboard_project() analyzes README + manifests + dev's description
7. → Graphiti extracts: Project, Organization, Technology entities + relations
8. → user_profile updated: [Developer] --works_on--> [Project]
```

### Scenario 2: Regular work session

```
1. SessionStart hook
2. → check_project(cwd) → project known
3. → get_context() → retrieves dev profile + project context
4. → get_reminders() → "Remember to fix the SQL perf bug"
5. → Everything injected into agent context
6. ... dev works normally ...
7. SessionEnd hook
8. → LLM generates session summary
9. → save_session(summary) → Graphiti extracts new entities/relations
```

### Scenario 3: Dev shares personal info

```
Dev: "Remember that I always prefer integration tests over mocks"
Agent: → save_memory("prefers integration tests over mocks", group_id="user_profile")
→ Graphiti creates: [Developer] --holds--> [Principle: integration tests > mocks]
```

### Scenario 4: Dev briefs a client project

```
Dev: "I work on Project Atlas for DataCorp, Julie is the tech lead,
      stack is Python/FastAPI/MongoDB, they have Mongo query latency issues"
Agent: → save_memory(content, group_id="project_datacorp-atlas")
→ Graphiti extracts:
   [Project: Atlas] --owned_by--> [Organization: DataCorp]
   [Colleague: Julie] --role--> "tech lead"
   [Project: Atlas] --uses--> [Technology: Python, FastAPI, MongoDB]
   [Blocker: Mongo query latency] --affects--> [Project: Atlas]
+ → save_memory("works on Atlas for DataCorp", group_id="user_profile")
   [Developer] --works_on--> [Project: Atlas]
```

### Scenario 5: Reminder

```
Dev: "Remind me to renew the OpenAI API key before April 15"
Agent: → add_reminder("renew OpenAI API key before April 15", group_id="user_profile")
... April 14 session ...
SessionStart → get_reminders() → "Reminder: renew OpenAI API key before April 15"
```

## 9. Configuration

### graph-mem MCP server (local)

Configuration via environment variables:

| Variable | Description | Default |
|---|---|---|
| `GRAPHITI_URL` | Graphiti REST API URL | `http://localhost:8000` |
| `GRAPHITI_API_KEY` | API key for Graphiti (if auth enabled) | - |

That's it. Our MCP server is minimal - it only needs to know where Graphiti is.

### Graphiti REST server (Docker)

Deployed via `docker compose up` using Graphiti's official setup:

```yaml
services:
  graphiti:
    image: zepai/graphiti:latest
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
  neo4j:
    image: neo4j:5.22.0
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}

volumes:
  neo4j_data:
```

Swagger docs available at `http://localhost:8000/docs`.

### MCP client configuration (Claude Code example)

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

## 10. REST API Limitation: No Node Search

The Graphiti REST API (`server/`) does not expose a `search_nodes` equivalent (entity search by type).
It only exposes fact/edge search via `POST /search` and `POST /get-memory`.

**Mitigation options (to evaluate during implementation):**
1. Use `POST /search` with targeted queries to find entities indirectly via their relationships
2. Contribute a `POST /search-nodes` endpoint upstream to Graphiti's REST server
3. Add a thin custom endpoint if we end up wrapping the Graphiti server

This is not a blocker for v1 - fact search covers most use cases.

## 11. Future Considerations

- **Privacy tags**: `<private>content</private>` to prevent storage
- **Memory decay**: Auto-archive old/stale entities
- **Reminder expiry**: Auto-dismiss past-date reminders
- **Multi-user**: Support for team knowledge graphs
- **UI viewer**: Web interface to explore the knowledge graph
- **Authentication**: OAuth 2.0 / API key auth for remote Graphiti deployments
- **Node search**: Contribute `search_nodes` endpoint to Graphiti REST API
