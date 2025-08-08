# claude-code-context v1.0

A semantic context system for Claude Code with local embeddings and intelligent search orchestration.

**⚠️ IMPORTANT**: The project architecture and requirements are defined, but **no specific sprint has been selected for implementation yet**.
**Wait for the user to specify which sprint to implement** before beginning any development work. Each sprint has specific deliverables, user stories, and technical requirements that must be followed precisely.
When a sprint is specified, refer to the detailed sprint documentation for:
- Specific features to implement
- Technical requirements and constraints
- User stories and acceptance criteria
- Testing requirements
- Integration points with other sprints

## Project Overview

Semantic context enrichment for Claude Code using:
- **MCP Server Architecture**: Python package with intelligent Claude-powered search orchestration
- **Local Embeddings**: Stella (stella_en_400M_v5 by default, 1024d)
- **Vector Database**: Qdrant with one instance, isolated collections per project
- **User Control**: Context injection via natural language queries or optional `<ccc>query</ccc>` tags for focused search
- **Zero External Dependencies**: 100% offline operation (except Claude Code CLI calls)

## Tech Stack

- Python 3.12+
- Tree-sitter 0.21.3 (multi-language AST parsing)
- Qdrant 1.7.0 (vector database)
- Stella embeddings (local, 400M model)
- FastMCP (Model Context Protocol server)
- Click (CLI framework)
- Pydantic 2.5.0 (data validation)
- Watchdog (file monitoring)
- Claude Code CLI (for intelligent orchestration)

## Project Structure

```
claude-code-context/
├── pyproject.toml              # Python package configuration
├── setup.py                    # Setup script (optional)
├── claude_code_context/        # Main Python package
│   ├── __init__.py
│   ├── core/
│   │   ├── models/            # Pydantic data models
│   │   ├── parser/            # Tree-sitter language parsers
│   │   ├── storage/           # Qdrant client and collections
│   │   ├── indexer/           # Indexing orchestration
│   │   ├── search/            # Search engine and ranking
│   │   └── embeddings/        # Stella local embeddings
│   ├── mcp_server/            # MCP server implementation
│   │   ├── __init__.py
│   │   ├── __main__.py        # Entry point for python -m
│   │   ├── server.py          # FastMCP server core
│   │   ├── orchestrator.py    # Claude CLI subprocess management
│   │   ├── context_builder.py # Project context assembly
│   │   ├── search_executor.py # Bridge to search engine
│   │   └── session_manager.py # Multi-turn conversation state
│   ├── cli.py                 # CLI commands (ccc tool)
│   └── utils/                 # Shared utilities
├── tests/                      # Test suite
├── examples/                   # Example projects
└── docs/                       # Documentation
```

## Key Commands

### Installation (One-time setup)

```bash
# Install Qdrant (Docker required)
docker run -d --name qdrant -p 6333:6333 \
  -v ~/qdrant_storage:/qdrant/storage \
  --restart always qdrant/qdrant

# Install claude-code-context package
pip install claude-code-context
# Or from source:
git clone https://github.com/user/claude-code-context.git
cd claude-code-context
pip install -e .
```

### Project Setup and Usage

```bash
# Initialize in your project
cd /your/project
ccc init

# Check status
ccc status

# Manual indexing (if needed)
ccc index --full

# Clean project data
ccc clean

# Use with Claude Code
claude
# Then in Claude:
# > /mcp                    # Check MCP server status
# > Find authentication functions
# > Show me <ccc>JWT token validation</ccc> examples
```

## Code Style

- Use type hints for all function signatures
- Follow PEP 8 with 88 char line limit (Black formatter)
- Docstrings for all public functions (Google style)
- Async/await for I/O operations
- Dataclasses/Pydantic for data models
- No raw dict manipulation - use typed models
- Explicit error handling with custom exceptions

## Architecture Principles

1. **MCP Integration**: FastMCP server that auto-starts with Claude Code
2. **Intelligent Orchestration**: Claude Code CLI determines optimal search strategies
3. **User Control**: Natural language queries with optional `<ccc>` tags for focus
4. **Performance First**: Target <10ms payload search, <100ms semantic search
5. **Isolation**: Collections per project using sanitized project names
6. **Offline Operation**: No external APIs except Claude Code CLI (local)

## Sprint Overview (3 months total)

### Sprint 1: Foundations (2 weeks) ✅ COMPLETED
- Project structure and multi-project setup
- Core data models (Entity, ASTNode, Relation)
- Stella installation automation
- Qdrant collections configuration

### Sprint 2: Embeddings & Storage (2 weeks) ✅ COMPLETED
- StellaEmbedder implementation
- Hybrid Qdrant client (payload + vector search)
- Collection management and indexing strategies
- Performance optimizations

### Sprint 3: Tree-sitter Parsers (2 weeks) ✅ COMPLETED
- Parser registry and base abstractions
- Language-specific parsers (Python, JS/TS, Go, etc.)
- Entity and relation extraction
- AST storage and navigation

### Sprint 4: Indexing & Search (2 weeks) ✅ COMPLETED
- Hybrid indexer with parallel processing
- Intelligent search engine with query classification
- Result ranking and fusion algorithms
- Delta-scan algorithm with real-time sync

### Sprint 5: MCP Server with Claude Orchestration (2 weeks)
#### Overview
Implement an MCP (Model Context Protocol) server that leverages Claude Code CLI as an intelligent orchestrator for code search. The server will be distributed as an installable Python package with a simple CLI tool for project initialization.

#### Key Features
- **Python Package Distribution**: Install globally via `pip install claude-code-context`
- **MCP Server Implementation**: FastMCP-based stdio server that auto-starts with Claude Code
- **Claude Orchestration**: Uses Claude Code CLI to intelligently determine search strategies
- **Iterative Search**: Up to 10 Claude calls per search session for refinement
- **Project Context**: Assembles project tree, documentation, and search engine guide for Claude
- **Automatic Initialization**: Delta-sync on startup, real-time monitoring during session

#### Technical Implementation
- **Entry Point**: `python -m claude_code_context.mcp_server`
- **CLI Tool**: `ccc` command for project management
  - `ccc init` - Creates `.mcp.json` configuration in project
  - `ccc status` - Checks Qdrant and project health
  - `ccc index` - Manual indexing trigger
  - `ccc clean` - Removes project from Qdrant
- **Collection Naming**: `ccc_{sanitized_project_name}`
- **Configuration**: `.mcp.json` in project root with MCP server settings

#### Search Flow
1. User queries Claude Code (with or without `<ccc>` tags)
2. MCP server receives the query
3. Server calls Claude CLI to analyze and generate search strategy
4. Claude returns JSON with search type, optimized query, and reasoning
5. Server executes search using existing HybridQdrantClient
6. Results fed back to Claude for next iteration or synthesis
7. Final analysis returned to user

#### Error Handling
- Graceful fallback if Claude CLI unavailable
- Direct search execution as backup
- Clear error messages for Qdrant connection issues
- Session state management for recovery

### Sprint 6: GitHub Repository & Production Deployment (2 weeks)
#### Overview
Create a production-ready GitHub repository with comprehensive documentation, automated installation, and professional distribution through PyPI. Focus on user experience from discovery to first successful search.

#### Repository Structure
```
claude-code-context/
├── README.md                    # Quick start guide (3-step process)
├── INSTALL.md                   # Detailed installation instructions
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # MIT License
├── pyproject.toml              # Modern Python packaging
├── setup.py                    # Backward compatibility
├── requirements.txt            # Direct dependencies
├── requirements-dev.txt        # Development dependencies
├── .github/                    # GitHub configuration
│   ├── workflows/             # CI/CD pipelines
│   │   ├── test.yml          # Run tests on PR
│   │   ├── release.yml       # PyPI publication
│   │   └── docs.yml          # Documentation build
│   └── ISSUE_TEMPLATE/        # Issue templates
├── claude_code_context/        # Source code (from Sprint 5)
├── tests/                      # Comprehensive test suite
├── examples/                   # Working examples
│   ├── python-fastapi/        # FastAPI project example
│   ├── javascript-react/      # React project example
│   ├── go-microservice/       # Go project example
│   └── mixed-language/        # Multi-language example
├── scripts/                    # Helper scripts
│   ├── install-qdrant.sh      # Qdrant setup helper
│   ├── test-installation.sh   # Verify installation
│   └── benchmark.py           # Performance testing
└── docs/                       # Documentation
    ├── architecture.md         # Technical architecture
    ├── troubleshooting.md     # Common issues & solutions
    ├── advanced-usage.md      # Power user features
    └── api-reference.md       # API documentation
```

#### Installation Experience
```bash
# Step 1: Install Qdrant (one-time, clearly separated)
docker run -d --name qdrant -p 6333:6333 \
  -v ~/qdrant_storage:/qdrant/storage \
  --restart always qdrant/qdrant

# Step 2: Install claude-code-context
pip install claude-code-context

# Step 3: Use in your project
cd /your/project
ccc init
claude
```

#### Key Deliverables
- **PyPI Package**: Published to Python Package Index for easy installation
- **Comprehensive Documentation**: From quick start to advanced usage
- **Working Examples**: Real project examples showing best practices
- **Automated Testing**: CI/CD with GitHub Actions
- **Installation Verification**: Scripts to verify successful setup
- **Performance Benchmarks**: Tools to measure indexing and search speed
- **Troubleshooting Guide**: Common issues with clear solutions

#### User Experience Focus
- **3-Step Installation**: Qdrant → Package → Project init
- **Clear Separation**: Infrastructure (Qdrant) vs Application (claude-code-context)
- **Error Messages**: Helpful guidance when things go wrong
- **Progress Indicators**: Visual feedback for long operations
- **Health Checks**: `ccc status` for diagnostics

#### Distribution Strategy
- **PyPI Primary**: Professional Python package distribution
- **GitHub Releases**: Binary releases with changelog
- **Docker Option**: Optional all-in-one container (future)
- **Homebrew Formula**: Mac users convenience (future)

## Testing Requirements

### Coverage Targets by Sprint

Sprint-specific coverage targets reflect the incremental nature of development:

- **Sprint 2**: 70% ✅ (Embeddings & Storage foundation)
- **Sprint 3**: 68% ✅ (Tree-sitter parsers complete)
- **Sprint 4**: 88% ✅ (Full indexing and search pipeline)
- **Sprint 5**: 80%+ (MCP server with orchestration)
- **Sprint 6**: 85%+ (Complete system with documentation)

### Test Requirements

- Unit tests for each sprint deliverable
- Integration tests for sprint completions
- Performance benchmarks:
  - Indexing: 2000+ files/minute
  - Search: <10ms payload, <100ms semantic
  - Memory: <2GB for 100k entities
- Multi-project isolation tests
- MCP server connection tests
- Claude orchestration mock tests

## Do NOT

- Add unnecessary abstraction layers (keep it simple)
- Use external embedding APIs (Stella is local)
- Create Docker containers per project (use collections)
- Implement automatic context injection without user query
- Store raw embeddings in Qdrant (compute on demand)
- Mix project data (strict isolation)
- Require internet connection (except Claude CLI, which is local)

## Configuration Files

### Per Project
- `.mcp.json` - MCP server configuration (created by `ccc init`)
- `.claude-indexer/config.json` - Project-specific settings (optional)
- `CLAUDE.md` - Project documentation (optional, used for context)

### Global
- `~/.cache/claude-code-context/` - Stella model cache
- No global configuration needed (all per-project)

### MCP Configuration Example
```json
{
  "mcpServers": {
    "claude-code-context": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "claude_code_context.mcp_server"],
      "env": {
        "MCP_PROJECT_PATH": ".",
        "MCP_COLLECTION_NAME": "auto",
        "MCP_MAX_CLAUDE_CALLS": "10",
        "MCP_DEBUG": "false",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

## Performance Targets

- Installation: <2 minutes (excluding Qdrant setup)
- MCP server startup: <5 seconds
- Indexing: 2000+ files/minute
- Search latency: <10ms payload, <100ms semantic
- Total search response: <30s for complex multi-turn
- Memory usage: <20MB per 1000 entities
- Claude orchestration: <2s per strategy call

## Error Handling

- Never block user actions
- Graceful degradation when services unavailable
- Fallback to direct search if Claude unavailable
- Automatic recovery attempts
- Clear error messages with solutions
- Debug mode for troubleshooting

## MCP Server Behavior

The MCP server with Claude orchestration:
1. Auto-starts when Claude Code launches in project with `.mcp.json`
2. Connects to Qdrant and verifies/creates collection
3. Runs delta-sync to update index
4. Receives queries from Claude Code
5. Uses Claude CLI to determine optimal search strategy
6. Executes iterative searches (up to 10 rounds)
7. Returns synthesized results with code snippets

Example interaction:
```
> Find authentication functions
[MCP server calls Claude to analyze query]
[Claude returns: {"search_type": "hybrid", "query": "authentication function login logout", ...}]
[Server executes search and returns results]

> Show me <ccc>JWT token validation</ccc> examples
[MCP server recognizes focused search]
[Claude optimizes specifically for JWT validation code]
[Returns targeted examples with explanations]
```

## Intelligent Search Optimization

The MCP server leverages Claude's intelligence to automatically select the optimal search strategy. Understanding how Claude analyzes queries helps you write more effective searches.

### Claude's Strategy Selection

**Payload Search** - Claude selects this for:
- Direct function/class names
- File path references
- Short, specific identifiers
- Quoted exact matches

**Semantic Search** - Claude selects this for:
- Conceptual queries ("how to implement X")
- Natural language descriptions
- Pattern-finding requests
- Explanation requests

**Hybrid Search** - Claude selects this for:
- Complex multi-part queries
- Mixed specific + conceptual terms
- Broad exploration requests
- Ambiguous queries needing both approaches

### Query Enhancement with `<ccc>` Tags

The `<ccc>` tags provide focus for your search:

```
# Without tags - Claude analyzes the entire query
> Find user authentication and session management code

# With tags - Claude focuses on the tagged portion
> I need to implement login. Show me <ccc>password hashing bcrypt</ccc> examples
```

### Iterative Refinement

Claude may perform multiple searches to find the best results:

1. **Initial broad search** to understand codebase structure
2. **Refined searches** based on initial findings
3. **Targeted searches** for specific implementations
4. **Final synthesis** combining all relevant findings

### Query Examples by Use Case

**Learning a New Codebase:**
```
> How is authentication implemented in this project?
> Show me the main API endpoints
> Explain the database schema and models
```

**Finding Specific Implementations:**
```
> <ccc>JWT token generation and validation</ccc>
> Find all async database operations
> Show error handling patterns in the API layer
```

**Debugging and Troubleshooting:**
```
> Find where <ccc>database connection errors</ccc> are handled
> Show me all try-catch blocks in authentication code
> Where are API responses validated?
```

**Code Review and Security:**
```
> Find potential security issues in <ccc>user input validation</ccc>
> Show me all places handling sensitive data
> Check error handling completeness
```

### Advanced Search Features

**Multi-Turn Conversations:**
Claude remembers context within a search session, allowing follow-up refinements:
```
> Find user authentication code
[Results shown]
> Now show me just the password validation part
[Refined results based on previous findings]
```

**Cross-File Relationships:**
Claude can trace connections across files:
```
> Show me how the User model is used across the application
> Find all imports of the authentication module
> Trace the data flow from API to database
```

## Current Status

Project architecture is fully defined with completed implementations through Sprint 4:
- ✅ Sprint 1: Foundations and architecture
- ✅ Sprint 2: Embeddings and storage (70% coverage)
- ✅ Sprint 3: Tree-sitter parsers (68% coverage)
- ✅ Sprint 4: Indexing and search (88% coverage)
- ⏳ Sprint 5: MCP server implementation pending
- ⏳ Sprint 6: GitHub repository and distribution pending

**Waiting for sprint selection before beginning development.**