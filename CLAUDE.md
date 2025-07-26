# claude-code-context v1.0

A semantic context system for Claude Code with local embeddings and direct search capabilities.

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
- **Simplified Architecture**: Python Indexer + Stella + Qdrant direct (no MCP layer)
- **Local Embeddings**: Stella (stella_en_400M_v5 by default, 1024d)
- **Dedicated Qdrant**: One instance, isolated collections per project
- **User Control**: Context injection via `<ccc>query</ccc>` tags only
- **Zero External Dependencies**: 100% offline operation

## Tech Stack

- Python 3.12+
- Tree-sitter 0.21.3 (multi-language AST parsing)
- Qdrant 1.7.0 (vector database)
- Stella embeddings (local, 400M model)
- Click (CLI framework)
- Pydantic 2.5.0 (data validation)
- Watchdog (file monitoring)

## Project Structure

```
claude-code-context/
├── core/
│   ├── models/          # Pydantic data models
│   ├── parser/          # Tree-sitter language parsers
│   ├── storage/         # Qdrant client and collections
│   ├── indexer/         # Indexing orchestration
│   ├── search/          # Search engine and ranking
│   └── embeddings/      # Stella local embeddings
├── hooks/               # Claude Code UserPromptSubmit hook
├── cli/                 # Command line interface
├── config/              # Configuration management
├── templates/           # Project setup templates
└── tests/               # Test suite
```

## Key Commands

```bash
# Global installation
./install-global.sh

# Project setup
claude-code-context init PROJECT_NAME

# Indexing
claude-indexer index -p . -c PROJECT_NAME-code

# Search test
claude-indexer search -c PROJECT_NAME-code -q "search query"

# Watch mode
claude-indexer watch -p . -c PROJECT_NAME-code
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

1. **Direct Integration**: Stella → Qdrant without intermediate layers
2. **User Control**: No automatic enrichment, only via `<ccc>` tags
3. **Performance First**: Target <10ms search (hardware dependent)
4. **Isolation**: Collections per project, not Docker instances
5. **Offline Operation**: No external APIs or services

## Sprint Overview (3 months total)

### Sprint 1: Foundations (2 weeks)
- Project structure and multi-project setup
- Core data models (Entity, ASTNode, Relation)
- Stella installation automation
- Qdrant collections configuration

### Sprint 2: Embeddings & Storage (2 weeks)
- StellaEmbedder implementation
- Hybrid Qdrant client (payload + vector search)
- Collection management and indexing strategies
- Performance optimizations

### Sprint 3: Tree-sitter Parsers (2 weeks)
- Parser registry and base abstractions
- Language-specific parsers (Python, JS/TS, Go, etc.)
- Entity and relation extraction
- AST storage and navigation

### Sprint 4: Indexing & Search (2 weeks)
- Hybrid indexer with parallel processing
- Intelligent search engine with query classification
- Result ranking and fusion algorithms
- Incremental indexing support

### Sprint 5: Hook & CLI Integration (2 weeks)
- UserPromptSubmit hook with `<ccc>` tag parsing
- Direct Stella integration (no MCP)
- Comprehensive CLI commands
- Multi-project collection management
- Performance monitoring

### Sprint 6: GitHub Repository (2 weeks)
- Public repository setup
- One-click installation scripts
- Multi-project automation
- Documentation and examples

## Testing Requirements

- Unit tests: 95% coverage minimum
- Integration tests for each sprint deliverable
- Performance benchmarks:
  - Indexing: 10k files < 5 min
  - Search: <10ms payload, <50ms semantic
  - Memory: <2GB for 100k files
- Multi-project isolation tests

## Do NOT

- Add MCP protocol layers (direct integration only)
- Use external embedding APIs (Stella is local)
- Create Docker containers per project (use collections)
- Implement automatic context injection (user control via tags)
- Store raw embeddings in Qdrant (compute on demand)
- Mix project data (strict isolation)

## Configuration Files

### Per Project
- `.claude-indexer/config.json` - Project-specific settings
- `.claude/settings.json` - Claude Code hooks configuration
- `CLAUDE.md` - Project instructions (this file)

### Global
- `~/.claude-indexer/` - Global configuration
- `~/.cache/claude-indexer/stella/` - Stella model cache

## Performance Targets

- Installation: <3 minutes
- Indexing: 2000 files/minute
- Search latency: <10ms (hardware dependent)
- Memory usage: <20MB per 1000 entities
- Startup time: <2 seconds

## Error Handling

- Never block user actions
- Silent failures with logging
- Graceful degradation
- Automatic recovery attempts
- Clear error messages in CLI

## Hook Behavior

The UserPromptSubmit hook:
1. Parses `<ccc>query</ccc>` tags in prompts
2. Performs direct Stella embedding + Qdrant search
3. Injects context ONLY when tags present
4. Returns enriched context to stdout
5. Fails silently on errors

Example usage:
```
How to implement auth <ccc>JWT authentication examples</ccc> in my API?
```

## Current Status

Project architecture and requirements are fully defined. Implementation has not started.
**Waiting for sprint selection before beginning development.**