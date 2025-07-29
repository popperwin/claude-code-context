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

### Coverage Targets by Sprint

Sprint-specific coverage targets reflect the incremental nature of development:

- **Sprint 2**: 35-45% ✅ (Embeddings & Storage foundation)
- **Sprint 3**: 60-65% (after Tree-sitter integration) 
- **Sprint 4**: 75-80% (after full indexing pipeline)
- **Sprint 5**: 85-90% (after CLI/Hook integration)
- **Sprint 6**: 95%+ (complete system with documentation)

### Test Requirements

- Unit tests for each sprint deliverable
- Integration tests for sprint completions
- Performance benchmarks:
  - Indexing: 10k files < 5 min
  - Search: <10ms payload, <50ms semantic
  - Memory: <2GB for 100k files
- Multi-project isolation tests

**Note**: Lower early-sprint coverage is expected due to future sprint dependencies and proper separation of concerns.

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

## Intelligent Search Optimization

The QueryAnalyzer automatically selects the optimal search strategy based on query patterns. Understanding these patterns helps you write more effective search queries.

### Search Mode Classification

**Payload Search** - Fast exact matching for:
- **Quoted strings**: `"ExactFunctionName"` or `'SpecificClass'`
- **Identifiers**: `name:UserManager`, `id:auth-service-123`
- **File references**: `file:auth.py`, `path:/src/models`
- **Short terms**: `login`, `database`, `config`

**Semantic Search** - AI-powered understanding for:
- **How-to queries**: `how to implement JWT authentication`
- **Explanatory requests**: `explain the database connection pattern`
- **Find patterns**: `find functions that handle user validation`
- **Show examples**: `show me error handling approaches`
- **Natural language**: `what manages user sessions in this app`

**Hybrid Search** - Balanced approach for:
- **Code with context**: `async function error handling`
- **Mixed patterns**: `class UserModel in models.py`
- **Medium complexity**: `authentication system architecture`
- **Ambiguous queries**: `database connection manager`

### Query Optimization Tips

**For Fastest Results (Payload Search):**
```
# Good: Direct, specific terms
<ccc>UserAuthentication</ccc>
<ccc>name:"LoginHandler"</ccc>
<ccc>file:auth.py</ccc>

# Avoid: Long descriptive phrases
<ccc>how does the user authentication system work</ccc>
```

**For Best Semantic Understanding:**
```
# Good: Natural language with clear intent
<ccc>how to implement password hashing securely</ccc>
<ccc>find all functions that validate user input</ccc>
<ccc>show me examples of error handling patterns</ccc>

# Avoid: Single words or code fragments
<ccc>hash</ccc>
<ccc>def validate</ccc>
```

**For Comprehensive Coverage (Hybrid):**
```
# Good: Mix of specific and contextual terms
<ccc>async database operations with error handling</ccc>
<ccc>user authentication middleware implementation</ccc>
<ccc>API endpoint validation and security</ccc>
```

### Advanced Search Patterns

**Entity-Specific Searches:**
- Functions: `def authenticate_user`, `async login handler`
- Classes: `class UserModel`, `authentication service class`
- Files: `models.py user schema`, `auth/*.py`
- Configuration: `config.json database`, `settings authentication`

**Relationship Searches:**
- Dependencies: `import user models`, `uses authentication service`
- Inheritance: `extends BaseUser`, `implements AuthInterface`
- Patterns: `decorator pattern auth`, `singleton database connection`

**Contextual Searches:**
- Error handling: `try catch authentication`, `error handling login`
- Security: `password validation`, `JWT token verification`
- Performance: `async database queries`, `caching user sessions`

### Query Examples by Use Case

**Learning Codebase:**
```
<ccc>how does user authentication work in this application</ccc>
<ccc>explain the database schema and relationships</ccc>
<ccc>show me the main application entry points</ccc>
```

**Implementation Help:**
```
<ccc>password hashing and validation examples</ccc>
<ccc>async error handling patterns</ccc>
<ccc>JWT token implementation and verification</ccc>
```

**Debugging Assistance:**
```
<ccc>find functions that handle database connection errors</ccc>
<ccc>authentication middleware and session management</ccc>
<ccc>logging and error reporting mechanisms</ccc>
```

**Code Review:**
```
<ccc>security validation and input sanitization</ccc>
<ccc>error handling and exception management</ccc>
<ccc>test coverage for authentication functions</ccc>
```

## Current Status

Project architecture and requirements are fully defined. Implementation has not started.
**Waiting for sprint selection before beginning development.**