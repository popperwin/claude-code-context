# claude-code-context v1.0.0

🚀 **Intelligent code search for Claude Code powered by local embeddings and Claude orchestration.**

A sophisticated MCP (Model Context Protocol) server that enhances Claude Code with intelligent code search capabilities. Using Claude's own intelligence to orchestrate searches, local Stella embeddings, and Qdrant vector storage, it provides contextual code understanding through natural language queries.

## ✨ Features

- 🧠 **Claude-Powered Search** - Claude intelligently orchestrates multi-step search strategies
- 🔍 **Natural Language Queries** - Ask questions in plain English, Claude understands context
- 🏠 **100% Local Operation** - Stella embeddings and Qdrant run entirely offline
- 📦 **Simple Installation** - One `pip install` command, per-project configuration
- ⚡ **High Performance** - <10ms payload search, <100ms semantic search
- 🎯 **Smart Query Analysis** - Automatic selection of optimal search strategy
- 🔄 **Real-time Sync** - Automatic indexing and live updates as you code

## 📋 Requirements

- **Python 3.8+**
- **Docker** (for Qdrant vector database)
- **Claude Code CLI** (authenticated via `claude login`)
- **5GB+ free disk space** (for Stella model)
- **macOS, Linux, or Windows** (via WSL)

## 🚀 Quick Start

### 1. Install Qdrant (One-time Setup)

```bash
# Start Qdrant vector database
docker run -d --name qdrant -p 6333:6333 \
  -v ~/qdrant_storage:/qdrant/storage \
  --restart always qdrant/qdrant
```

### 2. Install claude-code-context

```bash
# Install from PyPI
pip install claude-code-context

# Or install from source
git clone https://github.com/user/claude-code-context.git
cd claude-code-context
pip install -e .
```

### 3. Use in Your Project

```bash
# Navigate to your project
cd /path/to/your/project

# Initialize claude-code-context
ccc init

# Start Claude Code
claude

# Verify MCP connection
> /mcp
# Should show: claude-code-context: connected

# Start searching!
> Find all authentication functions
> How does error handling work in this codebase?
> Show me database connection examples
```

That's it! Claude will intelligently search your codebase and provide relevant context.

## 🎯 How It Works

### Intelligent Orchestration

When you ask a question, the MCP server:
1. **Analyzes your query** using Claude's intelligence
2. **Determines the best search strategy** (exact match, semantic, or hybrid)
3. **Executes iterative searches** refining results up to 10 times
4. **Synthesizes findings** into a coherent answer with code examples

### Search Examples

**Natural Language Understanding:**
```
> How is user authentication implemented in this project?
[Claude analyzes the conceptual query and searches for auth patterns]

> Explain the database schema and relationships
[Claude identifies this as an architectural query and searches accordingly]
```

**Specific Code Search:**
```
> Find all async functions that handle errors
[Claude recognizes the pattern query and uses hybrid search]

> Show me the UserModel class implementation
[Claude identifies this as an exact match query for fast results]
```

**Focused Search with `<ccc>` Tags (Optional):**
```
> I need help with login. Show me <ccc>password hashing bcrypt</ccc> examples
[Claude focuses specifically on the tagged concept]

> Debug this error <ccc>database connection timeout retry</ccc>
[Claude searches for relevant error handling patterns]
```

## 🔧 Configuration

### Project Structure

After running `ccc init`, your project will have:

```
your-project/
├── .mcp.json               # MCP server configuration
└── [your code files]
```

### MCP Configuration

The `.mcp.json` file configures the MCP server:

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
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

### Environment Variables

- `MCP_PROJECT_PATH`: Project root directory (default: current directory)
- `MCP_COLLECTION_NAME`: Qdrant collection name (default: auto-generated)
- `MCP_MAX_CLAUDE_CALLS`: Maximum search iterations (default: 10)
- `MCP_DEBUG`: Enable debug logging (default: false)
- `QDRANT_URL`: Qdrant server URL (default: http://localhost:6333)

## 📖 Command Reference

### CLI Commands (`ccc`)

```bash
# Initialize project
ccc init                    # Create .mcp.json in current directory
ccc init --force           # Overwrite existing configuration

# Check status
ccc status                 # Show Qdrant and project status

# Manual operations
ccc index                  # Manually trigger indexing
ccc index --full          # Force complete reindex
ccc clean                  # Remove project from Qdrant

# Debug
ccc test "query"           # Test search without Claude Code
ccc test "query" --debug   # Show Claude's reasoning
```

### Claude Code Commands

```bash
# In Claude Code
/mcp                       # Show MCP server status
/mcp list                  # List available MCP tools
/mcp debug                 # Enable debug mode
```

## 🚀 Advanced Usage

### Search Strategies

Claude automatically selects the optimal strategy:

**Payload Search (Fastest)** - For specific names and identifiers:
- Direct function/class names: `UserAuthentication`
- File references: `auth.py`, `models/user.js`
- Short specific terms: `login`, `validate`

**Semantic Search (Smartest)** - For conceptual understanding:
- How-to questions: "How to implement JWT authentication"
- Pattern finding: "Find all error handling examples"
- Explanations: "Explain the API structure"

**Hybrid Search (Balanced)** - For comprehensive results:
- Mixed queries: "async database operations"
- Contextual searches: "user model validation methods"
- Broad explorations: "authentication system overview"

### Multi-Project Usage

Each project gets its own isolated collection:

```bash
# Project 1
cd /path/to/project1
ccc init
# Collection: ccc_project1

# Project 2
cd /path/to/project2
ccc init
# Collection: ccc_project2

# Both projects share the same Qdrant instance
# but have completely isolated data
```

### Performance Optimization

For large codebases (>10k files), adjust settings in `.mcp.json`:

```json
{
  "env": {
    "MCP_ENABLE_DELTA_SYNC": "true",
    "MCP_SANITY_CHECK_INTERVAL": "600",
    "QDRANT_BATCH_SIZE": "200"
  }
}
```

## 🛠 Troubleshooting

### Common Issues

**MCP Server Not Connecting**
```bash
# Check if Qdrant is running
docker ps | grep qdrant
ccc status

# Restart Qdrant if needed
docker restart qdrant

# Verify Claude Code sees the MCP config
cat .mcp.json
```

**No Search Results**
```bash
# Check if project is indexed
ccc status

# Trigger manual indexing
ccc index --full

# Test search directly
ccc test "your search query" --debug
```

**Claude Not Responding**
```bash
# Ensure you're logged in to Claude
claude login

# Check Claude Code is working
claude --version

# Try with debug mode
MCP_DEBUG=true claude
```

**Performance Issues**
```bash
# Check Qdrant health
curl http://localhost:6333/health

# Monitor indexing progress
ccc index --verbose

# Reduce parallel operations
# Edit .mcp.json: "MCP_MAX_WORKERS": "2"
```

### Error Messages

**"Qdrant not running"**
- Start Qdrant: `docker start qdrant`
- Check port 6333 is not in use: `lsof -i :6333`

**"Collection not found"**
- Run `ccc init` in your project
- Check collection name: `ccc status`

**"Claude login required"**
- Run `claude login` and follow instructions
- Ensure you have an active Claude subscription

## 🏗 Architecture

### System Overview

```
┌─────────────────────┐
│   Claude Code CLI   │
│   (User Interface)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│    MCP Server       │────▶│   Claude CLI        │
│ (stdio transport)   │     │  (Orchestrator)     │
└──────────┬──────────┘     └─────────────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Search Engine      │     │  Search Strategy    │
│  (Hybrid Search)    │◀────│   (JSON Format)     │
└──────────┬──────────┘     └─────────────────────┘
           │
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Qdrant Database    │     │  Stella Embeddings  │
│  (Vector Storage)   │◀────│   (Local Model)     │
└─────────────────────┘     └─────────────────────┘
```

### Key Components

1. **MCP Server** - FastMCP-based server that integrates with Claude Code
2. **Claude Orchestrator** - Uses Claude CLI to intelligently plan searches
3. **Search Engine** - Hybrid search with payload and semantic capabilities
4. **Qdrant** - High-performance vector database for code storage
5. **Stella** - Local embedding model for semantic understanding

### Data Flow

1. **Query** → MCP Server receives natural language query
2. **Analysis** → Claude analyzes and creates search strategy
3. **Search** → Execute strategy using hybrid search engine
4. **Refinement** → Claude may request additional searches
5. **Synthesis** → Final results compiled and returned

## 📊 Performance

### Benchmarks

- **Indexing Speed**: 2000+ files/minute
- **Search Latency**: 
  - Payload: <10ms
  - Semantic: <100ms
  - Hybrid: <150ms
- **Memory Usage**: <20MB per 1000 entities
- **Startup Time**: <5 seconds

### Supported Languages

Full AST parsing and semantic understanding:
- **Python** (.py, .pyi)
- **JavaScript/TypeScript** (.js, .ts, .jsx, .tsx)
- **Go** (.go)
- **Rust** (.rs)
- **Java** (.java)
- **C/C++** (.c, .cpp, .h, .hpp)
- **C#** (.cs)
- **Ruby** (.rb)
- **PHP** (.php)
- **Swift** (.swift)
- **Kotlin** (.kt)
- **Scala** (.scala)

Configuration and documentation:
- **JSON, YAML, TOML**
- **Markdown, Text**
- **HTML, CSS**

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/user/claude-code-context.git
cd claude-code-context

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Format code
black claude_code_context
isort claude_code_context

# Type checking
mypy claude_code_context
```

## 📚 Resources

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **API Reference**: [docs/api.md](docs/api.md)
- **Troubleshooting**: [docs/troubleshooting.md](docs/troubleshooting.md)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Anthropic** - Claude and Model Context Protocol
- **Qdrant** - Vector database technology
- **Infgrad** - Stella embedding models
- **Tree-sitter** - Code parsing infrastructure

## 🔗 Links

- **MCP Documentation**: [modelcontextprotocol.io](https://modelcontextprotocol.io)
- **Claude Code**: [claude.ai](https://claude.ai)
- **Issues**: [GitHub Issues](https://github.com/user/claude-code-context/issues)
- **Discussions**: [GitHub Discussions](https://github.com/user/claude-code-context/discussions)

---

**⚡ Intelligent code search that understands your questions.**

Made with ❤️ for developers who think in natural language.