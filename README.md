# claude-code-context v1.0.0

🚀 **Semantic context enrichment for Claude Code with local embeddings and direct search capabilities.**

A powerful semantic search system that enhances Claude Code with intelligent code context injection using local Stella embeddings and Qdrant vector storage. Enable precise code search and automatic context enrichment through simple `<ccc>query</ccc>` tags in your prompts.

## ✨ Features

- 🔍 **Semantic Code Search** - Natural language queries find relevant code across your entire project
- 🏠 **100% Offline** - Local Stella embeddings, no external API calls required
- 🚀 **Claude Code Integration** - Seamless context injection via `<ccc>` tags  
- 📦 **Multi-Project Support** - Isolated collections for unlimited projects
- ⚡ **High Performance** - <10ms search, 2000 files/minute indexing
- 🔧 **Zero Configuration** - One-line setup with intelligent defaults
- 🎯 **Collection-Based Isolation** - Single Qdrant instance, project separation via collections

## 📋 Requirements

- **Python 3.12+**
- **Docker** (for Qdrant vector database)
- **5GB+ free disk space** (for Stella model cache)
- **macOS or Linux** (Windows support planned)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and install
git clone https://github.com/popperwin/claude-code-context.git
cd claude-code-context

# Global installation (recommended)
./install-global.sh

# OR development installation
./install-global.sh --dev
```

### 2. Setup Infrastructure

```bash
# Start Qdrant vector database
./scripts/setup-qdrant.sh

# Install Stella embedding model (800MB download)
python scripts/install_stella.py
```

### 3. Configure Your Project

```bash
# Setup current project
./setup-project.sh my-awesome-project

# OR setup specific project path
./setup-project.sh my-api --path /path/to/project
```

### 4. Index Your Code

```bash
# Index the project
claude-indexer index -p . -c my-awesome-project-code

# Watch for changes (optional)
claude-indexer watch -p . -c my-awesome-project-code &
```

### 5. Use with Claude Code

Now use `<ccc>query</ccc>` tags in your Claude Code prompts:

```
How do I implement user authentication <ccc>user auth login authentication</ccc> in this codebase?
```

```
Show me examples of error handling <ccc>error handling exceptions try catch</ccc> patterns.
```

```
Help me understand the API structure <ccc>API routes endpoints handlers</ccc> in this application.
```

## 🔧 Configuration

### Project Structure

After setup, your project will have:

```
your-project/
├── .claude-indexer/
│   └── config.json          # Project configuration
├── .claude/
│   └── settings.json        # Claude Code hook settings
└── [your code files]
```

### Collections Created

Each project gets isolated Qdrant collections:
- `{project-name}-code` - Code entities and functions
- `{project-name}-relations` - Code relationships  
- `{project-name}-embeddings` - Semantic embeddings

### Configuration Files

**Project Config** (`.claude-indexer/config.json`):
```json
{
  "name": "my-project",
  "collection_prefix": "my-project", 
  "qdrant": {
    "url": "http://localhost:6333",
    "collections": {
      "code": "my-project-code",
      "relations": "my-project-relations",
      "embeddings": "my-project-embeddings"
    }
  },
  "stella": {
    "model_name": "stella_en_400M_v5",
    "dimensions": 1024
  },
  "indexing": {
    "include_patterns": ["*.py", "*.js", "*.ts", "*.go", "*.rs"],
    "exclude_patterns": ["node_modules/*", ".git/*", "__pycache__/*"]
  }
}
```

## 📖 Usage Examples

### Basic Search
```bash
# Search for functions
claude-indexer search -c my-project-code -q "authentication function"

# Search with file type filter  
claude-indexer search -c my-project-code -q "database" --file-type python

# Get collection statistics
claude-indexer stats -c my-project-code
```

### Advanced Claude Code Integration

The system uses intelligent QueryAnalyzer to automatically optimize your searches. Understanding these patterns helps you get better results faster.

#### Search Strategy Guide

**🎯 For Exact Matches (Fastest - <10ms):**
```
# Use quotes for exact function/class names
I need to understand <ccc>"UserAuthenticationService"</ccc> implementation.

# Use prefixes for specific searches  
Help me debug <ccc>name:LoginHandler</ccc> issues.

# Reference specific files
Show me <ccc>file:auth.py</ccc> authentication logic.
```

**🧠 For Conceptual Understanding (AI-Powered):**
```
# Use natural language for explanations
<ccc>how does user authentication work in this application</ccc>

# Ask for patterns and examples
<ccc>show me error handling patterns used in this codebase</ccc>

# Request architectural insights
<ccc>explain the database connection and session management</ccc>
```

**⚖️ For Comprehensive Coverage (Balanced):**
```
# Combine specific terms with context
<ccc>async function error handling patterns</ccc>

# Mix code terms with descriptive context
<ccc>JWT token validation and middleware implementation</ccc>

# Search across related components
<ccc>user authentication database models and API endpoints</ccc>
```

#### Query Optimization Examples

**Function Implementation:**
```
# Good: Specific + contextual
I need to implement password validation <ccc>password hashing bcrypt security validation</ccc> for user registration.

# Better: Natural language for understanding
How should I implement <ccc>secure password validation with hashing and salting</ccc> in this codebase?
```

**Debugging Help:**
```
# Good: Specific error context
This error is confusing <ccc>database connection timeout retry logic</ccc>, how should I handle it?

# Better: Natural language for patterns
<ccc>find functions that handle database connection errors and recovery</ccc>
```

**Architecture Understanding:**
```
# Good: Conceptual query
Explain the data flow <ccc>data models database relationships schema</ccc> in this application.

# Better: Natural explanation request
<ccc>how does data flow from API endpoints through models to the database</ccc>
```

**Code Patterns:**
```
# Good: Pattern with context
Show me how to implement <ccc>async operations error handling concurrency</ccc> in this codebase style.

# Better: Example-focused query
<ccc>show me examples of async error handling and concurrency patterns</ccc>
```

#### Pro Tips for Better Results

**📊 Query Length Guidelines:**
- **1-2 words**: Perfect for exact matches (`login`, `UserModel`)
- **3-5 words**: Good for hybrid searches (`async error handling`)  
- **6+ words**: Best for semantic searches (`how to implement secure authentication`)

**🎯 Pattern Recognition:**
- **Quoted text**: `"ExactFunctionName"` → Exact payload search
- **Prefixes**: `name:`, `file:`, `id:` → Targeted payload search
- **Questions**: `how`, `what`, `explain` → Semantic understanding
- **Commands**: `find`, `show`, `help` → Semantic examples

**🚀 Performance Optimization:**
- Use exact matches when you know the specific name
- Use semantic searches for learning and exploration
- Combine both approaches for comprehensive coverage
- Keep queries focused and relevant to your current task

## 🛠 Management Commands

### Project Management
```bash
# List all configured projects
claude-indexer list-projects

# Validate project setup
./scripts/test-project.sh

# Recreate project with new settings
./setup-project.sh my-project --overwrite
```

### Index Management
```bash
# Full reindex
claude-indexer index -p . -c my-project-code --rebuild

# Index specific directory
claude-indexer index -p ./src -c my-project-code

# Delete collection (careful!)
claude-indexer delete -c my-project-code
```

### Infrastructure Management
```bash
# Qdrant operations
./scripts/setup-qdrant.sh status    # Check status
./scripts/setup-qdrant.sh restart  # Restart Qdrant
./scripts/setup-qdrant.sh logs     # View logs

# Stella model management
python scripts/install_stella.py --info     # Model info
python scripts/install_stella.py --cleanup  # Remove cached model
```

## ⚡ Performance & Optimization

### Performance Targets
- **Search Latency:** <10ms for payload queries, <50ms for semantic search
- **Indexing Speed:** 2000+ files per minute  
- **Memory Usage:** <20MB per 1000 entities
- **Startup Time:** <2 seconds

### Optimization Tips

**For Large Projects (>10k files):**
```json
{
  "qdrant": {
    "batch_size": 200,
    "parallel_requests": 8
  },
  "stella": {
    "batch_size": 64,
    "use_fp16": true
  }
}
```

**For Resource-Constrained Systems:**
```json
{
  "qdrant": {
    "batch_size": 50,
    "parallel_requests": 2
  },
  "stella": {
    "batch_size": 16,
    "use_fp16": false
  }
}
```

## 🔍 Supported Languages

Primary support with full AST parsing:
- **Python** (.py, .pyi)
- **JavaScript/TypeScript** (.js, .ts, .jsx, .tsx)  
- **Go** (.go)
- **Rust** (.rs)
- **Java** (.java)
- **C/C++** (.c, .cpp, .h, .hpp)

Additional indexing support:
- **C#, Ruby, PHP, Swift, Kotlin, Scala**
- **Configuration files** (.json, .yaml, .toml)
- **Documentation** (.md, .txt)

## 🔧 Troubleshooting

### Common Issues

**Qdrant Not Accessible**
```bash
# Check Qdrant status
curl http://localhost:6333/health

# Restart Qdrant
./scripts/setup-qdrant.sh restart

# View logs
./scripts/setup-qdrant.sh logs
```

**Collections Not Found**
```bash
# Check collections
curl http://localhost:6333/collections

# Recreate project collections
./setup-project.sh my-project --overwrite
```

**Search Returns No Results**
```bash
# Check if project is indexed
claude-indexer stats -c my-project-code

# Reindex if needed
claude-indexer index -p . -c my-project-code --rebuild
```

**Hook Not Working**
- Verify `.claude/settings.json` exists and is valid JSON
- Check `PROJECT_NAME` and `COLLECTION_PREFIX` environment variables
- Ensure Qdrant is running and accessible
- Test hook manually: `python -m hooks.user_prompt_submit`

**Stella Model Issues**
```bash
# Reinstall model
python scripts/install_stella.py --force

# Check model info
python scripts/install_stella.py --info

# Verify installation
python scripts/install_stella.py --verify-only
```

### Performance Issues

**Slow Indexing:**
- Increase `batch_size` in configuration
- Use SSD storage for better I/O
- Exclude large binary directories

**Slow Search:**
- Check Qdrant resource allocation
- Reduce `max_results_per_query` for faster responses
- Use more specific search terms

**High Memory Usage:**
- Reduce Stella `batch_size`
- Enable `use_fp16` for GPU acceleration
- Limit concurrent operations

## 🏗 Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Claude Code   │    │  User Prompt    │    │    Semantic     │
│     Hooks       │◄──►│   Processing    │◄──►│     Search      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Context Inject │    │   Collection    │    │     Stella      │
│   <ccc> Tags    │    │   Management    │    │   Embeddings    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │     Qdrant      │    │  Local Model    │
                       │ Vector Database │    │     Cache       │
                       └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Indexing:** Files → Tree-sitter AST → Entities → Stella Embeddings → Qdrant Collections
2. **Search:** Query → Stella Embedding → Qdrant Search → Ranked Results  
3. **Context Injection:** `<ccc>` Tags → Search → Context → Enhanced Prompt

### Project Isolation

Each project gets dedicated collections in the shared Qdrant instance:
- **Namespace Pattern:** `{project-name}-{collection-type}`
- **No Data Mixing:** Strict collection-based isolation
- **Resource Sharing:** Single Qdrant instance for efficiency
- **Independent Configuration:** Per-project settings and patterns

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/popperwin/claude-code-context.git
cd claude-code-context

# Development installation
./install-global.sh --dev

# Install development dependencies
pip install -e .[dev]

# Run tests
python -m pytest tests/ -v

# Code formatting
python -m black .
python -m isort .

# Type checking
python -m mypy core/ config/ scripts/
```

### Testing

```bash
# Unit tests
python -m pytest tests/test_*.py -v

# Integration tests  
python -m pytest tests/test_integration.py -v

# Performance tests
python -m pytest tests/test_integration.py::TestPerformanceIntegration -v

# Test project setup
./scripts/test-project.sh --quick
```

### Code Style

- **Python 3.12+** with type hints
- **Black** formatting (88 char limit)
- **Google-style** docstrings
- **Pydantic v2** for data validation
- **Async/await** for I/O operations

## 📚 Documentation

### API Documentation
- [Core Models API](docs/api/models.md)
- [Configuration API](docs/api/config.md) 
- [Search API](docs/api/search.md)
- [Embeddings API](docs/api/embeddings.md)

### Guides
- [Advanced Configuration](docs/guides/configuration.md)
- [Performance Tuning](docs/guides/performance.md)
- [Multi-Project Setup](docs/guides/multi-project.md)
- [Custom Embeddings](docs/guides/embeddings.md)

### Examples
- [Claude Code Workflows](examples/claude-workflows.md)
- [Search Patterns](examples/search-patterns.md)
- [Integration Examples](examples/integrations.md)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Anthropic** - Claude Code platform and inspiration
- **Qdrant** - High-performance vector database
- **Infgrad** - Stella embedding models
- **Tree-sitter** - Universal code parsing
- **Hugging Face** - Model hosting and transformers library

## 🔗 Links

- **Documentation:** [https://docs.anthropic.com/claude-code](https://docs.anthropic.com/claude-code)
- **Issues:** [https://github.com/popperwin/claude-code-context/issues](https://github.com/popperwin/claude-code-context/issues)
- **Discussions:** [https://github.com/popperwin/claude-code-context/discussions](https://github.com/popperwin/claude-code-context/discussions)
- **Claude Code:** [https://claude.ai/code](https://claude.ai/code)

---

**⚡ Built for developers who want intelligent code context without complexity.**

Made with ❤️ by the Claude Code team.