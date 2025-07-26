# data-pipeline

This project is configured with claude-code-context for semantic code search and context enrichment.

## Setup

The project has been automatically configured with:
- Qdrant collections for vector storage
- Stella embeddings for semantic search
- Claude Code hooks for context injection

## Usage

Use `<ccc>query</ccc>` tags in your Claude Code prompts to search for relevant context:

```
How do I implement data processing <ccc>data processing algorithms</ccc> in this codebase?
```

## Collections

This project uses the following Qdrant collections:
- `data-pipeline-code` - Code entities and functions
- `data-pipeline-relations` - Relationships between code elements  
- `data-pipeline-embeddings` - Semantic embeddings

## Commands

```bash
# Index the project
claude-indexer index -p . -c data-pipeline-code

# Search for code
claude-indexer search -c data-pipeline-code -q "your search query"

# Watch for changes
claude-indexer watch -p . -c data-pipeline-code
```
