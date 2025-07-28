#!/bin/bash
# Test environment variables

export TEST_ENV_DIR="/Users/goku/code_projects/claude-code-context/test-harness/test-env"
export TEST_VENV="/Users/goku/code_projects/claude-code-context/test-harness/test-env/venv"
export TEST_LOGS="/Users/goku/code_projects/claude-code-context/test-harness/test-env/logs"
export TEST_CONFIGS="/Users/goku/code_projects/claude-code-context/test-harness/test-env/configs"
export TEST_PROJECTS="/Users/goku/code_projects/claude-code-context/test-harness/test-env/projects"
export DOCKER_NETWORK="claude-test-network"
export QDRANT_CONTAINER="claude-test-qdrant"
export QDRANT_PORT="6334"
export QDRANT_URL="http://localhost:6334"

# Override default configurations for testing
export CLAUDE_INDEXER_CONFIG_DIR="/Users/goku/code_projects/claude-code-context/test-harness/test-env/configs"
export CLAUDE_INDEXER_CACHE_DIR="/Users/goku/code_projects/claude-code-context/test-harness/test-env/cache"
export CLAUDE_INDEXER_LOG_LEVEL="DEBUG"

# Integration and performance test settings
export CLAUDE_TEST_MODE="integration"
export INTEGRATION_COLLECTION="integration-test-code"
export PERFORMANCE_TEST_TIMEOUT=300
export EMBEDDING_PERFORMANCE_TARGET_MS=50
export PAYLOAD_SEARCH_TARGET_MS=5
export SEMANTIC_SEARCH_TARGET_MS=50
export HYBRID_SEARCH_TARGET_MS=100

# Test data settings
export REAL_DATA_SAMPLE_SIZE=100
export SEARCH_TEST_SAMPLE_SIZE=50
export CONCURRENT_TEST_WORKERS=5

# Python virtual environment activation
source "/Users/goku/code_projects/claude-code-context/test-harness/test-env/venv/bin/activate"

# Add project root to Python path
export PYTHONPATH="/Users/goku/code_projects/claude-code-context:${PYTHONPATH:-}"

alias test-python="/Users/goku/code_projects/claude-code-context/test-harness/test-env/venv/bin/python"
alias test-pytest="/Users/goku/code_projects/claude-code-context/test-harness/test-env/venv/bin/pytest"
alias test-cleanup="/Users/goku/code_projects/claude-code-context/test-harness/cleanup-test-env.sh"

echo "Test environment activated"
echo "  • Qdrant: http://localhost:6334"
echo "  • Python: /Users/goku/code_projects/claude-code-context/test-harness/test-env/venv/bin/python"
echo "  • Logs: /Users/goku/code_projects/claude-code-context/test-harness/test-env/logs"
echo "  • Projects: /Users/goku/code_projects/claude-code-context/test-harness/test-env/projects"
echo "  • Performance monitoring: enabled"
