#!/bin/bash
# Setup isolated testing environment for Sprint 1 validation
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_ENV_DIR="$SCRIPT_DIR/test-env"
TEST_VENV="$TEST_ENV_DIR/venv"
TEST_LOGS="$TEST_ENV_DIR/logs"
TEST_CONFIGS="$TEST_ENV_DIR/configs"
TEST_PROJECTS="$TEST_ENV_DIR/projects"
DOCKER_NETWORK="claude-test-network"
QDRANT_CONTAINER="claude-test-qdrant"
QDRANT_PORT="6334"  # Use non-standard port for isolation

log() {
    echo -e "${BLUE}[TEST-ENV]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[TEST-ENV]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[TEST-ENV]${NC} $1"
}

log_error() {
    echo -e "${RED}[TEST-ENV]${NC} $1"
}

setup_directories() {
    log "Setting up test directories..."
    
    # Clean up any existing test environment
    if [ -d "$TEST_ENV_DIR" ]; then
        log_warning "Removing existing test environment"
        rm -rf "$TEST_ENV_DIR"
    fi
    
    # Create directory structure
    mkdir -p "$TEST_ENV_DIR"
    mkdir -p "$TEST_LOGS"
    mkdir -p "$TEST_CONFIGS"
    mkdir -p "$TEST_PROJECTS"
    
    log_success "Test directories created"
}

setup_python_venv() {
    log "Creating Python virtual environment..."
    
    # Check Python version
    if ! python3.12 --version >/dev/null 2>&1; then
        if ! python3 --version | grep -E "3\.(12|13)" >/dev/null 2>&1; then
            log_error "Python 3.12+ required but not found"
            exit 1
        fi
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python3.12"
    fi
    
    # Create virtual environment
    $PYTHON_CMD -m venv "$TEST_VENV"
    
    # Activate and upgrade pip
    source "$TEST_VENV/bin/activate"
    pip install --upgrade pip setuptools wheel
    
    log_success "Python virtual environment created"
}

setup_docker_network() {
    log "Setting up Docker network..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    # Remove existing network if it exists
    if docker network ls | grep -q "$DOCKER_NETWORK"; then
        log_warning "Removing existing Docker network"
        docker network rm "$DOCKER_NETWORK" || true
    fi
    
    # Create isolated network
    docker network create "$DOCKER_NETWORK"
    
    log_success "Docker network created: $DOCKER_NETWORK"
}

setup_test_qdrant() {
    log "Setting up test Qdrant instance..."
    
    # Stop and remove existing container
    if docker ps -a | grep -q "$QDRANT_CONTAINER"; then
        log_warning "Stopping existing Qdrant container"
        docker stop "$QDRANT_CONTAINER" || true
        docker rm "$QDRANT_CONTAINER" || true
    fi
    
    # Start Qdrant container
    docker run -d \
        --name "$QDRANT_CONTAINER" \
        --network "$DOCKER_NETWORK" \
        -p "${QDRANT_PORT}:6333" \
        -v "$TEST_ENV_DIR/qdrant_storage:/qdrant/storage" \
        qdrant/qdrant:latest
    
    # Wait for Qdrant to be ready
    log "Waiting for Qdrant to be ready..."
    for i in {1..30}; do
        if curl -s "http://localhost:${QDRANT_PORT}/health" >/dev/null 2>&1; then
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Qdrant failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
    
    log_success "Test Qdrant instance ready on port $QDRANT_PORT"
}

install_test_dependencies() {
    log "Installing test dependencies..."
    
    source "$TEST_VENV/bin/activate"
    
    # Install the project in development mode
    cd "$PROJECT_ROOT"
    pip install -e ".[dev]"
    
    log_success "Test dependencies installed"
}

create_test_config() {
    log "Creating test configuration..."
    
    cat > "$TEST_CONFIGS/test-env.sh" << EOF
#!/bin/bash
# Test environment variables

export TEST_ENV_DIR="$TEST_ENV_DIR"
export TEST_VENV="$TEST_VENV"
export TEST_LOGS="$TEST_LOGS"
export TEST_CONFIGS="$TEST_CONFIGS"
export TEST_PROJECTS="$TEST_PROJECTS"
export DOCKER_NETWORK="$DOCKER_NETWORK"
export QDRANT_CONTAINER="$QDRANT_CONTAINER"
export QDRANT_PORT="$QDRANT_PORT"
export QDRANT_URL="http://localhost:$QDRANT_PORT"

# Override default configurations for testing
export CLAUDE_INDEXER_CONFIG_DIR="$TEST_CONFIGS"
export CLAUDE_INDEXER_CACHE_DIR="$TEST_ENV_DIR/cache"
export CLAUDE_INDEXER_LOG_LEVEL="DEBUG"

# Python virtual environment activation
source "$TEST_VENV/bin/activate"

# Add project root to Python path
export PYTHONPATH="$PROJECT_ROOT:\${PYTHONPATH:-}"

alias test-python="$TEST_VENV/bin/python"
alias test-pytest="$TEST_VENV/bin/pytest"
alias test-cleanup="$SCRIPT_DIR/cleanup-test-env.sh"

echo "Test environment activated"
echo "  • Qdrant: http://localhost:$QDRANT_PORT"
echo "  • Python: $TEST_VENV/bin/python"
echo "  • Logs: $TEST_LOGS"
echo "  • Projects: $TEST_PROJECTS"
EOF
    
    chmod +x "$TEST_CONFIGS/test-env.sh"
    
    log_success "Test configuration created"
}

main() {
    log "Setting up isolated test environment for Sprint 1 validation"
    
    setup_directories
    setup_python_venv
    setup_docker_network
    setup_test_qdrant
    install_test_dependencies
    create_test_config
    
    log_success "Test environment setup complete!"
    echo
    echo -e "${GREEN}To activate the test environment:${NC}"
    echo -e "  source $TEST_CONFIGS/test-env.sh"
    echo
    echo -e "${GREEN}Available commands:${NC}"
    echo -e "  test-python    - Use isolated Python"
    echo -e "  test-pytest    - Run tests"
    echo -e "  test-cleanup   - Clean up environment"
    echo
    echo -e "${GREEN}Next steps:${NC}"
    echo -e "  ./run-all-tests.sh"
}

main "$@"