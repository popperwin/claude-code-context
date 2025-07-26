#!/bin/bash
# Complete cleanup of test environment
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ENV_DIR="$SCRIPT_DIR/test-env"
DOCKER_NETWORK="claude-test-network"
QDRANT_CONTAINER="claude-test-qdrant"

log() {
    echo -e "${BLUE}[CLEANUP]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[CLEANUP]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[CLEANUP]${NC} $1"
}

log_error() {
    echo -e "${RED}[CLEANUP]${NC} $1"
}

cleanup_docker() {
    log "Cleaning up Docker resources..."
    
    # Stop and remove Qdrant container
    if docker ps -a --format "{{.Names}}" | grep -q "^${QDRANT_CONTAINER}$"; then
        log "Stopping Qdrant container"
        docker stop "$QDRANT_CONTAINER" || true
        sleep 2  # Give it time to stop
        docker rm "$QDRANT_CONTAINER" || true
    fi
    
    # Remove Docker network (only after container is removed)
    if docker network ls --format "{{.Name}}" | grep -q "^${DOCKER_NETWORK}$"; then
        log "Removing Docker network"
        # Wait a bit more to ensure container is fully removed
        sleep 1
        docker network rm "$DOCKER_NETWORK" || true
    fi
    
    log_success "Docker resources cleaned up"
}

cleanup_directories() {
    log "Cleaning up test directories..."
    
    # Deactivate virtual environment if active
    if [[ "${VIRTUAL_ENV:-}" == *"test-env"* ]]; then
        log "Deactivating virtual environment"
        deactivate || true
    fi
    
    # Remove test environment directory
    if [ -d "$TEST_ENV_DIR" ]; then
        log "Removing test environment directory"
        rm -rf "$TEST_ENV_DIR"
    fi
    
    log_success "Test directories cleaned up"
}

cleanup_processes() {
    log "Cleaning up any running processes..."
    
    # Kill any Python processes running in test environment
    pkill -f "test-env/venv" || true
    
    # Clean up any orphaned Qdrant processes
    pkill -f "qdrant" || true
    
    log_success "Processes cleaned up"
}

cleanup_temp_files() {
    log "Cleaning up temporary files..."
    
    # Remove any temporary project directories that might have been created during testing
    find /tmp -name "claude-test-*" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Clean up any Python cache files
    find "$SCRIPT_DIR/.." -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$SCRIPT_DIR/.." -name "*.pyc" -delete 2>/dev/null || true
    
    log_success "Temporary files cleaned up"
}

verify_cleanup() {
    log "Verifying cleanup..."
    
    local cleanup_complete=true
    
    # Check Docker resources
    if docker ps -a | grep -q "$QDRANT_CONTAINER"; then
        log_error "Qdrant container still exists"
        cleanup_complete=false
    fi
    
    if docker network ls | grep -q "$DOCKER_NETWORK"; then
        log_error "Docker network still exists"
        cleanup_complete=false
    fi
    
    # Check directories
    if [ -d "$TEST_ENV_DIR" ]; then
        log_error "Test environment directory still exists"
        cleanup_complete=false
    fi
    
    if [ "$cleanup_complete" = true ]; then
        log_success "Cleanup verification passed"
        return 0
    else
        log_error "Cleanup verification failed"
        return 1
    fi
}

main() {
    log "Starting complete test environment cleanup"
    
    cleanup_docker
    cleanup_processes
    cleanup_directories
    cleanup_temp_files
    
    if verify_cleanup; then
        log_success "Test environment cleanup complete!"
        echo
        echo -e "${GREEN}✅ All test resources have been cleaned up${NC}"
        echo -e "${GREEN}✅ System restored to pre-test state${NC}"
    else
        log_error "Cleanup completed with warnings"
        echo
        echo -e "${YELLOW}⚠️  Some resources may require manual cleanup${NC}"
        return 1
    fi
}

main "$@"