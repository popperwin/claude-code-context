#!/bin/bash
# scripts/setup-qdrant.sh - Setup single Qdrant instance for all projects

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
QDRANT_VERSION="v1.7.4"
QDRANT_CONTAINER_NAME="claude-indexer-qdrant"
QDRANT_HTTP_PORT="6333"
QDRANT_GRPC_PORT="6334"
QDRANT_URL="http://localhost:${QDRANT_HTTP_PORT}"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
STORAGE_DIR="${PROJECT_ROOT}/qdrant_storage"
CONFIG_DIR="${PROJECT_ROOT}/qdrant_config"

echo -e "${BLUE}🚀 Setting up Qdrant vector database...${NC}"

# Function to print colored output
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command_exists docker; then
        log_error "Docker is not installed or not in PATH"
        log_info "Please install Docker from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        log_info "Please start Docker and try again"
        exit 1
    fi
    
    # Check if docker-compose is available
    if command_exists docker-compose; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        log_error "Docker Compose is not available"
        log_info "Please install Docker Compose or update Docker to a version that includes it"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Check if Qdrant is already running
check_existing_qdrant() {
    log_info "Checking for existing Qdrant instance..."
    
    if docker ps --format "{{.Names}}" | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
        log_warning "Qdrant container '${QDRANT_CONTAINER_NAME}' is already running"
        
        # Test if it's healthy
        if curl -s "${QDRANT_URL}/health" >/dev/null 2>&1; then
            log_success "Existing Qdrant instance is healthy"
            
            read -p "Do you want to recreate the Qdrant instance? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "Stopping existing Qdrant instance..."
                docker stop "${QDRANT_CONTAINER_NAME}" >/dev/null
                docker rm "${QDRANT_CONTAINER_NAME}" >/dev/null
                log_success "Existing instance removed"
            else
                log_info "Using existing Qdrant instance"
                check_qdrant_health
                return 0
            fi
        else
            log_warning "Existing instance is not healthy, will recreate"
            docker stop "${QDRANT_CONTAINER_NAME}" >/dev/null 2>&1 || true
            docker rm "${QDRANT_CONTAINER_NAME}" >/dev/null 2>&1 || true
        fi
    fi
    
    # Also check if stopped container exists
    if docker ps -a --format "{{.Names}}" | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
        log_info "Removing stopped Qdrant container..."
        docker rm "${QDRANT_CONTAINER_NAME}" >/dev/null
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating storage directories..."
    
    mkdir -p "${STORAGE_DIR}"
    mkdir -p "${CONFIG_DIR}"
    
    # Set appropriate permissions
    chmod 755 "${STORAGE_DIR}"
    chmod 755 "${CONFIG_DIR}"
    
    log_success "Directories created: ${STORAGE_DIR}, ${CONFIG_DIR}"
}

# Create Qdrant configuration file
create_qdrant_config() {
    log_info "Creating Qdrant configuration..."
    
    cat > "${CONFIG_DIR}/config.yaml" << 'EOF'
# Qdrant configuration for claude-code-context
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
  enable_cors: true
  
storage:
  # Optimized for multiple collections (projects)
  storage_path: /qdrant/storage
  snapshots_path: /qdrant/storage/snapshots
  on_disk_payload: true
  wal_capacity_mb: 32
  wal_segments_ahead: 0

cluster:
  enabled: false

telemetry:
  disabled: true

log_level: INFO

# Performance optimizations for multi-project usage
optimizer:
  deleted_threshold: 0.2
  vacuum_min_vector_number: 1000
  default_segment_number: 0
  max_segment_size_mb: 100
  memmap_threshold_mb: 100
  indexing_threshold_mb: 100
EOF
    
    log_success "Configuration file created"
}

# Start Qdrant using Docker Compose
start_qdrant() {
    log_info "Starting Qdrant using Docker Compose..."
    
    cd "${PROJECT_ROOT}"
    
    # Pull latest image
    log_info "Pulling Qdrant image..."
    ${COMPOSE_CMD} pull qdrant
    
    # Start the service
    log_info "Starting Qdrant service..."
    ${COMPOSE_CMD} up -d qdrant
    
    log_success "Qdrant container started"
}

# Wait for Qdrant to be ready
wait_for_qdrant() {
    log_info "Waiting for Qdrant to be ready..."
    
    local max_attempts=60  # 60 seconds max
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "${QDRANT_URL}/health" >/dev/null 2>&1; then
            log_success "Qdrant is ready!"
            return 0
        fi
        
        if [ $((attempt % 10)) -eq 0 ]; then
            log_info "Still waiting... (${attempt}s)"
        fi
        
        sleep 1
        ((attempt++))
    done
    
    log_error "Qdrant failed to start within ${max_attempts} seconds"
    log_info "Check logs with: docker logs ${QDRANT_CONTAINER_NAME}"
    return 1
}

# Check Qdrant health and display info
check_qdrant_health() {
    log_info "Performing health check..."
    
    # Basic health check
    if ! curl -s "${QDRANT_URL}/health" >/dev/null; then
        log_error "Qdrant health check failed"
        return 1
    fi
    
    # Get version info
    local version_info
    version_info=$(curl -s "${QDRANT_URL}/" | grep -o '"version":"[^"]*"' | cut -d'"' -f4 2>/dev/null || echo "unknown")
    
    # Get collections info
    local collections_response
    collections_response=$(curl -s "${QDRANT_URL}/collections" 2>/dev/null || echo '{"result": {"collections": []}}')
    local collections_count
    collections_count=$(echo "$collections_response" | grep -o '"collections":\[[^]]*\]' | grep -o '\[.*\]' | grep -o '{' | wc -l 2>/dev/null || echo "0")
    
    log_success "Qdrant is healthy and running!"
    echo -e "${BLUE}📊 Qdrant Information:${NC}"
    echo -e "   Version: ${version_info}"
    echo -e "   HTTP endpoint: ${QDRANT_URL}"
    echo -e "   gRPC endpoint: localhost:${QDRANT_GRPC_PORT}"
    echo -e "   Collections: ${collections_count// /}"
    echo -e "   Storage: ${STORAGE_DIR}"
    
    return 0
}

# Create example collections for testing
create_example_collections() {
    if [ "${CREATE_EXAMPLES:-}" = "true" ]; then
        log_info "Creating example collections..."
        
        # Create a test collection
        curl -s -X PUT "${QDRANT_URL}/collections/test-collection" \
            -H "Content-Type: application/json" \
            -d '{
                "vectors": {
                    "size": 1024,
                    "distance": "Cosine"
                },
                "optimizers_config": {
                    "default_segment_number": 2
                }
            }' >/dev/null
        
        log_success "Example collection 'test-collection' created"
    fi
}

# Display usage information
show_usage_info() {
    echo -e "\n${GREEN}🎉 Qdrant setup complete!${NC}"
    echo -e "\n${BLUE}📖 Usage Information:${NC}"
    echo -e "   • Access Qdrant API: ${QDRANT_URL}"
    echo -e "   • View collections: ${QDRANT_URL}/collections"
    echo -e "   • Check health: ${QDRANT_URL}/health"
    echo -e "\n${BLUE}🔧 Management Commands:${NC}"
    echo -e "   • Stop Qdrant: ${COMPOSE_CMD} down"
    echo -e "   • View logs: docker logs ${QDRANT_CONTAINER_NAME}"
    echo -e "   • Restart: ${COMPOSE_CMD} restart qdrant"
    echo -e "   • Start dashboard: ${COMPOSE_CMD} --profile dashboard up -d"
    echo -e "\n${BLUE}📁 File Locations:${NC}"
    echo -e "   • Storage: ${STORAGE_DIR}"
    echo -e "   • Config: ${CONFIG_DIR}/config.yaml"
    echo -e "   • Docker Compose: ${PROJECT_ROOT}/docker-compose.yml"
}

# Cleanup function
cleanup() {
    if [ "${1:-}" = "full" ]; then
        log_info "Performing full cleanup..."
        
        # Stop and remove container
        docker stop "${QDRANT_CONTAINER_NAME}" >/dev/null 2>&1 || true
        docker rm "${QDRANT_CONTAINER_NAME}" >/dev/null 2>&1 || true
        
        # Remove data (ask for confirmation)
        read -p "Remove all Qdrant data? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "${STORAGE_DIR}"
            rm -rf "${CONFIG_DIR}"
            log_success "Data removed"
        fi
        
        # Remove Docker image
        read -p "Remove Qdrant Docker image? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker rmi "qdrant/qdrant:${QDRANT_VERSION}" >/dev/null 2>&1 || true
            log_success "Docker image removed"
        fi
    fi
}

# Parse command line arguments
case "${1:-}" in
    "start"|"")
        check_prerequisites
        check_existing_qdrant
        create_directories
        create_qdrant_config
        start_qdrant
        wait_for_qdrant
        check_qdrant_health
        create_example_collections
        show_usage_info
        ;;
    "stop")
        log_info "Stopping Qdrant..."
        cd "${PROJECT_ROOT}"
        ${COMPOSE_CMD} down
        log_success "Qdrant stopped"
        ;;
    "restart")
        log_info "Restarting Qdrant..."
        cd "${PROJECT_ROOT}"
        ${COMPOSE_CMD} restart qdrant
        wait_for_qdrant
        check_qdrant_health  
        ;;
    "status")
        check_qdrant_health
        ;;
    "logs")
        docker logs "${QDRANT_CONTAINER_NAME}" "${@:2}"
        ;;
    "cleanup")
        cleanup full
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start      Start Qdrant (default)"
        echo "  stop       Stop Qdrant"
        echo "  restart    Restart Qdrant"
        echo "  status     Check Qdrant status"
        echo "  logs       Show Qdrant logs"
        echo "  cleanup    Remove Qdrant and all data"
        echo "  help       Show this help"
        echo ""
        echo "Environment variables:"
        echo "  CREATE_EXAMPLES=true    Create example collections"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac