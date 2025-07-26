#!/bin/bash
# setup-project.sh - Automated project setup with collection isolation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
QDRANT_PORT=6333
QDRANT_URL="http://localhost:${QDRANT_PORT}"
QDRANT_TIMEOUT=10
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
PROJECT_NAME=""
PROJECT_PATH="."
COLLECTION_PREFIX=""
OVERWRITE=false
SKIP_QDRANT_CHECK=false
VERBOSE=false

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

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}🔍 $1${NC}"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 <project-name> [options]"
    echo ""
    echo "Arguments:"
    echo "  project-name          Name of the project (required)"
    echo ""
    echo "Options:"
    echo "  -p, --path PATH       Project path (default: current directory)"
    echo "  -c, --collection PREFIX  Collection prefix (default: same as project-name)"
    echo "  --port PORT          Qdrant port (default: 6333)"
    echo "  --overwrite          Overwrite existing configuration"
    echo "  --skip-qdrant        Skip Qdrant connection check"
    echo "  -v, --verbose        Verbose output"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 my-api-project"
    echo "  $0 frontend-app --path /path/to/project"
    echo "  $0 shared-lib --collection shared-components"
    echo ""
}

# Function to validate project name
validate_project_name() {
    local name="$1"
    
    if [[ -z "$name" ]]; then
        log_error "Project name cannot be empty"
        return 1
    fi
    
    # Check if name contains only valid characters
    if [[ ! "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$ ]] && [[ ${#name} -gt 1 ]]; then
        log_error "Project name must be alphanumeric with dashes/underscores (no spaces)"
        return 1
    fi
    
    # Single character names should be alphanumeric
    if [[ ${#name} -eq 1 ]] && [[ ! "$name" =~ ^[a-zA-Z0-9]$ ]]; then
        log_error "Single character project name must be alphanumeric"
        return 1
    fi
    
    return 0
}

# Function to normalize project name for collection usage
normalize_collection_prefix() {
    local name="$1"
    # Convert to lowercase and replace spaces/underscores with dashes
    echo "$name" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr ' ' '-'
}

# Function to check if Qdrant is accessible
check_qdrant_connection() {
    if [ "$SKIP_QDRANT_CHECK" = true ]; then
        log_warning "Skipping Qdrant connection check"
        return 0
    fi
    
    log_info "Checking Qdrant connection..."
    
    if ! command -v curl >/dev/null 2>&1; then
        log_warning "curl not found, skipping Qdrant check"
        return 0
    fi
    
    if curl -s --max-time "$QDRANT_TIMEOUT" "$QDRANT_URL/health" >/dev/null 2>&1; then
        log_success "Qdrant is accessible at $QDRANT_URL"
        return 0
    else
        log_error "Qdrant is not accessible at $QDRANT_URL"
        log_info "Start Qdrant with: ./scripts/setup-qdrant.sh"
        return 1
    fi
}

# Function to create project directory structure
create_project_structure() {
    local project_path="$1"
    
    log_info "Creating project directory structure..."
    
    # Create main directories
    mkdir -p "$project_path/.claude-indexer"
    mkdir -p "$project_path/.claude/hooks"
    
    # Set appropriate permissions
    chmod 755 "$project_path/.claude-indexer"
    chmod 755 "$project_path/.claude"
    chmod 755 "$project_path/.claude/hooks"
    
    log_verbose "Created directories:"
    log_verbose "  - $project_path/.claude-indexer"
    log_verbose "  - $project_path/.claude/hooks"
    
    log_success "Project structure created"
}

# Function to generate project configuration
generate_project_config() {
    local project_name="$1"
    local project_path="$2"
    local collection_prefix="$3"
    
    log_info "Generating project configuration..."
    
    local config_file="$project_path/.claude-indexer/config.json"
    
    # Check if config exists and overwrite flag
    if [[ -f "$config_file" ]] && [[ "$OVERWRITE" != true ]]; then
        log_warning "Configuration already exists at $config_file"
        read -p "Overwrite existing configuration? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Keeping existing configuration"
            return 0
        fi
    fi
    
    # Create configuration JSON
    cat > "$config_file" << EOF
{
  "name": "$project_name",
  "path": "$(cd "$project_path" && pwd)",
  "collection_prefix": "$collection_prefix",
  "qdrant": {
    "url": "$QDRANT_URL",
    "timeout": 60.0,
    "collections": {
      "code": "${collection_prefix}-code",
      "relations": "${collection_prefix}-relations",
      "embeddings": "${collection_prefix}-embeddings"
    },
    "batch_size": 100,
    "parallel_requests": 4,
    "vector_size": 1024,
    "distance_metric": "cosine"
  },
  "stella": {
    "model_name": "stella_en_400M_v5",
    "dimensions": 1024,
    "cache_dir": "$(echo ~)/.cache/claude-indexer/stella",
    "device": null,
    "batch_size": 32,
    "max_length": 512,
    "normalize_embeddings": true,
    "use_fp16": true
  },
  "indexing": {
    "include_patterns": [
      "*.py", "*.js", "*.ts", "*.jsx", "*.tsx",
      "*.go", "*.rs", "*.java", "*.cpp", "*.c", "*.h", "*.hpp",
      "*.cs", "*.rb", "*.php", "*.swift", "*.kt", "*.scala",
      "*.md", "*.txt", "*.json", "*.yaml", "*.yml", "*.toml"
    ],
    "exclude_patterns": [
      "node_modules/*", ".git/*", "__pycache__/*", "*.pyc",
      "venv/*", ".venv/*", "env/*", ".env/*",
      "dist/*", "build/*", "target/*", ".next/*",
      "*.log", "*.tmp", "*.cache", ".DS_Store",
      "*.min.js", "*.min.css", "coverage/*"
    ],
    "max_file_size_mb": 10,
    "max_files_per_batch": 50,
    "extract_docstrings": true,
    "extract_comments": true,
    "include_test_files": true
  },
  "description": null,
  "version": "1.0.0",
  "enable_watch_mode": false,
  "auto_index_on_change": true,
  "max_concurrent_operations": 4,
  "cache_embeddings": true
}
EOF
    
    log_verbose "Configuration written to: $config_file"
    log_success "Project configuration created"
}

# Function to setup Claude Code hooks
setup_claude_hooks() {
    local project_name="$1"
    local project_path="$2"
    local collection_prefix="$3"
    
    log_info "Setting up Claude Code hooks..."
    
    local settings_file="$project_path/.claude/settings.json"
    
    # Create Claude Code settings
    cat > "$settings_file" << EOF
{
  "hooks": {
    "user_prompt_submit": [
      {
        "command": ["python", "-m", "hooks.user_prompt_submit"],
        "working_directory": "$(cd "$project_path" && pwd)",
        "environment": {
          "PROJECT_NAME": "$project_name",
          "COLLECTION_PREFIX": "$collection_prefix",
          "QDRANT_URL": "$QDRANT_URL"
        }
      }
    ]
  },
  "project_info": {
    "name": "$project_name",
    "collection_prefix": "$collection_prefix",
    "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  }
}
EOF
    
    log_verbose "Claude Code settings written to: $settings_file"
    log_success "Claude Code hooks configured"
}

# Function to create Qdrant collections
create_qdrant_collections() {
    local collection_prefix="$1"
    
    if [ "$SKIP_QDRANT_CHECK" = true ]; then
        log_warning "Skipping Qdrant collection creation"
        return 0
    fi
    
    log_info "Creating Qdrant collections..."
    
    local collections=("code" "relations" "embeddings")
    
    for collection_type in "${collections[@]}"; do
        local collection_name="${collection_prefix}-${collection_type}"
        
        log_verbose "Creating collection: $collection_name"
        
        # Check if collection already exists
        local response=$(curl -s --max-time "$QDRANT_TIMEOUT" "$QDRANT_URL/collections/$collection_name" 2>/dev/null)
        if echo "$response" | grep -q '"status":"ok"'; then
            log_warning "Collection '$collection_name' already exists"
            continue
        fi
        
        # Create collection with appropriate configuration
        local config=""
        case $collection_type in
            "code")
                config='{
                    "vectors": {
                        "size": 1024,
                        "distance": "Cosine"
                    },
                    "optimizers_config": {
                        "default_segment_number": 2,
                        "max_segment_size_mb": 100
                    },
                    "hnsw_config": {
                        "m": 16,
                        "ef_construct": 100
                    }
                }'
                ;;
            "relations")
                config='{
                    "vectors": {
                        "size": 1024,
                        "distance": "Cosine"
                    },
                    "optimizers_config": {
                        "default_segment_number": 1,
                        "max_segment_size_mb": 50
                    }
                }'
                ;;
            "embeddings")
                config='{
                    "vectors": {
                        "size": 1024,
                        "distance": "Cosine"
                    },
                    "optimizers_config": {
                        "default_segment_number": 2,
                        "max_segment_size_mb": 200
                    }
                }'
                ;;
        esac
        
        # Create the collection
        if curl -s --max-time "$QDRANT_TIMEOUT" \
            -X PUT "$QDRANT_URL/collections/$collection_name" \
            -H "Content-Type: application/json" \
            -d "$config" >/dev/null 2>&1; then
            log_success "Created collection: $collection_name"
        else
            log_error "Failed to create collection: $collection_name"
            return 1
        fi
    done
    
    log_success "All collections created successfully"
}

# Function to create example files
create_example_files() {
    local project_path="$1"
    local project_name="$2"
    
    # Only create examples if directory is empty or nearly empty
    local file_count
    file_count=$(find "$project_path" -maxdepth 1 -type f | wc -l)
    
    if [ "$file_count" -gt 3 ]; then
        log_verbose "Project directory has files, skipping example creation"
        return 0
    fi
    
    if [[ "$OVERWRITE" != true ]]; then
        read -p "Create example files to test indexing? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    log_info "Creating example files..."
    
    # Create a simple Python example
    cat > "$project_path/example.py" << EOF
"""
Example Python module for $project_name

This file demonstrates indexable code for testing the claude-code-context system.
"""

class DataProcessor:
    """Process and analyze data with various algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
    
    def process_data(self, data: list) -> dict:
        """
        Process input data and return summary statistics.
        
        Args:
            data: List of numeric values to process
            
        Returns:
            Dictionary containing processing results
        """
        if not data:
            return {"error": "No data provided"}
        
        result = {
            "count": len(data),
            "sum": sum(data),
            "average": sum(data) / len(data),
            "min": min(data),
            "max": max(data)
        }
        
        self.processed_count += 1
        return result
    
    async def async_process(self, data: list) -> dict:
        """Asynchronous version of data processing."""
        import asyncio
        await asyncio.sleep(0.1)  # Simulate async work
        return self.process_data(data)

def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Example usage
if __name__ == "__main__":
    processor = DataProcessor("example")
    test_data = [1, 2, 3, 4, 5]
    result = processor.process_data(test_data)
    print(f"Processed data: {result}")
EOF
    
    # Create README
    cat > "$project_path/README.md" << EOF
# $project_name

This project is configured with claude-code-context for semantic code search and context enrichment.

## Setup

The project has been automatically configured with:
- Qdrant collections for vector storage
- Stella embeddings for semantic search
- Claude Code hooks for context injection

## Usage

Use \`<ccc>query</ccc>\` tags in your Claude Code prompts to search for relevant context:

\`\`\`
How do I implement data processing <ccc>data processing algorithms</ccc> in this codebase?
\`\`\`

## Collections

This project uses the following Qdrant collections:
- \`${COLLECTION_PREFIX}-code\` - Code entities and functions
- \`${COLLECTION_PREFIX}-relations\` - Relationships between code elements  
- \`${COLLECTION_PREFIX}-embeddings\` - Semantic embeddings

## Commands

\`\`\`bash
# Index the project
claude-indexer index -p . -c ${COLLECTION_PREFIX}-code

# Search for code
claude-indexer search -c ${COLLECTION_PREFIX}-code -q "your search query"

# Watch for changes
claude-indexer watch -p . -c ${COLLECTION_PREFIX}-code
\`\`\`
EOF
    
    log_success "Example files created"
}

# Function to validate setup
validate_setup() {
    local project_path="$1"
    local collection_prefix="$2"
    
    log_info "Validating setup..."
    
    local errors=0
    
    # Check required files
    local required_files=(
        "$project_path/.claude-indexer/config.json"
        "$project_path/.claude/settings.json"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Missing required file: $file"
            ((errors++))
        else
            log_verbose "Found: $file"
        fi
    done
    
    # Check Qdrant collections if not skipping
    if [ "$SKIP_QDRANT_CHECK" != true ]; then
        local collections=("code" "relations" "embeddings")
        
        for collection_type in "${collections[@]}"; do
            local collection_name="${collection_prefix}-${collection_type}"
            
            local response=$(curl -s --max-time "$QDRANT_TIMEOUT" "$QDRANT_URL/collections/$collection_name" 2>/dev/null)
            if echo "$response" | grep -q '"status":"ok"'; then
                log_verbose "Collection exists: $collection_name"
            else
                log_error "Collection missing: $collection_name"
                ((errors++))
            fi
        done
    fi
    
    if [ $errors -eq 0 ]; then
        log_success "Setup validation passed"
        return 0
    else
        log_error "Setup validation failed with $errors errors"
        return 1
    fi
}

# Function to show completion information
show_completion_info() {
    local project_name="$1"
    local project_path="$2"
    local collection_prefix="$3"
    
    echo -e "\n${GREEN}🎉 Project setup complete!${NC}"
    echo -e "\n${BLUE}📊 Project Information:${NC}"
    echo -e "   Name: $project_name"
    echo -e "   Path: $(cd "$project_path" && pwd)"
    echo -e "   Collection Prefix: $collection_prefix"
    echo -e "\n${BLUE}📁 Collections Created:${NC}"
    echo -e "   • ${collection_prefix}-code"
    echo -e "   • ${collection_prefix}-relations"
    echo -e "   • ${collection_prefix}-embeddings"
    echo -e "\n${BLUE}⚡ Next Steps:${NC}"
    echo -e "   1. Index your project: claude-indexer index -p '$project_path' -c '${collection_prefix}-code'"
    echo -e "   2. Test search: claude-indexer search -c '${collection_prefix}-code' -q 'function'"
    echo -e "   3. Use <ccc>query</ccc> tags in Claude Code prompts"
    echo -e "\n${BLUE}📖 Documentation:${NC}"
    echo -e "   • Configuration: $project_path/.claude-indexer/config.json"
    echo -e "   • Claude Settings: $project_path/.claude/settings.json"
    echo -e "   • Qdrant API: $QDRANT_URL"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            PROJECT_PATH="$2"
            shift 2
            ;;
        -c|--collection)
            COLLECTION_PREFIX="$2"
            shift 2
            ;;
        --port)
            QDRANT_PORT="$2"
            QDRANT_URL="http://localhost:${QDRANT_PORT}"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --skip-qdrant)
            SKIP_QDRANT_CHECK=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            if [[ -z "$PROJECT_NAME" ]]; then
                PROJECT_NAME="$1"
            else
                log_error "Multiple project names specified"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$PROJECT_NAME" ]]; then
    log_error "Project name is required"
    show_usage
    exit 1
fi

# Validate project name
if ! validate_project_name "$PROJECT_NAME"; then
    exit 1
fi

# Set collection prefix if not provided
if [[ -z "$COLLECTION_PREFIX" ]]; then
    COLLECTION_PREFIX=$(normalize_collection_prefix "$PROJECT_NAME")
fi

# Convert to absolute path
PROJECT_PATH=$(cd "$PROJECT_PATH" && pwd)

# Validate project path
if [[ ! -d "$PROJECT_PATH" ]]; then
    log_error "Project path does not exist: $PROJECT_PATH"
    exit 1
fi

# Main execution
main() {
    echo -e "${CYAN}🚀 Setting up project '$PROJECT_NAME'${NC}"
    echo -e "${CYAN}   Path: $PROJECT_PATH${NC}"
    echo -e "${CYAN}   Collections: ${COLLECTION_PREFIX}-*${NC}"
    echo
    
    # Check Qdrant connection
    if ! check_qdrant_connection; then
        if [ "$SKIP_QDRANT_CHECK" != true ]; then
            exit 1
        fi
    fi
    
    # Create project structure
    create_project_structure "$PROJECT_PATH"
    
    # Generate configuration
    generate_project_config "$PROJECT_NAME" "$PROJECT_PATH" "$COLLECTION_PREFIX"
    
    # Setup Claude hooks
    setup_claude_hooks "$PROJECT_NAME" "$PROJECT_PATH" "$COLLECTION_PREFIX"
    
    # Create Qdrant collections
    if ! create_qdrant_collections "$COLLECTION_PREFIX"; then
        if [ "$SKIP_QDRANT_CHECK" != true ]; then
            log_error "Failed to create collections, but continuing..."
        fi
    fi
    
    # Create example files
    create_example_files "$PROJECT_PATH" "$PROJECT_NAME"
    
    # Validate setup
    if ! validate_setup "$PROJECT_PATH" "$COLLECTION_PREFIX"; then
        log_warning "Setup validation had issues, but basic setup is complete"
    fi
    
    # Show completion info
    show_completion_info "$PROJECT_NAME" "$PROJECT_PATH" "$COLLECTION_PREFIX"
}

# Run main function
main