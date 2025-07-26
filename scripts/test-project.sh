#!/bin/bash
# scripts/test-project.sh - Project validation and testing script

set -euo pipefail

# Colors for output  
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
QDRANT_URL="http://localhost:6333"
QDRANT_TIMEOUT=10
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Test options
PROJECT_PATH="."
PROJECT_NAME=""
COLLECTION_PREFIX=""
VERBOSE=false
SKIP_QDRANT=false
SKIP_COLLECTIONS=false
QUICK_MODE=false

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

log_test() {
    echo -e "${CYAN}🧪 Testing: $1${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [project-path] [options]"
    echo ""
    echo "Arguments:"
    echo "  project-path          Path to project directory (default: current directory)"
    echo ""
    echo "Options:"
    echo "  -n, --name NAME       Project name (auto-detected if not provided)"
    echo "  -c, --collection PREFIX  Collection prefix (auto-detected if not provided)"
    echo "  --skip-qdrant         Skip Qdrant connectivity tests"
    echo "  --skip-collections    Skip collection validation tests"
    echo "  --quick               Quick mode - essential tests only"
    echo "  -v, --verbose         Verbose output"
    echo "  -h, --help            Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                    # Test current directory"
    echo "  $0 /path/to/project   # Test specific project"
    echo "  $0 --quick            # Quick validation only"
    echo ""
}

# Function to detect project information
detect_project_info() {
    local project_path="$1"
    
    log_verbose "Detecting project information from $project_path"
    
    # Try to read from config file
    local config_file="$project_path/.claude-indexer/config.json"
    if [[ -f "$config_file" ]]; then
        log_verbose "Found config file: $config_file"
        
        # Extract project name and collection prefix using basic JSON parsing
        if command -v python3 >/dev/null 2>&1; then
            PROJECT_NAME=$(python3 -c "
import json, sys
try:
    with open('$config_file', 'r') as f:
        data = json.load(f)
    print(data.get('name', ''))
except:
    pass
" 2>/dev/null || echo "")
            
            COLLECTION_PREFIX=$(python3 -c "
import json, sys
try:
    with open('$config_file', 'r') as f:
        data = json.load(f)
    print(data.get('collection_prefix', ''))
except:
    pass
" 2>/dev/null || echo "")
        fi
    fi
    
    # Fallback to directory name
    if [[ -z "$PROJECT_NAME" ]]; then
        PROJECT_NAME=$(basename "$(cd "$project_path" && pwd)" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr ' ' '-')
    fi
    
    if [[ -z "$COLLECTION_PREFIX" ]]; then
        COLLECTION_PREFIX="$PROJECT_NAME"
    fi
    
    log_verbose "Detected project name: $PROJECT_NAME"
    log_verbose "Detected collection prefix: $COLLECTION_PREFIX"
}

# Function to test file system structure
test_filesystem_structure() {
    log_test "File system structure"
    
    local errors=0
    local project_path="$1"
    
    # Required directories
    local required_dirs=(
        "$project_path/.claude-indexer"
        "$project_path/.claude"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            log_verbose "✅ Directory exists: $dir"
        else
            log_error "Missing directory: $dir"
            ((errors++))
        fi
    done
    
    # Required files
    local required_files=(
        "$project_path/.claude-indexer/config.json"
        "$project_path/.claude/settings.json"
    )
    
    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            log_verbose "✅ File exists: $file"
            
            # Test if file is valid JSON
            if command -v python3 >/dev/null 2>&1; then
                if python3 -c "import json; json.load(open('$file'))" 2>/dev/null; then
                    log_verbose "✅ Valid JSON: $file"
                else
                    log_error "Invalid JSON: $file"
                    ((errors++))
                fi
            fi
        else
            log_error "Missing file: $file"
            ((errors++))
        fi
    done
    
    if [ $errors -eq 0 ]; then
        log_success "File system structure is valid"
        return 0
    else
        log_error "File system structure has $errors errors"
        return 1
    fi
}

# Function to test configuration validity
test_configuration() {
    log_test "Configuration validity"
    
    local project_path="$1"
    local config_file="$project_path/.claude-indexer/config.json"
    local settings_file="$project_path/.claude/settings.json"
    
    if [[ ! -f "$config_file" ]]; then
        log_error "Configuration file not found: $config_file"
        return 1
    fi
    
    # Test configuration completeness using Python
    if ! command -v python3 >/dev/null 2>&1; then
        log_warning "Python3 not available, skipping configuration validation"
        return 0
    fi
    
    local config_errors
    config_errors=$(python3 -c "
import json
import sys

try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    
    errors = []
    
    # Check required fields
    required_fields = ['name', 'path', 'collection_prefix', 'qdrant', 'stella', 'indexing']
    for field in required_fields:
        if field not in config:
            errors.append(f'Missing required field: {field}')
    
    # Check Qdrant configuration
    if 'qdrant' in config:
        qdrant_config = config['qdrant']
        if 'url' not in qdrant_config:
            errors.append('Missing qdrant.url')
        if 'collections' not in qdrant_config:
            errors.append('Missing qdrant.collections')
    
    # Check collection names
    if 'collection_prefix' in config and 'qdrant' in config and 'collections' in config['qdrant']:
        prefix = config['collection_prefix']
        collections = config['qdrant']['collections']
        
        expected_collections = ['code', 'relations', 'embeddings']
        for coll_type in expected_collections:
            if coll_type not in collections:
                errors.append(f'Missing collection: {coll_type}')
            elif not collections[coll_type].startswith(prefix):
                errors.append(f'Collection {coll_type} does not use prefix {prefix}')
    
    if errors:
        for error in errors:
            print(error)
        sys.exit(1)
    else:
        print('Configuration is valid')
        sys.exit(0)

except Exception as e:
    print(f'Configuration validation failed: {e}')
    sys.exit(1)
" 2>&1)
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "Configuration is valid"
        return 0
    else
        log_error "Configuration validation failed:"
        echo "$config_errors" | while read -r line; do
            if [[ -n "$line" ]]; then
                log_error "  $line"
            fi
        done
        return 1
    fi
}

# Function to test Qdrant connectivity
test_qdrant_connectivity() {
    if [ "$SKIP_QDRANT" = true ]; then
        log_warning "Skipping Qdrant connectivity tests"
        return 0
    fi
    
    log_test "Qdrant connectivity"
    
    if ! command -v curl >/dev/null 2>&1; then
        log_warning "curl not available, skipping Qdrant tests"
        return 0
    fi
    
    # Test basic connectivity
    if curl -s --max-time "$QDRANT_TIMEOUT" "$QDRANT_URL/health" >/dev/null 2>&1; then
        log_success "Qdrant is accessible at $QDRANT_URL"
    else
        log_error "Qdrant is not accessible at $QDRANT_URL"
        log_info "Start Qdrant with: ./scripts/setup-qdrant.sh"
        return 1
    fi
    
    # Test collections endpoint
    if curl -s --max-time "$QDRANT_TIMEOUT" "$QDRANT_URL/collections" >/dev/null 2>&1; then
        log_verbose "✅ Collections endpoint accessible"
    else
        log_error "Collections endpoint not accessible"
        return 1
    fi
    
    return 0
}

# Function to test project collections
test_project_collections() {
    if [ "$SKIP_COLLECTIONS" = true ] || [ "$SKIP_QDRANT" = true ]; then
        log_warning "Skipping collection tests"
        return 0
    fi
    
    log_test "Project collections"
    
    if ! command -v curl >/dev/null 2>&1; then
        log_warning "curl not available, skipping collection tests"
        return 0
    fi
    
    local collections=("code" "relations" "embeddings")
    local errors=0
    
    for collection_type in "${collections[@]}"; do
        local collection_name="${COLLECTION_PREFIX}-${collection_type}"
        
        log_verbose "Checking collection: $collection_name"
        
        if curl -s --max-time "$QDRANT_TIMEOUT" "$QDRANT_URL/collections/$collection_name" >/dev/null 2>&1; then
            log_verbose "✅ Collection exists: $collection_name"
            
            # Get collection info
            local collection_info
            collection_info=$(curl -s --max-time "$QDRANT_TIMEOUT" "$QDRANT_URL/collections/$collection_name" 2>/dev/null || echo '{}')
            
            # Extract point count if possible
            if command -v python3 >/dev/null 2>&1; then
                local point_count
                point_count=$(echo "$collection_info" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    result = data.get('result', {})
    points_count = result.get('points_count', 0)
    print(points_count)
except:
    print('0')
" 2>/dev/null || echo "0")
                
                log_verbose "  Points: $point_count"
            fi
        else
            log_error "Collection missing: $collection_name"
            ((errors++))
        fi
    done
    
    if [ $errors -eq 0 ]; then
        log_success "All collections are accessible"
        return 0
    else
        log_error "$errors collections are missing or inaccessible"
        log_info "Recreate collections with: ./setup-project.sh \"$PROJECT_NAME\" --overwrite"
        return 1
    fi
}

# Function to test Stella model availability
test_stella_model() {
    log_test "Stella model availability"
    
    if ! command -v python3 >/dev/null 2>&1; then
        log_warning "Python3 not available, skipping Stella tests"
        return 0
    fi
    
    # Test if Stella dependencies are available
    local stella_test_result
    stella_test_result=$(python3 -c "
import sys
try:
    import sentence_transformers
    import transformers
    import torch
    print('dependencies_ok')
except ImportError as e:
    print(f'missing_dependencies:{e}')
except Exception as e:
    print(f'error:{e}')
" 2>&1)
    
    if [[ "$stella_test_result" == "dependencies_ok" ]]; then
        log_verbose "✅ Stella dependencies available"
    else
        log_warning "Stella dependencies issue: $stella_test_result"
        log_info "Install with: python scripts/install_stella.py"
        return 1
    fi
    
    # Test if model is cached
    local model_cache_dir="$HOME/.cache/claude-indexer/stella/stella_en_400M_v5"
    if [[ -d "$model_cache_dir" ]] && [[ "$(ls -A "$model_cache_dir")" ]]; then
        log_verbose "✅ Stella model is cached"
        log_success "Stella model is available"
        return 0
    else
        log_warning "Stella model not cached"
        log_info "Download with: python scripts/install_stella.py"
        return 1
    fi
}

# Function to test CLI commands
test_cli_commands() {
    if [ "$QUICK_MODE" = true ]; then
        log_warning "Skipping CLI tests in quick mode"
        return 0
    fi
    
    log_test "CLI commands availability"
    
    local commands=("claude-indexer" "claude-code-context")
    local available_commands=0
    
    for cmd in "${commands[@]}"; do
        if command -v "$cmd" >/dev/null 2>&1; then
            log_verbose "✅ Command available: $cmd"
            
            # Test version flag
            if "$cmd" --version >/dev/null 2>&1; then
                log_verbose "  Version flag works"
            else
                log_verbose "  Version flag not available"
            fi
            
            ((available_commands++))
        else
            log_verbose "❌ Command not found: $cmd"
        fi
    done
    
    if [ $available_commands -gt 0 ]; then
        log_success "$available_commands CLI commands available"
        return 0
    else
        log_warning "No CLI commands found"
        log_info "Install with: ./install-global.sh"
        return 1
    fi
}

# Function to test directory permissions
test_permissions() {
    log_test "Directory permissions"
    
    local project_path="$1"
    local errors=0
    
    # Test read permissions
    if [[ -r "$project_path" ]]; then
        log_verbose "✅ Project directory readable"
    else
        log_error "Project directory not readable"
        ((errors++))
    fi
    
    # Test write permissions for configuration
    local config_dir="$project_path/.claude-indexer"
    if [[ -w "$config_dir" ]]; then
        log_verbose "✅ Configuration directory writable"
    else
        log_error "Configuration directory not writable"
        ((errors++))
    fi
    
    # Test cache directory permissions
    local cache_dir="$HOME/.cache/claude-indexer"
    if [[ -d "$cache_dir" ]]; then
        if [[ -w "$cache_dir" ]]; then
            log_verbose "✅ Cache directory writable"
        else
            log_error "Cache directory not writable"
            ((errors++))
        fi
    else
        log_verbose "Cache directory doesn't exist (will be created)"
    fi
    
    if [ $errors -eq 0 ]; then
        log_success "Permissions are correct"
        return 0
    else
        log_error "Permission issues found"
        return 1
    fi
}

# Function to run all tests
run_all_tests() {
    local project_path="$1"
    local total_tests=0
    local passed_tests=0
    
    echo -e "${CYAN}🧪 Running project validation tests${NC}"
    echo -e "${CYAN}   Project: $PROJECT_NAME${NC}"
    echo -e "${CYAN}   Path: $project_path${NC}"
    echo -e "${CYAN}   Collections: ${COLLECTION_PREFIX}-*${NC}"
    echo
    
    # Test 1: File system structure
    ((total_tests++))
    if test_filesystem_structure "$project_path"; then
        ((passed_tests++))
    fi
    echo
    
    # Test 2: Configuration validity
    ((total_tests++))
    if test_configuration "$project_path"; then
        ((passed_tests++))
    fi
    echo
    
    # Test 3: Directory permissions
    ((total_tests++))
    if test_permissions "$project_path"; then
        ((passed_tests++))
    fi
    echo
    
    # Test 4: Qdrant connectivity (optional)
    if [ "$SKIP_QDRANT" != true ]; then
        ((total_tests++))
        if test_qdrant_connectivity; then
            ((passed_tests++))
        fi
        echo
    fi
    
    # Test 5: Project collections (optional)
    if [ "$SKIP_COLLECTIONS" != true ] && [ "$SKIP_QDRANT" != true ]; then
        ((total_tests++))
        if test_project_collections; then
            ((passed_tests++))
        fi
        echo
    fi
    
    # Test 6: Stella model (optional)
    if [ "$QUICK_MODE" != true ]; then
        ((total_tests++))
        if test_stella_model; then
            ((passed_tests++))
        fi
        echo
    fi
    
    # Test 7: CLI commands (optional)
    if [ "$QUICK_MODE" != true ]; then
        ((total_tests++))
        if test_cli_commands; then
            ((passed_tests++))
        fi
        echo
    fi
    
    # Summary
    echo -e "${CYAN}📊 Test Results${NC}"
    echo -e "   Passed: $passed_tests/$total_tests"
    
    local success_rate=$((passed_tests * 100 / total_tests))
    
    if [ $passed_tests -eq $total_tests ]; then
        log_success "All tests passed! Project is ready to use."
        return 0
    elif [ $success_rate -ge 75 ]; then
        log_warning "Most tests passed ($success_rate%). Project should work but may have minor issues."
        return 1
    else
        log_error "Many tests failed ($success_rate%). Project needs attention."
        return 2
    fi
}

# Function to show recommendations
show_recommendations() {
    local exit_code="$1"
    
    echo
    echo -e "${BLUE}💡 Recommendations:${NC}"
    
    case $exit_code in
        0)
            echo -e "   ✅ Project is fully configured and ready"
            echo -e "   • Index your project: claude-indexer index -p '$PROJECT_PATH' -c '${COLLECTION_PREFIX}-code'"
            echo -e "   • Test search: claude-indexer search -c '${COLLECTION_PREFIX}-code' -q 'function'"
            echo -e "   • Use <ccc>query</ccc> tags in Claude Code prompts"
            ;;
        1)
            echo -e "   ⚠️  Project has minor issues but should work"
            echo -e "   • Check failed tests above and address if needed"
            echo -e "   • Consider running: ./setup-project.sh '$PROJECT_NAME' --overwrite"
            ;;
        2)
            echo -e "   ❌ Project has significant issues"
            echo -e "   • Run setup again: ./setup-project.sh '$PROJECT_NAME' --overwrite"
            echo -e "   • Ensure Qdrant is running: ./scripts/setup-qdrant.sh"
            echo -e "   • Install Stella model: python scripts/install_stella.py"
            ;;
    esac
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -c|--collection)
            COLLECTION_PREFIX="$2"
            shift 2
            ;;
        --skip-qdrant)
            SKIP_QDRANT=true
            shift
            ;;
        --skip-collections)
            SKIP_COLLECTIONS=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
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
            if [[ -z "$PROJECT_PATH" ]] || [[ "$PROJECT_PATH" == "." ]]; then
                PROJECT_PATH="$1"
            else
                log_error "Multiple project paths specified"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Convert to absolute path and validate
PROJECT_PATH=$(cd "$PROJECT_PATH" && pwd)

if [[ ! -d "$PROJECT_PATH" ]]; then
    log_error "Project path does not exist: $PROJECT_PATH"
    exit 1
fi

# Detect project information
detect_project_info "$PROJECT_PATH"

# Run tests
run_all_tests "$PROJECT_PATH"
test_exit_code=$?

# Show recommendations
show_recommendations $test_exit_code

exit $test_exit_code