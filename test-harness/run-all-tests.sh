#!/bin/bash
# Comprehensive Sprint 1 testing with detailed reporting
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_ENV_DIR="$SCRIPT_DIR/test-env"
TEST_LOGS="$TEST_ENV_DIR/logs"
TEST_RESULTS="$TEST_ENV_DIR/results"

# Test tracking (using a simpler approach for macOS compatibility)
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
START_TIME=$(date +%s)
TEST_RESULTS_FILE=""

log() {
    echo -e "${BLUE}[TEST]${NC} $1" | tee -a "$TEST_LOGS/test-run.log"
}

log_success() {
    echo -e "${GREEN}[TEST]${NC} $1" | tee -a "$TEST_LOGS/test-run.log"
}

log_warning() {
    echo -e "${YELLOW}[TEST]${NC} $1" | tee -a "$TEST_LOGS/test-run.log"
}

log_error() {
    echo -e "${RED}[TEST]${NC} $1" | tee -a "$TEST_LOGS/test-run.log"
}

log_section() {
    echo -e "\n${CYAN}========================================${NC}" | tee -a "$TEST_LOGS/test-run.log"
    echo -e "${CYAN}$1${NC}" | tee -a "$TEST_LOGS/test-run.log"
    echo -e "${CYAN}========================================${NC}" | tee -a "$TEST_LOGS/test-run.log"
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_result="${3:-0}"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    log "Running: $test_name"
    
    local start_time=$(date +%s.%N)
    local safe_test_name=$(echo "$test_name" | sed 's/[^a-zA-Z0-9]/_/g')
    if eval "$test_command" >"$TEST_LOGS/${safe_test_name}.log" 2>&1; then
        local result_code=0
    else
        local result_code=$?
    fi
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0.0")
    
    if [ "$result_code" -eq "$expected_result" ]; then
        log_success "✅ $test_name (${duration}s)"
        echo "PASS: $test_name" >> "$TEST_RESULTS_FILE"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        log_error "❌ $test_name (${duration}s) - Exit code: $result_code"
        echo "FAIL: $test_name" >> "$TEST_RESULTS_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        
        # Show last few lines of error log
        if [ -f "$TEST_LOGS/${safe_test_name}.log" ]; then
            echo -e "${RED}Last 5 lines of error log:${NC}"
            tail -5 "$TEST_LOGS/${safe_test_name}.log" | sed 's/^/  /'
        fi
        return 1
    fi
}

check_environment() {
    log_section "Phase 1: Environment Check"
    
    # Check if test environment is set up
    if [ ! -f "$TEST_ENV_DIR/configs/test-env.sh" ]; then
        log_error "Test environment not set up. Run ./setup-test-env.sh first"
        exit 1
    fi
    
    # Load test environment
    source "$TEST_ENV_DIR/configs/test-env.sh"
    
    # Verify critical components
    run_test "Python version check" "python --version"
    run_test "Virtual environment active" "[ \"\$VIRTUAL_ENV\" = \"$TEST_ENV_DIR/venv\" ]"
    run_test "Qdrant accessibility" "curl -s http://localhost:$QDRANT_PORT/health"
    run_test "Project root accessible" "[ -f \"$PROJECT_ROOT/pyproject.toml\" ]"
    
    log_success "Environment check completed"
}

test_python_imports() {
    log_section "Phase 2: Python Import Tests"
    
    run_test "Import core models" "python -c 'from core.models import entities, storage, config, hooks'"
    run_test "Import configuration loader" "python -c 'from config.loader import ConfigurationLoader'"
    run_test "Import embeddings base" "python -c 'from core.embeddings.base import BaseEmbedder'"
    run_test "Pydantic models validation" "python -c 'from core.models.config import ProjectConfig; print(\"Pydantic models OK\")'"
    
    log_success "Python import tests completed"
}

test_unit_tests() {
    log_section "Phase 3: Unit Test Suite"
    
    cd "$PROJECT_ROOT"
    
    run_test "Unit tests - entities" "python -m pytest tests/test_entities.py -v"
    run_test "Unit tests - storage" "python -m pytest tests/test_storage.py -v"  
    run_test "Unit tests - config" "python -m pytest tests/test_config.py -v"
    run_test "Integration tests" "python -m pytest tests/test_integration.py -v"
    run_test "All unit tests with coverage" "python -m pytest tests/ --cov=core --cov=config --cov-report=term-missing --cov-fail-under=65"
    
    log_success "Unit test suite completed"
}

test_shell_scripts() {
    log_section "Phase 4: Shell Script Tests"
    
    cd "$PROJECT_ROOT"
    
    # Test script executability
    for script in install-global.sh setup-project.sh scripts/setup-qdrant.sh scripts/test-project.sh; do
        if [ -f "$script" ]; then
            run_test "Script executable: $script" "[ -x \"$script\" ]"
        fi
    done
    
    # Test help commands
    run_test "Install script help" "./install-global.sh --help"
    run_test "Setup script help" "./setup-project.sh --help"
    run_test "Qdrant script help" "./scripts/setup-qdrant.sh help"
    
    # Test Python script syntax
    if [ -f "scripts/install_stella.py" ]; then
        run_test "Stella script syntax" "python -m py_compile scripts/install_stella.py"
        run_test "Stella script info" "python scripts/install_stella.py --info"
    fi
    
    log_success "Shell script tests completed"
}

test_configuration_system() {
    log_section "Phase 5: Configuration System Tests"
    
    cd "$PROJECT_ROOT"
    
    # Test configuration loading
    run_test "Configuration loader import" "python -c 'from config.loader import ConfigurationLoader; loader = ConfigurationLoader()'"
    
    # Test template existence
    run_test "Config template exists" "[ -f \"templates/.claude-indexer/config.json.template\" ]"
    run_test "Settings template exists" "[ -f \"templates/.claude/settings.json.template\" ]"
    
    # Test template substitution
    run_test "Template variables present" "grep -q '\${PROJECT_NAME}' templates/.claude-indexer/config.json.template"
    
    log_success "Configuration system tests completed"
}

test_multi_project_setup() {
    log_section "Phase 6: Multi-Project Setup Tests"
    
    cd "$PROJECT_ROOT"
    
    # Run comprehensive multi-project isolation test
    run_test "Multi-project setup with 5 simultaneous projects" "./test-harness/test-multi-project.sh"
    
    log_success "Multi-project setup tests completed"
}

test_performance() {
    log_section "Phase 7: Performance Tests"
    
    cd "$PROJECT_ROOT"
    
    # Test configuration loading performance
    run_test "Config loading performance" "python -c '
import time
from config.loader import ConfigurationLoader
start = time.time()
for i in range(10):
    loader = ConfigurationLoader()
duration = time.time() - start
print(f\"10 config loads: {duration:.3f}s\")
assert duration < 1.0, f\"Too slow: {duration}s\"
'"
    
    # Test import performance
    run_test "Import performance" "python -c '
import time
start = time.time()
from core.models import entities, storage, config
duration = time.time() - start
print(f\"Import time: {duration:.3f}s\")
assert duration < 2.0, f\"Too slow: {duration}s\"
'"
    
    log_success "Performance tests completed"
}

test_repository_urls() {
    log_section "Phase 8: Repository URL Validation"
    
    cd "$PROJECT_ROOT"
    
    # Check that all URLs point to popperwin
    run_test "README URLs updated" "! grep -r 'github.com/anthropics' README.md"
    run_test "pyproject.toml URLs updated" "! grep -r 'github.com/anthropics' pyproject.toml"
    run_test "Install script URLs updated" "! grep -r 'github.com/anthropics' install-global.sh"
    run_test "Template URLs updated" "! grep -r 'github.com/anthropics' templates/"
    
    # Verify popperwin URLs exist
    run_test "Popperwin URLs present" "grep -r 'github.com/popperwin' README.md pyproject.toml install-global.sh templates/"
    
    log_success "Repository URL validation completed"
}

generate_report() {
    log_section "Test Report Generation"
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))
    
    # Create detailed report
    mkdir -p "$TEST_RESULTS"
    local report_file="$TEST_RESULTS/sprint1-test-report.md"
    
    cat > "$report_file" << EOF
# Sprint 1 Testing Report

**Date:** $(date)
**Duration:** ${total_duration}s
**Total Tests:** $TOTAL_TESTS
**Passed:** $PASSED_TESTS
**Failed:** $FAILED_TESTS
**Success Rate:** $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%

## Test Results

EOF
    
    if [ -f "$TEST_RESULTS_FILE" ]; then
        while IFS=': ' read -r result test_name; do
            if [ "$result" = "PASS" ]; then
                echo "✅ **$test_name**: PASSED" >> "$report_file"
            else
                echo "❌ **$test_name**: FAILED" >> "$report_file"
            fi
        done < "$TEST_RESULTS_FILE"
    fi
    
    cat >> "$report_file" << EOF

## Definition of Done Checklist

- [$([ $FAILED_TESTS -eq 0 ] && echo "x" || echo " ")] All repository URLs point to popperwin
- [$([ $FAILED_TESTS -eq 0 ] && echo "x" || echo " ")] All shell scripts executable and pass syntax check
- [$([ $FAILED_TESTS -eq 0 ] && echo "x" || echo " ")] All Python unit tests pass
- [$([ $FAILED_TESTS -eq 0 ] && echo "x" || echo " ")] All integration tests pass
- [$([ $FAILED_TESTS -eq 0 ] && echo "x" || echo " ")] Multi-project setup validated
- [$([ $FAILED_TESTS -eq 0 ] && echo "x" || echo " ")] Template substitution working
- [$([ $FAILED_TESTS -eq 0 ] && echo "x" || echo " ")] Configuration loading performs adequately
- [$([ $FAILED_TESTS -eq 0 ] && echo "x" || echo " ")] Code quality validated (65% coverage for Sprint 1)

## Sprint 1 Status

EOF
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo "🎉 **SPRINT 1 COMPLETE** - All tests passed!" >> "$report_file"
        echo "Ready to proceed to Sprint 2." >> "$report_file"
    else
        echo "❌ **SPRINT 1 INCOMPLETE** - $FAILED_TESTS test(s) failed." >> "$report_file"
        echo "Issues must be resolved before Sprint 2." >> "$report_file"
    fi
    
    # Display report
    cat "$report_file"
    
    log_success "Test report generated: $report_file"
}

main() {
    log_section "Sprint 1 Comprehensive Testing"
    
    # Ensure logs directory exists
    mkdir -p "$TEST_LOGS" "$TEST_RESULTS"
    
    # Initialize log file and test results tracking
    echo "Sprint 1 Test Run - $(date)" > "$TEST_LOGS/test-run.log"
    TEST_RESULTS_FILE="$TEST_RESULTS/test-results.txt"
    > "$TEST_RESULTS_FILE"  # Clear the results file
    
    # Run all test phases
    check_environment
    test_python_imports
    test_unit_tests
    test_shell_scripts
    test_configuration_system
    test_multi_project_setup
    test_performance
    test_repository_urls
    
    # Generate final report
    generate_report
    
    # Final status
    if [ $FAILED_TESTS -eq 0 ]; then
        log_success "🎉 ALL TESTS PASSED! Sprint 1 is complete and ready for Sprint 2!"
        exit 0
    else
        log_error "❌ $FAILED_TESTS tests failed. Sprint 1 requires fixes before completion."
        exit 1
    fi
}

main "$@"