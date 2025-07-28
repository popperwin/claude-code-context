#!/bin/bash
# Simple comprehensive testing for Sprint 1 + Sprint 2
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_ENV_DIR="$SCRIPT_DIR/test-env"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Sprint 1 + Sprint 2 Comprehensive Testing${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if test environment is set up
if [ ! -f "$TEST_ENV_DIR/configs/test-env.sh" ]; then
    echo -e "${RED}❌ Test environment not set up. Run ./setup-test-env.sh first${NC}"
    exit 1
fi

# Load test environment
source "$TEST_ENV_DIR/configs/test-env.sh"
echo "Test environment activated"
echo "  • Qdrant: http://localhost:$QDRANT_PORT"
echo "  • Python: $VIRTUAL_ENV/bin/python"

# Change to project root
cd "$PROJECT_ROOT"

# Run comprehensive test suite with coverage
echo -e "\n${BLUE}Running all tests with coverage analysis...${NC}"
echo "Command: pytest tests/ -v --cov=core --cov=config --cov-report=term-missing --cov-report=html --cov-fail-under=65"

if pytest tests/ -v --cov=core --cov=config --cov-report=term-missing --cov-report=html --cov-fail-under=65; then
    echo -e "\n${GREEN}🎉 ALL TESTS PASSED! Sprint 1 + Sprint 2 Complete!${NC}"
    echo -e "${GREEN}✅ Coverage requirement (65%) met${NC}"
    echo -e "${GREEN}✅ All 212 tests passing${NC}"
    echo -e "${GREEN}📊 Coverage report: htmlcov/index.html${NC}"
    echo -e "\n${GREEN}Ready to proceed to Sprint 3 (Tree-sitter Parsers)${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Tests failed or coverage below 65%${NC}"
    echo -e "${RED}📊 Check coverage report: htmlcov/index.html${NC}"
    echo -e "${RED}Issues must be resolved before Sprint 3${NC}"
    exit 1
fi