#!/bin/bash
# install-global.sh - Global installation script for claude-code-context

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PACKAGE_NAME="claude-code-context"
MINIMUM_PYTHON_VERSION="3.12"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Installation options
DEVELOPMENT_MODE=false
SKIP_DEPENDENCIES=false
FORCE_REINSTALL=false
QUIET=false
VERBOSE=false

# Function to print colored output
log_info() {
    if [ "$QUIET" != true ]; then
        echo -e "${BLUE}ℹ️  $1${NC}"
    fi
}

log_success() {
    if [ "$QUIET" != true ]; then
        echo -e "${GREEN}✅ $1${NC}"
    fi
}

log_warning() {
    if [ "$QUIET" != true ]; then
        echo -e "${YELLOW}⚠️  $1${NC}"
    fi
}

log_error() {
    echo -e "${RED}❌ $1${NC}" >&2
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}🔍 $1${NC}"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -d, --dev             Install in development mode"
    echo "  -f, --force           Force reinstall even if already installed"
    echo "  --skip-deps           Skip dependency installation"
    echo "  -q, --quiet           Quiet output (errors only)"
    echo "  -v, --verbose         Verbose output"
    echo "  -h, --help            Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                    # Standard installation"
    echo "  $0 --dev             # Development installation"
    echo "  $0 --force           # Force reinstall"
    echo ""
}

# Function to check Python version
check_python_version() {
    log_info "Checking Python version..."
    
    if ! command -v python3 >/dev/null 2>&1; then
        log_error "Python 3 is not installed or not in PATH"
        log_info "Please install Python 3.12+ from: https://www.python.org/downloads/"
        return 1
    fi
    
    local python_version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    log_verbose "Found Python version: $python_version"
    
    # Compare versions
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"; then
        log_error "Python $python_version is installed, but $MINIMUM_PYTHON_VERSION+ is required"
        log_info "Please upgrade Python to version $MINIMUM_PYTHON_VERSION or higher"
        return 1
    fi
    
    log_success "Python $python_version meets requirements"
    return 0
}

# Function to check if package is already installed
check_existing_installation() {
    log_info "Checking for existing installation..."
    
    if python3 -c "import $PACKAGE_NAME" 2>/dev/null; then
        local installed_version
        installed_version=$(python3 -c "import $PACKAGE_NAME; print($PACKAGE_NAME.__version__)" 2>/dev/null || echo "unknown")
        
        log_warning "Package is already installed (version: $installed_version)"
        
        if [ "$FORCE_REINSTALL" = true ]; then
            log_info "Force reinstall requested, will proceed"
            return 1  # Indicate we should reinstall
        else
            read -p "Reinstall anyway? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                return 1  # Reinstall
            else
                return 0  # Skip installation
            fi
        fi
    fi
    
    return 1  # Not installed, proceed
}

# Function to check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    local requirements_met=true
    
    # Check pip
    if ! command -v pip3 >/dev/null 2>&1; then
        log_error "pip3 is not installed or not in PATH"
        requirements_met=false
    else
        log_verbose "Found pip3"
    fi
    
    # Check git (optional but recommended)
    if ! command -v git >/dev/null 2>&1; then
        log_warning "git is not installed (optional but recommended)"
    else
        log_verbose "Found git"
    fi
    
    # Check curl (for Qdrant health checks)
    if ! command -v curl >/dev/null 2>&1; then
        log_warning "curl is not installed (recommended for Qdrant checks)"
    else
        log_verbose "Found curl"
    fi
    
    # Check Docker (optional)
    if command -v docker >/dev/null 2>&1; then
        log_verbose "Found Docker"
        if docker info >/dev/null 2>&1; then
            log_verbose "Docker daemon is running"
        else
            log_warning "Docker is installed but daemon is not running"
        fi
    else
        log_warning "Docker not found (required for Qdrant)"
        log_info "Install Docker from: https://docs.docker.com/get-docker/"
    fi
    
    if [ "$requirements_met" = false ]; then
        log_error "System requirements not met"
        return 1
    fi
    
    log_success "System requirements check passed"
    return 0
}

# Function to create virtual environment (optional)
setup_virtual_environment() {
    if [ "$DEVELOPMENT_MODE" != true ]; then
        return 0
    fi
    
    log_info "Setting up virtual environment for development..."
    
    local venv_path="$SCRIPT_DIR/.venv"
    
    if [ -d "$venv_path" ]; then
        log_warning "Virtual environment already exists"
        read -p "Recreate virtual environment? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$venv_path"
        else
            log_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    python3 -m venv "$venv_path"
    
    # Activate virtual environment
    source "$venv_path/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    log_success "Virtual environment created and activated"
}

# Function to install the package
install_package() {
    log_info "Installing $PACKAGE_NAME..."
    
    local install_cmd="pip3 install"
    local install_options=()
    
    if [ "$DEVELOPMENT_MODE" = true ]; then
        log_info "Installing in development mode..."
        install_options+=("-e" ".")
        if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
            install_options+=("[dev]")
        fi
    else
        log_info "Installing from source..."
        install_options+=(".")
    fi
    
    # Add upgrade flag
    install_options+=("--upgrade")
    
    # Change to script directory for local install
    cd "$SCRIPT_DIR"
    
    # Execute installation
    if [ "$VERBOSE" = true ]; then
        $install_cmd "${install_options[@]}" --verbose
    elif [ "$QUIET" = true ]; then
        $install_cmd "${install_options[@]}" --quiet
    else
        $install_cmd "${install_options[@]}"
    fi
    
    log_success "Package installation completed"
}

# Function to install system dependencies
install_system_dependencies() {
    if [ "$SKIP_DEPENDENCIES" = true ]; then
        log_info "Skipping dependency installation"
        return 0
    fi
    
    log_info "Installing system dependencies..."
    
    # Detect OS and install system packages if needed
    local os_type
    os_type=$(uname -s)
    
    case "$os_type" in
        "Darwin")
            log_verbose "Detected macOS"
            # macOS - check if Homebrew is available
            if command -v brew >/dev/null 2>&1; then
                log_verbose "Homebrew found, installing dependencies..."
                # Install any macOS-specific dependencies if needed
            else
                log_warning "Homebrew not found, manual dependency installation may be required"
            fi
            ;;
        "Linux")
            log_verbose "Detected Linux"
            # Linux - basic dependencies should be available via pip
            ;;
        *)
            log_warning "Unknown operating system: $os_type"
            ;;
    esac
    
    log_success "System dependencies check completed"
}

# Function to install Stella model
install_stella_model() {
    log_info "Installing Stella embedding model..."
    
    local stella_script="$SCRIPT_DIR/scripts/install_stella.py"
    
    if [ ! -f "$stella_script" ]; then
        log_warning "Stella installation script not found, skipping"
        return 0
    fi
    
    read -p "Install Stella embedding model now? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        log_info "Skipping Stella installation"
        log_info "Run later with: python scripts/install_stella.py"
        return 0
    fi
    
    if python3 "$stella_script"; then
        log_success "Stella model installed successfully"
    else
        log_warning "Stella installation failed, but continuing"
        log_info "You can install it later with: python scripts/install_stella.py"
    fi
}

# Function to setup Qdrant
setup_qdrant() {
    log_info "Setting up Qdrant vector database..."
    
    local qdrant_script="$SCRIPT_DIR/scripts/setup-qdrant.sh"
    
    if [ ! -f "$qdrant_script" ]; then
        log_warning "Qdrant setup script not found, skipping"
        return 0
    fi
    
    # Check if Docker is available
    if ! command -v docker >/dev/null 2>&1; then
        log_warning "Docker not found, skipping Qdrant setup"
        log_info "Install Docker and run: ./scripts/setup-qdrant.sh"
        return 0
    fi
    
    read -p "Setup Qdrant vector database now? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        log_info "Skipping Qdrant setup"
        log_info "Run later with: ./scripts/setup-qdrant.sh"
        return 0
    fi
    
    if "$qdrant_script"; then
        log_success "Qdrant setup completed"
    else
        log_warning "Qdrant setup failed, but continuing"
        log_info "You can set it up later with: ./scripts/setup-qdrant.sh"
    fi
}

# Function to verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Test package import
    if ! python3 -c "import $PACKAGE_NAME" 2>/dev/null; then
        log_error "Package import failed"
        return 1
    fi
    
    # Test CLI commands
    local cli_commands=("claude-indexer" "claude-code-context")
    
    for cmd in "${cli_commands[@]}"; do
        if command -v "$cmd" >/dev/null 2>&1; then
            log_verbose "CLI command available: $cmd"
            
            # Test version
            if "$cmd" --version >/dev/null 2>&1; then
                log_verbose "Version check passed for: $cmd"
            fi
        else
            log_warning "CLI command not found: $cmd"
            log_info "You may need to restart your shell or check your PATH"
        fi
    done
    
    # Check configuration directories
    local config_dir="$HOME/.claude-indexer"
    if [ ! -d "$config_dir" ]; then
        mkdir -p "$config_dir"
        log_verbose "Created config directory: $config_dir"
    fi
    
    local cache_dir="$HOME/.cache/claude-indexer"
    if [ ! -d "$cache_dir" ]; then
        mkdir -p "$cache_dir"
        log_verbose "Created cache directory: $cache_dir"
    fi
    
    log_success "Installation verification completed"
}

# Function to show completion information
show_completion_info() {
    echo -e "\n${GREEN}🎉 Installation complete!${NC}"
    echo -e "\n${BLUE}📦 Package Information:${NC}"
    echo -e "   Name: $PACKAGE_NAME"
    
    # Get version if available
    local version
    version=$(python3 -c "import $PACKAGE_NAME; print($PACKAGE_NAME.__version__)" 2>/dev/null || echo "unknown")
    echo -e "   Version: $version"
    
    echo -e "\n${BLUE}🔧 Available Commands:${NC}"
    echo -e "   • claude-indexer        - Main CLI tool"
    echo -e "   • claude-code-context   - Alternative CLI name"
    echo -e "   • ccc-setup            - Project setup helper"
    
    echo -e "\n${BLUE}📁 Configuration:${NC}"
    echo -e "   • Global config: ~/.claude-indexer/"
    echo -e "   • Cache directory: ~/.cache/claude-indexer/"
    echo -e "   • Per-project config: .claude-indexer/config.json"
    
    echo -e "\n${BLUE}🚀 Quick Start:${NC}"
    echo -e "   1. Setup a project: ./setup-project.sh my-project"
    echo -e "   2. Index code: claude-indexer index -p . -c my-project-code"
    echo -e "   3. Search: claude-indexer search -c my-project-code -q 'function'"
    echo -e "   4. Use <ccc>query</ccc> tags in Claude Code prompts"
    
    echo -e "\n${BLUE}📖 Documentation:${NC}"
    echo -e "   • README: ./README.md"
    echo -e "   • Examples: ./examples/"
    echo -e "   • GitHub: https://github.com/popperwin/claude-code-context"
    
    if [ "$DEVELOPMENT_MODE" = true ]; then
        echo -e "\n${BLUE}🔨 Development:${NC}"
        echo -e "   • Virtual env: $SCRIPT_DIR/.venv"
        echo -e "   • Run tests: python -m pytest"
        echo -e "   • Code style: python -m black ."
    fi
}

# Function to handle cleanup on exit
cleanup() {
    if [ "$DEVELOPMENT_MODE" = true ] && [ -n "${VIRTUAL_ENV:-}" ]; then
        deactivate 2>/dev/null || true
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dev)
            DEVELOPMENT_MODE=true
            shift
            ;;
        -f|--force)
            FORCE_REINSTALL=true
            shift
            ;;
        --skip-deps)
            SKIP_DEPENDENCIES=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
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
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main installation process
main() {
    echo -e "${CYAN}🚀 Installing $PACKAGE_NAME${NC}"
    if [ "$DEVELOPMENT_MODE" = true ]; then
        echo -e "${CYAN}   Mode: Development${NC}"
    else
        echo -e "${CYAN}   Mode: Production${NC}"
    fi
    echo
    
    # Check Python version
    if ! check_python_version; then
        exit 1
    fi
    
    # Check system requirements
    if ! check_system_requirements; then
        exit 1
    fi
    
    # Check existing installation
    if check_existing_installation; then
        log_success "Package is already installed and up to date"
        exit 0
    fi
    
    # Setup virtual environment if in development mode
    if [ "$DEVELOPMENT_MODE" = true ]; then
        setup_virtual_environment
    fi
    
    # Install system dependencies
    install_system_dependencies
    
    # Install the package
    install_package
    
    # Verify installation
    if ! verify_installation; then
        log_error "Installation verification failed"
        exit 1
    fi
    
    # Optional components
    install_stella_model
    setup_qdrant
    
    # Show completion info
    show_completion_info
    
    log_success "Installation completed successfully!"
}

# Run main function
main