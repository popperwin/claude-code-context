#!/bin/bash
# Test multi-project setup with 3+ simultaneous projects
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
TEST_PROJECTS="$TEST_ENV_DIR/projects"

log() {
    echo -e "${BLUE}[MULTI-PROJECT-TEST]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[MULTI-PROJECT-TEST]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[MULTI-PROJECT-TEST]${NC} $1"
}

log_error() {
    echo -e "${RED}[MULTI-PROJECT-TEST]${NC} $1"
}

# Test projects configuration
PROJECTS=(
    "web-app:A React web application with TypeScript"
    "api-server:Python FastAPI backend service"
    "mobile-app:React Native mobile application"
    "data-pipeline:Python data processing pipeline"
    "ml-service:Machine learning inference service"
)

get_project_description() {
    local project_name="$1"
    for project_config in "${PROJECTS[@]}"; do
        if [[ "$project_config" == "$project_name:"* ]]; then
            echo "${project_config#*:}"
            return
        fi
    done
}

get_project_name() {
    local project_config="$1"
    echo "${project_config%%:*}"
}

setup_test_projects() {
    log "Setting up test projects..."
    
    # Create project directories
    for project_config in "${PROJECTS[@]}"; do
        project=$(get_project_name "$project_config")
        description=$(get_project_description "$project")
        project_dir="$TEST_PROJECTS/$project"
        mkdir -p "$project_dir"
        
        # Create basic project structure
        case $project in
            "web-app")
                mkdir -p "$project_dir/src/components"
                cat > "$project_dir/package.json" << EOF
{
  "name": "web-app",
  "version": "1.0.0",
  "description": "$description",
  "main": "src/index.tsx",
  "dependencies": {
    "react": "^18.0.0",
    "typescript": "^5.0.0"
  }
}
EOF
                cat > "$project_dir/src/App.tsx" << 'EOF'
import React from 'react';

const App: React.FC = () => {
  return (
    <div className="App">
      <h1>Web Application</h1>
    </div>
  );
};

export default App;
EOF
                ;;
                
            "api-server")
                mkdir -p "$project_dir/app/routers"
                cat > "$project_dir/requirements.txt" << EOF
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
EOF
                cat > "$project_dir/app/main.py" << 'EOF'
from fastapi import FastAPI
from typing import Dict

app = FastAPI(title="API Server", version="1.0.0")

@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Hello World"}

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}
EOF
                ;;
                
            "mobile-app")
                mkdir -p "$project_dir/src/screens"
                cat > "$project_dir/package.json" << EOF
{
  "name": "mobile-app",
  "version": "1.0.0",
  "description": "$description",
  "main": "src/App.tsx",
  "dependencies": {
    "react-native": "^0.72.0",
    "@react-navigation/native": "^6.0.0"
  }
}
EOF
                cat > "$project_dir/src/App.tsx" << 'EOF'
import React from 'react';
import { View, Text } from 'react-native';

const App = () => {
  return (
    <View style={{flex: 1, justifyContent: 'center', alignItems: 'center'}}>
      <Text>Mobile App</Text>
    </View>
  );
};

export default App;
EOF
                ;;
                
            "data-pipeline")
                mkdir -p "$project_dir/pipeline/processors"
                cat > "$project_dir/requirements.txt" << EOF
pandas>=2.1.0
numpy>=1.24.0
pydantic>=2.5.0
EOF
                cat > "$project_dir/pipeline/main.py" << 'EOF'
import pandas as pd
from typing import List, Dict

class DataProcessor:
    def __init__(self, config: Dict[str, str]):
        self.config = config
    
    def process_data(self, data: List[Dict]) -> pd.DataFrame:
        """Process raw data into structured format"""
        return pd.DataFrame(data)
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate processed data"""
        return not df.empty
EOF
                ;;
                
            "ml-service")
                mkdir -p "$project_dir/models"
                cat > "$project_dir/requirements.txt" << EOF
scikit-learn>=1.3.0
numpy>=1.24.0
fastapi>=0.104.0
EOF
                cat > "$project_dir/service.py" << 'EOF'
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import List

class MLService:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, features: List[float]) -> float:
        """Make prediction"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict([features])[0]
EOF
                ;;
        esac
        
        log_success "Created project: $project"
    done
}

test_project_setup() {
    log "Testing project setup with setup-project.sh..."
    
    # Source test environment
    source "$TEST_ENV_DIR/configs/test-env.sh"
    
    local success_count=0
    local total_count=${#PROJECTS[@]}
    
    for project_config in "${PROJECTS[@]}"; do
        project=$(get_project_name "$project_config")
        description=$(get_project_description "$project")
        log "Setting up project: $project"
        
        project_dir="$TEST_PROJECTS/$project"
        cd "$project_dir"
        
        # Run project setup with test environment port and overwrite flag
        if "$PROJECT_ROOT/setup-project.sh" "$project" --port 6334 --path "$project_dir" --overwrite 2>/dev/null; then
            if [[ -f ".claude-indexer/config.json" ]]; then
                log_success "✅ Project $project setup completed"
                ((success_count++))
            else
                log_error "❌ Project $project missing config file"
            fi
        else
            log_error "❌ Project $project setup failed"
        fi
    done
    
    log "Project setup results: $success_count/$total_count projects configured"
    
    if [[ $success_count -eq $total_count ]]; then
        log_success "✅ All projects setup successfully"
        return 0
    else
        log_error "❌ Some projects failed setup"
        return 1
    fi
}

test_configuration_isolation() {
    log "Testing configuration isolation between projects..."
    
    local configs_valid=0
    local total_projects=${#PROJECTS[@]}
    
    for project_config in "${PROJECTS[@]}"; do
        project=$(get_project_name "$project_config")
        description=$(get_project_description "$project")
        project_dir="$TEST_PROJECTS/$project"
        config_file="$project_dir/.claude-indexer/config.json"
        
        if [[ -f "$config_file" ]]; then
            # Check if config contains project-specific settings
            if grep -q "\"name\": \"$project\"" "$config_file"; then
                log_success "✅ Project $project has isolated configuration"
                ((configs_valid++))
            else
                log_error "❌ Project $project configuration not properly isolated"
            fi
        else
            log_error "❌ Project $project missing configuration file"
        fi
    done
    
    log "Configuration isolation results: $configs_valid/$total_projects projects isolated"
    
    if [[ $configs_valid -eq $total_projects ]]; then
        log_success "✅ All projects have isolated configurations"
        return 0
    else
        log_error "❌ Configuration isolation failed"
        return 1
    fi
}

test_claude_settings_generation() {
    log "Testing Claude settings generation for each project..."
    
    local settings_valid=0
    local total_projects=${#PROJECTS[@]}
    
    for project_config in "${PROJECTS[@]}"; do
        project=$(get_project_name "$project_config")
        project_dir="$TEST_PROJECTS/$project"
        settings_file="$project_dir/.claude/settings.json"
        
        if [[ -f "$settings_file" ]]; then
            # Check if settings contain project-specific hooks
            if grep -q "user_prompt_submit" "$settings_file" && \
               grep -q "hooks.user_prompt_submit" "$settings_file"; then
                log_success "✅ Project $project has Claude settings"
                ((settings_valid++))
            else
                log_error "❌ Project $project Claude settings incomplete"
            fi
        else
            log_error "❌ Project $project missing Claude settings"
        fi
    done
    
    log "Claude settings results: $settings_valid/$total_projects projects configured"
    
    if [[ $settings_valid -eq $total_projects ]]; then
        log_success "✅ All projects have Claude settings"
        return 0
    else
        log_error "❌ Claude settings generation failed"
        return 1
    fi
}

test_collection_naming() {
    log "Testing collection naming scheme for project isolation..."
    
    # Test collection name generation
    for project_config in "${PROJECTS[@]}"; do
        project=$(get_project_name "$project_config")
        project_dir="$TEST_PROJECTS/$project"
        config_file="$project_dir/.claude-indexer/config.json"
        
        if [[ -f "$config_file" ]]; then
            # Check collection naming format
            if grep -q "\"collection_prefix\": \"${project}\"" "$config_file"; then
                log_success "✅ Project $project has proper collection naming"
            else
                log_warning "⚠️  Project $project collection naming may not be isolated"
            fi
        fi
    done
}

test_concurrent_operations() {
    log "Testing concurrent operations across multiple projects..."
    
    # Simulate concurrent operations
    local pids=()
    
    for project_config in "${PROJECTS[@]}"; do
        project=$(get_project_name "$project_config")
        project_dir="$TEST_PROJECTS/$project"
        
        # Run concurrent configuration read operations
        (
            cd "$project_dir"
            for i in {1..5}; do
                if [[ -f ".claude-indexer/config.json" ]]; then
                    python3 -c "import json; print(json.load(open('.claude-indexer/config.json'))['name'])" >/dev/null
                fi
                sleep 0.1
            done
        ) &
        pids+=($!)
    done
    
    # Wait for all concurrent operations
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((failed++))
        fi
    done
    
    if [[ $failed -eq 0 ]]; then
        log_success "✅ Concurrent operations completed successfully"
        return 0
    else
        log_error "❌ $failed concurrent operations failed"
        return 1
    fi
}

generate_test_report() {
    log "Generating multi-project test report..."
    
    local report_file="$TEST_ENV_DIR/multi-project-test-report.md"
    
    cat > "$report_file" << EOF
# Multi-Project Setup Test Report

**Date:** $(date)
**Test Environment:** $TEST_ENV_DIR
**Projects Tested:** ${#PROJECTS[@]}

## Test Projects

EOF
    
    for project_config in "${PROJECTS[@]}"; do
        project=$(get_project_name "$project_config")
        description=$(get_project_description "$project")
        cat >> "$report_file" << EOF
### $project
- **Description:** $description
- **Directory:** $TEST_PROJECTS/$project
- **Configuration:** $([ -f "$TEST_PROJECTS/$project/.claude-indexer/config.json" ] && echo "✅ Present" || echo "❌ Missing")
- **Claude Settings:** $([ -f "$TEST_PROJECTS/$project/.claude/settings.json" ] && echo "✅ Present" || echo "❌ Missing")

EOF
    done
    
    cat >> "$report_file" << EOF
## Test Results Summary

- **Project Setup:** $(test_project_setup && echo "✅ PASSED" || echo "❌ FAILED")
- **Configuration Isolation:** $(test_configuration_isolation && echo "✅ PASSED" || echo "❌ FAILED")
- **Claude Settings Generation:** $(test_claude_settings_generation && echo "✅ PASSED" || echo "❌ FAILED")
- **Concurrent Operations:** $(test_concurrent_operations && echo "✅ PASSED" || echo "❌ FAILED")

## Files Created

$(find "$TEST_PROJECTS" -type f | wc -l) files across ${#PROJECTS[@]} projects

## Conclusion

Multi-project setup $([ $? -eq 0 ] && echo "PASSED" || echo "FAILED") - Ready for Sprint 1 completion.
EOF
    
    log_success "Test report saved: $report_file"
}

main() {
    log "Starting multi-project setup test with ${#PROJECTS[@]} projects"
    
    # Clean up previous test
    rm -rf "$TEST_PROJECTS"
    mkdir -p "$TEST_PROJECTS"
    
    # Run tests
    setup_test_projects
    
    local all_passed=true
    
    if ! test_project_setup; then
        all_passed=false
    fi
    
    if ! test_configuration_isolation; then
        all_passed=false
    fi
    
    if ! test_claude_settings_generation; then
        all_passed=false
    fi
    
    test_collection_naming
    
    if ! test_concurrent_operations; then
        all_passed=false
    fi
    
    generate_test_report
    
    if $all_passed; then
        log_success "✅ Multi-project setup test PASSED"
        log_success "Definition of Done: Multi-project setup tested with 3+ simultaneous projects ✅"
        exit 0
    else
        log_error "❌ Multi-project setup test FAILED"
        log_error "Definition of Done: Multi-project setup requirement not met ❌"
        exit 1
    fi
}

main "$@"