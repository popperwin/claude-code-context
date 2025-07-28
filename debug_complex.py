#!/usr/bin/env python3

from pathlib import Path
from core.parser.go_parser import GoParser

def debug_complex_structures():
    """Debug complex Go structures test"""
    complex_go_code = '''package server

import (
    "context"
    "net/http"
    "time"
    
    "github.com/gorilla/mux"
    "github.com/sirupsen/logrus"
)

// Generic type definition (Go 1.18+)
type Result[T any] struct {
    Value T
    Error error
}

// Constructor function
func NewServer(cfg Config) *Server {
    return &Server{
        config: cfg,
    }
}

// Method with context and error handling
func (s *Server) Start(ctx context.Context) error {
    return nil
}'''
    
    # Write to temporary file
    test_file = Path("/tmp/debug_complex.go")
    test_file.write_text(complex_go_code)
    
    # Parse with Go parser
    parser = GoParser()
    result = parser.parse_file(test_file)
    
    print(f"Parse success: {result.success}")
    print(f"Total entities: {len(result.entities)}")
    
    # Show all entities by type
    from core.models.entities import EntityType
    
    entity_types = {}
    for entity in result.entities:
        if entity.entity_type not in entity_types:
            entity_types[entity.entity_type] = []
        entity_types[entity.entity_type].append(entity.name)
    
    for entity_type, names in entity_types.items():
        print(f"\n{entity_type.value}s: {names}")
    
    # Check specifically for functions
    functions = [e for e in result.entities if e.entity_type == EntityType.FUNCTION]
    print(f"\nFunctions found: {len(functions)}")
    for func in functions:
        print(f"  - {func.name}: {func.signature}")
        
    methods = [e for e in result.entities if e.entity_type == EntityType.METHOD]
    print(f"\nMethods found: {len(methods)}")
    for method in methods:
        print(f"  - {method.name}: {method.signature}")
    
    # Debug the receiver detection issue
    print("\n=== Debugging receiver detection ===")
    import tree_sitter
    tree = parser.parser.parse(complex_go_code.encode('utf-8'))
    
    function_nodes = parser.find_nodes_by_type(tree, ["function_declaration"])
    print(f"Found {len(function_nodes)} function_declaration nodes")
    
    for i, func_node in enumerate(function_nodes):
        func_text = parser.get_node_text(func_node, complex_go_code)
        print(f"\nFunction {i+1}: {func_text[:50]}...")
        
        # Check receiver detection
        params = parser.find_child_by_type(func_node, "parameter_list")
        if params:
            param_text = parser.get_node_text(params, complex_go_code)
            has_receiver = parser._has_receiver_syntax(params, complex_go_code)
            print(f"  Parameter text: '{param_text}'")
            print(f"  Detected as receiver: {has_receiver}")
            
            # Check regex pattern
            import re
            pattern = re.compile(r'\(\s*(\w+)\s+\*?(\w+)\s*\)')
            match = pattern.search(param_text)
            print(f"  Regex match: {match.groups() if match else None}")
    
    # Also check method nodes for comparison
    method_nodes = parser.find_nodes_by_type(tree, ["method_declaration"])
    print(f"\nFound {len(method_nodes)} method_declaration nodes")
    
    for i, method_node in enumerate(method_nodes):
        method_text = parser.get_node_text(method_node, complex_go_code)
        print(f"\nMethod {i+1}: {method_text[:50]}...")
    
    # All entity names
    entity_names = [e.name for e in result.entities]
    print(f"\nAll entity names: {entity_names}")
    print(f"NewServer in names: {'NewServer' in entity_names}")
    
    # Cleanup
    test_file.unlink()

if __name__ == "__main__":
    debug_complex_structures()