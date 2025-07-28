#!/usr/bin/env python3

"""Test Tree-sitter's native distinction between function_declaration and method_declaration"""

from pathlib import Path
from core.parser.go_parser import GoParser

def test_tree_sitter_distinction():
    """Test that Tree-sitter correctly distinguishes functions from methods"""
    
    go_code = '''package test

// Regular function (no receiver)
func NewServer(cfg Config) *Server {
    return &Server{config: cfg}
}

// Another function with multiple parameters
func ProcessData(input string, count int) error {
    return nil
}

// Method with pointer receiver
func (s *Server) Start(ctx context.Context) error {
    return nil
}

// Method with value receiver  
func (s Server) GetConfig() Config {
    return s.config
}
'''

    # Write to temporary file
    test_file = Path("/tmp/test_distinction.go")
    test_file.write_text(go_code)
    
    # Parse with Go parser
    parser = GoParser()
    
    # Parse the tree directly 
    tree = parser.parser.parse(go_code.encode('utf-8'))
    
    # Find function_declaration nodes
    function_nodes = parser.find_nodes_by_type(tree, ["function_declaration"])
    print(f"Found {len(function_nodes)} function_declaration nodes:")
    
    for i, func_node in enumerate(function_nodes):
        func_text = parser.get_node_text(func_node, go_code)
        print(f"  {i+1}. {func_text.split('{')[0]}...")
    
    # Find method_declaration nodes  
    method_nodes = parser.find_nodes_by_type(tree, ["method_declaration"])
    print(f"\nFound {len(method_nodes)} method_declaration nodes:")
    
    for i, method_node in enumerate(method_nodes):
        method_text = parser.get_node_text(method_node, go_code)
        print(f"  {i+1}. {method_text.split('{')[0]}...")
    
    # Test the hypothesis: functions should be in function_declaration, methods in method_declaration
    function_names = []
    for func_node in function_nodes:
        name_node = parser.find_child_by_type(func_node, "identifier")
        if name_node:
            name = parser.get_node_text(name_node, go_code)
            function_names.append(name)
    
    method_names = []
    for method_node in method_nodes:
        name_node = parser.find_child_by_type(method_node, "field_identifier") or parser.find_child_by_type(method_node, "identifier")
        if name_node:
            name = parser.get_node_text(name_node, go_code)
            method_names.append(name)
    
    print(f"\nFunction names from function_declaration: {function_names}")
    print(f"Method names from method_declaration: {method_names}")
    
    # Verify Tree-sitter's classification is correct
    expected_functions = ["NewServer", "ProcessData"]
    expected_methods = ["Start", "GetConfig"]
    
    print(f"\nVerification:")
    print(f"Expected functions: {expected_functions}")
    print(f"Actual functions: {function_names}")
    print(f"Functions correct: {set(function_names) == set(expected_functions)}")
    
    print(f"Expected methods: {expected_methods}")
    print(f"Actual methods: {method_names}")
    print(f"Methods correct: {set(method_names) == set(expected_methods)}")
    
    # Cleanup
    test_file.unlink()
    
    return set(function_names) == set(expected_functions) and set(method_names) == set(expected_methods)

if __name__ == "__main__":
    result = test_tree_sitter_distinction()
    print(f"\nTree-sitter distinction is reliable: {result}")