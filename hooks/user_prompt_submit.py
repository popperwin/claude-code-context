"""
User prompt submit hook for Claude Code integration.

This will be fully implemented in Sprint 5: Hook & CLI Integration.
Currently provides minimal placeholder functionality.
"""

import sys
import json
from typing import Dict, Any


def main():
    """
    Placeholder implementation for user prompt submit hook.
    
    In Sprint 5, this will:
    1. Parse <ccc>query</ccc> tags in prompts
    2. Perform direct Stella embedding + Qdrant search
    3. Inject context ONLY when tags present
    4. Return enriched context to stdout
    5. Fail silently on errors
    """
    
    # For now, just pass through the input unchanged
    try:
        # Read input from stdin
        if not sys.stdin.isatty():
            prompt_data = sys.stdin.read()
            
            # In Sprint 5, this will parse <ccc> tags and inject context
            # For now, just return the original prompt
            print(prompt_data, end='')
        else:
            # Called directly - show status
            status = {
                "status": "placeholder",
                "sprint": "1 (Foundations)",
                "note": "Full hook functionality will be available in Sprint 5"
            }
            print(json.dumps(status, indent=2))
            
    except Exception as e:
        # Silent failure as per requirements
        if not sys.stdin.isatty():
            # Pass through original input on error
            try:
                prompt_data = sys.stdin.read()
                print(prompt_data, end='')
            except:
                pass
        else:
            print(f"Hook error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()