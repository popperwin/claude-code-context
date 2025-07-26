"""
Project setup script for claude-code-context.

This provides a Python entry point for the ccc-setup command referenced in pyproject.toml.
The main functionality is in setup-project.sh shell script.
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Main entry point for ccc-setup command."""
    
    # Get the script directory
    script_dir = Path(__file__).parent.parent
    setup_script = script_dir / "setup-project.sh"
    
    if not setup_script.exists():
        print("Error: setup-project.sh not found", file=sys.stderr)
        return 1
    
    # Pass all arguments to the shell script
    try:
        result = subprocess.run(
            [str(setup_script)] + sys.argv[1:],
            check=False
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\nSetup cancelled by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error running setup script: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())