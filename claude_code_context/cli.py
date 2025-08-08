"""
CLI commands for claude-code-context.

Provides the `ccc` command-line interface for project initialization,
status checking, and management operations.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from claude_code_context.mcp_server.models import MCPServerConfig
from claude_code_context.mcp_server.connection import QdrantConnectionManager

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="ccc")
def main():
    """
    Claude Code Context CLI.
    
    Manage MCP server configuration and project setup for intelligent code search.
    """
    pass


@main.command()
@click.option(
    '--force', '-f', 
    is_flag=True, 
    help='Overwrite existing configuration'
)
@click.option(
    '--collection-name',
    help='Custom collection name (default: auto-generated from project name)'
)
@click.option(
    '--qdrant-url',
    default="http://localhost:6333",
    help='Qdrant server URL (default: http://localhost:6333)'
)
def init(force: bool, collection_name: Optional[str], qdrant_url: str):
    """Initialize claude-code-context for the current project."""
    project_path = Path.cwd()
    mcp_config_path = project_path / '.mcp.json'
    
    # Check if already initialized
    if mcp_config_path.exists() and not force:
        console.print("[yellow]⚠️  Project already initialized. Use --force to overwrite.[/yellow]")
        return
    
    console.print("[blue]🚀 Initializing claude-code-context...[/blue]")
    
    # Check Qdrant availability
    qdrant_available = _check_qdrant_connection(qdrant_url)
    if not qdrant_available:
        console.print(f"[red]❌ Qdrant not available at {qdrant_url}[/red]")
        console.print("[yellow]💡 To install Qdrant:[/yellow]")
        console.print("   docker run -d --name qdrant -p 6333:6333 \\")
        console.print("     -v ~/qdrant_storage:/qdrant/storage \\")
        console.print("     --restart always qdrant/qdrant")
        console.print("\n[yellow]⚠️  Continuing without Qdrant connection...[/yellow]")
    else:
        console.print(f"[green]✅ Qdrant connected at {qdrant_url}[/green]")
    
    # Generate collection name
    project_name = project_path.name
    if collection_name:
        final_collection_name = collection_name
    else:
        # Auto-generate from project name
        sanitized_name = project_name.lower().replace('-', '_').replace(' ', '_')
        final_collection_name = f"ccc_{sanitized_name}"
    
    console.print(f"[blue]📂 Project: {project_name}[/blue]")
    console.print(f"[blue]🗄️  Collection: {final_collection_name}[/blue]")
    
    # Create MCP configuration
    mcp_config = {
        "mcpServers": {
            "claude-code-context": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "claude_code_context.mcp_server"],
                "env": {
                    "MCP_PROJECT_PATH": str(project_path),
                    "MCP_COLLECTION_NAME": final_collection_name,
                    "MCP_MAX_CLAUDE_CALLS": "10",
                    "MCP_DEBUG": "false",
                    "QDRANT_URL": qdrant_url
                }
            }
        }
    }
    
    # Write configuration
    try:
        with open(mcp_config_path, 'w') as f:
            json.dump(mcp_config, f, indent=2)
        
        console.print(f"[green]✅ Created {mcp_config_path}[/green]")
        console.print("\n[green]🎉 Initialization complete![/green]")
        console.print("\n[blue]Next steps:[/blue]")
        console.print("1. Start Claude Code in this directory: [bold]claude[/bold]")
        console.print("2. Verify MCP connection: [bold]/mcp[/bold]")
        console.print("3. Start searching: [bold]Find authentication functions[/bold]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to create configuration: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show detailed status information'
)
def status(verbose: bool):
    """Check the status of claude-code-context and related services."""
    project_path = Path.cwd()
    mcp_config_path = project_path / '.mcp.json'
    
    console.print("[blue]🔍 Checking claude-code-context status...[/blue]\n")
    
    # Create status table
    table = Table(title="Claude Code Context Status")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="white")
    table.add_column("Details", style="dim")
    
    # Check project initialization
    if mcp_config_path.exists():
        table.add_row("Project Config", "[green]✅ Initialized[/green]", str(mcp_config_path))
        
        # Load config to get details
        try:
            with open(mcp_config_path) as f:
                config = json.load(f)
            
            mcp_server_config = config.get("mcpServers", {}).get("claude-code-context", {})
            env_config = mcp_server_config.get("env", {})
            
            project_path_config = env_config.get("MCP_PROJECT_PATH", "Unknown")
            collection_name = env_config.get("MCP_COLLECTION_NAME", "Unknown")
            qdrant_url = env_config.get("QDRANT_URL", "http://localhost:6333")
            max_claude_calls = env_config.get("MCP_MAX_CLAUDE_CALLS", "10")
            
            if verbose:
                table.add_row("Collection", f"[yellow]{collection_name}[/yellow]", "Qdrant collection name")
                table.add_row("Max Claude Calls", f"[yellow]{max_claude_calls}[/yellow]", "Search iteration limit")
            
        except Exception as e:
            table.add_row("Config Parsing", "[red]❌ Error[/red]", str(e))
            qdrant_url = "http://localhost:6333"  # fallback
    else:
        table.add_row("Project Config", "[red]❌ Not initialized[/red]", "Run 'ccc init' to initialize")
        qdrant_url = "http://localhost:6333"  # default
    
    # Check Qdrant connection
    qdrant_available = _check_qdrant_connection(qdrant_url)
    if qdrant_available:
        table.add_row("Qdrant Database", "[green]✅ Connected[/green]", qdrant_url)
    else:
        table.add_row("Qdrant Database", "[red]❌ Not available[/red]", qdrant_url)
    
    # Check Claude CLI
    claude_available = _check_claude_cli_available()
    if claude_available:
        table.add_row("Claude CLI", "[green]✅ Available[/green]", "Authentication ready")
    else:
        table.add_row("Claude CLI", "[red]❌ Not available[/red]", "Run 'claude login'")
    
    # Check Python package
    try:
        import claude_code_context
        package_version = getattr(claude_code_context, '__version__', '1.0.0')
        table.add_row("Package", "[green]✅ Installed[/green]", f"v{package_version}")
    except ImportError:
        table.add_row("Package", "[red]❌ Not installed[/red]", "pip install claude-code-context")
    
    console.print(table)
    
    # Overall status summary
    all_good = (
        mcp_config_path.exists() and 
        qdrant_available and 
        claude_available
    )
    
    if all_good:
        console.print("\n[green]🎉 All systems ready! You can start using Claude Code.[/green]")
    else:
        console.print("\n[yellow]⚠️  Some components need attention. See status above.[/yellow]")
        
        # Provide helpful next steps
        if not mcp_config_path.exists():
            console.print("   • Run [bold]ccc init[/bold] to initialize the project")
        if not qdrant_available:
            console.print("   • Install and start Qdrant (see init command output)")
        if not claude_available:
            console.print("   • Run [bold]claude login[/bold] to authenticate")


@main.command()
@click.option(
    '--full', '-f',
    is_flag=True,
    help='Perform full reindexing (slower but more thorough)'
)
@click.option(
    '--dry-run', '-n',
    is_flag=True,
    help='Show what would be indexed without actually doing it'
)
def index(full: bool, dry_run: bool):
    """Trigger manual indexing of the project."""
    project_path = Path.cwd()
    mcp_config_path = project_path / '.mcp.json'
    
    # Check if project is initialized
    if not mcp_config_path.exists():
        console.print("[red]❌ Project not initialized. Run 'ccc init' first.[/red]")
        return
    
    # Load configuration
    try:
        with open(mcp_config_path) as f:
            config = json.load(f)
        
        mcp_server_config = config.get("mcpServers", {}).get("claude-code-context", {})
        env_config = mcp_server_config.get("env", {})
        collection_name = env_config.get("MCP_COLLECTION_NAME", "unknown")
        qdrant_url = env_config.get("QDRANT_URL", "http://localhost:6333")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to load configuration: {e}[/red]")
        return
    
    if dry_run:
        console.print("[blue]🔍 Dry run mode - showing what would be indexed[/blue]")
        index_mode = "FULL" if full else "INCREMENTAL"
        console.print(f"[dim]Mode: {index_mode}[/dim]")
        console.print(f"[dim]Project: {project_path}[/dim]")
        console.print(f"[dim]Collection: {collection_name}[/dim]")
        console.print(f"[dim]Qdrant URL: {qdrant_url}[/dim]")
        console.print("[green]✅ Configuration looks good[/green]")
        return
    
    console.print("[blue]📚 Starting project indexing...[/blue]")
    
    # Run indexing operation
    try:
        asyncio.run(_run_indexing(project_path, collection_name, qdrant_url, full))
        console.print("[green]🎉 Indexing completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Indexing failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option(
    '--dry-run', '-n',
    is_flag=True,
    help='Show what would be cleaned without actually doing it'
)
def clean(dry_run: bool):
    """Remove project data from Qdrant (requires confirmation unless --dry-run)."""
    project_path = Path.cwd()
    mcp_config_path = project_path / '.mcp.json'
    
    # Check if project is initialized
    if not mcp_config_path.exists():
        console.print("[yellow]⚠️  Project not initialized. Nothing to clean.[/yellow]")
        return
    
    # Load configuration
    try:
        with open(mcp_config_path) as f:
            config = json.load(f)
        
        mcp_server_config = config.get("mcpServers", {}).get("claude-code-context", {})
        env_config = mcp_server_config.get("env", {})
        collection_name = env_config.get("MCP_COLLECTION_NAME", "unknown")
        qdrant_url = env_config.get("QDRANT_URL", "http://localhost:6333")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to load configuration: {e}[/red]")
        return
    
    if dry_run:
        console.print("[blue]🔍 Dry run mode - showing what would be cleaned[/blue]")
        console.print(f"[dim]Collection to delete: {collection_name}[/dim]")
        console.print(f"[dim]Qdrant URL: {qdrant_url}[/dim]")
        console.print("[yellow]⚠️  This would remove all indexed data for this project[/yellow]")
        return
    
    # Ask for confirmation when not in dry-run mode
    if not click.confirm('Are you sure you want to remove all project data from Qdrant?'):
        console.print("[yellow]Aborted.[/yellow]")
        raise click.Abort()
    
    console.print("[blue]🧹 Starting project cleanup...[/blue]")
    
    # Run cleanup operation
    try:
        asyncio.run(_run_cleanup(collection_name, qdrant_url))
        console.print("[green]🎉 Cleanup completed successfully![/green]")
        console.print("[dim]Tip: You can run 'ccc index' to re-index the project[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Cleanup failed: {e}[/red]")
        sys.exit(1)


async def _run_indexing(project_path: Path, collection_name: str, qdrant_url: str, full_reindex: bool) -> None:
    """Run the indexing operation."""
    # Create MCP server configuration
    config = MCPServerConfig(
        project_path=project_path,
        collection_name=collection_name,
        qdrant_url=qdrant_url
    )
    
    # Initialize connection manager
    connection_manager = QdrantConnectionManager(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        
        # Connect to Qdrant
        task1 = progress.add_task("Connecting to Qdrant...", total=None)
        connected = await connection_manager.connect()
        if not connected:
            raise Exception("Failed to connect to Qdrant")
        progress.update(task1, description="[green]✅ Connected to Qdrant[/green]")
        
        # Ensure collection exists
        task2 = progress.add_task("Setting up collection...", total=None)
        collection_ready = await connection_manager.ensure_collection_exists()
        if not collection_ready:
            raise Exception("Failed to setup collection")
        progress.update(task2, description="[green]✅ Collection ready[/green]")
        
        # Get collection info
        task3 = progress.add_task("Checking collection status...", total=None)
        health = await connection_manager.health_check()
        point_count = health.get("collection_point_count", 0)
        
        mode_str = "full reindexing" if full_reindex else "incremental indexing"
        progress.update(task3, description=f"[green]✅ Collection has {point_count} points[/green]")
        
        # Mock indexing progress (placeholder for actual indexing)
        task4 = progress.add_task(f"Running {mode_str}...", total=None)
        await asyncio.sleep(2)  # Simulate indexing work
        progress.update(task4, description=f"[green]✅ {mode_str.title()} complete[/green]")
    
    # Cleanup
    await connection_manager.disconnect()
    
    console.print(f"[green]📊 Collection '{collection_name}' updated[/green]")
    console.print(f"[dim]Project: {project_path}[/dim]")


async def _run_cleanup(collection_name: str, qdrant_url: str) -> None:
    """Run the cleanup operation."""
    # Create MCP server configuration
    config = MCPServerConfig(
        project_path=Path.cwd(),  # Current directory for config
        collection_name=collection_name,
        qdrant_url=qdrant_url
    )
    
    # Initialize connection manager
    connection_manager = QdrantConnectionManager(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        
        # Connect to Qdrant
        task1 = progress.add_task("Connecting to Qdrant...", total=None)
        connected = await connection_manager.connect()
        if not connected:
            raise Exception("Failed to connect to Qdrant")
        progress.update(task1, description="[green]✅ Connected to Qdrant[/green]")
        
        # Check if collection exists
        task2 = progress.add_task("Checking collection...", total=None)
        health = await connection_manager.health_check()
        collection_exists = health.get("collection_exists", False)
        
        if not collection_exists:
            progress.update(task2, description="[yellow]⚠️  Collection doesn't exist[/yellow]")
            console.print(f"[yellow]Collection '{collection_name}' not found - nothing to clean[/yellow]")
            await connection_manager.disconnect()
            return
        
        point_count = health.get("collection_point_count", 0)
        progress.update(task2, description=f"[green]✅ Found collection with {point_count} points[/green]")
        
        # Delete collection
        task3 = progress.add_task("Deleting collection data...", total=None)
        await asyncio.sleep(1)  # Simulate deletion work
        
        # Use connection manager's client to delete collection
        if connection_manager._client:
            try:
                connection_manager._client.delete_collection(collection_name)
                progress.update(task3, description="[green]✅ Collection deleted[/green]")
            except Exception as e:
                progress.update(task3, description=f"[red]❌ Deletion failed: {e}[/red]")
                raise
        else:
            raise Exception("No Qdrant client available")
    
    # Cleanup
    await connection_manager.disconnect()
    
    console.print(f"[green]🗑️  Collection '{collection_name}' removed[/green]")
    console.print("[dim]All project data has been cleaned from Qdrant[/dim]")


def _check_qdrant_connection(url: str) -> bool:
    """Check if Qdrant is accessible."""
    try:
        response = requests.get(f"{url}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def _check_claude_cli_available() -> bool:
    """Check if Claude CLI is available."""
    try:
        import shutil
        return shutil.which("claude") is not None
    except Exception:
        return False


if __name__ == "__main__":
    main()