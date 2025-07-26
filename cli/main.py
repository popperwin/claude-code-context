"""
Main CLI entry point for claude-code-context.

This will be fully implemented in Sprint 5: Hook & CLI Integration.
Currently provides minimal placeholder functionality.
"""

import click


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Claude Code Context - Semantic search for your codebase.
    
    Full CLI functionality will be available in Sprint 5.
    """
    pass


@cli.command()
def status():
    """Show the current status of claude-code-context."""
    click.echo("🔧 Claude Code Context v1.0.0 (Sprint 1 - Foundations)")
    click.echo("📝 Full CLI commands will be available in Sprint 5")
    click.echo("✅ Core models and configuration system ready")


@cli.command()
def info():
    """Show information about the installation."""
    click.echo("Claude Code Context - Semantic Context Enrichment")
    click.echo("Current Sprint: 1 (Foundations and Architecture)")
    click.echo("Coming in Sprint 5: Full CLI commands and hook integration")


if __name__ == "__main__":
    cli()