"""
Configuration management for claude-code-context

Handles loading, validation, and template-based configuration.
"""

from .loader import ConfigurationLoader
from .defaults import DEFAULT_SETTINGS

__all__ = ["ConfigurationLoader", "DEFAULT_SETTINGS"]