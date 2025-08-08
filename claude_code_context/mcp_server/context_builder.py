"""
Project context builder for Claude orchestration.

This module provides the ProjectContextBuilder class that generates project trees,
loads documentation, and assembles comprehensive context for Claude CLI interactions,
with a 20k word limit to ensure optimal performance.
"""

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime

from .models import MCPServerConfig

logger = logging.getLogger(__name__)


@dataclass
class ProjectStructure:
    """Project structure information"""
    name: str
    description: str
    main_language: str
    languages_detected: List[str]
    total_files: int
    total_lines: int
    directory_tree: str
    key_files: List[str]
    
    
@dataclass 
class DocumentationInfo:
    """Documentation information"""
    readme_content: Optional[str] = None
    claude_md_content: Optional[str] = None
    other_docs: List[Tuple[str, str]] = None  # (filename, content)
    has_readme: bool = False
    has_claude_md: bool = False
    docs_word_count: int = 0


class ProjectContextBuilder:
    """
    Builds comprehensive project context for Claude orchestration.
    
    Generates project trees, loads documentation, detects languages,
    and assembles context within the 20k word limit for optimal Claude performance.
    """
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.project_path = config.project_path
        self.max_words = config.context_word_limit  # 20k words from config
        self.max_tree_depth = 5
        self.max_files_in_tree = 1000
        
        # File patterns for different languages
        self.language_patterns = {
            'Python': ['.py', '.pyw'],
            'JavaScript': ['.js', '.mjs'],
            'TypeScript': ['.ts', '.tsx'],
            'Java': ['.java'],
            'Go': ['.go'],
            'Rust': ['.rs'],
            'C++': ['.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.h'],
            'C': ['.c', '.h'],
            'C#': ['.cs'],
            'Ruby': ['.rb'],
            'PHP': ['.php'],
            'HTML': ['.html', '.htm'],
            'CSS': ['.css', '.scss', '.sass'],
            'JSON': ['.json'],
            'YAML': ['.yml', '.yaml'],
            'Markdown': ['.md', '.markdown'],
            'Shell': ['.sh', '.bash', '.zsh'],
        }
        
        # Files to ignore in tree generation
        self.ignore_patterns = {
            # Version control
            '.git', '.svn', '.hg',
            # Dependencies
            'node_modules', '__pycache__', '.venv', 'venv', 'env',
            # Build outputs
            'build', 'dist', 'target', 'out', '.next',
            # IDE files
            '.vscode', '.idea', '.eclipse',
            # OS files
            '.DS_Store', 'Thumbs.db',
            # Cache
            '.cache', '.pytest_cache', '.mypy_cache',
            # Logs
            '*.log', 'logs',
        }
        
        # Key files that should always be included if present
        self.key_files = {
            'README.md', 'README.rst', 'README.txt', 'README',
            'CLAUDE.md', 'CLAUDE.txt',
            'package.json', 'pyproject.toml', 'Cargo.toml', 'go.mod',
            'requirements.txt', 'setup.py', 'Makefile', 'docker-compose.yml',
            'tsconfig.json', '.gitignore', 'LICENSE',
        }
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored"""
        name = path.name
        
        # Check exact matches
        if name in self.ignore_patterns:
            return True
            
        # Check if it's a hidden file/directory (but allow some key files)
        if name.startswith('.') and name not in self.key_files:
            return True
            
        # Check parent directories for ignore patterns
        for parent in path.parents:
            if parent.name in self.ignore_patterns:
                return True
                
        return False
    
    def _count_words(self, text: str) -> int:
        """Count words in text"""
        if not text:
            return 0
        # Simple word count - split on whitespace
        return len(text.split())
    
    def _truncate_to_word_limit(self, text: str, max_words: int) -> str:
        """Truncate text to word limit"""
        if not text:
            return text
            
        words = text.split()
        if len(words) <= max_words:
            return text
            
        truncated = ' '.join(words[:max_words])
        return truncated + f"\n\n[... truncated at {max_words} words ...]"
    
    async def _detect_languages(self) -> Tuple[str, List[str]]:
        """
        Detect programming languages used in the project.
        
        Returns:
            Tuple of (main_language, all_languages_detected)
        """
        language_counts = {}
        
        try:
            # Walk through project files
            for root, dirs, files in os.walk(self.project_path):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if not self._should_ignore(Path(root) / d)]
                
                for file in files:
                    file_path = Path(root) / file
                    
                    if self._should_ignore(file_path):
                        continue
                    
                    # Count by file extension
                    suffix = file_path.suffix.lower()
                    for language, extensions in self.language_patterns.items():
                        if suffix in extensions:
                            language_counts[language] = language_counts.get(language, 0) + 1
                            break
        
        except Exception as e:
            logger.warning(f"Error detecting languages: {e}")
            return "Unknown", ["Unknown"]
        
        if not language_counts:
            return "Unknown", ["Unknown"]
        
        # Sort by count
        sorted_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
        main_language = sorted_languages[0][0]
        all_languages = [lang for lang, _ in sorted_languages]
        
        logger.debug(f"Detected languages: {dict(sorted_languages)}")
        return main_language, all_languages
    
    async def _generate_directory_tree(self, max_depth: int = None) -> Tuple[str, int, int]:
        """
        Generate ASCII directory tree.
        
        Returns:
            Tuple of (tree_string, total_files, total_lines)
        """
        if max_depth is None:
            max_depth = self.max_tree_depth
            
        tree_lines = []
        total_files = 0
        total_lines = 0
        file_count = 0
        
        def add_tree_item(path: Path, prefix: str, is_last: bool, depth: int):
            nonlocal total_files, total_lines, file_count
            
            if file_count >= self.max_files_in_tree:
                if file_count == self.max_files_in_tree:
                    tree_lines.append(f"{prefix}... [truncated at {self.max_files_in_tree} items]")
                return
            
            if depth >= max_depth:
                return
                
            connector = "└── " if is_last else "├── "
            tree_lines.append(f"{prefix}{connector}{path.name}")
            file_count += 1
            
            if path.is_file():
                total_files += 1
                # Try to count lines for code files
                try:
                    if path.suffix in ['.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.h']:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = sum(1 for _ in f)
                            total_lines += lines
                except:
                    pass  # Skip files we can't read
            
            elif path.is_dir() and not self._should_ignore(path):
                # Get children
                try:
                    children = sorted([
                        child for child in path.iterdir() 
                        if not self._should_ignore(child)
                    ], key=lambda x: (x.is_file(), x.name))
                    
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    
                    for i, child in enumerate(children):
                        if file_count >= self.max_files_in_tree:
                            break
                        is_child_last = (i == len(children) - 1)
                        add_tree_item(child, new_prefix, is_child_last, depth + 1)
                        
                except PermissionError:
                    pass  # Skip directories we can't read
        
        try:
            # Start with project root
            tree_lines.append(f"{self.project_path.name}/")
            
            # Get top-level items
            children = sorted([
                child for child in self.project_path.iterdir()
                if not self._should_ignore(child)
            ], key=lambda x: (x.is_file(), x.name))
            
            for i, child in enumerate(children):
                if file_count >= self.max_files_in_tree:
                    if file_count == self.max_files_in_tree:
                        tree_lines.append(f"... [truncated at {self.max_files_in_tree} items]")
                    break
                is_last = (i == len(children) - 1)
                add_tree_item(child, "", is_last, 1)
                
        except Exception as e:
            logger.error(f"Error generating directory tree: {e}")
            return f"Error generating tree: {e}", 0, 0
        
        tree_string = "\n".join(tree_lines)
        return tree_string, total_files, total_lines
    
    async def _load_documentation(self) -> DocumentationInfo:
        """Load documentation files from the project"""
        doc_info = DocumentationInfo()
        docs_found = []
        
        # Look for key documentation files
        for doc_file in ['README.md', 'README.rst', 'README.txt', 'README']:
            doc_path = self.project_path / doc_file
            if doc_path.exists():
                try:
                    content = doc_path.read_text(encoding='utf-8', errors='ignore')
                    doc_info.readme_content = content
                    doc_info.has_readme = True
                    docs_found.append(('README', content))
                    break
                except Exception as e:
                    logger.warning(f"Error reading {doc_file}: {e}")
        
        # Look for CLAUDE.md
        claude_md_path = self.project_path / 'CLAUDE.md'
        if claude_md_path.exists():
            try:
                content = claude_md_path.read_text(encoding='utf-8', errors='ignore')
                doc_info.claude_md_content = content
                doc_info.has_claude_md = True
                docs_found.append(('CLAUDE.md', content))
            except Exception as e:
                logger.warning(f"Error reading CLAUDE.md: {e}")
        
        # Look for other documentation files
        doc_extensions = {'.md', '.rst', '.txt'}
        doc_names = {'CONTRIBUTING', 'CHANGELOG', 'HISTORY', 'INSTALL', 'USAGE'}
        
        other_docs = []
        for file_path in self.project_path.iterdir():
            if self._should_ignore(file_path) or not file_path.is_file():
                continue
                
            # Check if it's a documentation file
            is_doc = (
                file_path.suffix.lower() in doc_extensions and
                (file_path.stem.upper() in doc_names or 
                 'doc' in file_path.name.lower())
            )
            
            if is_doc and file_path.name not in ['README.md', 'CLAUDE.md']:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    other_docs.append((file_path.name, content))
                    docs_found.append((file_path.name, content))
                except Exception as e:
                    logger.warning(f"Error reading {file_path.name}: {e}")
        
        doc_info.other_docs = other_docs
        
        # Count total words in all documentation
        total_words = 0
        for _, content in docs_found:
            total_words += self._count_words(content)
        doc_info.docs_word_count = total_words
        
        return doc_info
    
    async def _identify_key_files(self) -> List[str]:
        """Identify key files in the project"""
        key_files_found = []
        
        for file_name in self.key_files:
            file_path = self.project_path / file_name
            if file_path.exists():
                key_files_found.append(file_name)
        
        # Also look for main entry points
        possible_mains = [
            'main.py', 'app.py', 'server.py', 'index.js', 'index.ts',
            'main.go', 'main.rs', 'Main.java', 'App.java'
        ]
        
        for main_file in possible_mains:
            main_path = self.project_path / main_file
            if main_path.exists() and main_file not in key_files_found:
                key_files_found.append(main_file)
        
        return sorted(key_files_found)
    
    async def build_project_context(self) -> str:
        """
        Build comprehensive project context within word limit.
        
        Returns:
            Formatted project context string
        """
        logger.info(f"Building project context for {self.project_path}")
        
        try:
            # Gather all information concurrently
            (main_language, all_languages), \
            (tree_string, total_files, total_lines), \
            doc_info, \
            key_files = await asyncio.gather(
                self._detect_languages(),
                self._generate_directory_tree(), 
                self._load_documentation(),
                self._identify_key_files()
            )
            
            # Build project structure
            project_structure = ProjectStructure(
                name=self.project_path.name,
                description=f"A {main_language} project",
                main_language=main_language,
                languages_detected=all_languages,
                total_files=total_files,
                total_lines=total_lines,
                directory_tree=tree_string,
                key_files=key_files
            )
            
            # Start building context
            context_parts = []
            current_word_count = 0
            
            # 1. Project overview
            overview = f"""# Project: {project_structure.name}

**Main Language**: {project_structure.main_language}
**Languages**: {', '.join(project_structure.languages_detected[:5])}
**Files**: {project_structure.total_files} files, {project_structure.total_lines:,} lines
**Key Files**: {', '.join(project_structure.key_files[:10])}

"""
            overview_words = self._count_words(overview)
            context_parts.append(overview)
            current_word_count += overview_words
            
            # 2. Directory structure (limit to reasonable size)
            max_tree_words = min(2000, self.max_words // 4)  # Up to 1/4 of limit for tree
            tree_section = f"## Directory Structure\n\n```\n{tree_string}\n```\n\n"
            tree_words = self._count_words(tree_section)
            
            if tree_words > max_tree_words:
                # Regenerate with less depth
                shorter_tree, _, _ = await self._generate_directory_tree(max_depth=2)
                tree_section = f"## Directory Structure\n\n```\n{shorter_tree}\n```\n\n"
                tree_words = self._count_words(tree_section)
            
            if current_word_count + tree_words <= self.max_words:
                context_parts.append(tree_section)
                current_word_count += tree_words
            
            # 3. Documentation (prioritize CLAUDE.md, then README)
            remaining_words = self.max_words - current_word_count
            
            if doc_info.has_claude_md and remaining_words > 100:
                claude_content = doc_info.claude_md_content
                max_doc_words = min(remaining_words - 50, 3000)  # Reserve some space
                
                if self._count_words(claude_content) > max_doc_words:
                    claude_content = self._truncate_to_word_limit(claude_content, max_doc_words)
                
                doc_section = f"## Project Documentation (CLAUDE.md)\n\n{claude_content}\n\n"
                doc_words = self._count_words(doc_section)
                
                if current_word_count + doc_words <= self.max_words:
                    context_parts.append(doc_section)
                    current_word_count += doc_words
                    remaining_words = self.max_words - current_word_count
            
            # 4. README if space available and not already included
            if doc_info.has_readme and remaining_words > 100 and not doc_info.has_claude_md:
                readme_content = doc_info.readme_content
                max_readme_words = min(remaining_words - 50, 2000)
                
                if self._count_words(readme_content) > max_readme_words:
                    readme_content = self._truncate_to_word_limit(readme_content, max_readme_words)
                
                readme_section = f"## Project Documentation (README)\n\n{readme_content}\n\n"
                readme_words = self._count_words(readme_section)
                
                if current_word_count + readme_words <= self.max_words:
                    context_parts.append(readme_section)
                    current_word_count += readme_words
            
            # Final assembly
            full_context = "".join(context_parts)
            final_word_count = self._count_words(full_context)
            
            logger.info(f"Generated project context: {final_word_count} words (limit: {self.max_words})")
            
            return full_context
            
        except Exception as e:
            logger.error(f"Error building project context: {e}")
            return f"Error building project context: {str(e)}"
    
    async def get_project_summary(self) -> Dict[str, any]:
        """
        Get a summary of project information for health checks.
        
        Returns:
            Dictionary with project summary
        """
        try:
            main_language, all_languages = await self._detect_languages()
            _, total_files, total_lines = await self._generate_directory_tree()
            doc_info = await self._load_documentation()
            key_files = await self._identify_key_files()
            
            return {
                "project_name": self.project_path.name,
                "project_path": str(self.project_path),
                "main_language": main_language,
                "languages_detected": all_languages,
                "total_files": total_files,
                "total_lines": total_lines,
                "key_files_count": len(key_files),
                "has_readme": doc_info.has_readme,
                "has_claude_md": doc_info.has_claude_md,
                "docs_word_count": doc_info.docs_word_count,
                "max_context_words": self.max_words,
            }
            
        except Exception as e:
            logger.error(f"Error getting project summary: {e}")
            return {
                "error": str(e),
                "project_name": self.project_path.name,
                "project_path": str(self.project_path),
            }
    
    def is_valid_project(self) -> bool:
        """Check if the project path is valid"""
        return (
            self.project_path.exists() and 
            self.project_path.is_dir() and
            not self._should_ignore(self.project_path)
        )