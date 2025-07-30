"""
Deterministic Entity ID Generation.

Provides stable, content-based entity identification for atomic replacement
operations in the synchronization system.
"""

import hashlib
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from ..models.entities import Entity


class DeterministicEntityId:
    """
    Generates deterministic entity IDs for stable identification across 
    file modifications and synchronization operations.
    
    This class ensures that entities have stable IDs that change only when 
    their semantic content changes, enabling atomic replacement operations
    where old entities can be reliably replaced with new ones.
    """
    
    # Cache for computed IDs to avoid redundant calculations
    _id_cache: Dict[str, str] = {}
    
    @staticmethod
    def generate(entity: 'Entity', file_hash: str) -> str:
        """
        Generate a deterministic ID for an entity based on its content and file context.
        
        The ID is constructed from:
        - Entity name (unique within the file)
        - Entity type (function, class, variable, etc.)
        - Start line number (position within file)
        - File hash (ensures uniqueness across file versions)
        
        Args:
            entity: The entity to generate an ID for
            file_hash: SHA-256 hash of the file content
            
        Returns:
            A 16-character deterministic ID
            
        Example:
            For a function named "calculate_sum" at line 42 in a file with hash "abc123...",
            this might generate: "d4f8c2a1b3e5f678"
        """
        # Create cache key to avoid redundant computation
        cache_key = f"{entity.name}:{entity.entity_type.value}:{entity.location.start_line}:{entity.location.start_column}:{file_hash[:8]}"
        
        if cache_key in DeterministicEntityId._id_cache:
            return DeterministicEntityId._id_cache[cache_key]
        
        # Construct stable content identifier
        content_parts = [
            entity.name,
            entity.entity_type.value,
            str(entity.location.start_line),
            str(entity.location.start_column),
            file_hash[:8]  # First 8 chars of file hash for uniqueness
        ]
        
        # Add qualified name if different from name for additional uniqueness
        if entity.qualified_name != entity.name:
            content_parts.append(entity.qualified_name)
        
        # Include signature hash if available for more precise identification
        if entity.signature:
            sig_hash = hashlib.sha256(entity.signature.encode('utf-8')).hexdigest()[:8]
            content_parts.append(sig_hash)
        
        # Create deterministic content string
        content = ":".join(content_parts)
        
        # Generate deterministic hash
        entity_id = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        
        # Cache the result
        DeterministicEntityId._id_cache[cache_key] = entity_id
        
        return entity_id
    
    @staticmethod
    def generate_from_content(
        name: str,
        entity_type: str, 
        start_line: int,
        start_column: int,
        file_hash: str,
        qualified_name: Optional[str] = None,
        signature: Optional[str] = None
    ) -> str:
        """
        Generate a deterministic ID from entity content without an Entity object.
        
        Useful for scenarios where you need to generate an ID for comparison
        or lookup without constructing a full Entity object.
        
        Args:
            name: Entity name
            entity_type: Type of entity (function, class, etc.)
            start_line: Starting line number
            start_column: Starting column number  
            file_hash: File content hash
            qualified_name: Fully qualified name (optional)
            signature: Entity signature (optional)
            
        Returns:
            A 16-character deterministic ID
        """
        content_parts = [
            name,
            entity_type,
            str(start_line),
            str(start_column),
            file_hash[:8]
        ]
        
        if qualified_name and qualified_name != name:
            content_parts.append(qualified_name)
        
        if signature:
            sig_hash = hashlib.sha256(signature.encode('utf-8')).hexdigest()[:8]
            content_parts.append(sig_hash)
        
        content = ":".join(content_parts)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    @staticmethod
    def update_entity_with_deterministic_id(entity: 'Entity', file_hash: str) -> 'Entity':
        """
        Create a new entity with a deterministic ID, replacing the original ID.
        
        This method is used during synchronization to ensure entities have
        stable IDs that enable atomic replacement operations.
        
        Args:
            entity: The entity to update
            file_hash: Hash of the file containing the entity
            
        Returns:
            A new Entity object with deterministic ID
        """
        deterministic_id = DeterministicEntityId.generate(entity, file_hash)
        
        # Create updated entity with deterministic ID
        return entity.model_copy(update={'id': deterministic_id})
    
    @staticmethod
    def clear_cache() -> None:
        """
        Clear the internal ID cache.
        
        This should be called periodically to prevent unbounded memory growth
        in long-running synchronization processes.
        """
        DeterministicEntityId._id_cache.clear()
    
    @staticmethod
    def get_cache_size() -> int:
        """
        Get the current size of the ID cache.
        
        Returns:
            Number of cached ID mappings
        """
        return len(DeterministicEntityId._id_cache)
    
    @staticmethod
    def validate_deterministic_id(entity_id: str) -> bool:
        """
        Validate that an ID appears to be deterministically generated.
        
        Args:
            entity_id: The ID to validate
            
        Returns:
            True if the ID appears to be deterministically generated
        """
        # Deterministic IDs should be exactly 16 hexadecimal characters
        if len(entity_id) != 16:
            return False
        
        try:
            # Should be valid hexadecimal
            int(entity_id, 16)
            return True
        except ValueError:
            return False