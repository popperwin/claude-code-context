"""
Storage utilities for consistent operations across the codebase.

Provides centralized utilities for storage operations including entity ID
to Qdrant point ID conversion and other storage-related helper functions.
"""

import hashlib


def entity_id_to_qdrant_id(entity_id: str) -> int:
    """
    Convert entity ID to Qdrant point ID using consistent SHA256 hashing.
    
    This is the canonical conversion function used throughout the codebase
    to ensure consistent point ID generation for Qdrant storage operations.
    All callers should use this function rather than implementing their own.
    
    Args:
        entity_id: Entity identifier to convert (e.g., "test.py::test_function")
        
    Returns:
        Integer point ID for Qdrant storage
        
    Example:
        >>> entity_id_to_qdrant_id("test.py::test_function")
        12345678901234567890
    """
    # Use SHA256 hash and take first 8 bytes as unsigned integer
    hash_digest = hashlib.sha256(entity_id.encode('utf-8')).digest()
    return int.from_bytes(hash_digest[:8], byteorder='big', signed=False)