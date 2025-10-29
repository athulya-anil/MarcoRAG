"""
Validation utilities for chunk and metadata structure consistency.
"""

from typing import Dict, List, Any, Optional


def validate_chunk_structure(chunk: Dict[str, Any], require_fields: Optional[List[str]] = None) -> bool:
    """
    Validate that a chunk has the expected structure.

    Args:
        chunk: The chunk dictionary to validate
        require_fields: List of required field names. If None, uses default required fields.

    Returns:
        True if valid, False otherwise
    """
    if require_fields is None:
        require_fields = ["chunk_id"]

    if not isinstance(chunk, dict):
        return False

    # Check required fields exist
    for field in require_fields:
        if field not in chunk:
            return False

    # Check that chunk has either 'text' or 'content'
    if "text" not in chunk and "content" not in chunk:
        return False

    return True


def validate_embedding_structure(embedding: Dict[str, Any]) -> bool:
    """
    Validate that an embedding has the expected structure.

    Args:
        embedding: The embedding dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(embedding, dict):
        return False

    # Must have chunk_id
    if "chunk_id" not in embedding:
        return False

    # Must have either 'vector' or 'embedding' field
    if "vector" not in embedding and "embedding" not in embedding:
        return False

    return True


def validate_metadata_structure(metadata: Dict[str, Any]) -> bool:
    """
    Validate that metadata has expected structure.

    Args:
        metadata: The metadata dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(metadata, dict):
        return False

    # Should have at least one of these fields
    expected_fields = ["summary", "keywords", "entities", "category"]
    has_field = any(field in metadata for field in expected_fields)

    return has_field


def normalize_chunk_fields(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize chunk fields to have consistent field names.
    Converts 'content' to 'text' if needed.

    Args:
        chunk: The chunk to normalize

    Returns:
        Normalized chunk dictionary
    """
    normalized = chunk.copy()

    # Normalize text field
    if "content" in normalized and "text" not in normalized:
        normalized["text"] = normalized["content"]
    elif "text" not in normalized and "content" not in normalized:
        normalized["text"] = ""

    return normalized


def normalize_embedding_fields(embedding: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize embedding fields to have consistent field names.
    Converts 'vector' to 'embedding' if needed.

    Args:
        embedding: The embedding to normalize

    Returns:
        Normalized embedding dictionary
    """
    normalized = embedding.copy()

    # Normalize embedding field
    if "vector" in normalized and "embedding" not in normalized:
        normalized["embedding"] = normalized["vector"]
    elif "embedding" not in normalized and "vector" not in normalized:
        normalized["embedding"] = []

    return normalized
