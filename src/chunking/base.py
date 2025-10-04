"""
base.py — Defines the abstract interface for all chunking strategies.

Every concrete chunker (semantic, structural, sliding, hybrid, etc.)
must inherit from this class and implement the `chunk()` method.
"""

from abc import ABC, abstractmethod


class Chunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, doc_name: str = "document"):
        """
        Split a text into chunks and return a list of dictionaries.

        Each returned dictionary should have the format:
        {
            "chunk_id": str,
            "text": str,
            "metadata": {
                "num_sentences": int,
                "num_words": int
            }
        }

        Args:
            text (str): The full input text.
            doc_name (str): The name or identifier of the source document.

        Returns:
            List[dict]: A list of chunk dictionaries.
        """
        pass

