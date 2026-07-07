"""Vectorstore package initializer.

This file makes the `vectorstore` directory importable as a Python package.
"""

# NOTE: azure_ai_search.py now contains the free ChromaDB implementation
# (old Azure AI Search code kept there as comments for reference).
__all__ = ["azure_ai_search"]
