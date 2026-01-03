"""
RAG Tutorial Utilities

Helpful functions for building RAG systems.
"""

from .embeddings import EmbeddingManager, embed_text, compute_similarity
from .retrieval import Retriever, HybridRetriever

# Optional imports (require additional dependencies)
try:
    from .generation import Generator, PromptTemplate
    __all__ = [
        'EmbeddingManager',
        'embed_text',
        'compute_similarity',
        'Retriever',
        'HybridRetriever',
        'Generator',
        'PromptTemplate',
    ]
except ImportError:
    __all__ = [
        'EmbeddingManager',
        'embed_text',
        'compute_similarity',
        'Retriever',
        'HybridRetriever',
    ]
