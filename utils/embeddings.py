"""
Utility functions for embedding text using various models.
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingManager:
    """Manage embedding models and operations"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model

        Popular models:
        - 'all-MiniLM-L6-v2': Fast, 384 dimensions
        - 'all-mpnet-base-v2': Better quality, 768 dimensions
        - 'all-MiniLM-L12-v2': Good balance
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed text(s) into vector representation

        Args:
            texts: Single text string or list of strings

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts

        Returns:
            Similarity score between -1 and 1 (1 = identical)
        """
        emb1, emb2 = self.embed([text1, text2])

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        return float(dot_product / (norm1 * norm2))

    def batch_similarity(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute similarity between query and multiple documents

        Returns:
            List of similarity scores
        """
        query_emb = self.embed(query)
        doc_embs = self.embed(documents)

        # Compute cosine similarity for all documents
        similarities = []
        for doc_emb in doc_embs:
            dot_product = np.dot(query_emb, doc_emb)
            norm1 = np.linalg.norm(query_emb)
            norm2 = np.linalg.norm(doc_emb)
            similarities.append(float(dot_product / (norm1 * norm2)))

        return similarities

    def get_info(self) -> dict:
        """Get information about the embedding model"""
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'max_seq_length': self.model.max_seq_length
        }


# Convenience functions
def embed_text(text: str, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Quick function to embed a single text"""
    embedder = EmbeddingManager(model_name)
    return embedder.embed(text)[0]


def compute_similarity(text1: str, text2: str, model_name: str = 'all-MiniLM-L6-v2') -> float:
    """Quick function to compute similarity between two texts"""
    embedder = EmbeddingManager(model_name)
    return embedder.similarity(text1, text2)


if __name__ == "__main__":
    # Example usage
    embedder = EmbeddingManager()

    print("Embedding Model Info:")
    print(embedder.get_info())

    print("\nTesting similarity:")
    texts = [
        "The cat sat on the mat",
        "A feline rested on the rug",
        "Python is a programming language"
    ]

    query = "Where is the cat?"
    similarities = embedder.batch_similarity(query, texts)

    for text, sim in zip(texts, similarities):
        print(f"Similarity: {sim:.3f} - {text}")
