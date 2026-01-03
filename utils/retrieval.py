"""
Utility functions for text retrieval and vector search.
"""

from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings


class Retriever:
    """Simple retrieval system using ChromaDB"""

    def __init__(self, collection_name: str = "documents", persist_directory: Optional[str] = None):
        """
        Initialize retriever

        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the database (None for in-memory)
        """
        if persist_directory:
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document collection for retrieval"}
        )

    def add_documents(
        self,
        documents: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None
    ) -> int:
        """
        Add documents to the collection

        Args:
            documents: List of document texts
            ids: Optional list of IDs (auto-generated if None)
            metadatas: Optional list of metadata dicts

        Returns:
            Number of documents added
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

        return len(documents)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant documents

        Args:
            query: Search query
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {"source": "file.txt"})

        Returns:
            List of dicts with 'content', 'metadata', 'distance'
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i],
                'id': results['ids'][0][i]
            })

        return formatted_results

    def get_by_ids(self, ids: List[str]) -> List[Dict]:
        """Get documents by their IDs"""
        results = self.collection.get(ids=ids)

        formatted_results = []
        for i in range(len(results['documents'])):
            formatted_results.append({
                'content': results['documents'][i],
                'metadata': results['metadatas'][i] if results['metadatas'] else {},
                'id': results['ids'][i]
            })

        return formatted_results

    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None):
        """Delete documents by IDs or metadata filter"""
        if ids:
            self.collection.delete(ids=ids)
        elif where:
            self.collection.delete(where=where)

    def count(self) -> int:
        """Get total number of documents"""
        return self.collection.count()

    def reset(self):
        """Delete all documents from collection"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(self.collection.name)


class HybridRetriever(Retriever):
    """Retriever with additional filtering and ranking capabilities"""

    def search_with_threshold(
        self,
        query: str,
        threshold: float = 0.5,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search and filter by similarity threshold

        Args:
            query: Search query
            threshold: Minimum similarity score (0-1, higher = more similar)
            max_results: Maximum results to consider

        Returns:
            List of results above threshold
        """
        results = self.search(query, n_results=max_results)

        # Filter by threshold (distance is inverse of similarity)
        filtered = [
            r for r in results
            if (1 - r['distance']) >= threshold
        ]

        return filtered

    def search_with_mmr(
        self,
        query: str,
        n_results: int = 5,
        lambda_mult: float = 0.5
    ) -> List[Dict]:
        """
        Maximum Marginal Relevance search for diverse results

        Args:
            query: Search query
            n_results: Number of results
            lambda_mult: Diversity vs relevance (0=diverse, 1=relevant)

        Note: ChromaDB doesn't natively support MMR, this is a placeholder
        """
        # For now, just return regular search
        # In production, implement proper MMR algorithm
        return self.search(query, n_results=n_results)


if __name__ == "__main__":
    # Example usage
    retriever = Retriever(collection_name="test")

    # Add documents
    docs = [
        "Python is a programming language",
        "Machine learning is a subset of AI",
        "RAG combines retrieval and generation"
    ]

    retriever.add_documents(
        documents=docs,
        metadatas=[{"topic": "programming"}, {"topic": "AI"}, {"topic": "AI"}]
    )

    print(f"Total documents: {retriever.count()}")

    # Search
    results = retriever.search("What is AI?", n_results=2)

    print("\nSearch results:")
    for r in results:
        print(f"  Score: {1 - r['distance']:.3f}")
        print(f"  Content: {r['content']}")
        print(f"  Metadata: {r['metadata']}\n")
