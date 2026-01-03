#!/usr/bin/env python3
"""
Quick demo of RAG concepts using the tutorial utilities.
This demonstrates the basic RAG workflow without needing an API key.
"""

from utils import EmbeddingManager, Retriever
print("=" * 70)
print("RAG Tutorial - Quick Demo")
print("=" * 70)

# Step 1: Import utilities
print("\n📦 Step 1: Importing RAG utilities...")

# Step 2: Create embedding model
print("🔧 Step 2: Loading embedding model...")
embedder = EmbeddingManager('all-MiniLM-L6-v2')
print(f"   Model info: {embedder.get_info()}")

# Step 3: Create knowledge base
print("\n📚 Step 3: Creating knowledge base...")
documents = [
    "Python is a high-level programming language created by Guido van Rossum.",
    "RAG stands for Retrieval-Augmented Generation, combining retrieval with LLMs.",
    "Vector embeddings convert text into numerical representations that capture meaning.",
    "ChromaDB is an open-source vector database designed for AI applications.",
    "The Eiffel Tower is located in Paris, France and was built in 1889.",
    "Machine learning is a subset of AI that learns patterns from data.",
]

# Step 4: Initialize retriever and add documents
print("💾 Step 4: Storing documents in vector database...")
retriever = Retriever(collection_name="demo")
retriever.add_documents(documents)
print(f"   Stored {retriever.count()} documents")

# Step 5: Test semantic search
print("\n🔍 Step 5: Testing semantic search...")
test_queries = [
    "Who created Python?",
    "What is RAG?",
    "Tell me about embeddings",
]

for query in test_queries:
    print(f"\n   Query: '{query}'")
    results = retriever.search(query, n_results=2)

    for i, result in enumerate(results, 1):
        similarity = 1 - result['distance']
        print(f"   [{i}] Similarity: {similarity:.3f}")
        print(f"       {result['content'][:70]}...")

# Step 6: Demonstrate similarity computation
print("\n\n🎯 Step 6: Computing text similarity...")
text1 = "machine learning algorithms"
text2 = "AI and neural networks"
text3 = "cooking recipes"

sim_12 = embedder.similarity(text1, text2)
sim_13 = embedder.similarity(text1, text3)

print(f"   Similarity '{text1}' vs '{text2}': {sim_12:.3f}")
print(f"   Similarity '{text1}' vs '{text3}': {sim_13:.3f}")
print(f"   → Related topics have high similarity!")

# Summary
print("\n" + "=" * 70)
print("✅ Demo Complete!")
print("=" * 70)
print("\nWhat you just saw:")
print("  1. Text converted to embeddings (vectors)")
print("  2. Embeddings stored in a vector database")
print("  3. Semantic search finds relevant documents")
print("  4. Works even with different words (RAG vs Retrieval-Augmented)")
print("\nNext steps:")
print("  - Read lessons/01-introduction-to-rag.md")
print("  - Try examples/01-simple-qa/simple_qa.py")
print("  - Build your own RAG system!")
print("=" * 70)
