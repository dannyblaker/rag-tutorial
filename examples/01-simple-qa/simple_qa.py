#!/usr/bin/env python3
"""
Example 1: Simple Question Answering

A minimal RAG system that answers questions from a small knowledge base.
Perfect for understanding the basics.
"""

import os
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI


class SimpleQA:
    def __init__(self, openai_api_key=None):
        """Initialize the simple QA system"""
        # Setup embedding model (free, runs locally)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Setup vector database
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("qa_knowledge")

        # Setup LLM (optional)
        self.llm = None
        if openai_api_key:
            self.llm = OpenAI(api_key=openai_api_key)

    def add_knowledge(self, facts: list[str]):
        """Add facts to the knowledge base"""
        print(f"Adding {len(facts)} facts to knowledge base...")

        ids = [f"fact_{i}" for i in range(len(facts))]
        self.collection.add(
            documents=facts,
            ids=ids
        )

        print("✓ Knowledge base updated")

    def answer(self, question: str, n_results: int = 2) -> dict:
        """Answer a question using the knowledge base"""
        # Retrieve relevant facts
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )

        relevant_facts = results['documents'][0]
        distances = results['distances'][0]

        # Generate answer
        if self.llm:
            answer = self._generate_with_llm(question, relevant_facts)
        else:
            answer = self._generate_simple(question, relevant_facts)

        return {
            'question': question,
            'answer': answer,
            'sources': [
                {'fact': fact, 'similarity': 1 - dist}
                for fact, dist in zip(relevant_facts, distances)
            ]
        }

    def _generate_with_llm(self, question: str, facts: list[str]) -> str:
        """Generate answer using LLM"""
        context = "\n".join([f"- {fact}" for fact in facts])

        prompt = f"""Answer the question based on these facts.
If the facts don't contain the answer, say so.

Facts:
{context}

Question: {question}

Answer:"""

        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )

        return response.choices[0].message.content

    def _generate_simple(self, question: str, facts: list[str]) -> str:
        """Generate simple answer without LLM"""
        if not facts:
            return "I don't have information to answer this question."

        return f"Based on the knowledge base: {facts[0]}"


def main():
    """Run the simple QA example"""
    print("=" * 60)
    print("Simple Question Answering System")
    print("=" * 60)

    # Initialize system
    api_key = os.getenv("OPENAI_API_KEY", "")
    qa = SimpleQA(openai_api_key=api_key)

    # Add knowledge base
    knowledge = [
        "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "RAG stands for Retrieval-Augmented Generation, a technique that combines retrieval with LLMs.",
        "Vector embeddings are numerical representations of text that capture semantic meaning.",
        "ChromaDB is an open-source vector database designed for AI applications.",
        "The Eiffel Tower is located in Paris, France and was completed in 1889.",
        "Machine learning is a subset of artificial intelligence that learns from data.",
    ]

    qa.add_knowledge(knowledge)

    # Example questions
    questions = [
        "Who created Python?",
        "What is RAG?",
        "Where is the Eiffel Tower?",
        "What is machine learning?",
    ]

    print("\n" + "=" * 60)
    print("Asking Questions")
    print("=" * 60)

    for question in questions:
        result = qa.answer(question)

        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print("\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(
                f"  [{i}] (similarity: {source['similarity']:.3f}) {source['fact'][:60]}...")
        print("-" * 60)


if __name__ == "__main__":
    main()
