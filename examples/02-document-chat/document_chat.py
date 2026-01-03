#!/usr/bin/env python3
"""
Example 2: Document Chat System

Chat with your documents! Upload text files and ask questions about them.
Demonstrates document loading, chunking, and conversational RAG.
"""

import os
from pathlib import Path
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI


class DocumentChatSystem:
    def __init__(self, openai_api_key: str):
        """Initialize the document chat system"""
        self.llm = OpenAI(api_key=openai_api_key)

        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Vector database
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="documents",
            metadata={"description": "Document chat system"}
        )

        # Conversation history
        self.conversation_history = []

    def load_document(self, file_path: str) -> int:
        """Load and process a document"""
        print(f"Loading document: {file_path}")

        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Chunk the document
        chunks = self._chunk_text(content, chunk_size=500, overlap=50)

        # Add to vector database
        ids = [f"{Path(file_path).stem}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{
            'source': os.path.basename(file_path),
            'chunk_id': i
        } for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

        print(
            f"✓ Added {len(chunks)} chunks from {os.path.basename(file_path)}")
        return len(chunks)

    def load_directory(self, directory: str) -> int:
        """Load all text files from a directory"""
        total_chunks = 0

        for file_path in Path(directory).glob("*.txt"):
            total_chunks += self.load_document(str(file_path))

        return total_chunks

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to end at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.5:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk.strip())
            start = end - overlap

        return [c for c in chunks if c]  # Remove empty chunks

    def chat(self, message: str, n_results: int = 3) -> Dict:
        """Chat with documents"""
        # Retrieve relevant chunks
        results = self.collection.query(
            query_texts=[message],
            n_results=n_results
        )

        chunks = results['documents'][0]
        metadatas = results['metadatas'][0]

        # Format context with sources
        context_parts = []
        for i, (chunk, metadata) in enumerate(zip(chunks, metadatas), 1):
            context_parts.append(
                f"[Source {i}: {metadata['source']}]\n{chunk}"
            )
        context = "\n\n".join(context_parts)

        # Build conversation
        messages = [
            {"role": "system", "content": """You are a helpful assistant that answers questions about documents.
Always cite which source you're referencing (e.g., "According to Source 1...").
If the documents don't contain the answer, say so."""}
        ]

        # Add conversation history (last 3 exchanges)
        messages.extend(self.conversation_history[-6:])

        # Add current query with context
        user_message = f"""Documents:
{context}

Question: {message}"""

        messages.append({"role": "user", "content": user_message})

        # Generate response
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        assistant_message = response.choices[0].message.content

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_message})

        return {
            'question': message,
            'answer': assistant_message,
            'sources': [
                {
                    'content': chunk,
                    'source': metadata['source'],
                    'chunk_id': metadata['chunk_id']
                }
                for chunk, metadata in zip(chunks, metadatas)
            ]
        }

    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'total_chunks': self.collection.count(),
            'conversation_turns': len(self.conversation_history) // 2
        }


def main():
    """Run the document chat example"""
    print("=" * 60)
    print("Document Chat System")
    print("=" * 60)

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: Please set OPENAI_API_KEY environment variable")
        print("export OPENAI_API_KEY='your-key-here'")
        return

    # Initialize system
    chat_system = DocumentChatSystem(openai_api_key=api_key)

    # Load sample documents
    print("\nLoading sample documents...")

    # Create sample documents if they don't exist
    sample_dir = Path("sample_docs")
    sample_dir.mkdir(exist_ok=True)

    # Sample document 1
    doc1_path = sample_dir / "ai_overview.txt"
    if not doc1_path.exists():
        with open(doc1_path, 'w') as f:
            f.write("""Artificial Intelligence Overview

Artificial Intelligence (AI) is the simulation of human intelligence by machines.
AI systems can perform tasks such as learning, reasoning, and problem-solving.

Machine Learning is a subset of AI that enables systems to learn from data without
explicit programming. Deep Learning is a subset of ML using neural networks with
multiple layers.

Natural Language Processing (NLP) is an AI field focused on enabling computers to
understand, interpret, and generate human language. Applications include chatbots,
translation, and sentiment analysis.

Computer Vision enables machines to interpret and understand visual information from
the world. It's used in facial recognition, autonomous vehicles, and medical imaging.
""")

    # Sample document 2
    doc2_path = sample_dir / "rag_details.txt"
    if not doc2_path.exists():
        with open(doc2_path, 'w') as f:
            f.write("""Retrieval-Augmented Generation Details

RAG combines information retrieval with text generation. It consists of three main steps:

1. Indexing: Documents are split into chunks, embedded, and stored in a vector database.
   This happens once during setup.

2. Retrieval: When a query comes in, it's embedded and used to search the vector database
   for similar chunks. Typically, the top 3-5 most relevant chunks are retrieved.

3. Generation: The retrieved chunks are added to the prompt as context, and an LLM
   generates a response based on this context.

Key advantages of RAG:
- Access to external knowledge beyond training data
- Reduced hallucinations by grounding responses in documents
- Easy to update knowledge by adding/removing documents
- Source attribution for transparency

RAG is widely used in customer support, document Q&A, and knowledge management systems.
""")

    # Load documents
    total_chunks = chat_system.load_directory(str(sample_dir))

    print(f"\n✓ Loaded {total_chunks} total chunks")

    # Interactive chat
    print("\n" + "=" * 60)
    print("Chat with Your Documents")
    print("Type 'quit' to exit, 'stats' for statistics")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if user_input.lower() == 'stats':
                stats = chat_system.get_stats()
                print(f"\nStatistics:")
                print(f"  Total chunks: {stats['total_chunks']}")
                print(f"  Conversation turns: {stats['conversation_turns']}\n")
                continue

            # Get response
            result = chat_system.chat(user_input)

            print(f"\nAssistant: {result['answer']}")

            # Show sources (compact)
            print(
                f"\n(Sources: {', '.join(set(s['source'] for s in result['sources']))})\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}\n")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
