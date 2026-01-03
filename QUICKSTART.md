# Quick Start Guide

Get up and running with RAG in 5 minutes!

## Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) OpenAI API key for generation

## Installation

### 1. Clone or Navigate to Repository

```bash
cd /path/to/rag_tutorial
```

### 2. Create Virtual Environment

```bash
# Create virtual environment (tested with Python 3.13)
python -m venv .venv

# Activate it
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# This will install:
# - chromadb (vector database)
# - sentence-transformers (embeddings)
# - openai (for GPT models)
# - other utilities
```

### 4. Set Up Environment Variables (Optional)

```bash
# Copy example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# (Not required for basic examples)
```

## Running Examples

### Example 1: Simple QA (No API Key Required)

```bash
cd examples/01-simple-qa
python simple_qa.py
```

This example works without an API key and demonstrates basic RAG concepts.

### Example 2: Document Chat (Requires API Key)

```bash
# Set your API key first
export OPENAI_API_KEY="your-key-here"

cd examples/02-document-chat
python document_chat.py
```

This provides an interactive chat interface with your documents.

## Testing Your Setup

Quick test to verify everything works:

```bash
python -c "
from sentence_transformers import SentenceTransformer
import chromadb

print('Loading embedding model...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('✓ Embeddings working')

print('Testing ChromaDB...')
client = chromadb.Client()
collection = client.create_collection('test')
collection.add(documents=['test'], ids=['1'])
print('✓ Vector database working')

print('\nSetup successful! You are ready to learn RAG.')
"
```

## Learning Path

1. **Read the lessons** in order, starting with [lessons/01-introduction-to-rag.md](lessons/01-introduction-to-rag.md)
2. **Run the examples** to see concepts in action
3. **Experiment** with the utility functions in `utils/`
4. **Build your own** RAG system!
