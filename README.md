# Retrieval-Augmented Generation (RAG) Tutorial

A comprehensive, hands-on tutorial that takes you from the basics through to building production-ready RAG systems.

[![A Danny Blaker project badge](https://github.com/dannyblaker/dannyblaker.github.io/blob/main/danny_blaker_project_badge.svg)](https://github.com/dannyblaker/)

## 🎯 What You'll Learn

By the end of this tutorial, you'll understand:
- What RAG is and why it's revolutionary for AI applications
- How vector embeddings capture semantic meaning
- How to build and query vector databases
- How to integrate retrieval with language models
- Advanced techniques for production RAG systems

## 🚀 Who This Is For

- **Beginners**: No prior RAG knowledge required
- **Developers**: Learn to build RAG applications
- **ML Engineers**: Understand the architecture and implementation details
- **Product Managers**: Gain technical insight into RAG capabilities

## 📚 Learning Path

### Part 1: Foundations
1. **[Introduction to RAG](lessons/01-introduction-to-rag.md)**
   - The problem RAG solves
   - How RAG works at a high level
   - Real-world applications

2. **[Understanding Embeddings](lessons/02-understanding-embeddings.md)**
   - What are vector embeddings?
   - Semantic similarity
   - Embedding models

3. **[Vector Databases & Retrieval](lessons/03-vector-databases-retrieval.md)**
   - Storing and indexing vectors
   - Similarity search algorithms
   - Retrieval strategies

4. **[Language Models & Generation](lessons/04-language-models-generation.md)**
   - How LLMs work
   - Prompting techniques
   - Integrating retrieved context

### Part 2: Building RAG Systems
5. **[Building Your First RAG System](lessons/05-building-simple-rag.md)**
   - Step-by-step implementation
   - Working code example
   - Testing and evaluation

6. **[Advanced RAG Techniques](lessons/06-advanced-rag-techniques.md)**
   - Document chunking strategies
   - Re-ranking retrieved results
   - Hybrid search
   - Query optimization
   - Handling edge cases

### Part 3: Practical Examples
- **[Example 1: Simple Question Answering](examples/01-simple-qa/)**
- **[Example 2: Document Chat System](examples/02-document-chat/)**
- **[Example 3: Advanced RAG Pipeline](examples/03-advanced-rag/)**

## 🛠️ Setup

### Prerequisites
- Python 3.13 (or 3.10+)
- Basic Python knowledge
- Understanding of APIs (helpful but not required)

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd myproject
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys (you'll need at least one):
```bash
# Create a .env file
cp .env.example .env

# Add your API keys:
# OPENAI_API_KEY=your-key-here
# Or for free alternatives:
# HUGGINGFACE_API_KEY=your-key-here
```

## 📖 How to Use This Tutorial

1. **Start with Lesson 1** and work through sequentially
2. **Read the theory** in each lesson markdown file
3. **Run the code examples** to see concepts in action
4. **Experiment** with the examples and modify them
5. **Build your own** RAG system for your use case

## 🎓 Learning Tips

- **Take your time**: RAG combines multiple complex concepts
- **Run every code example**: Hands-on practice is crucial
- **Experiment**: Modify parameters and see what happens
- **Build projects**: Apply what you learn to real problems
- **Ask questions**: Open issues if something isn't clear

## 📊 What is RAG?

**Retrieval-Augmented Generation** is a technique that enhances large language models by giving them access to external knowledge. Instead of relying solely on their training data, RAG systems:

1. **Retrieve** relevant information from a knowledge base
2. **Augment** the LLM's prompt with this information
3. **Generate** accurate, grounded responses

This solves key problems like:
- ✅ Hallucinations (making up facts)
- ✅ Outdated information
- ✅ Lack of domain-specific knowledge
- ✅ Inability to cite sources

## 🌟 Why RAG Matters

RAG has become the standard approach for:
- **Customer support chatbots**: Answer from company documentation
- **Research assistants**: Search through papers and documents
- **Enterprise search**: Query internal knowledge bases
- **Personal assistants**: Access your notes and files
- **Legal/Medical AI**: Ground responses in verified sources

## 🔧 Technology Stack

This tutorial uses:
- **Python**: Primary programming language
- **OpenAI/Anthropic APIs**: For embeddings and LLMs (with free alternatives)
- **ChromaDB**: Vector database (simple, local, no setup)
- **LangChain** (optional): Framework for building LLM apps
- **Sentence Transformers**: Free, open-source embeddings

## 📁 Repository Structure

```
rag_tutorial/
├── README.md                          # You are here
├── lessons/                           # Step-by-step lessons
│   ├── 01-introduction-to-rag.md
│   ├── 02-understanding-embeddings.md
│   ├── 03-vector-databases-retrieval.md
│   ├── 04-language-models-generation.md
│   ├── 05-building-simple-rag.md
│   └── 06-advanced-rag-techniques.md
├── examples/                          # Working code examples
│   ├── 01-simple-qa/
│   ├── 02-document-chat/
│   └── 03-advanced-rag/
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── embeddings-exploration.ipynb
│   └── rag-from-scratch.ipynb
├── utils/                            # Helper utilities
│   ├── embeddings.py
│   ├── retrieval.py
│   └── generation.py
├── data/                             # Sample data for examples
├── requirements.txt                  # Python dependencies
└── .env.example                      # Environment variables template
```


---

**Ready to start?** Head to [Lesson 1: Introduction to RAG](lessons/01-introduction-to-rag.md)
