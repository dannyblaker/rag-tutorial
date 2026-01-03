# RAG Glossary

Common terms and acronyms used in RAG systems.

## A

**ANN (Approximate Nearest Neighbor)**  
Algorithm that finds similar vectors quickly by trading perfect accuracy for speed. Essential for scaling vector search to millions of documents.

**Augmentation**  
The process of adding retrieved context to a prompt before sending it to an LLM.

## C

**Chunk/Chunking**  
Splitting documents into smaller pieces for embedding and retrieval. Common sizes: 200-1000 characters.

**ChromaDB**  
Open-source vector database designed for AI applications. Easy to use, runs locally.

**Cosine Similarity**  
Metric for measuring how similar two vectors are. Range: -1 (opposite) to 1 (identical). Commonly used in RAG retrieval.

**Context Window**  
Maximum number of tokens an LLM can process at once (input + output). GPT-3.5: 16K, GPT-4: 128K, Claude 3: 200K.

**Cross-Encoder**  
Model that scores query-document pairs directly for re-ranking. More accurate but slower than embeddings.

## E

**Embedding**  
Vector (list of numbers) representing text in high-dimensional space. Captures semantic meaning.

**Embedding Model**  
Neural network that converts text to embeddings. Examples: text-embedding-ada-002, all-MiniLM-L6-v2.

## F

**Faiss**  
Facebook AI Similarity Search. High-performance library for vector search, especially at scale.

**Few-Shot Prompting**  
Providing examples in the prompt to guide the LLM's response style.

**Fine-Tuning**  
Training a model on specific data to specialize it. Alternative to RAG but more expensive to update.

## G

**Generation**  
The "G" in RAG. Using an LLM to create text based on retrieved context.

**GPT**  
Generative Pre-trained Transformer. Family of LLMs by OpenAI (GPT-3.5, GPT-4).

## H

**Hallucination**  
When an LLM generates plausible-sounding but false information. RAG reduces this by grounding responses in retrieved facts.

**HNSW (Hierarchical Navigable Small World)**  
Graph-based ANN algorithm. Very fast queries, high accuracy. Used by ChromaDB, Weaviate, Qdrant.

**Hybrid Search**  
Combining semantic search (embeddings) with keyword search (BM25) for better results.

## I

**Indexing**  
Process of preparing documents for retrieval: chunking, embedding, and storing in vector database.

**Inference**  
Using a trained model to make predictions (generate text, create embeddings, etc.).

**IVF (Inverted File Index)**  
ANN algorithm that clusters vectors into regions for faster search. More memory efficient than HNSW.

## K

**Knowledge Base**  
Collection of documents/data that a RAG system retrieves from.

**K-NN (K-Nearest Neighbors)**  
Finding the K most similar items. In RAG: finding K most relevant document chunks.

## L

**LLM (Large Language Model)**  
Neural network trained on massive text to understand and generate language. Examples: GPT-4, Claude, Llama.

## M

**Metadata**  
Information about documents (source, date, author, etc.). Used for filtering in RAG.

**MMR (Maximum Marginal Relevance)**  
Retrieval strategy that balances relevance with diversity to avoid redundant results.

## O

**Ollama**  
Tool for running open-source LLMs locally. Useful for private/offline RAG systems.

## P

**Pinecone**  
Managed vector database service. Fully serverless, scales automatically.

**Prompt Engineering**  
Art/science of crafting effective prompts to get desired LLM outputs.

**Prompt Template**  
Reusable pattern for creating prompts. Example: "Context: {context}\nQuestion: {question}\nAnswer:"

## Q

**Query**  
User's question or search term in a RAG system.

**Query Expansion**  
Generating variations of a query to retrieve more comprehensive results.

## R

**RAG (Retrieval-Augmented Generation)**  
Technique combining information retrieval with LLM generation for grounded, accurate responses.

**Re-ranking**  
Re-ordering retrieved results using a more sophisticated model (often cross-encoder) for better relevance.

**Retrieval**  
The "R" in RAG. Finding relevant documents/chunks from a knowledge base.

## S

**Semantic Search**  
Finding information based on meaning rather than exact keywords. Enabled by embeddings.

**Sentence Transformers**  
Library and collection of models for creating embeddings. Popular for open-source RAG.

**Similarity Score**  
Numerical measure of how related two pieces of text are. Higher = more similar.

## T

**Temperature**  
LLM parameter controlling randomness. 0 = deterministic, 1 = creative. For RAG, typically 0.3-0.7.

**Token**  
Smallest unit of text for LLMs. Roughly 4 characters or 3/4 of a word. "Hello world" ≈ 2-3 tokens.

**Top-K**  
Retrieving the K highest-scoring results. Common in RAG: K=3-5 chunks.

## V

**Vector**  
List of numbers representing data in mathematical space. Text embeddings are vectors.

**Vector Database**  
Specialized database for storing and searching vectors efficiently. Examples: ChromaDB, Pinecone, Weaviate.

**Vector Search**  
Finding similar vectors using distance/similarity metrics. Core of RAG retrieval.

## Common Abbreviations

| Abbreviation | Meaning |
|-------------|---------|
| AI | Artificial Intelligence |
| ANN | Approximate Nearest Neighbor |
| API | Application Programming Interface |
| BM25 | Best Matching 25 (keyword search algorithm) |
| DB | Database |
| FAISS | Facebook AI Similarity Search |
| GPT | Generative Pre-trained Transformer |
| HNSW | Hierarchical Navigable Small World |
| IVF | Inverted File Index |
| K-NN | K-Nearest Neighbors |
| LLM | Large Language Model |
| ML | Machine Learning |
| MMR | Maximum Marginal Relevance |
| NLP | Natural Language Processing |
| RAG | Retrieval-Augmented Generation |

## Quick Reference

### Similarity Scores
- `1.0` = Identical
- `0.8-1.0` = Very similar
- `0.5-0.8` = Somewhat similar
- `0.0-0.5` = Not very similar
- `-1.0` = Opposite (rare)

### Typical RAG Parameters
- Chunk size: 500-1000 characters
- Chunk overlap: 50-200 characters
- Top-K retrieval: 3-5 chunks
- Temperature: 0.3-0.7
- Embedding dimensions: 384-1536

### Model Comparison
| Model Type | Speed | Quality | Cost |
|-----------|-------|---------|------|
| GPT-3.5 | Fast | Good | Low |
| GPT-4 | Slow | Excellent | High |
| Claude 3 Haiku | Fast | Good | Low |
| Claude 3 Opus | Slow | Excellent | High |
| Llama 3 (local) | Fast* | Good | Free** |

*Depends on hardware  
**Hosting costs only

---

**Need more help?** Check the [lessons](lessons/) for detailed explanations!
