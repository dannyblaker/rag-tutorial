# Example 2: Document Chat System

An interactive chat system that lets you ask questions about your documents with conversation history.

## Features

- ✅ Load text documents from files or directories
- ✅ Smart text chunking with overlap
- ✅ Conversational interface with history
- ✅ Source attribution for answers
- ✅ Multi-turn conversations

## Setup

```bash
cd examples/02-document-chat

# Set API key
export OPENAI_API_KEY="your-key-here"

# Run
python document_chat.py
```

## Usage

The system will:
1. Create sample documents about AI and RAG
2. Load and chunk them
3. Start an interactive chat session

### Chat Commands

- Type your question and press Enter
- Type `stats` to see system statistics
- Type `quit` to exit

## Example Conversation

```
You: What is machine learning?