"""
Utility functions for text generation using LLMs.
"""

from typing import List, Dict, Optional
from openai import OpenAI
import os


class Generator:
    """Text generation using OpenAI models"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize generator

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (gpt-3.5-turbo, gpt-4, etc.)
            temperature: Randomness (0=deterministic, 1=creative)
            max_tokens: Maximum response length
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: User prompt
            system_message: Optional system message to set behavior
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated text
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        return response.choices[0].message.content

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate answer using retrieved context (RAG)

        Args:
            query: User question
            context: Retrieved document chunks
            system_message: Optional system message

        Returns:
            Generated answer
        """
        # Format context
        context_str = "\n\n".join([
            f"[{i+1}] {chunk}"
            for i, chunk in enumerate(context)
        ])

        # Default system message for RAG
        if system_message is None:
            system_message = """You are a helpful assistant that answers questions based on provided context.
Always cite the source numbers you reference (e.g., "According to [1]...").
If the context doesn't contain the answer, say so clearly."""

        # Create prompt
        prompt = f"""Context:
{context_str}

Question: {query}

Answer:"""

        return self.generate(prompt, system_message=system_message)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate response in a conversation

        Args:
            messages: List of message dicts with 'role' and 'content'
                     e.g., [{"role": "user", "content": "Hello"}]
            temperature: Override default temperature

        Returns:
            Generated response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content


class PromptTemplate:
    """Helper for creating consistent prompts"""

    # Common templates
    QA_TEMPLATE = """Answer the question based on the context below.
If the context doesn't contain the answer, say "I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer:"""

    QA_WITH_SOURCES = """Answer the question based on the context below.
Cite which sources you use (e.g., "According to [1]...").
If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""

    SUMMARIZE_TEMPLATE = """Summarize the following text concisely:

{text}

Summary:"""

    @staticmethod
    def format_qa(question: str, context: List[str], with_sources: bool = True) -> str:
        """Format QA prompt with context"""
        # Format context with numbers
        context_str = "\n\n".join([
            f"[{i+1}] {chunk}"
            for i, chunk in enumerate(context)
        ])

        template = PromptTemplate.QA_WITH_SOURCES if with_sources else PromptTemplate.QA_TEMPLATE

        return template.format(context=context_str, question=question)

    @staticmethod
    def format_summarize(text: str) -> str:
        """Format summarization prompt"""
        return PromptTemplate.SUMMARIZE_TEMPLATE.format(text=text)


if __name__ == "__main__":
    # Example usage
    try:
        generator = Generator()

        # Simple generation
        response = generator.generate(
            "Explain RAG in one sentence.",
            system_message="You are a helpful AI assistant."
        )
        print("Simple generation:")
        print(response)

        # RAG generation
        print("\n" + "="*60)
        print("RAG generation:")

        context = [
            "RAG stands for Retrieval-Augmented Generation.",
            "It combines information retrieval with text generation.",
            "RAG helps reduce hallucinations in LLM responses."
        ]

        answer = generator.generate_with_context(
            query="What is RAG?",
            context=context
        )
        print(answer)

    except ValueError as e:
        print(f"Error: {e}")
        print("Please set OPENAI_API_KEY environment variable")
