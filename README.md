# RAG Pipeline — Effective Python Q&A

A Retrieval-Augmented Generation (RAG) system that answers questions about any PDF document using semantic search and LLM.

## How It Works

1. Load PDF document
2. Split into chunks with overlap
3. Convert chunks to embeddings using HuggingFace
4. Store embeddings in ChromaDB vector database
5. User asks a question → find relevant chunks → LLM answers

## Tech Stack

- **LangChain** — document loading and text splitting
- **HuggingFace Embeddings** — free local embeddings (all-MiniLM-L6-v2)
- **ChromaDB** — vector database for semantic search
- **Groq LLaMA** — LLM for answer generation
- **Python** — core language

## Running Locally

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Add `.env` file with `GROQ_API_KEY=your_key`
6. Add your PDF as `sample.pdf`
7. Run: `python main.py`

## Features

- Persistent vector store — builds once, reloads instantly
- Semantic search — finds relevant content by meaning not keywords
- Context-grounded answers — LLM answers only from document content
- Handles large documents (tested with 475 page book)

## Environment Variables

```
GROQ_API_KEY=your_groq_api_key
```