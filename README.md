# RAG PDF Q&A System

![CI](https://github.com/Ani1070git/rag-pipeline/actions/workflows/ci.yml/badge.svg)

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions about any PDF document using semantic search and LLM.

## Live Demo
**Frontend:** [rag-frontend](https://github.com/Ani1070git/rag-frontend)  
**Backend API:** Run locally with `python main.py`

## How It Works

1. Upload any PDF document
2. System splits it into chunks with overlap
3. HuggingFace embeddings convert chunks to vectors
4. ChromaDB stores vectors for semantic search
5. User asks a question → system finds relevant chunks → Groq LLaMA answers

## Tech Stack

- **FastAPI** — Backend API framework
- **LangChain** — Document loading and text splitting
- **HuggingFace Embeddings** — Free local embeddings (all-MiniLM-L6-v2)
- **ChromaDB** — Vector database for semantic search
- **Groq LLaMA** — LLM for answer generation
- **React** — Frontend UI
- **pytest** — Automated testing
- **GitHub Actions** — CI/CD pipeline

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload | Upload and process PDF |
| POST | /query | Ask a question |
| POST | /reset | Clear vector store |

## Features

- ✅ Semantic search — finds content by meaning not keywords
- ✅ Persistent vector store — builds once, reloads instantly
- ✅ Context-grounded answers — LLM answers only from document content
- ✅ Handles large documents (tested with 475 page book)
- ✅ Fast queries — embeddings loaded at startup
- ✅ Automated tests with pytest
- ✅ CI/CD via GitHub Actions

## Running Locally

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install: `pip install -r requirements.txt`
5. Add `.env` with `GROQ_API_KEY=your_key`
6. Run: `python main.py`
7. API available at `http://127.0.0.1:8000`

## Environment Variables

```
GROQ_API_KEY=your_groq_api_key
```