# Local Multi-PDF RAG System

## 📌 Overview

This project implements a fully local Retrieval-Augmented Generation (RAG) system using FastAPI, FAISS, and Ollama.

The system allows users to:
- Upload multiple PDF documents
- Generate embeddings using `nomic-embed-text`
- Store vectors in FAISS
- Perform semantic search
- Generate context-aware answers using TinyLlama

Everything runs locally — no OpenAI API, no cloud dependency.

---

## 🏗 Architecture

1. PDF Upload  
2. Text Extraction (PyPDF)  
3. Chunking with overlap  
4. Embedding generation (Ollama: nomic-embed-text)  
5. FAISS vector indexing  
6. Question embedding  
7. Top-k similarity search  
8. Context injection into TinyLlama  
9. Final answer generation with source tracking  

---

## 🚀 Features

- Multi-PDF upload support
- Semantic vector search (FAISS)
- Local LLM inference (TinyLlama)
- Source attribution in responses
- Persistent index storage
- Fully offline execution

---

## 🛠 Tech Stack

- Python
- FastAPI
- FAISS
- Ollama
- TinyLlama
- nomic-embed-text
- PyPDF
- NumPy
- Requests

