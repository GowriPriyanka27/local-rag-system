import json
import os
import faiss
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from pypdf import PdfReader

app = FastAPI()

documents = []
index = None

# -------------------------
# Extract text from PDF
# -------------------------
def load_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text


# -------------------------
# Split text into chunks
# -------------------------
def chunk_text(text, chunk_size=400, overlap=100):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # move back for overlap

    return chunks


# -------------------------
# Create embedding (Ollama)
# -------------------------
def create_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )

    if response.status_code != 200:
        raise Exception("Embedding failed")

    return response.json()["embedding"]


# -------------------------
# Create FAISS index
# -------------------------
def create_faiss_index(vectors):
    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors).astype("float32"))
    return index

def save_index(index):
    faiss.write_index(index, "faiss_index.bin")

def save_documents(docs):
    with open("documents.json", "w", encoding="utf-8") as f:
        json.dump(docs, f)

def load_documents():
    if os.path.exists("documents.json"):
        with open("documents.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def load_index():
    if os.path.exists("faiss_index.bin"):
        return faiss.read_index("faiss_index.bin")
    return None

documents = load_documents()
index = load_index()
# -------------------------
# Upload Endpoint
# -------------------------
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global documents, index

    text = load_pdf(file.file)

    if not text.strip():
        return {"message": "Empty PDF"}

    chunks = chunk_text(text)

    for chunk in chunks:
        documents.append({
            "text": chunk,
            "source": file.filename
        })

    print("Creating embeddings...")

    vectors = [create_embedding(doc["text"]) for doc in documents]

    index = create_faiss_index(vectors)

    save_documents(documents)
    save_index(index)

    return {"message": "File uploaded successfully"}
# -------------------------
# Ask Endpoint
# -------------------------
@app.post("/ask/")
async def ask_question(question: str):
    global documents, index

    if index is None:
        raise HTTPException(status_code=400, detail="Upload a PDF first")

    question_vector = np.array(
        [create_embedding(question)]
    ).astype("float32")

    D, I = index.search(question_vector, k=3)

    retrieved_docs = [documents[i] for i in I[0]]

    context = " ".join([doc["text"] for doc in retrieved_docs])
    sources = list(set([doc["source"] for doc in retrieved_docs]))

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "tinyllama",
            "prompt": f"""
You are a helpful assistant.
Answer ONLY from the context below.

Context:
{context}

Question:
{question}
""",
            "stream": False
        }
    )

    if response.status_code != 200:
        raise Exception("LLM generation failed")

    return {
    "answer": response.json()["response"],
    "sources": sources
    }
@app.get("/stats/")
def get_stats():
    global documents

    if not documents:
        return {"message": "No documents uploaded"}

    sources = [doc["source"] for doc in documents]
    unique_sources = list(set(sources))

    return {
        "total_chunks": len(documents),
        "total_pdfs": len(unique_sources),
        "pdf_names": unique_sources
    }