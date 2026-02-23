STEP 1 — Create Project Folder
mkdir rag_project
cd rag_project

STEP 2 — Create Virtual Environment
python -m venv env
env\Scripts\activate

STEP 3 — Create requirements.txt
fastapi
uvicorn
faiss-cpu
pypdf
numpy
requests
python-multipart

Install:
pip install -r requirements.txt

STEP 4 — Install Ollama Models

Install Ollama first from official website.

Then:

ollama pull tinyllama
ollama pull nomic-embed-text

STEP 5 — Create main.py

Your final clean version should look like this:
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

# ---------- PDF Loader ----------
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# ---------- Chunking ----------
def chunk_text(text, chunk_size=400, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ---------- Embedding ----------
def create_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return response.json()["embedding"]

# ---------- FAISS ----------
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

# ---------- Upload ----------
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global documents, index

    text = load_pdf(file.file)
    chunks = chunk_text(text)

    for chunk in chunks:
        documents.append({
            "text": chunk,
            "source": file.filename
        })

    vectors = [create_embedding(doc["text"]) for doc in documents]
    index = create_faiss_index(vectors)

    save_documents(documents)
    save_index(index)

    return {"message": "File uploaded successfully"}

# ---------- Ask ----------
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
Answer only from the context.

Context:
{context}

Question:
{question}
""",
            "stream": False
        }
    )

    return {
        "answer": response.json()["response"],
        "sources": sources
    }
▶️ RUN PROJECT
uvicorn main:app --reload

Open:

http://127.0.0.1:8000/docs

Upload PDF → Ask questions.