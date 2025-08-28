# api/ingest.py
import os
import glob
import math
import time
from typing import List, Dict

import requests
from PyPDF2 import PdfReader
import chromadb

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "company-faq")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

# --- Helpers ---
def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t.strip())
    return "\n\n".join(texts).strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple char-based chunking (robust & fast for PoC)."""
    text = " ".join(text.split())  # collapse whitespace
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks

def embed(Texts: List[str]) -> List[List[float]]:
    """Call Ollama embeddings API."""
    url = f"{OLLAMA_URL}/api/embeddings"
    vectors = []
    for t in Texts:
        r = requests.post(url, json={"model": EMBED_MODEL, "input": t}, timeout=60)
        r.raise_for_status()
        vectors.append(r.json()["embedding"])
    return vectors

# --- Main ingest ---
def main():
    t0 = time.time()
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    try:
        coll = client.get_collection(COLLECTION_NAME)
    except Exception:
        coll = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    pdfs = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdfs:
        print(f"[ingest] Keine PDFs in {DATA_DIR} gefunden.")
        return

    doc_count = 0
    chunk_count = 0

    for path in pdfs:
        doc_id = os.path.basename(path)
        print(f"[ingest] {doc_id} einlesen …")
        text = read_pdf_text(path)
        chunks = chunk_text(text, 1000, 200)
        if not chunks:
            print(f"[warn] Keine Textinhalte in {doc_id}")
            continue

        # IDs & Metadaten bauen
        ids = [f"{doc_id}::chunk{idx}" for idx in range(len(chunks))]
        metas = [{"source": doc_id, "chunk": idx} for idx in range(len(chunks))]

        # Embeddings (Batch-weise falls groß)
        batch = 12
        vectors_all = []
        for i in range(0, len(chunks), batch):
            vectors_all.extend(embed(chunks[i : i + batch]))

        # Upsert
        coll.upsert(ids=ids, documents=chunks, embeddings=vectors_all, metadatas=metas)

        doc_count += 1
        chunk_count += len(chunks)
        print(f"[ingest] {doc_id}: {len(chunks)} Chunks upserted.")

    dt = time.time() - t0
    print(f"[ingest] Fertig: {doc_count} Dokument(e), {chunk_count} Chunks in {dt:.1f}s.")

if __name__ == "__main__":
    main()
