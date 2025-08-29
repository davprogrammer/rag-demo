import os, glob, time
from typing import List
from PyPDF2 import PdfReader
from . import config
from .chroma_client import get_collection
from .ollama_client import embed as embed_one

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    texts = [(p.extract_text() or "").strip() for p in reader.pages]
    return "\n\n".join(texts).strip()

def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, step = [], size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start+size]
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks

def ingest():
    t0 = time.time()
    coll = get_collection()
    pdfs = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.pdf")))
    if not pdfs:
        print(f"[ingest] Keine PDFs in {config.DATA_DIR}")
        return
    docs, total = 0, 0
    for path in pdfs:
        doc = os.path.basename(path)
        print(f"[ingest] {doc} …")
        chunks = chunk_text(read_pdf_text(path), 1000, 200)
        if not chunks:
            print(f"[warn] Keine Textinhalte in {doc}")
            continue
        ids = [f"{doc}::chunk{idx}" for idx in range(len(chunks))]
        metas = [{"source": doc, "chunk": idx} for idx in range(len(chunks))]
        vectors = []
        B = 12
        for i in range(0, len(chunks), B):
            vectors.extend([embed_one(t) for t in chunks[i:i+B]])
        coll.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=vectors)
        docs += 1; total += len(chunks)
        print(f"[ingest] {doc}: {len(chunks)} Chunks")
    print(f"[ingest] Fertig: {docs} Dokument(e), {total} Chunks in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    ingest()