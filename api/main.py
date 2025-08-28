# api/main.py
import os
from typing import List, Dict, Any
import requests
from pydantic import BaseModel
from fastapi import FastAPI
import chromadb

# --- Env ---
MODEL = os.getenv("MODEL", "llama3.1:8b-instruct-q4_K_M")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "company-faq")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# --- App & DB client ---
app = FastAPI(title="RAG FAQ API")

def get_coll():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_collection(COLLECTION_NAME)

# --- Schemas ---
class QueryIn(BaseModel):
    query: str
    top_k: int | None = None

# --- Utils ---
def embed_query(q: str) -> List[float]:
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "input": q}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]

def generate_answer(system: str, user: str) -> str:
    payload = {
        "model": MODEL,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        "messages": [
            {"role": "system", "content":
                (system or "Du bist ein Unternehmens-FAQ-Assistent. Antworte nur basierend auf dem bereitgestellten Kontext. "
                 "Wenn Information fehlt, sage 'Ich weiß es nicht'. Nenne am Ende die Quellen.")}
            ,
            {"role": "user", "content": user},
        ],
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]

def build_prompt(query: str, hits: Dict[str, Any]) -> str:
    ctx_lines = []
    for doc, meta in zip(hits["documents"][0], hits["metadatas"][0]):
        src = meta.get("source", "unbekannt")
        chunk = meta.get("chunk", "?")
        ctx_lines.append(f"[Quelle: {src}#{chunk}]\n{doc}\n")
    context = "\n---\n".join(ctx_lines)
    return (
        f"Frage: {query}\n\n"
        f"Kontext (aus Snippets, nutze nur diese Info):\n{context}\n\n"
        f"Antworte kurz und präzise auf Deutsch. Wenn unklar, sage 'Ich weiß es nicht'. "
        f"Gib am Ende eine Quellenliste an im Format: Quelle: <Datei>#<Chunk>."
    )

# --- Routes ---
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/query")
def query(inp: QueryIn):
    top_k = inp.top_k or TOP_K
    # 1) Retrieve
    coll = get_coll()
    qvec = embed_query(inp.query)
    results = coll.query(query_embeddings=[qvec], n_results=top_k, include=["documents", "metadatas", "distances"])
    if not results.get("documents") or not results["documents"][0]:
        return {"answer": "Ich weiß es nicht.", "sources": []}

    # 2) Build prompt & generate
    prompt = build_prompt(inp.query, results)
    answer = generate_answer(system=None, user=prompt)

    # 3) Sources
    srcs = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        srcs.append({"source": meta.get("source"), "chunk": meta.get("chunk"), "score": float(dist)})

    return {"answer": answer, "sources": srcs}
