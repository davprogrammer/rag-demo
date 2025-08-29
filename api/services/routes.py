from fastapi import APIRouter
from .chroma_client import get_collection
from pydantic import BaseModel
from .ingest import ingest
from .retrieval import retrieve, build_prompt
from .ollama_client import chat

router = APIRouter()

class QueryIn(BaseModel):
    query: str
    top_k: int | None = None

@router.get("/health")
def health():
    return {"ok": True}

@router.get("/stats")
def stats():
    """Schritt 7: Einfache Statistik über Collection (Dokument & Embedding Count)."""
    try:
        coll = get_collection()
        count = coll.count()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "collection": coll.name, "count": count}

@router.post("/ingest/run")
def run_ingest():
    ingest()
    return {"ok": True}

@router.post("/query")
def query(inp: QueryIn):
    res = retrieve(inp.query, inp.top_k)
    docs = res.get("documents", [[]])[0]
    if not docs:
        return {"answer": "Ich weiß es nicht.", "sources": []}
    prompt = build_prompt(inp.query, res)
    answer = chat(None, prompt)
    sources = [
        {"source": m.get("source"), "chunk": m.get("chunk"), "score": float(s)}
        for m, s in zip(res["metadatas"][0], res["distances"][0])
    ]
    return {"answer": answer, "sources": sources}