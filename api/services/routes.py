from fastapi import APIRouter
from .chroma_client import get_collection
from pydantic import BaseModel
from .ingest import ingest
from .retrieval import retrieve, build_prompt
from .ollama_client import chat
import time
import logging
import os

# Performance-Logging einrichten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class QueryIn(BaseModel):
    query: str
    top_k: int | None = None

# OpenWebUI RAG-Interface
class OpenWebUIQuery(BaseModel):
    query: str
    knowledge_base: str | None = None

@router.get("/health")
def health():
    return {"ok": True}

@router.get("/stats")
def stats():
    """Collection-Statistiken."""
    try:
        coll = get_collection()
        count = coll.count()
        return {
            "ok": True,
            "collection": coll.name,
            "count": count,
            "config": {
                "model": os.getenv("MODEL"),
                "embed_model": os.getenv("EMBED_MODEL"),
                "max_tokens": os.getenv("MAX_TOKENS"),
                "context_docs": os.getenv("CONTEXT_DOCS"),
                "num_ctx": os.getenv("NUM_CTX")
            }
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@router.post("/ingest/run")
def run_ingest():
    ingest()
    return {"ok": True}

@router.post("/query")
def query(inp: QueryIn):
    start_time = time.time()
    performance_log = {}
    
    try:
        # 1. Embedding-Phase messen
        embed_start = time.time()
        logger.info(f"Starting query: '{inp.query[:50]}...'")
        res = retrieve(inp.query, inp.top_k)
        embed_time = time.time() - embed_start
        performance_log["embedding_retrieval_ms"] = round(embed_time * 1000, 2)
        logger.info(f"Embedding + Retrieval took: {embed_time:.2f}s")
        
        # 2. Dokument-Check
        docs = res.get("documents", [[]])[0]
        if not docs:
            total_time = time.time() - start_time
            logger.warning("No documents found for query")
            return {
                "answer": "Ich weiß es nicht.", 
                "sources": [], 
                "performance": {**performance_log, "total_ms": round(total_time * 1000, 2)}
            }
        
        # 3. Prompt-Building messen
        prompt_start = time.time()
        prompt = build_prompt(inp.query, res)
        prompt_time = time.time() - prompt_start
        performance_log["prompt_build_ms"] = round(prompt_time * 1000, 2)
        performance_log["prompt_length"] = len(prompt)
        logger.info(f"Prompt building took: {prompt_time:.3f}s, length: {len(prompt)} chars")
        
        # 4. Chat-Phase detailliert messen
        chat_start = time.time()
        logger.info("Starting chat request...")
        answer = chat(None, prompt)
        chat_time = time.time() - chat_start
        performance_log["chat_ms"] = round(chat_time * 1000, 2)
        logger.info(f"Chat took: {chat_time:.2f}s")
        
        # 5. Sources aufbereiten
        sources_start = time.time()
        sources = [
            {"source": m.get("source"), "chunk": m.get("chunk"), "score": float(s)}
            for m, s in zip(res["metadatas"][0], res["distances"][0])
        ]
        sources_time = time.time() - sources_start
        performance_log["sources_build_ms"] = round(sources_time * 1000, 2)
        
        total_time = time.time() - start_time
        performance_log["total_ms"] = round(total_time * 1000, 2)
        
        logger.info(f"Total query time: {total_time:.2f}s")
        
        return {
            "answer": answer, 
            "sources": sources,
            "performance": performance_log
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Query failed after {total_time:.2f}s: {str(e)}")
        return {
            "error": str(e),
            "performance": {"total_ms": round(total_time * 1000, 2)}
        }


@router.post("/rag")
def openwebui_rag(inp: OpenWebUIQuery):
    """RAG-Endpoint speziell für OpenWebUI Integration."""
    start_time = time.time()
    
    try:
        logger.info(f"OpenWebUI RAG query: '{inp.query[:50]}...'")
        
        # Standard RAG-Pipeline nutzen
        res = retrieve(inp.query, top_k=3)
        docs = res.get("documents", [[]])[0]
        
        if not docs:
            return {
                "response": "Ich weiß es nicht - keine relevanten Informationen gefunden.",
                "sources": [],
                "response_time": round((time.time() - start_time) * 1000, 2)
            }
        
        # Kompakter Prompt für bessere Performance
        prompt = build_prompt(inp.query, res)
        answer = chat(None, prompt)
        
        # Sources für OpenWebUI formatieren
        sources = [
            {
                "name": m.get("source", "unknown"),
                "content": d[:200] + "..." if len(d) > 200 else d,
                "score": float(s)
            }
            for d, m, s in zip(docs, res["metadatas"][0], res["distances"][0])
        ]
        
        total_time = time.time() - start_time
        logger.info(f"OpenWebUI RAG completed in {total_time:.2f}s")
        
        return {
            "response": answer,
            "sources": sources,
            "response_time": round(total_time * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"OpenWebUI RAG error: {str(e)}")
        return {
            "response": f"Fehler bei der Suche: {str(e)[:100]}",
            "sources": [],
            "response_time": round((time.time() - start_time) * 1000, 2)
        }