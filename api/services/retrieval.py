
from . import config
from .chroma_client import get_collection
from .ollama_client import embed as embed_one
import time
import logging

logger = logging.getLogger(__name__)

def retrieve(query: str, k: int | None = None):
    start_time = time.time()
    k = k or config.TOP_K
    logger.info(f"Starting retrieval for query, k={k}")
    
    # Embedding-Phase
    embed_start = time.time()
    qvec = embed_one(query)
    embed_time = time.time() - embed_start
    logger.info(f"Query embedding took: {embed_time:.2f}s")
    
    # Chroma-Abfrage
    chroma_start = time.time()
    coll = get_collection()
    results = coll.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    chroma_time = time.time() - chroma_start
    logger.info(f"Chroma query took: {chroma_time:.2f}s")
    
    total_time = time.time() - start_time
    doc_count = len(results.get("documents", [[]])[0]) if results else 0
    logger.info(f"Retrieval completed in {total_time:.2f}s, found {doc_count} documents")
    
    return results

def build_prompt(query: str, results) -> str:
    max_docs = config.CONTEXT_DOCS
    max_len = config.MAX_CHUNK_CHARS
    raw_docs = results.get("documents", [[]])[0][:max_docs]
    
    # Minimaler Kontext - nur Text, keine Metadaten
    docs_text = []
    for d in raw_docs:
        d_trim = d[:max_len] if len(d) > max_len else d
        docs_text.append(d_trim)
    
    context = " ".join(docs_text)
    
    # Klarerer Prompt mit Kontext-Abgrenzung
    if context.strip():
        return f"Kontext:\n{context}\n\nFrage: {query}\n\nBitte beantworte die Frage basierend auf dem obigen Kontext:"
    else:
        return f"Frage: {query}\n\nKein relevanter Kontext verfügbar."
