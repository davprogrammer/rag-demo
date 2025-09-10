from typing import Tuple, List, Dict, Any
from .ollama_client import OllamaClient
from .qdrant_client import QdrantStore
from .config import settings

def _build_context(hits: List[Dict[str, Any]], max_chars) -> str:
    
    context: List[str] = []
    used = 0
    for h in hits:
        src = h.get("source", "")
        sec = h.get("section", "")
        txt = h.get("text", "") or ""
        header = f"--- {src} {('(' + sec + ')') if sec else ''} ---".strip()
        block = f"{header}\n{txt}\n"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 0:
                context.append(block[:remaining])
                used += remaining
            break
        context.append(block)
        used += len(block)
    return "\n".join(context).strip()

def retrieve(question: str) -> Tuple[str, List[Dict[str, Any]]]:

    top_k = settings.RAG_TOPK
    max_tokens = settings.RAG_MAX_CTX_TOKENS
    ollama = OllamaClient()
    store = QdrantStore()

    q_vec = ollama.embed(question)
    hits = store.search(q_vec, top_k)
    context = _build_context(hits, max_chars=max_tokens)
    return context, hits
