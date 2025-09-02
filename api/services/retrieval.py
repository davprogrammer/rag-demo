# api/services/retrieval.py
from typing import Tuple, List, Dict, Any
from .ollama_client import OllamaClient
from .qdrant_client import QdrantStore
from .config import settings

# Default-Werte, falls nicht in .env/config gesetzt
_DEF_TOPK = getattr(settings, "RAG_RETURN", 5)
_DEF_MAX_CTX_CHARS = int(getattr(settings, "RAG_MAX_CTX_CHARS", 1800))

def _build_context(hits: List[Dict[str, Any]], max_chars: int = _DEF_MAX_CTX_CHARS) -> str:
    """
    Baut einen knappen Kontextstring aus den Top-Hits, begrenzt auf max_chars.
    Format: --- Quelle (Abschnitt) ---\nText\n\n...
    """
    parts: List[str] = []
    used = 0
    for h in hits:
        src = h.get("source", "")
        sec = h.get("section", "")
        txt = h.get("text", "") or ""
        header = f"--- {src} {('(' + sec + ')') if sec else ''} ---".strip()
        block = f"{header}\n{txt}\n"
        # simple Budgetierung
        if used + len(block) > max_chars:
            # ggf. noch ein Reststück mitnehmen
            remaining = max_chars - used
            if remaining > 0:
                parts.append(block[:remaining])
                used += remaining
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts).strip()

def retrieve(
    question: str,
    top_k: int | None = None,
    max_ctx_chars: int | None = None,
    ollama: OllamaClient | None = None,
    store: QdrantStore | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Einfaches Retrieval:
      1) Frage einbetten
      2) Qdrant-Search (Top-k)
      3) Kontextstring bauen (char-basiert begrenzt)
    Rückgabe: (context_str, hits)
    """
    k = top_k or _DEF_TOPK
    budget = max_ctx_chars or _DEF_MAX_CTX_CHARS

    ollama = ollama or OllamaClient()
    store = store or QdrantStore()

    q_vec = ollama.embed(question)
    hits = store.search(q_vec, top_k=k)
    context = _build_context(hits, max_chars=budget)
    return context, hits
