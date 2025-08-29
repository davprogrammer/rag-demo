
from . import config
from .chroma_client import get_collection
from .ollama_client import embed as embed_one

def retrieve(query: str, k: int | None = None):
    k = k or config.TOP_K
    qvec = embed_one(query)
    coll = get_collection()
    return coll.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

def build_prompt(query: str, results) -> str:
    max_docs = config.CONTEXT_DOCS
    max_len = config.MAX_CHUNK_CHARS
    raw_docs = results.get("documents", [[]])[0][:max_docs]
    raw_metas = results.get("metadatas", [[]])[0][:max_docs]
    parts = []
    for d, m in zip(raw_docs, raw_metas):
        d_trim = (d[:max_len] + "…") if len(d) > max_len else d
        src = m.get("source", "?")
        chk = m.get("chunk", "?")
        parts.append(f"[Quelle: {src}#{chk}]\n{d_trim}\n")
    context = "\n---\n".join(parts)
    return (
        f"Frage: {query}\n\n"
        f"Kontext (nur diese Ausschnitte):\n{context}\n\n"
        "Antworte knapp auf Deutsch. Wenn nicht beantwortbar: 'Ich weiß es nicht'. "
        "Ende mit 'Quellen:' Liste Datei#Chunk."
    )
