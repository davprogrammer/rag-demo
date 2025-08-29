
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
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    parts = []
    for d, m in zip(docs, metas):
        src = m.get("source", "?")
        chk = m.get("chunk", "?")
        parts.append(f"[Quelle: {src}#{chk}]\n{d}\n")
    context = "\n---\n".join(parts)
    return (
        f"Frage: {query}\n\n"
        f"Kontext (verwende ausschließlich diese Snippets):\n{context}\n\n"
        "Antworte kurz und präzise auf Deutsch. Wenn unklar, sage 'Ich weiß es nicht'. "
        "Am Ende eine Quellenliste im Format: Quelle: <Datei>#<Chunk>."
    )
