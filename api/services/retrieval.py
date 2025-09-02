from .ollama_client import OllamaClient
from .qdrant_client import QdrantStore

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Einfache Similarity-Suche.
    Rückgabe: Liste Dicts: id, score, text, source, section
    """
    query = (query or "").strip()
    if not query:
        return []
    vec = OllamaClient().embed(query)
    store = QdrantStore()
    return store.search(vec, top_k=top_k)

if __name__ == "__main__":
    import sys, json
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        print("Usage: python -m services.retrieval 'Frage hier'")
        raise SystemExit(1)
    res = retrieve(q, top_k=5)
    print(json.dumps(res, ensure_ascii=False, indent=2))