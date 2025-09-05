import os

#Helpers
def _get_int(key: str, default: str) -> int:
    v = os.getenv(key, default)
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"Env {key} expected int, got '{v}'")


def _get_float(key: str, default: str) -> float:
    v = os.getenv(key, default)
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"Env {key} expected float, got '{v}'")


class Settings():
    # Auth / Security
    AUTH_TOKEN: str = os.getenv("AUTH_TOKEN", "demo-key")

    # Modelle
    MODEL: str = os.getenv("MODEL", "llama3.2")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

    # Retrieval / RAG
    RAG_TOPK: int = _get_int("RAG_TOPK", "12")
    RAG_RETURN: int = _get_int("RAG_RETURN", "5")
    RAG_MAX_CTX_TOKENS: int = _get_int("RAG_MAX_CTX_TOKENS", "1500")

    # LLM Optionen
    NUM_CTX: int = _get_int("NUM_CTX", "3072")
    TEMPERATURE: float = _get_float("TEMPERATURE", "0.2")
    MAX_TOKENS: int = _get_int("MAX_TOKENS", "160")

    # Chunking / Embedding Pipeline
    MAX_TOKENS_PER_CHUNK: int = _get_int("MAX_TOKENS_PER_CHUNK", "350")
    OVERLAP_TOKENS: int = _get_int("OVERLAP_TOKENS", "40")
    MIN_CHUNK_CHARS: int = _get_int("MIN_CHUNK_CHARS", "40")
    EMBED_DIM: int = _get_int("EMBED_DIM", "768")

    # Services
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://ollama:11434")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "docs")

    # Ollama Runtime
    OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "12h")
    OLLAMA_NUM_PARALLEL: int = _get_int("OLLAMA_NUM_PARALLEL", "1")
    OLLAMA_MAX_LOADED_MODELS: int = _get_int("OLLAMA_MAX_LOADED_MODELS", "2")


settings = Settings()