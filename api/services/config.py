import os


class Settings():
    # Auth / Security
    AUTH_TOKEN: str = os.getenv("AUTH_TOKEN", "demo-key")

    # Modelle
    MODEL: str = os.getenv("MODEL", "llama3.2")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

    # Retrieval / RAG (direkt aus Env; Pydantic castet Strings zu int/float)
    RAG_TOPK: int = os.getenv("RAG_TOPK", "12")
    RAG_RETURN: int = os.getenv("RAG_RETURN", "5")
    RAG_MMR_LAMBDA: float = os.getenv("RAG_MMR_LAMBDA", "0.5")
    RAG_MAX_CTX_TOKENS: int = os.getenv("RAG_MAX_CTX_TOKENS", "1500")

    # LLM Optionen
    NUM_CTX: int = os.getenv("NUM_CTX", "3072")
    TEMPERATURE: float = os.getenv("TEMPERATURE", "0.2")
    MAX_TOKENS: int = os.getenv("MAX_TOKENS", "160")

    # Chunking / Embedding Pipeline
    MAX_TOKENS_PER_CHUNK: int = os.getenv("MAX_TOKENS_PER_CHUNK", "350")
    OVERLAP_TOKENS: int = os.getenv("OVERLAP_TOKENS", "40")
    MIN_CHUNK_CHARS: int = os.getenv("MIN_CHUNK_CHARS", "40")
    EMBED_DIM: int = os.getenv("EMBED_DIM", "768")

    # Services
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://ollama:11434")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "docs")

    # Ollama Runtime
    OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "12h")
    OLLAMA_NUM_PARALLEL: int = os.getenv("OLLAMA_NUM_PARALLEL", "1")
    OLLAMA_MAX_LOADED_MODELS: int = os.getenv("OLLAMA_MAX_LOADED_MODELS", "2")

settings = Settings()