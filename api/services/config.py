from pydantic import BaseModel
import os

class Settings(BaseModel):
    AUTH_TOKEN: str = os.getenv("AUTH_TOKEN", "demo-key")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llama3.2")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "bge-m3")

    RAG_TOPK: int = int(os.getenv("RAG_TOPK", "12"))
    RAG_RETURN: int = int(os.getenv("RAG_RETURN", "5"))
    RAG_MMR_LAMBDA: float = float(os.getenv("RAG_MMR_LAMBDA", "0.5"))
    RAG_MAX_CTX_TOKENS: int = int(os.getenv("RAG_MAX_CTX_TOKENS", "1500"))
    NUM_CTX: int = int(os.getenv("NUM_CTX", "3072"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))

    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://ollama:11434")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "docs")

settings = Settings()
