import os

MODEL = os.getenv("MODEL", "llama3.1:8b-instruct-q4_K_M")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "company-faq")
TENANT = os.getenv("CHROMA_TENANT", "default_tenant")
DATABASE = os.getenv("CHROMA_DATABASE", "default_database")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")