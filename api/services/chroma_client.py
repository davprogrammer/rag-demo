# api/services/chroma_client.py
import chromadb
from . import config

def get_client():
    # tenant/database hier direkt an HttpClient, NICHT in Settings
    return chromadb.HttpClient(
        host=config.CHROMA_HOST,
        port=config.CHROMA_PORT,
        tenant=config.TENANT,
        database=config.DATABASE,
    )

def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
