import chromadb
from . import config

def get_client():
    # tenant/database hier direkt an HttpClient, NICHT in Settings
    base_url = f"http://{config.CHROMA_HOST}:{config.CHROMA_PORT}"
    return chromadb.HttpClient(
        host=base_url,
        tenant=config.TENANT,
        database=config.DATABASE,
    )

def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
