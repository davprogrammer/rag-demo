import chromadb
from . import config

def get_client():
    
    return chromadb.HttpClient(
        host=config.CHROMA_HOST,
        port=config.CHROMA_PORT,
    )

def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
