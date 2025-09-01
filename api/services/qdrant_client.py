from qdrant_client import QdrantClient
from qdrant_client.http import models
from .config import settings

class QdrantStore:
    def __init__(self, timeout: float = 30.0):
        self.url = settings.QDRANT_URL
        self.collection = settings.QDRANT_COLLECTION
        self.client = QdrantClient(url=self.url, timeout=timeout)

    def ping(self) -> bool:
        self.client.get_collections()
        return True

    def upsert(self, vectors: list[list[float]], payloads: list[dict], ids: list[str] | None = None):
        assert len(vectors) == len(payloads), "vectors/payloads length mismatch"
        if ids is not None:
            assert len(ids) == len(vectors), "ids length mismatch"
        self.client.upsert(
            collection_name=self.collection,
            points=models.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )

    def search(self, query_vec: list[float]):
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=settings.RAG_TOPK,
            with_payload=True,
            with_vectors=False,
        )
        hits = []
        for p in res:
            hits.append({
                "id": str(p.id),
                "score": p.score,
                "text": (p.payload or {}).get("text", ""),
                "source": (p.payload or {}).get("source", ""),
                "section": (p.payload or {}).get("section", ""),
            })
        return hits
