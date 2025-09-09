from qdrant_client import QdrantClient
from qdrant_client.http import models
from .config import settings

class QdrantStore:
    def __init__(self, timeout: float = 30.0):
        self.url = settings.QDRANT_URL
        self.collection = settings.QDRANT_COLLECTION
        self.client = QdrantClient(url=self.url, timeout=timeout)
    
    #Helpers
    def _collection_exists(self, name: str) -> bool:
        cols = self.client.get_collections().collections
        return any(c.name == name for c in cols)

    def _get_collection_info(self, name: str):
        return self.client.get_collection(name)

    def _create_collection(self, name: str, vector_size: int, distance=models.Distance.COSINE):
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )

    def check_collection(self, vector_size: int, distance=models.Distance.COSINE) -> str:
        base = self.collection

        if not self._collection_exists(base):
            print(f"[QDRANT] collection '{base}' nicht gefunden -> anlegen (dim={vector_size}, dist={distance}).")
            self._create_collection(base, vector_size, distance)
            return base

        # existiert -> prüfen
        info = self._get_collection_info(base)
        exist_size = info.config.params.vectors.size
        exist_dist = info.config.params.vectors.distance

        if exist_size != vector_size or exist_dist != distance:
            raise ValueError(
                f"[QDRANT] Inkompatible Collection '{base}': "
                f"erwartet (size={vector_size}, dist={distance}) "
                f"vorhanden (size={exist_size}, dist={exist_dist}). Abbruch."
            )

        print(f"[QDRANT] collection '{base}' vorhanden und kompatibel (dim={exist_size}, dist={exist_dist}).")
        return base

    def upsert(self, vectors: list[list[float]], payloads: list[dict], ids: list[str] | None = None):
        assert len(vectors) == len(payloads), "vectors/payloads length mismatch"
        if ids is not None:
            assert len(ids) == len(vectors), "ids length mismatch"
        self.client.upsert(
            collection_name=self.collection,
            points=models.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )

    def search(self, query_vec: list[float], top_k: int = settings.RAG_TOPK):
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
        )
        return [{
            "id": str(p.id),
            "score": p.score,
            "text": (p.payload or {}).get("text", ""),
            "source": (p.payload or {}).get("source", ""),
            "section": (p.payload or {}).get("section", ""),
        } for p in res]
