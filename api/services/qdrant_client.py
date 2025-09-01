from qdrant_client import QdrantClient
from qdrant_client.http import models
from .config import settings

class QdrantStore:
    def __init__(self, url: str | None = None, collection: str | None = None, timeout: float = 30.0):
        self.url = url or settings.QDRANT_URL
        self.collection = collection or settings.QDRANT_COLLECTION
        self.client = QdrantClient(url=self.url, timeout=timeout)

    def ping(self) -> bool:
        self.client.get_collections()
        return True

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

    def ensure_or_migrate(self, vector_size: int, distance=models.Distance.COSINE) -> str:
        """
        - Existiert die Collection nicht -> anlegen.
        - Existiert mit gleicher Dimension/Distanz -> weiterverwenden.
        - Existiert, aber Dimension/Distanz weicht ab -> NEUE Collection mit Suffix anlegen und auf die umschalten.
        Gibt den tatsächlich verwendeten Collection-Namen zurück und setzt self.collection entsprechend.
        """
        base = self.collection

        if not self._collection_exists(base):
            print(f"[qdrant] collection '{base}' nicht gefunden -> anlegen (dim={vector_size}, dist={distance}).")
            self._create_collection(base, vector_size, distance)
            return base

        # existiert -> prüfen
        info = self._get_collection_info(base)
        exist_size = info.config.params.vectors.size
        exist_dist = info.config.params.vectors.distance

        if exist_size == vector_size and exist_dist == distance:
            print(f"[qdrant] collection '{base}' vorhanden und kompatibel (dim={exist_size}, dist={exist_dist}).")
            return base

        # Dim/Dist-Mismatch -> neue Version anlegen
        new_name = f"{base}_d{vector_size}"
        if not self._collection_exists(new_name):
            print(
                f"[qdrant] WARN: Collection '{base}' dim/dist mismatch "
                f"(exist={exist_size}/{exist_dist}, want={vector_size}/{distance}).\n"
                f"[qdrant] -> Lege neue Collection '{new_name}' an (alte bleibt unberührt)."
            )
            self._create_collection(new_name, vector_size, distance)
        else:
            print(f"[qdrant] benutze bereits existierende kompatible Collection '{new_name}'.")
        self.collection = new_name
        return new_name

    def upsert(self, vectors: list[list[float]], payloads: list[dict], ids: list[str] | None = None):
        assert len(vectors) == len(payloads), "vectors/payloads length mismatch"
        if ids is not None:
            assert len(ids) == len(vectors), "ids length mismatch"
        self.client.upsert(
            collection_name=self.collection,
            points=models.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )

    def search(self, query_vec: list[float], top_k: int = 5):
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )
        return [{
            "id": str(p.id),
            "score": p.score,
            "text": (p.payload or {}).get("text", ""),
            "source": (p.payload or {}).get("source", ""),
            "section": (p.payload or {}).get("section", ""),
            "vector": p.vector,
        } for p in res]
