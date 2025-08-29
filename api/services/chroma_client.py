# ...existing code...
import chromadb
from . import config
import requests

def _probe_server(base_url: str) -> str:
    try:
        r = requests.get(base_url + "/api/v2", timeout=3)
        return f"/api/v2 -> {r.status_code}"
    except Exception as e:
        return f"probe failed: {e}"

def get_client():
    """
    Erstelle einen HttpClient KORREKT mit Host/Port getrennt für bessere Performance.
    """
    try:
        # Korrekte Chroma-Client-Initialisierung mit getrennten Host/Port
        return chromadb.HttpClient(
            host=config.CHROMA_HOST,
            port=config.CHROMA_PORT,
            ssl=False,
            headers={"Connection": "keep-alive"}
        )
    except Exception as e:
        chroma_ver = getattr(chromadb, "__version__", "<unknown>")
        base_url = f"http://{config.CHROMA_HOST}:{config.CHROMA_PORT}"
        server_probe = _probe_server(base_url)
        raise RuntimeError(
            "Fehler beim Verbinden zum Chroma-Server.\n"
            f"chromadb-Paket: {chroma_ver}\n"
            f"Server-Probe: {server_probe}\n"
            f"Originalfehler: {e}\n\n"
            "Prüfe Chroma-Server-Verfügbarkeit und Netzwerk-Verbindung."
        ) from e

def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
# ...existing code...