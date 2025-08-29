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
    Erstelle einen HttpClient OHNE tenant/database für Single‑Tenant-Betrieb.
    Das vermeidet die v1-tenant-Validierung und ist langfristig sinnvoll,
    wenn du keine Multi‑Tenant-Funktionalität benötigst.
    """
    base_url = f"http://{config.CHROMA_HOST}:{config.CHROMA_PORT}"
    try:
        # kein tenant/database übergeben -> einfache Single‑Tenant-Nutzung
        return chromadb.HttpClient(host=base_url)
    except Exception as e:
        chroma_ver = getattr(chromadb, "__version__", "<unknown>")
        server_probe = _probe_server(base_url)
        raise RuntimeError(
            "Fehler beim Verbinden zum Chroma-Server (Single‑Tenant-Modus).\n"
            f"chromadb-Paket: {chroma_ver}\n"
            f"Server-Probe: {server_probe}\n"
            f"Originalfehler: {e}\n\n"
            "Wenn du Multi‑Tenant brauchst, lege den Tenant über die Chroma v2 API an oder gleiche Client/Server-Versionen ab."
        ) from e

def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
# ...existing code...