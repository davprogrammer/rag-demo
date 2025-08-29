import httpx
from .config import settings

class OllamaClient:
    def __init__(self, timeout: float = 60.0):
        self.base_url = settings.OLLAMA_URL
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def ping(self) -> bool:
        r = self._client.get("/api/tags")
        r.raise_for_status()
        return True
    
    def embed(self, text: str) -> list[float]:
        payload = {"model": settings.EMBED_MODEL, "prompt": text}
        r = self._client.post("/api/embeddings", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["embedding"]
    
    def generate(self, prompt: str) -> str:

        options = {"temperature": settings.TEMPERATURE, "num_ctx": settings.NUM_CTX}

        payload = {
            "model": settings.MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        r = self._client.post("/api/generate", json=payload)

        if r.status_code == 404:
            # Modelle vom Server auflisten für Debug
            try:
                tags = self._client.get("/api/tags").json()
                available = [m.get("name") for m in tags.get("models", [])]
            except Exception:
                available = []
            raise RuntimeError(
                f"Model '{settings.MODEL_NAME}' not found. "
                f"Available models: {available}"
            )

        r.raise_for_status()
        data = r.json()

        # /api/generate liefert {"response": "..."}
        if isinstance(data, dict) and "response" in data:
            return data["response"]

        # Fallback: falls Format anders ist
        msg = data.get("message", {})
        return msg.get("content", "")





