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
        payload = {
            "model": settings.MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.TEMPERATURE,
                "num_ctx": settings.NUM_CTX
            },
        }
        r = self._client.post("/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")



