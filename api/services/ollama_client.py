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
        """
        Fallback bei 404 auf /api/chat zurück.
        """
        options = {"temperature": settings.TEMPERATURE, "num_ctx": settings.NUM_CTX}

        # 1) Versuch: /api/generate
        payload_gen = {"model": settings.MODEL_NAME, "prompt": prompt, "stream": False, "options": options}
        r = self._client.post("/api/generate", json=payload_gen)
        if r.status_code == 404:
            
        # 2) Fallback: /api/chat
            payload_chat = {
                "model": settings.MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": options,
            }
            r = self._client.post("/api/chat", json=payload_chat)

        r.raise_for_status()
        data = r.json()
        # /api/generate => {"response": "..."}
        if isinstance(data, dict) and "response" in data:
            return data["response"]
        # /api/chat => {"message": {"content": "..."}}
        msg = data.get("message", {})
        return msg.get("content", "")




