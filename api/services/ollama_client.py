import httpx
from .config import settings

class OllamaClient:
    def __init__(self, timeout: float = 60.0):
        self.base_url = settings.OLLAMA_URL
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
