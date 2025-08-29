import hashlib
import json
import time
from typing import Dict, Any, Optional

class SimpleCache:
    """Einfacher In-Memory Cache für RAG Responses mit TTL."""
    
    def __init__(self, ttl_seconds: int = 300):  # 5 Minuten default TTL
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds
    
    def _get_key(self, query: str, top_k: int) -> str:
        """Erstellt Cache-Key aus Query und Parametern."""
        data = {"q": query.lower().strip(), "k": top_k}
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def get(self, query: str, top_k: int) -> Optional[Dict[str, Any]]:
        """Holt Antwort aus Cache falls vorhanden und nicht abgelaufen."""
        key = self._get_key(query, top_k)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["data"]
            else:
                del self.cache[key]
        return None
    
    def set(self, query: str, top_k: int, response: Dict[str, Any]) -> None:
        """Speichert Antwort im Cache."""
        key = self._get_key(query, top_k)
        self.cache[key] = {
            "data": response,
            "timestamp": time.time()
        }
    
    def clear(self) -> None:
        """Leert den kompletten Cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Anzahl gecachter Einträge."""
        return len(self.cache)
    
    def cleanup_expired(self) -> int:
        """Entfernt abgelaufene Einträge, gibt Anzahl zurück."""
        now = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now - entry["timestamp"] >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)

# Global cache instance
response_cache = SimpleCache(ttl_seconds=600)  # 10 Minuten
