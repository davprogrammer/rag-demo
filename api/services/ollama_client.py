import requests
from typing import Any, List
from . import config

def _extract_embedding(j: Any) -> List[float] | None:
    """Versucht verschiedene mögliche Ollama-Embedding-Response-Layouts zu interpretieren."""
    if not isinstance(j, dict):
        return None
    # Direktes Feld
    if isinstance(j.get("embedding"), list) and j["embedding"]:
        return [float(x) for x in j["embedding"]]
    # data[0].embedding
    data = j.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and isinstance(first.get("embedding"), list) and first["embedding"]:
            return [float(x) for x in first["embedding"]]
        if isinstance(first, list) and first:
            return [float(x) for x in first]
    # embeddings[0]
    embs = j.get("embeddings")
    if isinstance(embs, list) and embs and isinstance(embs[0], list) and embs[0]:
        return [float(x) for x in embs[0]]
    return None

def embed(text: str) -> List[float]:
    """Holt ein Embedding; wirft klaren Fehler, wenn keins geliefert wird."""
    r = requests.post(
        f"{config.OLLAMA_URL}/api/embeddings",
        json={"model": config.EMBED_MODEL, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    emb = _extract_embedding(r.json())
    if not emb:
        raise RuntimeError(
            f"Ollama lieferte kein Embedding (Model={config.EMBED_MODEL}, InputLen={len(text)}). "
            "Prüfe Ollama-Logs oder Modellverfügbarkeit."
        )
    return emb

def chat(system: str | None, user: str) -> str:
    payload = {
        "model": config.MODEL,
        "stream": False,
        "options": {"temperature": config.TEMPERATURE, "num_predict": config.MAX_TOKENS},
        "messages": [
            {"role": "system", "content": system or (
                "Du bist ein Unternehmens-FAQ-Assistent. Antworte nur anhand des Kontextes. "
                "Wenn Information fehlt, sage 'Ich weiß es nicht'. Nenne am Ende die Quellen."
            )},
            {"role": "user", "content": user},
        ],
    }
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    j = r.json()
    # robuste Extraktion
    if isinstance(j, dict):
        msg = j.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"].strip()
        if isinstance(j.get("response"), str):
            return j["response"].strip()
        # OpenAI-ähnliche Struktur
        choices = j.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                msg = c0.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"].strip()
                txt = c0.get("text")
                if isinstance(txt, str):
                    return txt.strip()
    raise RuntimeError("Konnte Chat-Antwort nicht extrahieren: unbekanntes Format")