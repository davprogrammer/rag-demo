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

def _model_present(model: str) -> bool:
    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=10)
        if r.status_code != 200:
            return False
        js = r.json()
        mods = js.get("models") if isinstance(js, dict) else None
        if isinstance(mods, list):
            names = {m.get("name") for m in mods if isinstance(m, dict)}
            return model in names or any(str(model).startswith(n or "") for n in names)
    except Exception:
        return False
    return False

def _pull_model(model: str) -> None:
    # Trigger Model Pull (non-streaming best-effort)
    try:
        requests.post(f"{config.OLLAMA_URL}/api/pull", json={"model": model}, timeout=300)
    except Exception:
        pass

def embed(text: str) -> List[float]:
    """Holt ein Embedding; versucht einmal Pull+Retry; liefert Fehler mit Diagnose."""
    model = config.EMBED_MODEL
    # Ollama nutzt für Embeddings das Feld "prompt" (nicht "input"). Wir senden primär prompt, fallback testweise input.
    payload_prompt = {"model": model, "prompt": text}
    payload_input  = {"model": model, "input": text}

    def _call() -> tuple[List[float] | None, Any]:
        # Erst mit "prompt"
        r = requests.post(f"{config.OLLAMA_URL}/api/embeddings", json=payload_prompt, timeout=120)
        status = r.status_code
        body: Any = None
        try:
            body = r.json()
        except Exception:
            body = r.text[:500]
        if status < 400:
            emb = _extract_embedding(body)
            if emb:
                return emb, body
        # Fallback mit "input" falls erste Form leer / Fehler
        r2 = requests.post(f"{config.OLLAMA_URL}/api/embeddings", json=payload_input, timeout=120)
        status2 = r2.status_code
        body2: Any = None
        try:
            body2 = r2.json()
        except Exception:
            body2 = r2.text[:500]
        if status2 >= 400:
            return None, body2
        return _extract_embedding(body2), body2

    emb, raw = _call()
    if not emb:
        # Falls Modell nicht vorhanden: Pull + Retry
        if not _model_present(model):
            _pull_model(model)
            emb, raw = _call()
    if not emb:
        raise RuntimeError(
            "Ollama lieferte kein Embedding. "
            f"Model={model} InputLen={len(text)} ResponseSnippet={str(raw)[:300]}"
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