import requests
from typing import Any, List
from . import config
import time
import logging

logger = logging.getLogger(__name__)

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
    start_time = time.time()
    model = config.EMBED_MODEL
    logger.info(f"Starting embedding for {len(text)} chars with model {model}")
    
    # Ollama nutzt für Embeddings das Feld "prompt" (nicht "input"). Wir senden primär prompt, fallback testweise input.
    payload_prompt = {"model": model, "prompt": text}
    payload_input  = {"model": model, "input": text}

    def _call() -> tuple[List[float] | None, Any]:
        # Erst mit "prompt"
        call_start = time.time()
        r = requests.post(f"{config.OLLAMA_URL}/api/embeddings", json=payload_prompt, timeout=120)
        call_time = time.time() - call_start
        logger.info(f"Embedding API call took: {call_time:.2f}s, status: {r.status_code}")
        
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
            logger.warning(f"Model {model} not present, attempting pull...")
            pull_start = time.time()
            _pull_model(model)
            pull_time = time.time() - pull_start
            logger.info(f"Model pull took: {pull_time:.2f}s")
            emb, raw = _call()
    
    total_time = time.time() - start_time
    logger.info(f"Total embedding time: {total_time:.2f}s, vector dimension: {len(emb) if emb else 0}")
    
    if not emb:
        raise RuntimeError(
            "Ollama lieferte kein Embedding. "
            f"Model={model} InputLen={len(text)} ResponseSnippet={str(raw)[:300]}"
        )
    return emb

def chat(system: str | None, user: str) -> str:
    start_time = time.time()
    logger.info(f"Starting chat with model {config.MODEL}, prompt length: {len(user)} chars")
    
    # Nur User-Message, kein System-Prompt für maximale Geschwindigkeit
    base_payload = {
        "model": config.MODEL,
        "stream": False,
        "options": {
            "temperature": config.TEMPERATURE,
            "num_predict": config.MAX_TOKENS,
            "num_ctx": config.NUM_CTX,
            "num_thread": 2,  # Weniger Threads
            "repeat_penalty": 1.1,
        },
        "messages": [
            {"role": "user", "content": user},  # Nur User-Message
        ],
    }

    # Kürzere Timeouts für schnellere Diagnose
    timeouts = [15, 45]  # 15s dann 45s statt 30s, 60s
    last_exc = None
    attempt = 0
    
    for to in timeouts:
        attempt += 1
        try:
            attempt_start = time.time()
            logger.info(f"Chat attempt {attempt} with {to}s timeout...")
            
            # Session verwenden für Connection-Reuse
            import requests
            session = requests.Session()
            session.headers.update({'Content-Type': 'application/json'})
            
            r = session.post(f"{config.OLLAMA_URL}/api/chat", json=base_payload, timeout=to)
            attempt_time = time.time() - attempt_start
            logger.info(f"Chat attempt {attempt} completed in {attempt_time:.2f}s, status: {r.status_code}")
            r.raise_for_status()
            j = r.json()
            session.close()
            break
        except Exception as e:
            attempt_time = time.time() - attempt_start
            logger.warning(f"Chat attempt {attempt} failed after {attempt_time:.2f}s: {str(e)[:100]}")
            last_exc = e
            if 'session' in locals():
                session.close()
    else:
        total_time = time.time() - start_time
        logger.error(f"All chat attempts failed after {total_time:.2f}s")
        raise RuntimeError(f"Chat fehlgeschlagen nach Retries: {last_exc}")
    
    total_time = time.time() - start_time
    logger.info(f"Chat completed successfully in {total_time:.2f}s")
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