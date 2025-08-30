import time, uuid, json
from fastapi import APIRouter, Body, Header, HTTPException
from fastapi.responses import StreamingResponse
from .config import settings
from .ollama_client import OllamaClient

router = APIRouter()

@router.get("/healthz")
def healthz():
    # gültiges JSON-Objekt statt {True}
    try:
        ok_ollama = OllamaClient().ping()
    except Exception:
        ok_ollama = False
    return {"ollama": ok_ollama}

# OpenAI: Model-Liste (für OWUI Dropdown)
@router.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": settings.MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }]
    }

def _build_prompt(user_msg: str) -> str:
    # Für den Start: einfacher Prompt (ohne RAG)
    return f"Antworte präzise und kurz auf Deutsch.\n\nFrage: {user_msg}\nAntwort:"

# OpenAI: Chat Completions (Passthrough → Ollama)
@router.post("/v1/chat/completions")
def chat_completions(payload: dict = Body(...), authorization: str | None = Header(None)):
    if not authorization or authorization.split()[-1] != settings.AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="invalid API key")

    messages = payload.get("messages", [])
    stream = bool(payload.get("stream", False))

    # letzte User-Nachricht
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    client = OllamaClient()
    prompt = _build_prompt(user_msg)
    t0 = time.time()

    if not stream:
        text = client.generate(prompt).strip()
        latency = round(time.time() - t0, 3)
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": settings.MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {},
            "latency_sec": latency
        }

    # Streaming (OpenAI-kompatible SSE-Chunks)
    def event_stream():
        start = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": settings.MODEL_NAME,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(start)}\n\n"

        # leichtes Streaming über /api/generate mit stream=true
        # wir nutzen hier eine minimale Inline-Variante
        from httpx import Client
        options = {"temperature": settings.TEMPERATURE, "num_ctx": settings.NUM_CTX}
        if settings.MAX_TOKENS:
            options["num_predict"] = settings.MAX_TOKENS
        with Client(base_url=client.base_url, timeout=None) as s:
            payload = {"model": settings.MODEL_NAME, "prompt": prompt, "stream": True, "options": options}
            with s.stream("POST", "/api/generate", json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("done"):
                        break
                    piece = obj.get("response", "")
                    if not piece:
                        continue
                    chunk = {
                        "id": start["id"],
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": settings.MODEL_NAME,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

        end = {
            "id": start["id"],
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": settings.MODEL_NAME,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
