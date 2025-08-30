import time, uuid, json, logging
from fastapi import APIRouter, Body, Header, HTTPException
from fastapi.responses import StreamingResponse
from .config import settings
from .ollama_client import OllamaClient
from httpx import Client

logger = logging.getLogger(__name__)

router = APIRouter()

def _current_model() -> str:
    return getattr(settings, "MODEL", None) or getattr(settings, "MODEL_NAME", "unknown-model")

@router.get("/healthz")
def healthz():
    try:
        ok_ollama = OllamaClient().ping()
    except Exception:
        ok_ollama = False
    return {"ollama": ok_ollama}

@router.get("/v1/models")
def list_models():
    m = _current_model()
    return {
        "object": "list",
        "data": [{
            "id": m,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }]
    }

def _build_prompt(user_msg: str) -> str:
    return (
        "System: Du bist ein hilfreicher Assistent. Antworte auf Deutsch, "
        "kurz und präzise. Wenn unklar oder kein Kontext: Sage, dass dir Informationen fehlen.\n\n"
        f"Frage: {user_msg}\nAntwort:"
    )

@router.post("/v1/chat/completions")
def chat_completions(
    payload: dict = Body(...),
    authorization: str | None = Header(None)
):
    if not authorization or authorization.split()[-1] != settings.AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="invalid API key")

    messages = payload.get("messages", [])
    stream_flag = payload.get("stream", False) is True

    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break
    if not user_msg:
        raise HTTPException(status_code=400, detail="no user message found")

    prompt = _build_prompt(user_msg)
    client = OllamaClient()
    model_name = _current_model()
    t0 = time.time()

    if not stream_flag:
        logger.info("Non-stream chat start model=%s prompt_len=%d", model_name, len(prompt))
        text = client.generate(prompt).strip()
        latency = round(time.time() - t0, 3)
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {},
            "latency_sec": latency
        }

    logger.info("Stream chat start model=%s prompt_len=%d", model_name, len(prompt))
    return StreamingResponse(
        event_stream(prompt, client, model_name),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

def event_stream(prompt: str, client: OllamaClient, model_name: str):
    start_id = f"chatcmpl-{uuid.uuid4()}"
    start = {
        "id": start_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(start)}\n\n"

    options = {}
    if hasattr(settings, "TEMPERATURE"):
        options["temperature"] = settings.TEMPERATURE
    if hasattr(settings, "NUM_CTX"):
        options["num_ctx"] = settings.NUM_CTX
    if getattr(settings, "MAX_TOKENS", None):
        options["num_predict"] = settings.MAX_TOKENS

    base_url = getattr(client, "base_url", "http://ollama:11434")
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": options
    }

    try:
        with Client(base_url=base_url, timeout=None) as s:
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
                        "id": start_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": None
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        err = {
            "id": start_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n[Fehler: {type(e).__name__}: {e}]"},
                "finish_reason": None
            }],
        }
        yield f"data: {json.dumps(err)}\n\n"

    end = {
        "id": start_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
    }
    yield f"data: {json.dumps(end)}\n\n"
    yield "data: [DONE]\n\n"