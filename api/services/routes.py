import time, uuid, json, logging
from fastapi import APIRouter, Body, Header, HTTPException
from fastapi.responses import StreamingResponse
from .config import settings
from .ollama_client import OllamaClient
from .qdrant_client import QdrantClient
from httpx import Client
from typing import List
from . import retrieval
router = APIRouter()

@router.get("/healthz")
def healthz():
    try:
        ok_ollama = OllamaClient().ping()
    except Exception:
        ok_ollama = False
    return {"ollama": ok_ollama}

@router.get("/v1/models")
#Es ist nur ein Modell aktiv, Endpunkt "faken"
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": settings.MODEL,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }]
    }

def _build_prompt(user_msg: str) -> str:
    return (
        "System: Du bist ein hilfreicher Assistent. Antworte auf Deutsch, "
        "kurz und präzise. Wenn unklar oder du kein Kontext hast, sage das ehrlich! \n\n"
        f"Frage: {user_msg}\n Antwort:"
    )
def _build_rag_prompt(question: str, context: str) -> str:
    return (
        "System: Du bist ein hilfreicher Assistent. Antworte auf Deutsch NUR mit Hilfe des bereitgestellten Kontextes. "
        "Wenn die Information nicht klar im Kontext steht, antworte exakt: 'Keine ausreichenden Informationen.'\n\n"
        f"Kontext:\n{context}\n\nFrage: {question}\nAntwort:"
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
    model_name = settings.MODEL
    t0 = time.time()

    if not stream_flag:

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

    options = {"temperature": settings.TEMPERATURE, "num_ctx": settings.NUM_CTX, "num_predict": settings.MAX_TOKENS}

    base_url = settings.OLLAMA_URL
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
@router.post("/v1/rag")
def rag_answer(
    payload: dict = Body(...),
    authorization: str | None = Header(None)
):
    if not authorization or authorization.split()[-1] != settings.AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="invalid API key")

    question = (payload.get("query") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="missing 'query'")

    top_k = int(payload.get("top_k", 5))
    max_ctx_chars = int(payload.get("max_ctx_chars", 4000))

    # Embedding der Frage
    client = OllamaClient()
    q_vec = client.embed(question)

    # Similarity Search
    store = QdrantStore()
    hits = store.search(q_vec, top_k=top_k)  # raw Qdrant Objekte

    # In handliche Dicts wandeln
    docs = []
    for h in hits:
        payload = h.payload or {}
        docs.append({
            "id": h.id,
            "score": h.score,
            "text": payload.get("text", ""),
            "source": payload.get("source"),
            "section": payload.get("section"),
        })

    if not docs:
        return {
            "query": question,
            "model": settings.MODEL,
            "answer": "Keine ausreichenden Informationen.",
            "chunks": [],
            "used_chunks": 0
        }

    # Kontext zusammenbauen (bis max_ctx_chars)
    parts = []
    used = 0
    for d in docs:
        t = d["text"].strip()
        if used >= max_ctx_chars:
            break
        room = max_ctx_chars - used
        if len(t) > room:
            t = t[:room] + "..."
        parts.append(f"[{d.get('source')} {d.get('section')}] {t}")
        used += len(t)
    context = "\n\n".join(parts)

    prompt = _build_rag_prompt(question, context)
    answer = client.generate(prompt).strip()

    return {
        "query": question,
        "model": settings.MODEL,
        "answer": answer,
        "used_chunks": len(parts),
        "chunks": [
            {
                "source": d.get("source"),
                "section": d.get("section"),
                "score": d["score"],
                "preview": (d["text"][:160] + "...") if len(d["text"]) > 160 else d["text"]
            } for d in docs
        ]
    }