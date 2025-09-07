import time, uuid, json, logging
from fastapi import APIRouter, Body, Header, HTTPException
from fastapi.responses import StreamingResponse
from .config import settings
from .ollama_client import OllamaClient
from .qdrant_client import QdrantClient
from httpx import Client
from .retrieval import retrieve  
from services.logging import Timer


router = APIRouter()

@router.get("/healthz")
def healthz():
    try:
        ok_ollama = OllamaClient().ping()
    except Exception:
        ok_ollama = False
    return {"ollama": ok_ollama}

@router.get("/v1/models")
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

def _build_prompt(question: str, context: str) -> str:
    return (
        "System: Du bist ein hilfreicher Assistent. Antworte auf Deutsch NUR mit Hilfe des bereitgestellten Kontextes."
        f"Kontext:\n{context}\n\nFrage: {question}\nAntwort:"
    )

def _format_context_block(hits: list) -> str:
    
    parts = []
    total = 0
    for h in hits:
        src = h.get("source") or ""
        sec = h.get("section") or ""
        score = h.get("score", 0.0)
        txt = (h.get("text") or "").replace("\n", " ").strip()
        if len(txt)> 200
            txt = txt[:200]
        line_core = f"{src} {sec}".strip()
        line = f"- {line_core} (score={score:.2f}) – {txt}"
        parts.append(line)
        total += len(line)
    if not parts:
        return ""
    bullets = "\n".join(parts)
    return "Quellen:\n" + bullets

@router.post("/v1/chat/completions")
def chat_completions(payload: dict = Body(...),authorization: str | None = Header(None)):
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
   
    with Timer("[RAG] Retrieval") as t:
        ctx, hits = retrieve(user_msg)
    logging.info(f"[RAG] Antwort in {t.ms/1000:.1f} s, Treffer: {len(hits)}, Kontext: {len(ctx)} Zeichen")

    prompt = _build_prompt(user_msg, ctx)
    context_block = _format_context_block(hits)
    client = OllamaClient()
    model_name = settings.MODEL
    t0 = time.time()

    if not stream_flag:
        with Timer("[OLLAMA] Generate") as tgen:
            answer = client.generate(prompt).strip()
            latency = round(time.time() - t0, 3)
            answer = f"{answer}\n\n{context_block}"
            sources = [
            {
                "source": h.get("source"),
                "section": h.get("section", ""),
                "score": h.get("score")
            } for h in hits
        ]
        logging.info(f"[OLLAMA] Antwort in {tgen.ms/1000:.1f} s, non-stream")
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }],
            "rag": {
                "chunks": len(hits),
                "context_chars": len(ctx),
                "sources": sources
            },
            "latency_sec": latency
        }

    return StreamingResponse(
        event_stream(
            prompt=prompt,
            client=client,
            model_name=model_name,
            hits=hits,
            context_block=context_block,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

def event_stream(prompt: str, client: OllamaClient, model_name: str,
                 hits: list, context_block: str, context_position: str):
    total_chars = 0
    with Timer("[Ollama] Stream") as tstream:
        start_id = f"chatcmpl-{uuid.uuid4()}"


        # Modell-Streaming
        options = {"temperature": settings.TEMPERATURE, "num_ctx": settings.NUM_CTX, "num_predict": settings.MAX_TOKENS}
        base_url = settings.OLLAMA_URL
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": options
        }
        error_mode = False
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
                        total_chars += len(piece)
                        yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            error_mode = True
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

     
        tail = {
                "id": start_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": "\n\n" + context_block},
                    "finish_reason": None
                }],
            }
        yield f"data: {json.dumps(tail)}\n\n"

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
    logging.info(f"[Ollama] Antwort in {tstream.ms/1000:.1f} s (stream, {total_chars} Zeichen)")