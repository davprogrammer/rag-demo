import time, uuid, json, logging
from fastapi import APIRouter, Body, Header, HTTPException
from fastapi.responses import StreamingResponse
from .config import settings
from .ollama_client import OllamaClient
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
        if len(txt)> 200:
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
    user_msg = ""
    with Timer("") as t:
        logging.info(f"[User] {m.get("role")}")
    
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

    
