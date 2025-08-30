import time
from fastapi import APIRouter, Body, Header, HTTPException
from .config import settings

router = APIRouter()

@router.get("/healthz")
def healthz():
    return {True}

# NEU: Modelle-Liste (für OpenWebUI)
@router.get("/v1/models")
def list_models():
    # Liefere mind. 1 Eintrag zurück – OWUI nimmt das als „verfügbar“
    return {
        "object": "list",
        "data": [
            {
                "id": settings.MODEL_NAME,   # z.B. "llama3.2"
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ]
    }