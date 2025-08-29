from fastapi import FastAPI
import threading, time
import requests
from services.routes import router
from services import config

def _warmup():
	"""Schritt 1: Warmup-Aufruf an Ollama, damit erstes User-Query nicht kalt startet."""
	# kleine Verzögerung, damit Server vollständig läuft
	time.sleep(2)
	payload = {
		"model": config.MODEL,
		"messages": [{"role": "user", "content": "Hallo"}],
		"stream": False,
	}
	try:
		requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=45)
	except Exception:
		pass

app = FastAPI(title="RAG FAQ API")
app.include_router(router)

@app.on_event("startup")
def _startup():
	threading.Thread(target=_warmup, daemon=True).start()
