import requests
from . import config

def embed(text: str) -> list[float]:
    r = requests.post(
        f"{config.OLLAMA_URL}/api/embeddings",
        json={"model": config.EMBED_MODEL, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]

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
    return r.json()["message"]["content"]