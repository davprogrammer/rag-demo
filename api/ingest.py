import os, sys, hashlib
from pathlib import Path
from typing import Iterable, List
from services.ollama_client import OllamaClient
from services.qdrant_client import QdrantStore
from bs4 import BeautifulSoup  

# Konfiguration
MAX_TOKENS_PER_CHUNK = 350          
OVERLAP_TOKENS = 40
MIN_CHUNK_CHARS = 40
EMBED_DIM_ENV = "EMBED_DIM"

def sha16(string: str) -> str:
    return hashlib.sha1(string.encode("utf-8")).hexdigest()[:16]

def iter_html_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".html", ".htm"}:
            yield p

def read_html_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    # Optional: Skripte/Styles entfernen
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
     
    # Normalisieren
    lines = [l.strip() for l in text.splitlines()]
    cleaned = "\n".join([l for l in lines if l])
    return cleaned

def chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    if not text:
        return []
    # Guards
    if overlap_tokens >= max_tokens:
        overlap_tokens = max_tokens // 4
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        chunk = text[i:j].strip()
        if chunk and len(chunk) >= MIN_CHUNK_CHARS:
            out.append(chunk)
        if j == n:
            break
        i_next = j - overlap_chars
        if i_next <= i:
            i = j
        else:
            i = i_next
    return out

def determine_dim(ollama: OllamaClient) -> int:
    env_val = os.getenv(EMBED_DIM_ENV)
    if env_val and env_val.isdigit():
        return int(env_val)
    # Probe (einmalig)
    return len(ollama.embed("probe"))

def ingest(folder: str):
    root = Path(folder)
    assert root.exists(), f"Pfad nicht gefunden: {folder}"

    ollama = OllamaClient()
    dim = determine_dim(ollama)
    store = QdrantStore()

    # Collection sicherstellen / migrieren (existierende Methode verwenden)
    if hasattr(store, "ensure_or_migrate"):
        used = store.ensure_or_migrate(vector_size=dim)
    elif hasattr(store, "ensure_collection"):
        created = store.ensure_collection(vector_size=dim)
        used = store.collection
        if created:
            print(f"[ingest] Collection '{used}' neu angelegt.")
    else:
        used = store.collection
        print("[warn] Keine ensure_* Methode gefunden – setze vorhandene Collection voraus.")

    print(f"[ingest] benutze Collection: {used} (dim={dim})")

    html_files = list(iter_html_files(root))
    if not html_files:
        print("[ingest] Keine HTML Dateien gefunden.")
        return

    total_chunks = 0
    for f in html_files:
        text = read_html_text(f)
        if not text:
            print(f"[warn] leer/ungültig: {f.name}")
            continue
        chunks = chunk_text(text, MAX_TOKENS_PER_CHUNK, OVERLAP_TOKENS)
        if not chunks:
            print(f"[warn] keine Chunks erzeugt: {f.name}")
            continue

        embeddings = []
        for c in chunks:
            try:
                embeddings.append(ollama.embed(c))
            except Exception as e:
                print(f"[warn] Embedding Fehler {f.name}: {e}")
                embeddings.append([0.0] * dim)

        payloads = [{
            "text": c,
            "source": f.name,
            "section": f"{i+1}/{len(chunks)}"
        } for i, c in enumerate(chunks)]

        ids = [f"{sha16(f.name)}-{sha16(c)}" for c in chunks]

        try:
            store.upsert(embeddings, payloads, ids)
        except Exception as e:
            print(f"[error] Upsert Fehler {f.name}: {e}")
            continue

        total_chunks += len(chunks)
        print(f"[ingest] {f.name}: {len(chunks)} Chunks upserted")

    print(f"[ingest] fertig. total chunks: {total_chunks}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./data"
    ingest(target)