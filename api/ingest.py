import os, sys, hashlib, uuid
from pathlib import Path
from typing import Iterable, List
from services.ollama_client import OllamaClient
from services.qdrant_client import QdrantStore
from bs4 import BeautifulSoup  
from services.config import settings

def sha16(string: str) -> str:
    return hashlib.sha1(string.encode("utf-8")).hexdigest()[:16]

def iter_html_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".html", ".htm"}:
            yield p

def read_html_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    # Skripte/Styles entfernen
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
     
    # Normalisieren
    lines = [l.strip() for l in text.splitlines()]
    cleaned = "\n".join([l for l in lines if l])
    return cleaned

def chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    
    if max_tokens <= 0:
        return [text] if text else []
    if overlap_tokens < 0:
        overlap_tokens = 0
    if not text:
        return []
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        chunk = text[i:j].strip()
        if chunk and len(chunk) >= settings.MIN_CHUNK_CHARS:
            out.append(chunk)
        if j == n:
            break
        i_next = j - overlap_chars
        if i_next <= i:
            i = j
        else:
            i = i_next
    return out

def ingest(folder: str):
    root = Path(folder)
    assert root.exists(), f"Pfad nicht gefunden: {folder}"

    ollama = OllamaClient()
    dim = settings.EMBED_DIM
    store = QdrantStore()
    used = store.ensure_or_migrate(dim)
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
        chunks = chunk_text(text, settings.MAX_TOKENS_PER_CHUNK, settings.OVERLAP_TOKENS)
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

        base_ns = uuid.NAMESPACE_URL
        ids = [str(uuid.uuid5(base_ns, f"{f.name}:{sha16(c)}")) for c in chunks]

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