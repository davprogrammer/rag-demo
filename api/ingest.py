import os, sys, hashlib, uuid
from pathlib import Path
from typing import Iterable, List
from services.ollama_client import OllamaClient
from services.qdrant_client import QdrantStore
from bs4 import BeautifulSoup  
from services.config import settings

#Verbesserungen
# - CLI, mehre Datapfade
# - Tracken der Dateien, automatisch geänderte Dateien ingesten.
# - Metadaten ingesten, für Filterung und Sortierung
# - Mehr Datentypen als HTML, Dokumentenstruktur wie Überschriften berücksichtigen, dynamischen Chunking

def sha16(string: str) -> str:
    return hashlib.sha1(string.encode("utf-8")).hexdigest()[:16]

def iterate_html_files(root: Path) -> Iterable[Path]:
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

def chunk_text(text: str) -> List[str]:
    max_chars = settings.MAX_TOKENS_PER_CHUNK * 4
    overlap_chars = settings.OVERLAP_TOKENS * 4
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
    ollama = OllamaClient()
    vec_store = QdrantStore()
    dim = settings.EMBED_DIM

    collection = vec_store.check_collection(dim)
    print(f"[INGEST] benutze Collection: {collection} (dim={dim})")

    html_files = list(iterate_html_files(root))
    total_chunks = 0
    for f in html_files:
        text = read_html_text(f)
        if not text:
            print(f"[WARN] leer/ungültig: {f.name}")
            continue
        chunks = chunk_text(text)
        if not chunks:
            print(f"[WARN] keine Chunks erzeugt: {f.name}")
            continue

        embeddings = []
        for c in chunks:
            try:
                embeddings.append(ollama.embed(c))
            except Exception as e:
                print(f"[WARN] Embedding Fehler {f.name}: {e}")
                embeddings.append([0.0] * dim)

        payloads = [{
            "text": c,
            "source": f.name,
            "section": f"{i+1}/{len(chunks)}"
        } for i, c in enumerate(chunks)]

        base_ns = uuid.NAMESPACE_URL
        ids = [str(uuid.uuid5(base_ns, f"{f.name}:{sha16(c)}")) for c in chunks]

        try:
            vec_store.upsert(embeddings, payloads, ids)
        except Exception as e:
            print(f"[ERROR] Upsert Fehler {f.name}: {e}")
            continue

        total_chunks += len(chunks)
        print(f"[INGEST] {f.name}: {len(chunks)} Chunks upserted")

    print(f"[INGEST] fertig. total chunks: {total_chunks}")

if __name__ == "__main__":
    target = "./data"
    ingest(target)