import sys, hashlib
from pathlib import Path
from pypdf import PdfReader
from services.ollama_client import OllamaClient
from services.qdrant_client import QdrantStore

def read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for pg in reader.pages:
        parts.append(pg.extract_text() or "")
    return "\n".join(parts)

def chunk_text(text: str, max_tokens: int = 350, overlap: int = 40) -> list[str]:
    max_chars = max_tokens * 4
    ov = overlap * 4
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + max_chars)
        chunk = text[i:j].strip()
        if chunk:
            out.append(chunk)
        if j == n:
            break
        i = max(0, j - ov)
    return out

def sh16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def ingest(data_dir: str):
    data_path = Path(data_dir)
    assert data_path.exists(), f"Pfad nicht gefunden: {data_dir}"

    ollama = OllamaClient()
    dim = len(ollama.embed("probe"))
    store = QdrantStore()

    used = store.ensure_or_migrate(vector_size=dim)  # legt an oder migriert bei Mismatch
    print(f"[ingest] benutze Collection: {used}")

    total = 0
    for pdf in sorted(data_path.glob("*.pdf")):
        text = read_pdf_text(pdf)
        chunks = chunk_text(text, 350, 40)
        vecs = [ollama.embed(c) for c in chunks]
        payloads = [{"text": c, "source": pdf.name, "section": f"{i+1}/{len(chunks)}"} for i, c in enumerate(chunks)]
        ids = [f"{sh16(pdf.name)}-{sh16(c)}" for c in chunks]
        store.upsert(vecs, payloads, ids)
        total += len(chunks)
        print(f"[ingest] {pdf.name}: {len(chunks)} Chunks upserted")

    print(f"[ingest] fertig. total chunks: {total}")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "./data"
    ingest(folder)
