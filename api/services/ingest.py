
import os, glob, time
from typing import List, Tuple
from PyPDF2 import PdfReader
from services import config
from .chroma_client import get_collection, get_client
from services import ollama_client

# -------------------- Helpers --------------------

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    texts = [(p.extract_text() or "").strip() for p in reader.pages]
    # kompakte Normalisierung (Mehrfach‑Whitespace -> Einzel‑Space)
    joined = "\n\n".join(t for t in texts if t)
    return " ".join(joined.split()).strip()

def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Sehr einfache, nachvollziehbare Chunks über Zeichenlänge.
    - Überspringt leere/Whitespace-Chunks.
    - Überlappung: 'overlap' Zeichen vom Ende des vorherigen Chunks.
    """
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, step = [], max(1, size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start:start+size]
        if chunk and chunk.strip():
            chunks.append(chunk.strip())
        if start + size >= len(text):
            break
    return chunks

def _filter_valid(
    ids: List[str],
    chunks: List[str],
    metas: List[dict],
    vectors: List
) -> Tuple[List[str], List[str], List[dict], List[List[float]], int]:
    """
    Entfernt leere/ungültige Embeddings + dazugehörige Daten.
    Bricht NICHT ab, sondern filtert und zählt verworfene Einträge.
    """
    from math import isfinite

    keep_ids, keep_chunks, keep_metas, keep_vecs = [], [], [], []
    dropped = 0

    for i, (id_, ch, meta, vec) in enumerate(zip(ids, chunks, metas, vectors)):
        if not isinstance(ch, str) or not ch.strip():
            dropped += 1
            continue
        if not isinstance(vec, (list, tuple)) or len(vec) == 0:
            dropped += 1
            continue

        try:
            vec_clean = [float(x) for x in vec if isinstance(x, (int, float)) and isfinite(float(x))]
        except Exception:
            dropped += 1
            continue

        if not vec_clean:
            dropped += 1
            continue

        keep_ids.append(id_)
        keep_chunks.append(ch)
        keep_metas.append(meta)
        keep_vecs.append(vec_clean)

    return keep_ids, keep_chunks, keep_metas, keep_vecs, dropped

# -------------------- Ingest --------------------

def ingest():
    t0 = time.time()
    coll = get_collection()
    pdfs = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.pdf")))
    if not pdfs:
        print(f"[ingest] Keine PDFs in {config.DATA_DIR}")
        return

    docs, total_chunks = 0, 0

    for path in pdfs:
        doc = os.path.basename(path)
        print(f"[ingest] {doc} …")
        try:
            text = read_pdf_text(path)
        except Exception as e:
            print(f"[warn] Konnte {doc} nicht lesen: {e}")
            continue

        chunks = chunk_text(text, 1000, 200)
        if not chunks:
            print(f"[warn] Keine Textinhalte in {doc}")
            continue

        ids   = [f"{doc}::chunk{idx}" for idx in range(len(chunks))]
        metas = [{"source": doc, "chunk": idx} for idx in range(len(chunks))]

        # Embeddings erzeugen (fehlertolerant: einzelne Chunks können übersprungen werden)
        vectors = []
        ok_ids, ok_chunks, ok_metas = [], [], []
        for i, (cid, ch, meta) in enumerate(zip(ids, chunks, metas)):
            try:
                vec = ollama_client.embed(ch)
                vectors.append(vec)
                ok_ids.append(cid)
                ok_chunks.append(ch)
                ok_metas.append(meta)
            except Exception as e:
                print(f"[warn] {doc} chunk {i} Embedding fehlgeschlagen: {e}")
        ids, chunks, metas = ok_ids, ok_chunks, ok_metas

        # Anstatt hart abzubrechen: defensiv filtern
        ids, chunks, metas, vectors, dropped = _filter_valid(ids, chunks, metas, vectors)
        if dropped:
            print(f"[warn] {doc}: {dropped} Einträge mit leeren/ungültigen Embeddings verworfen.")

        if not vectors:
            print(f"[warn] {doc}: Alle Embeddings waren leer/ungültig – Datei übersprungen.")
            continue

        # Upsert in Chroma (mit Dimensions-Mismatch Auto-Reset Option)
        try:
            coll.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=vectors)
        except Exception as e:
            msg = str(e)
            if (
                "dimension" in msg.lower()
                and os.getenv("RESET_ON_DIM_MISMATCH", "1") == "1"
            ):
                print("[warn] Dimensions-Mismatch erkannt. Versuche Collection neu zu erstellen …")
                try:
                    client = get_client()
                    client.delete_collection(config.COLLECTION_NAME)
                    coll = get_collection()
                    coll.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=vectors)
                    print("[info] Collection wegen Dimensionswechsel neu aufgebaut.")
                except Exception as e2:
                    print(f"[error] Neuaufbau nach Dimensions-Mismatch fehlgeschlagen: {e2}")
                    raise
            else:
                raise

        docs += 1
        total_chunks += len(chunks)
        print(f"[ingest] {doc}: {len(chunks)} Chunks (nach Filter)")

    print(f"[ingest] Fertig: {docs} Dokument(e), {total_chunks} Chunks in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    ingest()