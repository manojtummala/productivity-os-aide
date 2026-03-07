# app/embeddings/ingest_text.py
import sys, json, hashlib
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import Tuple, Any

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "app/data/chroma"
COLLECTION = "docs"

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_ID)
    return _model


def embed_text(text: str) -> np.ndarray:
    model = get_model()

    # sentence-transformers returns a 1D numpy array for a single string
    vec = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Ensure float32 for Chroma compatibility
    return vec.astype(np.float32)

def chunk_text(text, size=800, overlap=120):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return [c.strip() for c in chunks if c.strip()]

def stable_id(s):
    return hashlib.sha1(s.encode()).hexdigest()

def main():
    path = sys.argv[1]
    text = open(path, "r", encoding="utf-8").read()

    client = chromadb.Client(Settings(
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    ))

    col = client.get_or_create_collection(COLLECTION)

    chunks = chunk_text(text)
    ids, docs, metas, embs = [], [], [], []

    for i, c in enumerate(chunks):
        ids.append(stable_id(f"{path}:{i}"))
        docs.append(c)
        metas.append({"source": path, "chunk": i})
        embs.append(embed_text(c).tolist())

    col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    print(json.dumps({"ok": True, "chunks": len(chunks)}))

if __name__ == "__main__":
    main()
