# app/embeddings/retrieve.py
import sys, json
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

def main():
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    client = chromadb.Client(Settings(
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    ))

    col = client.get_or_create_collection(COLLECTION)

    qvec = embed_text(query).tolist()
    res = col.query(query_embeddings=[qvec], n_results=max(k, 5),
                    include=["documents", "metadatas", "distances"])

    docs = res.get("documents")
    if not docs or not docs[0]:
        print(json.dumps({"ok": True, "results": []}))
        return

    meta = res.get("metadatas")
    if not meta or not meta[0]:
        print(json.dumps({"ok": True, "results": []}))
        return
    
    dist = res.get("distances")
    if not dist or not dist[0]:
        print(json.dumps({"ok": True, "results": []}))
        return

    out = []
    for i in range(len(docs[0])):
        out.append({
            "text": docs[0][i],
            "meta": meta[0][i],
            "distance": dist[0][i],
        })

    print(json.dumps({"ok": True, "results": out}))

if __name__ == "__main__":
    main()
