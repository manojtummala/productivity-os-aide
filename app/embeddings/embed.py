# app/embeddings/embed.py

import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

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
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing text"}))
        sys.exit(1)

    text = sys.argv[1]
    vec = embed_text(text)

    print(json.dumps(vec.tolist()))


if __name__ == "__main__":
    main()
