import sys
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

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
    vec = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return vec.astype(np.float32)


def retrieve_context(query: str, k: int = 4) -> list[str]:
    client = chromadb.Client(Settings(
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    ))

    col = client.get_or_create_collection(COLLECTION)

    qvec = embed_text(query).tolist()

    res = col.query(
        query_embeddings=[qvec],
        n_results=max(k, 5),
        include=["documents"]
    )

    docs = res.get("documents")
    if not docs or not docs[0]:
        return []

    return docs[0]


def build_prompt(query: str, context_chunks: list[str]) -> str:
    if not context_chunks:
        return f"User question:\n{query}"

    context = "\n\n".join(
        f"- {chunk}" for chunk in context_chunks
    )

    return f"""You are a helpful assistant.

Context:
{context}

User question:
{query}
"""


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing query"}))
        sys.exit(1)

    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    context_chunks = retrieve_context(query, k)
    prompt = build_prompt(query, context_chunks)

    print(json.dumps({
        "ok": True,
        "prompt": prompt,
        "chunks_used": len(context_chunks)
    }))


if __name__ == "__main__":
    main()
