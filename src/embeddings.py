"""
Embed all chunks and store in ChromaDB.
Run:  python -m src.embeddings
"""
import json

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    PROCESSED_DIR, CHROMA_DIR, COLLECTION_NAME,
    EMBEDDING_MODEL,
)

CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
BATCH_SIZE = 32


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)["chunks"]


def prepare_text_for_embedding(text: str) -> str:
    # multilingual-e5 requires "passage: " prefix on documents
    if "e5" in EMBEDDING_MODEL.lower():
        return f"passage: {text}"
    return text


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads ~470MB — please wait)\n")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [prepare_text_for_embedding(c["text"]) for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE}...\n")
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_meta = metadatas[i:i+BATCH_SIZE]

        batch_emb = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=batch_ids,
            embeddings=batch_emb,
            documents=batch_texts,
            metadatas=batch_meta,
        )

    print(f"\nIndexed {collection.count()} chunks into {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
