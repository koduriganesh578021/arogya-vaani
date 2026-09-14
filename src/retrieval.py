"""
Hybrid retrieval: BM25 + vector search, fused with RRF.
Run:  python -m src.retrieval "Aarogyasri ki em documents kavali?"
"""
import json
import sys

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.config import (
    PROCESSED_DIR, CHROMA_DIR, COLLECTION_NAME,
    EMBEDDING_MODEL, TOP_K_VECTOR, TOP_K_BM25,
    TOP_K_FINAL, RRF_K,
)

CHUNKS_PATH = PROCESSED_DIR / "chunks.json"


class Retriever:
    def __init__(self):
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            self.chunks = json.load(f)["chunks"]
        self.chunk_by_id = {c["id"]: c for c in self.chunks}

        # BM25
        self.bm25_ids = [c["id"] for c in self.chunks]
        self.bm25 = BM25Okapi([self._tokenize(c["text"]) for c in self.chunks])

        # Vector
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)

    @staticmethod
    def _tokenize(text: str):
        return text.lower().split()

    def _prepare_query(self, query: str) -> str:
        if "e5" in EMBEDDING_MODEL.lower():
            return f"query: {query}"
        return query

    def _vector_search(self, query: str, k: int, scheme: str | None = None):
        q_emb = self.model.encode(
            [self._prepare_query(query)],
            normalize_embeddings=True,
        ).tolist()
        # Optional metadata filter: only filter when a specific scheme is
        # requested ("both" or None means search across both schemes).
        where = None
        if scheme is not None and scheme != "both":
            where = {"scheme": scheme}
        results = self.collection.query(
            query_embeddings=q_emb, n_results=k, where=where
        )
        return results["ids"][0]

    def _bm25_search(self, query: str, k: int, scheme: str | None = None):
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # Build ranked list, skipping non-matching schemes
        ranked = []
        for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True):
            if scheme and scheme not in ("both", ""):
                if self.chunks[i]["metadata"]["scheme"] != scheme:
                    continue
            ranked.append(self.bm25_ids[i])
            if len(ranked) >= k:
                break
        return ranked

    @staticmethod
    def _rrf_fuse(ranked_lists, k: int):
        scores = {}
        for ranked in ranked_lists:
            for rank, doc_id in enumerate(ranked, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:k]

    def search(self, query: str, k: int = TOP_K_FINAL, scheme: str | None = None):
        vec_ids = self._vector_search(query, TOP_K_VECTOR, scheme)
        bm25_ids = self._bm25_search(query, TOP_K_BM25, scheme=scheme)
        fused = self._rrf_fuse([vec_ids, bm25_ids], k)
        # Final safety filter: drop any chunks not matching the requested scheme
        if scheme and scheme not in ("both", ""):
            fused = [(doc_id, score) for doc_id, score in fused
                     if self.chunk_by_id[doc_id]["metadata"]["scheme"] == scheme]

        # Deduplicate: keep best-scoring chunk per (source_file, page)
        seen_pages = set()
        deduped = []
        for doc_id, score in fused:
            meta = self.chunk_by_id[doc_id]["metadata"]
            key = (meta["source_file"], meta["page"])
            if key in seen_pages:
                continue
            seen_pages.add(key)
            deduped.append((doc_id, score))
        fused = deduped[:k]

        out = []
        for doc_id, score in fused:
            c = self.chunk_by_id[doc_id]
            out.append({
                "id": doc_id,
                "score": round(score, 4),
                "text": c["text"],
                "metadata": c["metadata"],
            })
        return out


def format_result(r: dict, n: int):
    m = r["metadata"]
    print(f"\n[{n}] {m['source_file']} (page {m['page']})")
    print(f"    scheme={m['scheme']}  doc_type={m['doc_type']}  score={r['score']}")
    snippet = r["text"][:280].replace("\n", " ")
    print(f"    {snippet}...")


def main():
    query = " ".join(sys.argv[1:]) or "Aarogyasri ki em documents kavali?"
    print(f"Query: {query}\n")
    print("Loading retriever...")
    r = Retriever()
    print("Searching...")
    results = r.search(query, k=TOP_K_FINAL)

    print(f"\nTop {len(results)} results:")
    for i, res in enumerate(results, 1):
        format_result(res, i)


if __name__ == "__main__":
    main()
