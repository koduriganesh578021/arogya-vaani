"""
Extract text from PDFs, chunk, and save to data/processed/chunks.json
Run:  python -m src.ingestion
"""
import json
import re
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    PDF_DIR, PROCESSED_DIR, PDF_METADATA,
    CHUNK_SIZE, CHUNK_OVERLAP,
)


def clean_text(text: str) -> str:
    """Whitespace normalization that preserves paragraph breaks."""
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def extract_pages(pdf_path: Path):
    """Yield (page_number, text) for each page."""
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"  [warn] page {i} failed: {e}")
            text = ""
        yield i, clean_text(text)


def chunk_pdf(pdf_path: Path, metadata: dict) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for page_num, page_text in extract_pages(pdf_path):
        if len(page_text) < 50:
            # skip near-empty pages (covers, images)
            continue
        for sub_chunk in splitter.split_text(page_text):
            chunks.append({
                "text": sub_chunk,
                "metadata": {
                    "source_file": pdf_path.name,
                    "page": page_num,
                    "scheme": metadata["scheme"],
                    "authority": metadata["authority"],
                    "doc_type": metadata["doc_type"],
                },
            })
    return chunks


def main():
    all_chunks = []
    print(f"Scanning {PDF_DIR}\n")

    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        name = pdf_path.name
        if name not in PDF_METADATA:
            print(f"[skip] {name} — not in PDF_METADATA")
            continue

        print(f"[ingest] {name}")
        chunks = chunk_pdf(pdf_path, PDF_METADATA[name])
        for i, c in enumerate(chunks):
            c["id"] = f"{name}::p{c['metadata']['page']}::c{i:03d}"
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks\n")

    out_path = PROCESSED_DIR / "chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks}, f, ensure_ascii=False, indent=2)

    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
