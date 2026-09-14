"""
Central configuration for Arogya Vaani.
All paths, model names, and metadata live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"

for _d in [PDF_DIR, PROCESSED_DIR, CHROMA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------- Groq (for Step 3) ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MODEL_FAST = "openai/gpt-oss-20b"

# ---------- Embeddings ----------
# Multilingual, ~470MB, good Telugu <-> English matching.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
# Alternative (larger, higher quality, ~2.2GB):
# EMBEDDING_MODEL = "BAAI/bge-m3"

# ---------- Chunking ----------
CHUNK_SIZE = 1000       # characters
CHUNK_OVERLAP = 200     # characters

# ---------- ChromaDB ----------
COLLECTION_NAME = "arogya_vaani"

# ---------- PDF Metadata ----------
# Maps each PDF filename to its scheme, authority, and document type.
# Update this if you rename PDFs or add more.
PDF_METADATA = {
    "Schedules.pdf": {
        "scheme": "pmjay",
        "authority": "National Health Authority / Arogyakeralam",
        "doc_type": "beneficiary_identification",
        "description": "PM-JAY beneficiary identification and eligibility",
    },
    "HBP.pdf": {
        "scheme": "pmjay",
        "authority": "National Health Authority",
        "doc_type": "health_benefit_package",
        "description": "PM-JAY Health Benefit Packages — covered treatments",
    },
    "NEHS-Guidelines-DoCoVP_I.pdf": {
        "scheme": "nehs",
        "authority": "Aarogyasri Health Care Trust, Telangana",
        "doc_type": "scheme_guidelines",
        "description": "Telangana New Employees Health Scheme (NEHS) guidelines",
    },
    "m5.pdf": {
        "scheme": "ap_ehs",
        "authority": "Government of Telangana",
        "doc_type": "eligibility_criteria",
        "description": "Andhra Pradesh Employees Health Scheme (2013) — legacy reference",
    },
    "m7.pdf": {
        "scheme": "ap_ehs",
        "authority": "Government of Telangana",
        "doc_type": "therapy_coverage",
        "description": "Andhra Pradesh Employees Health Scheme — therapy prices (2013) — legacy reference",
    },
    "aarogyasri_kamareddy.pdf": {
        "scheme": "aarogyasri",
        "authority": "Aarogyasri Health Care Trust, Government of Telangana",
        "doc_type": "scheme_overview",
        "description": "Aarogyasri scheme overview — coverage amount, procedures, hospital network, dialysis and high-end therapies (Telugu)",
    },
}

# ---------- Retrieval ----------
TOP_K_VECTOR = 5
TOP_K_BM25 = 5
TOP_K_FINAL = 3
RRF_K = 60   # reciprocal rank fusion constant
