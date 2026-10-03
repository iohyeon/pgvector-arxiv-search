"""Runtime settings, read once from the environment (.env supported)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    chunk_tokens: int = int(os.getenv("CHUNK_TOKENS", "200"))
    chunk_overlap: float = float(os.getenv("CHUNK_OVERLAP", "0.2"))
    hnsw_ef_search: int = int(os.getenv("HNSW_EF_SEARCH", "100"))
    arxiv_delay_seconds: float = float(os.getenv("ARXIV_DELAY_SECONDS", "3"))
    pdf_dir: Path = PROJECT_ROOT / os.getenv("PDF_DIR", "data/pdfs")
    schema_path: Path = PROJECT_ROOT / "db" / "schema.sql"


settings = Settings()
