"""Ingestion: arXiv metadata -> (optional) PDF -> chunks -> embeddings -> PostgreSQL.

Each paper is written in its own transaction. Embeddings are computed before
the transaction opens, so a slow model call never holds row locks, and a
failure on one paper never rolls back the others.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

from .arxiv_client import ArxivPaper
from .chunker import Chunk, TokenAwareChunker
from .config import settings
from .db import connect, is_embedded, mark_paper, replace_chunks, upsert_paper
from .embeddings import get_embedder
from .pdf import DownloadError, PDFDownloader, PDFExtractor

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    fetched: int = 0
    skipped: int = 0
    embedded: int = 0
    fulltext: int = 0
    pdf_failed: int = 0
    chunks: int = 0


def ingest(papers: Iterable[ArxivPaper], with_pdf: bool = False, refresh: bool = False,
           on_paper: Callable[[ArxivPaper, int, str], None] | None = None) -> IngestStats:
    embedder = get_embedder(settings.embedding_model, settings.embedding_device)
    if settings.chunk_tokens + 2 > embedder.max_seq_length:  # +2: [CLS] and [SEP]
        raise ValueError(
            f"CHUNK_TOKENS={settings.chunk_tokens} exceeds the model limit "
            f"({embedder.max_seq_length} incl. special tokens); the tail would be truncated"
        )
    chunker = TokenAwareChunker(embedder.count_tokens, settings.chunk_tokens, settings.chunk_overlap)
    downloader = PDFDownloader(settings.pdf_dir, settings.arxiv_delay_seconds)
    extractor = PDFExtractor()

    stats = IngestStats()
    conn = connect()
    try:
        for paper in papers:
            stats.fetched += 1
            if not refresh:
                with conn, conn.cursor() as cur:
                    if is_embedded(cur, paper.arxiv_id, paper.updated):
                        stats.skipped += 1
                        continue

            chunks: list[Chunk] = chunker.chunk_text(paper.abstract, section="abstract")
            pdf_path, error = None, None
            if with_pdf:
                try:
                    pdf_path = downloader.download(paper)
                    body = chunker.chunk_sections(extractor.extract(pdf_path))
                    chunks += body
                    stats.fulltext += bool(body)
                except (DownloadError, RuntimeError, ValueError) as exc:
                    # degrade to abstract-only instead of dropping the paper
                    error = str(exc)[:500]
                    stats.pdf_failed += 1
                    log.warning("pdf failed for %s: %s", paper.arxiv_id, error)

            embeddings = embedder.encode([c.text for c in chunks])

            with conn, conn.cursor() as cur:
                paper_id = upsert_paper(cur, paper)
                replace_chunks(cur, paper_id, chunks, embeddings, embedder.model_name)
                mark_paper(cur, paper_id, status="embedded",
                           has_fulltext=len(chunks) > 1 and pdf_path is not None,
                           pdf_path=str(pdf_path) if pdf_path else None, error=error)

            stats.embedded += 1
            stats.chunks += len(chunks)
            if on_paper:
                on_paper(paper, len(chunks), "fulltext" if len(chunks) > 1 and pdf_path else "abstract")
    finally:
        conn.close()
    return stats
