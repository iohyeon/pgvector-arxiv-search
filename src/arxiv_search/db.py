"""PostgreSQL access: connection, schema bootstrap and write paths."""

import unicodedata
from typing import Sequence

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from .arxiv_client import ArxivPaper
from .chunker import Chunk
from .config import settings


def connect():
    """Connection parameters come from PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD."""
    conn = psycopg2.connect("")
    register_vector(conn)  # numpy.ndarray <-> vector; needs the extension to exist
    return conn


def init_schema() -> None:
    conn = psycopg2.connect("")  # no register_vector: the extension may not exist yet
    try:
        with conn, conn.cursor() as cur:
            cur.execute(settings.schema_path.read_text())
    finally:
        conn.close()


def normalize_author(name: str) -> str:
    """'Yoshua Bengio', 'yoshua  bengio', 'Yoshua Béngio' -> 'yoshua bengio'."""
    ascii_name = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(ch for ch in ascii_name if not unicodedata.combining(ch))
    return " ".join(ascii_name.lower().replace(".", " ").split())


def upsert_paper(cur, paper: ArxivPaper) -> int:
    cur.execute(
        """
        INSERT INTO papers (arxiv_id, title, abstract, authors, categories,
                            primary_category, published, updated, pdf_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (arxiv_id) DO UPDATE SET
            title = EXCLUDED.title,
            abstract = EXCLUDED.abstract,
            authors = EXCLUDED.authors,
            categories = EXCLUDED.categories,
            primary_category = EXCLUDED.primary_category,
            updated = EXCLUDED.updated,
            pdf_url = EXCLUDED.pdf_url
        RETURNING id
        """,
        (paper.arxiv_id, paper.title, paper.abstract, list(paper.authors), list(paper.categories),
         paper.primary_category, paper.published, paper.updated, paper.pdf_url),
    )
    paper_id = cur.fetchone()[0]

    # Rebuild the author relation so re-ingesting a revised paper stays consistent.
    cur.execute("DELETE FROM paper_authors WHERE paper_id = %s", (paper_id,))
    for position, name in enumerate(paper.authors, start=1):
        # DO UPDATE (not DO NOTHING) so RETURNING yields the id on conflict too:
        # no extra SELECT and no race between insert and lookup.
        cur.execute(
            """
            INSERT INTO authors (name, normalized_name) VALUES (%s, %s)
            ON CONFLICT (normalized_name) DO UPDATE SET name = authors.name
            RETURNING id
            """,
            (name, normalize_author(name)),
        )
        author_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO paper_authors (paper_id, author_id, position) VALUES (%s, %s, %s) "
            "ON CONFLICT DO NOTHING",
            (paper_id, author_id, position),
        )
    return paper_id


def replace_chunks(cur, paper_id: int, chunks: Sequence[Chunk], embeddings: np.ndarray,
                   model_name: str) -> None:
    cur.execute("DELETE FROM chunks WHERE paper_id = %s", (paper_id,))
    rows = [
        (paper_id, i, c.section, c.page, c.text, c.tokens, emb, model_name)
        for i, (c, emb) in enumerate(zip(chunks, embeddings))
    ]
    execute_values(
        cur,
        "INSERT INTO chunks (paper_id, chunk_index, section, page, content, token_count, "
        "embedding, embedding_model) VALUES %s",
        rows,
    )


def mark_paper(cur, paper_id: int, *, status: str, has_fulltext: bool,
               pdf_path: str | None, error: str | None) -> None:
    cur.execute(
        "UPDATE papers SET status = %s, has_fulltext = %s, pdf_path = %s, error = %s WHERE id = %s",
        (status, has_fulltext, pdf_path, error, paper_id),
    )


def is_embedded(cur, arxiv_id: str, updated) -> bool:
    cur.execute(
        "SELECT 1 FROM papers WHERE arxiv_id = %s AND status = 'embedded' AND updated = %s",
        (arxiv_id, updated),
    )
    return cur.fetchone() is not None
