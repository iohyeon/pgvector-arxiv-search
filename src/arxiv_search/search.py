"""Paper search over chunk-level vectors and full text.

* vector  : HNSW (cosine) over chunk embeddings
* keyword : PostgreSQL full-text search (tsvector + GIN, ts_rank_cd)
* hybrid  : Reciprocal Rank Fusion of the two candidate lists

Metadata filters (category, author, date) are part of the same SQL statement.
With an HNSW index scan the filter is applied to rows the index returns, so a
selective filter can leave fewer than `limit` rows. pgvector >= 0.8 solves this
with iterative index scans (`hnsw.iterative_scan`), which this module enables
per transaction. See docs/DESIGN.md.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import numpy as np
from psycopg2.extras import RealDictCursor

from .config import settings
from .embeddings import Embedder

import re

_VECTOR_LITERAL = re.compile(r"'\[[^\]]*\]'::vector")

RRF_K = 60  # standard constant from the RRF paper; damps the weight of top ranks


class Mode(str, Enum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class Filters:
    categories: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    date_from: date | None = None
    date_to: date | None = None


@dataclass
class MatchedChunk:
    section: str
    page: int | None
    content: str
    distance: float | None = None


@dataclass
class PaperHit:
    paper_id: int
    arxiv_id: str
    title: str
    authors: list[str]
    categories: list[str]
    published: date
    score: float
    best_similarity: float | None
    chunks: list[MatchedChunk] = field(default_factory=list)


def build_filter(f: Filters, alias: str = "p") -> tuple[str, dict]:
    """WHERE fragment + named params. Every value is bound, never interpolated."""
    clauses, params = ["TRUE"], {}
    if f.categories:
        clauses.append(f"{alias}.categories && %(f_categories)s")  # any of the categories
        params["f_categories"] = list(f.categories)
    if f.authors:
        from .db import normalize_author
        clauses.append(
            f"{alias}.id IN (SELECT pa.paper_id FROM paper_authors pa "
            f"JOIN authors a ON a.id = pa.author_id WHERE a.normalized_name = ANY(%(f_authors)s))"
        )
        params["f_authors"] = [normalize_author(a) for a in f.authors]
    if f.date_from:
        clauses.append(f"{alias}.published >= %(f_from)s")
        params["f_from"] = f.date_from
    if f.date_to:
        clauses.append(f"{alias}.published <= %(f_to)s")
        params["f_to"] = f.date_to
    return " AND ".join(clauses), params


VECTOR_SQL = """
SELECT c.id AS chunk_id, c.paper_id, c.section, c.page, c.content,
       c.embedding <=> %(qvec)s AS distance
FROM chunks c
JOIN papers p ON p.id = c.paper_id
WHERE {where}
ORDER BY c.embedding <=> %(qvec)s
LIMIT %(k)s
"""

# Without filters the join is dropped: ORDER BY distance LIMIT k on a single
# table is the shape the planner can answer straight from the HNSW index.
VECTOR_SQL_NOFILTER = """
SELECT c.id AS chunk_id, c.paper_id, c.section, c.page, c.content,
       c.embedding <=> %(qvec)s AS distance
FROM chunks c
ORDER BY c.embedding <=> %(qvec)s
LIMIT %(k)s
"""


def vector_sql(where: str) -> str:
    return VECTOR_SQL_NOFILTER if where == "TRUE" else VECTOR_SQL.format(where=where)


KEYWORD_SQL = """
SELECT c.id AS chunk_id, c.paper_id, c.section, c.page, c.content,
       ts_rank_cd(c.tsv, q) AS rank
FROM chunks c
JOIN papers p ON p.id = c.paper_id,
     websearch_to_tsquery('english', %(qtext)s) AS q
WHERE c.tsv @@ q AND {where}
ORDER BY rank DESC
LIMIT %(k)s
"""


class SearchEngine:
    def __init__(self, conn, embedder: Embedder, ef_search: int | None = None,
                 iterative_scan: bool = True):
        self.conn = conn
        self.embedder = embedder
        self.ef_search = ef_search or settings.hnsw_ef_search
        self.iterative_scan = iterative_scan

    # ---- public API -------------------------------------------------------

    def search(self, query: str, mode: Mode = Mode.HYBRID, limit: int = 10,
               filters: Filters = Filters(), candidates_per_paper: int = 5) -> list[PaperHit]:
        k = limit * candidates_per_paper  # chunk candidates; several chunks may hit one paper
        qvec = self.embedder.encode([query])[0]

        vec_rows = self.vector_candidates(qvec, k, filters) if mode != Mode.KEYWORD else []
        kw_rows = self.keyword_candidates(query, k, filters) if mode != Mode.VECTOR else []

        if mode == Mode.VECTOR:
            scored = [(r, 1.0 - r["distance"]) for r in vec_rows]
        elif mode == Mode.KEYWORD:
            scored = [(r, float(r["rank"])) for r in kw_rows]
        else:
            scored = rrf(vec_rows, kw_rows)
        return self._to_papers(scored, limit)

    def similar_papers(self, arxiv_id: str, limit: int = 5) -> list[PaperHit]:
        """Papers close to the centroid (avg) of all chunk vectors of `arxiv_id`."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT p.id, avg(c.embedding) FROM papers p JOIN chunks c ON c.paper_id = p.id "
                "WHERE p.arxiv_id = %s GROUP BY p.id",
                (arxiv_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise KeyError(f"unknown paper {arxiv_id}")
        paper_id, centroid = row
        centroid = centroid / np.linalg.norm(centroid)
        rows = [r for r in self.vector_candidates(centroid, limit * 10, Filters())
                if r["paper_id"] != paper_id]
        return self._to_papers([(r, 1.0 - r["distance"]) for r in rows], limit)

    def vector_candidates(self, qvec: np.ndarray, k: int, filters: Filters) -> list[dict]:
        where, params = build_filter(filters)
        with self.conn:  # SET LOCAL lives until the end of this transaction
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                self._tune_hnsw(cur)
                cur.execute(vector_sql(where), {"qvec": qvec, "k": k, **params})
                rows = cur.fetchall()
        # relaxed_order iterative scans may return slightly out-of-order rows
        return sorted(rows, key=lambda r: r["distance"])

    def keyword_candidates(self, query: str, k: int, filters: Filters) -> list[dict]:
        where, params = build_filter(filters)
        with self.conn:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(KEYWORD_SQL.format(where=where), {"qtext": query, "k": k, **params})
                return cur.fetchall()

    def explain(self, query: str, filters: Filters = Filters(), k: int = 50,
                force_index: bool = False) -> list[str]:
        """force_index disables seq scans, to show the HNSW path on a small corpus."""
        where, params = build_filter(filters)
        qvec = self.embedder.encode([query])[0]
        with self.conn:
            with self.conn.cursor() as cur:
                self._tune_hnsw(cur)
                if force_index:
                    cur.execute("SET LOCAL enable_seqscan = off")
                cur.execute("EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF, SUMMARY ON) "
                            + vector_sql(where), {"qvec": qvec, "k": k, **params})
                # the 384 floats of the query vector make the plan unreadable
                return [_VECTOR_LITERAL.sub("$query_vector", r[0]) for r in cur.fetchall()]

    # ---- internals ----------------------------------------------------------

    def _tune_hnsw(self, cur) -> None:
        cur.execute("SET LOCAL hnsw.ef_search = %s", (self.ef_search,))
        cur.execute("SET LOCAL hnsw.iterative_scan = %s",
                    ("relaxed_order" if self.iterative_scan else "off",))

    def _to_papers(self, scored: list[tuple[dict, float]], limit: int) -> list[PaperHit]:
        """Chunk scores -> paper ranking. A paper scores as its best chunk."""
        by_paper: dict[int, list[tuple[dict, float]]] = defaultdict(list)
        for row, score in scored:
            by_paper[row["paper_id"]].append((row, score))
        ranked = sorted(by_paper.items(), key=lambda kv: max(s for _, s in kv[1]), reverse=True)[:limit]
        if not ranked:
            return []

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, arxiv_id, title, authors, categories, published FROM papers "
                "WHERE id = ANY(%s)",
                ([pid for pid, _ in ranked],),
            )
            meta = {r["id"]: r for r in cur.fetchall()}

        hits = []
        for pid, rows in ranked:
            rows.sort(key=lambda rs: rs[1], reverse=True)
            m = meta[pid]
            distances = [r["distance"] for r, _ in rows if r.get("distance") is not None]
            hits.append(PaperHit(
                paper_id=pid, arxiv_id=m["arxiv_id"], title=m["title"], authors=m["authors"],
                categories=m["categories"], published=m["published"], score=rows[0][1],
                best_similarity=(1.0 - min(distances)) if distances else None,
                chunks=[MatchedChunk(r["section"], r["page"], r["content"], r.get("distance"))
                        for r, _ in rows[:2]],
            ))
        return hits


def rrf(*ranked_lists: list[dict]) -> list[tuple[dict, float]]:
    """Reciprocal Rank Fusion over chunk lists (already sorted best-first).

    Cosine distance and ts_rank_cd live on unrelated scales, so adding raw
    scores would let one side dominate. RRF only uses ranks.
    """
    scores: dict[int, float] = defaultdict(float)
    rows: dict[int, dict] = {}
    for lst in ranked_lists:
        for rank, row in enumerate(lst, start=1):
            scores[row["chunk_id"]] += 1.0 / (RRF_K + rank)
            rows.setdefault(row["chunk_id"], {}).update(row)
    return sorted(((rows[cid], s) for cid, s in scores.items()), key=lambda rs: rs[1], reverse=True)
