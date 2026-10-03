"""Two measurements that justify the index settings:

1. recall@k and latency of the HNSW index against exact (sequential) search
2. how many rows a selective metadata filter returns with and without
   pgvector's iterative index scan
"""

import random
import statistics
import time
from dataclasses import dataclass

from psycopg2.extras import RealDictCursor

from .search import Filters, SearchEngine, build_filter, vector_sql


@dataclass
class RecallResult:
    queries: int
    k: int
    ef_search: int
    recall: float
    hnsw_ms_p50: float
    exact_ms_p50: float


@dataclass
class FilterResult:
    category: str
    papers_in_category: int
    k: int
    rows_without_iterative: int
    rows_with_iterative: int


def _sample_query_vectors(conn, n: int, seed: int = 7):
    with conn.cursor() as cur:
        cur.execute("SELECT embedding FROM chunks")
        vecs = [r[0] for r in cur.fetchall()]
    random.Random(seed).shuffle(vecs)
    return vecs[:n]


def _run(conn, sql: str, params: dict, settings_sql: list[str]) -> tuple[list[int], float]:
    with conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for s in settings_sql:
                cur.execute(s)
            t0 = time.perf_counter()
            cur.execute(sql, params)
            rows = cur.fetchall()
            elapsed = (time.perf_counter() - t0) * 1000
    return [r["chunk_id"] for r in rows], elapsed


def recall_vs_exact(conn, n_queries: int = 50, k: int = 10, ef_search: int = 40) -> RecallResult:
    sql = vector_sql("TRUE")
    hits, hnsw_ms, exact_ms = 0, [], []
    for qvec in _sample_query_vectors(conn, n_queries):
        params = {"qvec": qvec, "k": k}
        approx, t_a = _run(conn, sql, params, [
            "SET LOCAL enable_seqscan = off",          # force the HNSW path even on a small table
            f"SET LOCAL hnsw.ef_search = {int(ef_search)}",
        ])
        exact, t_e = _run(conn, sql, params, ["SET LOCAL enable_indexscan = off"])  # seq scan + sort
        hits += len(set(approx) & set(exact))
        hnsw_ms.append(t_a)
        exact_ms.append(t_e)
    return RecallResult(n_queries, k, ef_search, hits / (n_queries * k),
                        statistics.median(hnsw_ms), statistics.median(exact_ms))


def filtered_shortfall(engine: SearchEngine, query: str, category: str, k: int = 10,
                       ef_search: int = 40) -> FilterResult:
    conn = engine.conn
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM papers WHERE %s = ANY(categories)", (category,))
        in_cat = cur.fetchone()[0]

    qvec = engine.embedder.encode([query])[0]
    where, fparams = build_filter(Filters(categories=(category,)))
    sql = vector_sql(where)
    params = {"qvec": qvec, "k": k, **fparams}
    base = ["SET LOCAL enable_seqscan = off", f"SET LOCAL hnsw.ef_search = {int(ef_search)}"]
    off, _ = _run(conn, sql, params, base + ["SET LOCAL hnsw.iterative_scan = off"])
    on, _ = _run(conn, sql, params, base + ["SET LOCAL hnsw.iterative_scan = relaxed_order"])
    return FilterResult(category, in_cat, k, len(off), len(on))
