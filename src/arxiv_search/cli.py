"""Command line interface.

    python -m arxiv_search init-db
    python -m arxiv_search fetch --query "cat:cs.IR AND abs:retrieval" --max 30 --pdf
    python -m arxiv_search fetch --categories cs.CL cs.IR --days 7 --max 50
    python -m arxiv_search search "long context retrieval" --mode hybrid --category cs.CL
    python -m arxiv_search similar 2610.01981
    python -m arxiv_search stats
    python -m arxiv_search explain "graph neural networks" --category cs.LG
    python -m arxiv_search bench
"""

import argparse
import logging
from datetime import date

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .config import settings

console = Console()


def _engine(**kwargs):
    from .db import connect
    from .embeddings import get_embedder
    from .search import SearchEngine
    return SearchEngine(connect(), get_embedder(settings.embedding_model, settings.embedding_device), **kwargs)


def _filters(args):
    from .search import Filters
    return Filters(
        categories=tuple(args.category or ()),
        authors=tuple(args.author or ()),
        date_from=args.since,
        date_to=args.until,
    )


# ---------------------------------------------------------------------------

def cmd_init_db(args, out: Console = console) -> None:
    from .db import init_schema
    init_schema()
    out.print("[green]schema applied[/] " + str(settings.schema_path.relative_to(settings.schema_path.parents[1])))


def cmd_fetch(args, out: Console = console) -> None:
    from .arxiv_client import ArxivClient
    from .pipeline import ingest

    client = ArxivClient(delay_seconds=settings.arxiv_delay_seconds)
    papers = (client.search(args.query, args.max) if args.query
              else client.recent(args.categories, args.days, args.max))

    def progress(paper, n_chunks, kind):
        out.print(f"  [dim]{paper.arxiv_id}[/]  {kind:<8} {n_chunks:>4} chunks  {paper.title[:70]}")

    out.print(f"[bold]fetch[/] {'pdf+abstract' if args.pdf else 'abstract only'}")
    stats = ingest(papers, with_pdf=args.pdf, refresh=args.refresh, on_paper=progress)
    out.print(
        f"[green]done[/] fetched={stats.fetched} embedded={stats.embedded} skipped={stats.skipped} "
        f"fulltext={stats.fulltext} pdf_failed={stats.pdf_failed} chunks={stats.chunks}"
    )


def cmd_search(args, out: Console = console) -> None:
    from .search import Mode
    engine = _engine()
    hits = engine.search(args.query, Mode(args.mode), args.limit, _filters(args))
    render_hits(out, args.query, args.mode, _filters(args), hits)


def render_hits(out: Console, query: str, mode: str, filters, hits) -> None:
    flt = []
    if filters.categories:
        flt.append("category=" + ",".join(filters.categories))
    if filters.authors:
        flt.append("author=" + ",".join(filters.authors))
    if filters.date_from:
        flt.append(f"since={filters.date_from}")
    out.print(f'[bold]query[/] "{query}"  [bold]mode[/] {mode}' + (f"  [bold]filter[/] {' '.join(flt)}" if flt else ""))
    if not hits:
        out.print("[yellow]no results[/]")
        return
    table = Table(show_lines=True, expand=True)
    table.add_column("#", justify="right", width=2)
    table.add_column("paper", ratio=3)
    table.add_column("cos", justify="right", width=5)
    table.add_column("best matching passage", ratio=4)
    for i, h in enumerate(hits, start=1):
        paper = Text()
        paper.append(h.title + "\n", style="bold")
        paper.append(f"{h.arxiv_id} · {h.published} · {', '.join(h.categories[:3])}", style="dim")
        c = h.chunks[0]
        where = c.section if c.page is None else f"{c.section} (p.{c.page})"
        passage = Text()
        passage.append(f"[{where[:40]}] ", style="cyan")
        passage.append(c.content[:220] + ("..." if len(c.content) > 220 else ""))
        sim = f"{h.best_similarity:.3f}" if h.best_similarity is not None else "-"
        table.add_row(str(i), paper, sim, passage)
    out.print(table)


def cmd_similar(args, out: Console = console) -> None:
    engine = _engine()
    hits = engine.similar_papers(args.arxiv_id, args.limit)
    render_hits(out, f"similar to {args.arxiv_id}", "centroid", _filters_none(), hits)


def _filters_none():
    from .search import Filters
    return Filters()


def cmd_stats(args, out: Console = console) -> None:
    from .db import connect
    conn = connect()
    with conn, conn.cursor() as cur:
        cur.execute("""
            SELECT count(*), count(*) FILTER (WHERE has_fulltext), min(published), max(published)
            FROM papers WHERE status = 'embedded'""")
        papers, fulltext, d_min, d_max = cur.fetchone()
        cur.execute("""
            SELECT count(*), count(*) FILTER (WHERE section = 'abstract'),
                   round(avg(token_count)), max(token_count), count(DISTINCT section)
            FROM chunks""")
        chunks, abstract_chunks, avg_tok, max_tok, sections = cur.fetchone()
        cur.execute("SELECT count(*) FROM authors")
        authors = cur.fetchone()[0]
        cur.execute("""
            SELECT c, count(*) FROM papers, unnest(categories) AS c
            GROUP BY c ORDER BY count(*) DESC LIMIT 6""")
        cats = cur.fetchall()
        cur.execute("""
            SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid))
            FROM pg_stat_user_indexes WHERE relname = 'chunks' ORDER BY pg_relation_size(indexrelid) DESC""")
        idx = cur.fetchall()
        cur.execute("SELECT pg_size_pretty(pg_total_relation_size('chunks'))")
        chunks_size = cur.fetchone()[0]
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        pgv = cur.fetchone()[0]
    conn.close()

    t = Table(title="corpus", show_header=False, expand=False)
    t.add_column(style="bold")
    t.add_column()
    t.add_row("papers", f"{papers}  (full text {fulltext}, abstract only {papers - fulltext})")
    t.add_row("published", f"{d_min} ~ {d_max}")
    t.add_row("authors (deduplicated)", str(authors))
    t.add_row("chunks", f"{chunks}  (abstract {abstract_chunks}, body {chunks - abstract_chunks}, sections {sections})")
    t.add_row("tokens / chunk", f"avg {avg_tok}, max {max_tok}  (model limit {256} incl. special tokens)")
    t.add_row("top categories", ", ".join(f"{c} {n}" for c, n in cats))
    t.add_row("chunks table size", chunks_size)
    t.add_row("indexes on chunks", ", ".join(f"{n} {s}" for n, s in idx))
    t.add_row("pgvector", pgv)
    out.print(t)


def cmd_explain(args, out: Console = console) -> None:
    engine = _engine()
    force = getattr(args, "force_index", False)
    out.print(f'[bold]EXPLAIN ANALYZE[/] vector search "{args.query}"'
              + (f" with category={','.join(args.category)}" if args.category else "")
              + (" [dim](enable_seqscan=off)[/]" if force else ""))
    for line in engine.explain(args.query, _filters(args), k=args.limit, force_index=force):
        style = "bold green" if "hnsw" in line.lower() or "Index Scan using chunks_embedding" in line else None
        out.print(Text(line, style=style))


def cmd_bench(args, out: Console = console) -> None:
    from .bench import filtered_shortfall, recall_vs_exact
    engine = _engine()

    t = Table(title="HNSW vs exact search (query = random stored chunk vectors)")
    for col in ("ef_search", "queries", "recall@10", "HNSW p50 ms", "exact p50 ms"):
        t.add_column(col, justify="right")
    for ef in args.ef:
        r = recall_vs_exact(engine.conn, n_queries=args.queries, k=10, ef_search=ef)
        t.add_row(str(ef), str(r.queries), f"{r.recall:.3f}", f"{r.hnsw_ms_p50:.2f}", f"{r.exact_ms_p50:.2f}")
    out.print(t)

    f = filtered_shortfall(engine, args.filter_query, args.filter_category, k=10, ef_search=40)
    t2 = Table(title=f'filtered search: "{args.filter_query}" AND category={f.category} (LIMIT {f.k})')
    for col in ("papers in category", "iterative_scan=off", "iterative_scan=relaxed_order"):
        t2.add_column(col, justify="right")
    t2.add_row(str(f.papers_in_category), f"{f.rows_without_iterative} rows", f"{f.rows_with_iterative} rows")
    out.print(t2)


# ---------------------------------------------------------------------------

def _date(s: str) -> date:
    return date.fromisoformat(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arxiv_search", description="Semantic search over arXiv papers (PostgreSQL + pgvector)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="apply db/schema.sql").set_defaults(func=cmd_init_db)

    f = sub.add_parser("fetch", help="fetch papers from arXiv and index them")
    src = f.add_mutually_exclusive_group(required=True)
    src.add_argument("--query", help='raw arXiv query, e.g. "cat:cs.IR AND abs:retrieval"')
    src.add_argument("--categories", nargs="+", help="e.g. cs.CL cs.IR (use with --days)")
    f.add_argument("--days", type=int, default=7)
    f.add_argument("--max", type=int, default=20)
    f.add_argument("--pdf", action="store_true", help="download PDFs and index full text")
    f.add_argument("--refresh", action="store_true", help="re-index papers already embedded")
    f.set_defaults(func=cmd_fetch)

    def add_filters(sp):
        sp.add_argument("--category", nargs="+", help="any of these arXiv categories")
        sp.add_argument("--author", nargs="+")
        sp.add_argument("--since", type=_date)
        sp.add_argument("--until", type=_date)

    s = sub.add_parser("search", help="search papers")
    s.add_argument("query")
    s.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="hybrid")
    s.add_argument("--limit", type=int, default=5)
    add_filters(s)
    s.set_defaults(func=cmd_search)

    sm = sub.add_parser("similar", help="papers similar to a stored paper")
    sm.add_argument("arxiv_id")
    sm.add_argument("--limit", type=int, default=5)
    sm.set_defaults(func=cmd_similar)

    sub.add_parser("stats", help="corpus and index statistics").set_defaults(func=cmd_stats)

    e = sub.add_parser("explain", help="show the query plan of a vector search")
    e.add_argument("query")
    e.add_argument("--limit", type=int, default=50)
    e.add_argument("--force-index", action="store_true", help="disable seq scan to show the HNSW plan")
    add_filters(e)
    e.set_defaults(func=cmd_explain)

    b = sub.add_parser("bench", help="recall/latency and filtered-search measurements")
    b.add_argument("--queries", type=int, default=50)
    b.add_argument("--ef", type=int, nargs="+", default=[10, 40, 100])
    b.add_argument("--filter-query", default="retrieval augmented generation")
    b.add_argument("--filter-category", default="cs.CV")
    b.set_defaults(func=cmd_bench)
    return p


def main(argv=None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)
