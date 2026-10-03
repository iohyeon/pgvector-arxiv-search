"""Render CLI output as SVG terminal screenshots for the README.

    python scripts/capture.py

Writes docs/images/*.svg. Every image is produced by running the real
commands against the local database, so the README shows actual results.
"""

import io
import sys
from argparse import Namespace
from pathlib import Path

from rich.console import Console
from rich.terminal_theme import MONOKAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arxiv_search import cli  # noqa: E402

OUT = ROOT / "docs" / "images"
WIDTH = 132


def shot(name: str, title: str, fn, args: Namespace | None = None, prompt: str = "") -> None:
    console = Console(record=True, width=WIDTH, file=io.StringIO(), force_terminal=True)
    if prompt:
        console.print(f"[bold green]$[/] {prompt}")
    fn(args, out=console) if args is not None else fn(console)
    OUT.mkdir(parents=True, exist_ok=True)
    console.save_svg(str(OUT / f"{name}.svg"), title=title, theme=MONOKAI)
    print("wrote", OUT / f"{name}.svg")


def search_args(query, mode="hybrid", limit=5, category=None, since=None):
    return Namespace(query=query, mode=mode, limit=limit, category=category, author=None,
                     since=since, until=None)


def ingest_log(console: Console) -> None:
    log = (ROOT / "data" / "ingest_combined.log").read_text().splitlines()
    lines = [l for l in log if l.strip() and "Warning" not in l]
    head = [l for l in lines if "fulltext" in l][:8]
    tail = [l for l in lines if l.startswith("done")]
    console.print("[bold green]$[/] python -m arxiv_search fetch --query '(cat:cs.IR OR cat:cs.CL) AND (abs:retrieval OR abs:RAG)' --max 30 --pdf")
    for l in head:
        console.print(l, highlight=False)
    console.print("  [dim]...[/]")
    if tail:
        console.print(tail[0], highlight=False)
    console.print("[bold green]$[/] python -m arxiv_search fetch --categories cs.CL cs.LG cs.CV cs.IR --days 3 --max 200")
    console.print("  [dim]...[/]")
    if len(tail) > 1:
        console.print(tail[1], highlight=False)


def main() -> None:
    console = Console(record=True, width=WIDTH, file=io.StringIO(), force_terminal=True)
    ingest_log(console)
    console.save_svg(str(OUT / "01_ingest.svg"), title="fetch", theme=MONOKAI)
    print("wrote", OUT / "01_ingest.svg")

    shot("02_stats", "stats", cli.cmd_stats, Namespace(), "python -m arxiv_search stats")

    q = "how to reduce hallucination in retrieval augmented generation"
    shot("03_search_hybrid", "hybrid search", cli.cmd_search, search_args(q),
         f'python -m arxiv_search search "{q}" --mode hybrid')

    q2 = "efficient attention for long context"
    shot("04_search_filtered", "vector search + metadata filter", cli.cmd_search,
         search_args(q2, mode="vector", category=["cs.CL", "cs.LG"]),
         f'python -m arxiv_search search "{q2}" --mode vector --category cs.CL cs.LG')

    q3 = "BM25"
    for mode in ("keyword", "vector"):
        shot(f"05_mode_{mode}", f"{mode} search", cli.cmd_search, search_args(q3, mode=mode, limit=3),
             f'python -m arxiv_search search "{q3}" --mode {mode} --limit 3')

    shot("06_explain_nofilter", "query plan without filter", cli.cmd_explain,
         Namespace(query="retrieval augmented generation", limit=10, category=None, author=None,
                   since=None, until=None),
         'python -m arxiv_search explain "retrieval augmented generation" --limit 10')
    shot("06_explain_filter", "query plan with selective filter", cli.cmd_explain,
         Namespace(query="retrieval augmented generation", limit=10, category=["cs.CV"], author=None,
                   since=None, until=None, force_index=False),
         'python -m arxiv_search explain "retrieval augmented generation" --category cs.CV --limit 10')
    shot("06_explain_filter_hnsw", "filtered query forced onto HNSW", cli.cmd_explain,
         Namespace(query="retrieval augmented generation", limit=10, category=["cs.CV"], author=None,
                   since=None, until=None, force_index=True),
         'python -m arxiv_search explain "retrieval augmented generation" --category cs.CV --limit 10 --force-index')

    shot("07_bench", "benchmark", cli.cmd_bench,
         Namespace(queries=50, ef=[10, 40, 100], filter_query="retrieval augmented generation",
                   filter_category="cs.CV"),
         "python -m arxiv_search bench")


if __name__ == "__main__":
    main()
