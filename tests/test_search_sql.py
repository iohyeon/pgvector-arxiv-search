from datetime import date

from arxiv_search.db import normalize_author
from arxiv_search.pdf import _Block, _is_heading, order_blocks
from arxiv_search.search import Filters, build_filter, rrf


def test_build_filter_binds_every_value():
    where, params = build_filter(Filters(categories=("cs.CL",), authors=("Yoshua Bengio",),
                                         date_from=date(2026, 1, 1)))
    assert "%(f_categories)s" in where and "%(f_authors)s" in where and "%(f_from)s" in where
    assert "cs.CL" not in where  # values never interpolated into SQL text
    assert params["f_authors"] == ["yoshua bengio"]


def test_empty_filter_is_true():
    assert build_filter(Filters()) == ("TRUE", {})


def test_normalize_author():
    assert normalize_author("  Yoshua  Béngio ") == "yoshua bengio"
    assert normalize_author("J. R. R. Tolkien") == "j r r tolkien"


def test_rrf_rewards_agreement():
    vec = [{"chunk_id": 1, "paper_id": 1}, {"chunk_id": 2, "paper_id": 2}]
    kw = [{"chunk_id": 2, "paper_id": 2}, {"chunk_id": 3, "paper_id": 3}]
    ranked = [row["chunk_id"] for row, _ in rrf(vec, kw)]
    assert ranked[0] == 2  # second in vector, first in keyword -> best fused


def test_two_column_reading_order():
    b = lambda x0, y0, x1, t: _Block(1, x0, y0, x1, y0 + 10, t)  # noqa: E731
    blocks = [b(320, 100, 560, "R1"), b(50, 100, 290, "L1"), b(320, 200, 560, "R2"),
              b(50, 200, 290, "L2"), b(50, 40, 560, "TITLE")]
    assert [x.text for x in order_blocks(blocks, page_width=612)] == ["TITLE", "L1", "L2", "R1", "R2"]


def test_heading_detection():
    assert _is_heading("3 Method")
    assert _is_heading("4.2 Ablation Study")
    assert _is_heading("References")
    assert not _is_heading("We propose a new method for retrieval.")
    assert not _is_heading("A Matryoshka Hierarchical RAG for Efficient Multi-Hop Question Answering")
    assert not _is_heading("Method CoreQ SubQ")


def test_glued_heading_is_split():
    from arxiv_search.pdf import _split_glued_headings
    out = _split_glued_headings([_Block(2, 0, 0, 1, 1, "1 INTRODUCTION To invent is to discern, to choose.")])
    assert [b.text for b in out] == ["1 INTRODUCTION", "To invent is to discern, to choose."]
