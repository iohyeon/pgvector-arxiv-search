"""arXiv metadata access.

Wraps the `arxiv` package client, which already handles paging, the delay
between API pages and retries. This module only converts results into an
immutable `ArxivPaper` and builds category/date queries.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterator, Sequence

import arxiv

_VERSION_SUFFIX = re.compile(r"v\d+$")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class ArxivPaper:
    arxiv_id: str  # version-less, so re-ingesting v2 updates the same row
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    primary_category: str
    published: date
    updated: date
    pdf_url: str


class ArxivClient:
    def __init__(self, delay_seconds: float = 3.0, page_size: int = 100):
        self._client = arxiv.Client(page_size=page_size, delay_seconds=delay_seconds, num_retries=3)

    def search(self, query: str, max_results: int) -> Iterator[ArxivPaper]:
        """Raw arXiv query syntax, e.g. 'cat:cs.IR AND abs:retrieval'."""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        for result in self._client.results(search):
            yield self._to_paper(result)

    def recent(self, categories: Sequence[str], days: int, max_results: int) -> Iterator[ArxivPaper]:
        """Papers submitted to any of `categories` in the last `days` days."""
        return self.search(recent_query(categories, days, today=date.today()), max_results)

    @staticmethod
    def _to_paper(r: "arxiv.Result") -> ArxivPaper:
        return ArxivPaper(
            arxiv_id=_VERSION_SUFFIX.sub("", r.get_short_id()),
            title=_WS.sub(" ", r.title).strip(),
            abstract=_WS.sub(" ", r.summary).strip(),
            authors=tuple(a.name for a in r.authors),
            categories=tuple(r.categories),
            primary_category=r.primary_category,
            published=r.published.date(),
            updated=r.updated.date(),
            pdf_url=r.pdf_url,
        )


def recent_query(categories: Sequence[str], days: int, today: date) -> str:
    cats = " OR ".join(f"cat:{c}" for c in categories)
    start = today - timedelta(days=days)
    return f"({cats}) AND submittedDate:[{start:%Y%m%d}0000 TO {today:%Y%m%d}2359]"
