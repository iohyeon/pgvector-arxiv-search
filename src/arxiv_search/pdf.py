"""PDF download and layout-aware text extraction for academic papers.

Extraction works on PyMuPDF text blocks (with coordinates) instead of the
plain page text, because two-column papers otherwise interleave lines from
the left and right columns.
"""

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import requests

from .arxiv_client import ArxivPaper


class DownloadError(Exception):
    pass


class PDFDownloader:
    """Downloads into <root>/<YYYY-MM>/<arxiv_id>.pdf, skipping valid files already on disk."""

    def __init__(self, root: Path, delay_seconds: float = 3.0, timeout: int = 60, max_retries: int = 3):
        self.root = root
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "pgvector-arxiv-search/0.1 (personal research tool)"

    def path_for(self, paper: ArxivPaper) -> Path:
        safe_id = paper.arxiv_id.replace("/", "_")  # old-style ids look like hep-th/9901001
        return self.root / f"{paper.published:%Y-%m}" / f"{safe_id}.pdf"

    def download(self, paper: ArxivPaper) -> Path:
        path = self.path_for(paper)
        if is_pdf(path):
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        url = paper.pdf_url.replace("http://", "https://", 1)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                if not resp.content.startswith(b"%PDF"):
                    raise DownloadError(f"not a PDF response ({resp.headers.get('content-type')})")
                tmp = path.with_suffix(".part")
                tmp.write_bytes(resp.content)
                tmp.replace(path)  # atomic: a crash never leaves a half-written .pdf
                time.sleep(self.delay_seconds)
                return path
            except (requests.RequestException, DownloadError) as exc:
                last_error = exc
                time.sleep(self.delay_seconds * attempt)
        raise DownloadError(f"{paper.arxiv_id}: {last_error}")


def is_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as f:
        return f.read(4) == b"%PDF"


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

@dataclass
class Section:
    title: str
    paragraphs: list[tuple[int, str]] = field(default_factory=list)  # (page, text)


@dataclass(frozen=True)
class _Block:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


_HEADING_NAMES = (
    "abstract|introduction|related work|background|preliminaries|method|methods|methodology|"
    "approach|model|experiments?|experimental setup|results|evaluation|analysis|discussion|"
    "conclusions?|limitations|future work|references|bibliography|acknowledg(?:e)?ments?|appendix"
)
# Numbered headings: "3 Method", "4.2. Ablation", "II. RELATED WORK". Letter prefixes ("A Proofs")
# are not accepted: they collide with titles like "A Matryoshka ..." and appendices come
# after References anyway.
_NUMBERED = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,2}|[IVX]{1,4})\.?\s+[A-Z][\w\- ,:&()]{1,70}$")
# Unnumbered headings must be exactly a known section name ("Method", not "Method CoreQ SubQ").
_NAMED = re.compile(rf"^(?:\d{{1,2}}\.?\s+)?(?:{_HEADING_NAMES})$", re.IGNORECASE)
# Some templates put an upper-case heading and the first sentence in one block:
# "1 INTRODUCTION To invent is to discern ..." -> ("1 INTRODUCTION", "To invent ...")
_GLUED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2}\.?\s+[A-Z][A-Z0-9\-:& ]{2,60}?)\s+(?=[A-Z][a-z])(.+)$")
_STOP = re.compile(r"^(?:\d{1,2}\.?\s+)?(?:references|bibliography)$", re.IGNORECASE)
_SKIP = re.compile(r"^(?:\d{1,2}\.?\s+)?(?:abstract|acknowledg(?:e)?ments?)\b", re.IGNORECASE)
_ARXIV_STAMP = re.compile(r"^arXiv:\d{4}\.\d{4,5}")
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")


class PDFExtractor:
    """Returns body sections between the first heading and References.

    Front matter (title, authors, affiliations) and the Abstract heading are
    skipped: the abstract is indexed from arXiv metadata, which is cleaner.
    """

    def extract(self, pdf_path: Path) -> list[Section]:
        with fitz.open(pdf_path) as doc:
            pages = [self._page_blocks(page, i) for i, page in enumerate(doc, start=1)]
        repeated = _repeated_lines(pages)

        sections: list[Section] = []
        current: Section | None = None
        for blocks in pages:
            for b in _split_glued_headings(blocks):
                if b.text in repeated or _is_noise(b.text):
                    continue
                if _is_heading(b.text):
                    if _STOP.match(b.text):
                        return [s for s in sections if s.paragraphs]
                    current = Section(title=b.text)
                    sections.append(current)
                    continue
                if current is not None and not _SKIP.match(current.title):
                    current.paragraphs.append((b.page, b.text))

        sections = [s for s in sections if s.paragraphs]
        if not sections:  # no recognizable headings: keep everything as one section
            body = [(b.page, b.text) for blocks in pages for b in blocks
                    if b.text not in repeated and not _is_noise(b.text)]
            sections = [Section(title="body", paragraphs=body)]
        return sections

    def _page_blocks(self, page: "fitz.Page", page_no: int) -> list[_Block]:
        blocks = [
            _Block(page_no, x0, y0, x1, y1, _clean(text))
            for (x0, y0, x1, y1, text, _bno, btype) in page.get_text("blocks")
            if btype == 0 and text.strip()
        ]
        return order_blocks(blocks, page.rect.width)


def order_blocks(blocks: list[_Block], page_width: float) -> list[_Block]:
    """Reading order for one- or two-column pages.

    Full-width blocks above the columns (title, abstract) come first, then the
    whole left column, then the right column, then full-width blocks below.
    """
    mid = page_width / 2
    left = [b for b in blocks if b.x1 <= mid + 15]
    right = [b for b in blocks if b.x0 >= mid - 15]
    if len(left) < 2 or len(right) < 2:
        return sorted(blocks, key=lambda b: (round(b.y0), b.x0))

    full = [b for b in blocks if b not in left and b not in right]
    top = min(b.y0 for b in left + right)
    by_y = lambda b: b.y0  # noqa: E731
    return (
        sorted([b for b in full if b.y0 < top], key=by_y)
        + sorted(left, key=by_y)
        + sorted(right, key=by_y)
        + sorted([b for b in full if b.y0 >= top], key=by_y)
    )


_BLOB = re.compile(r"\S{80,}")  # base64 / embedded LaTeX image data, long URLs


def _clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    text = text.replace("\n", " ")
    text = _BLOB.sub(" ", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_glued_headings(blocks: list[_Block]) -> list[_Block]:
    out = []
    for b in blocks:
        m = _GLUED_HEADING.match(b.text)
        if m:
            out.append(_Block(b.page, b.x0, b.y0, b.x1, b.y1, m.group(1).strip()))
            out.append(_Block(b.page, b.x0, b.y0, b.x1, b.y1, m.group(2).strip()))
        else:
            out.append(b)
    return out


def _is_heading(text: str) -> bool:
    if len(text) > 80 or len(text.split()) > 10 or text.endswith((".", ",")):
        return False
    return bool(_NAMED.match(text) or _NUMBERED.match(text))


def _is_noise(text: str) -> bool:
    return bool(_PAGE_NUMBER.match(text) or _ARXIV_STAMP.match(text)) or len(text) < 3


def _repeated_lines(pages: list[list[_Block]]) -> set[str]:
    """Running headers/footers: short lines that appear on at least a third of the pages."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for blocks in pages:
        counts.update({b.text for b in blocks if len(b.text) <= 80})
    threshold = max(2, len(pages) // 3)
    return {text for text, n in counts.items() if n >= threshold}
