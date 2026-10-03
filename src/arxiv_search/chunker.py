"""Token-aware, sentence-preserving chunking.

Chunk size is measured with the embedding model's own tokenizer, not words or
characters: all-MiniLM-L6-v2 truncates input at 256 tokens, so a chunk that
looks fine in characters can silently lose its tail at embedding time.
"""

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from .pdf import Section

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class Chunk:
    section: str
    page: int | None
    text: str
    tokens: int


class TokenAwareChunker:
    def __init__(self, count_tokens: TokenCounter, target_tokens: int = 200,
                 overlap: float = 0.2, min_tokens: int = 20):
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        self.count = count_tokens
        self.target = target_tokens
        self.overlap_budget = int(target_tokens * overlap)
        self.min_tokens = min_tokens

    def chunk_text(self, text: str, section: str, page: int | None = None) -> list[Chunk]:
        return self._pack(((page, s) for s in split_sentences(text)), section)

    def chunk_sections(self, sections: Iterable[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for sec in sections:  # never let a chunk span two sections
            units = ((page, s) for page, para in sec.paragraphs for s in split_sentences(para))
            chunks.extend(self._pack(units, sec.title))
        return chunks

    def _pack(self, units: Iterable[tuple[int | None, str]], section: str) -> list[Chunk]:
        out: list[Chunk] = []
        cur: list[tuple[int | None, str, int]] = []  # (page, sentence, tokens)
        cur_tokens = 0

        def emit() -> None:
            if cur and cur_tokens >= self.min_tokens:
                out.append(Chunk(section, cur[0][0], " ".join(s for _, s, _ in cur), cur_tokens))

        for page, sentence in units:
            for piece in self._split_oversized(sentence):
                n = self.count(piece)
                if cur and cur_tokens + n > self.target:
                    emit()
                    # carry the tail sentences into the next chunk (sliding window)
                    tail, tail_tokens = [], 0
                    for item in reversed(cur):
                        if tail_tokens + item[2] > self.overlap_budget:
                            break
                        tail.insert(0, item)
                        tail_tokens += item[2]
                    cur, cur_tokens = tail, tail_tokens
                    if cur_tokens + n > self.target:  # overlap would not fit next to this piece
                        cur, cur_tokens = [], 0
                cur.append((page, piece, n))
                cur_tokens += n
        emit()
        return out

    def _split_oversized(self, sentence: str) -> list[str]:
        """Sentences longer than the target (tables, equations) are cut on word boundaries.

        A single "word" over the target (base64, hashes) is cut by characters, so the
        token limit holds for any input.
        """
        if self.count(sentence) <= self.target:
            return [sentence]
        pieces, buf = [], []
        words = []
        for word in sentence.split():
            words.extend(self._split_word(word) if self.count(word) > self.target else [word])
        for word in words:
            if buf and self.count(" ".join(buf + [word])) > self.target:
                pieces.append(" ".join(buf))
                buf = []
            buf.append(word)
        if buf:
            pieces.append(" ".join(buf))
        return pieces

    def _split_word(self, word: str) -> list[str]:
        parts, start = [], 0
        while start < len(word):
            end = len(word)
            while self.count(word[start:end]) > self.target:  # shrink until it fits
                end = start + max(1, (end - start) // 2)
            parts.append(word[start:end])
            start = end
        return parts


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
