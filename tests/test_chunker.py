from arxiv_search.chunker import TokenAwareChunker, split_sentences
from arxiv_search.pdf import Section


def words(text: str) -> int:  # stand-in tokenizer: 1 token per word
    return len(text.split())


def sentence(n: int, tag: str) -> str:
    return " ".join([tag] * (n - 1)) + " end."


def test_no_chunk_exceeds_target():
    text = " ".join(sentence(30, f"S{i}") for i in range(20))
    chunks = TokenAwareChunker(words, target_tokens=100, overlap=0.2).chunk_text(text, "body")
    assert chunks
    assert max(c.tokens for c in chunks) <= 100


def test_consecutive_chunks_overlap():
    sents = [sentence(15, f"S{i}") for i in range(12)]
    chunks = TokenAwareChunker(words, target_tokens=60, overlap=0.3).chunk_text(" ".join(sents), "body")
    for a, b in zip(chunks, chunks[1:]):
        last_sentence_of_a = split_sentences(a.text)[-1]
        assert b.text.startswith(last_sentence_of_a)


def test_oversized_sentence_is_split_on_words():
    long = " ".join(["w"] * 250) + "."
    chunks = TokenAwareChunker(words, target_tokens=100, overlap=0.0, min_tokens=1).chunk_text(long, "body")
    assert len(chunks) == 3
    assert all(c.tokens <= 100 for c in chunks)


def test_chunks_do_not_cross_sections_and_keep_page():
    sections = [
        Section("1 Introduction", [(2, sentence(40, "intro"))]),
        Section("2 Method", [(3, sentence(40, "method"))]),
    ]
    chunks = TokenAwareChunker(words, target_tokens=100).chunk_sections(sections)
    assert [(c.section, c.page) for c in chunks] == [("1 Introduction", 2), ("2 Method", 3)]


def test_tiny_fragments_are_dropped():
    chunks = TokenAwareChunker(words, target_tokens=100, min_tokens=20).chunk_text("Too short.", "body")
    assert chunks == []


def test_limit_holds_for_huge_unbroken_token():
    blob = "sha1_base64=" + "A" * 3000
    chars = lambda s: max(1, len(s) // 4)  # noqa: E731  (BPE-like: ~4 chars per token)
    chunks = TokenAwareChunker(chars, target_tokens=100, overlap=0.2, min_tokens=1).chunk_text(
        f"Intro sentence here. {blob} Tail sentence.", "body")
    assert chunks and max(c.tokens for c in chunks) <= 100
