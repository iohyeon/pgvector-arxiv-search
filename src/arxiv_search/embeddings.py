"""Embedding model wrapper (one model instance per process)."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.max_seq_length = self.model.max_seq_length
        self._tokenizer = self.model.tokenizer

    def count_tokens(self, text: str) -> int:
        # verbose=False: counting a long sentence is fine, we split it before embedding
        return len(self._tokenizer.encode(text, add_special_tokens=False, verbose=False))

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # L2-normalized, so cosine distance (<=>) and inner product rank identically.
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


@lru_cache(maxsize=2)
def get_embedder(model_name: str, device: str = "cpu") -> Embedder:
    return Embedder(model_name, device)
