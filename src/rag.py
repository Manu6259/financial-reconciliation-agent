"""Retrieval memory for transaction categorization.

The agent gets better when it can see how *similar past transactions* were
labelled. This module owns that retrieval entirely, behind a small interface
(`build`, `retrieve`). It is deliberately isolated so the retrieval strategy can
be upgraded to production grade later — better representations, hybrid
lexical+vector search, a reranker, an ANN index, caching — without touching the
agent or the eval.

v1 (here): in-memory embeddings + exact cosine top-k. Honest and simple. The
README's "production roadmap" section lists exactly what changes from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from model import embed


@dataclass
class Example:
    description: str
    category: str


@dataclass
class Retrieved:
    description: str
    category: str
    score: float


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class MemoryIndex:
    """A tiny vector store of (description -> category) memories."""

    def __init__(self) -> None:
        self._examples: list[Example] = []
        self._vectors: list[list[float]] = []

    def build(self, examples: list[Example]) -> "MemoryIndex":
        self._examples = list(examples)
        if self._examples:
            self._vectors = embed([e.description for e in self._examples])
        return self

    def retrieve(self, description: str, k: int = 5) -> list[Retrieved]:
        if not self._examples:
            return []
        q = embed([description])[0]
        scored = [
            Retrieved(e.description, e.category, _cosine(q, v))
            for e, v in zip(self._examples, self._vectors)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self._examples)
