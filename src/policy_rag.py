"""RAG over the accounting knowledge base.

Retrieves the policy passages relevant to a transaction so the categorization
agent can ground its decision in written rules — and cite them. This is the
"explainable bookkeeping" angle: the agent doesn't just say "Advertising", it
says "Advertising, per rule kb-0042: memos containing FACEBK are Meta ads."

Embeddings come from the shared model wrapper (OpenAI, or the offline mock), and
are cached to disk so we embed the KB once. Kept isolated so the retrieval
strategy can be hardened later (hybrid lexical+vector, reranking) exactly like
the filings project.
"""

from __future__ import annotations

import os

import numpy as np

from knowledge_base import load
from model import embed

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
EMB_CACHE = os.path.join(DATA_DIR, "kb_embeddings.npy")


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


class KnowledgeBaseIndex:
    def __init__(self) -> None:
        self.docs = load()
        self.vecs = _normalize(self._embed())

    def _embed(self) -> np.ndarray:
        texts = [f"{d['title']}. {d['text']}" for d in self.docs]
        probe_dim = len(embed(["dimension probe"])[0])  # detects backend (real vs mock)
        if os.path.exists(EMB_CACHE):
            cached = np.load(EMB_CACHE)
            if cached.shape[0] == len(texts) and cached.shape[1] == probe_dim:
                return cached
        vecs = np.asarray(embed(texts), dtype="float32")
        np.save(EMB_CACHE, vecs)
        return vecs

    def retrieve(self, description: str, k: int = 4) -> list[dict]:
        qv = _normalize(np.asarray(embed([description]), dtype="float32"))[0]
        sims = self.vecs @ qv
        out = []
        for i in np.argsort(-sims)[:k]:
            d = dict(self.docs[i])
            d["score"] = round(float(sims[i]), 3)
            out.append(d)
        return out

    def __len__(self) -> int:
        return len(self.docs)


if __name__ == "__main__":
    import sys
    kb = KnowledgeBaseIndex()
    q = sys.argv[1] if len(sys.argv) > 1 else "FACEBK *7H2K9"
    print(f"KB size: {len(kb)} passages\nTop policy for: {q!r}\n")
    for d in kb.retrieve(q, k=4):
        print(f"  [{d['doc_id']}] {d['score']}  {d['category']:24} {d['title']}")
