"""Thin provider wrapper around the LLM + embeddings.

Two reasons this layer exists:

1. **Provider independence.** Every model call goes through here, so swapping
   OpenAI for Anthropic (or anything else) is a one-file change. In a production
   financial system you do not want the provider hard-wired through your code.

2. **An offline mock.** If no API key is present (or APP_USE_MOCK=1), we fall
   back to a deterministic, rule-based "model". That keeps the whole project
   runnable, free, and reproducible for a reviewer with no key — and gives the
   eval harness a sensible baseline to beat.

Nothing here ever does financial arithmetic. The LLM's only jobs are
classification and free-text reasoning; numbers are the reconciliation engine's
responsibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("APP_LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("APP_EMBED_MODEL", "text-embedding-3-small")
_FORCE_MOCK = os.getenv("APP_USE_MOCK", "0") == "1"
_HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))

USING_MOCK = _FORCE_MOCK or not _HAS_KEY

_client = None
if not USING_MOCK:
    try:
        from openai import OpenAI

        _client = OpenAI()
    except Exception:  # pragma: no cover - missing dep / bad key falls back safely
        USING_MOCK = True


# --- Mock heuristics ----------------------------------------------------------
# A transparent keyword baseline. It is intentionally decent-but-imperfect so the
# eval can show the LLM (and RAG) adding measurable value on top of it.
_MOCK_RULES: list[tuple[str, str]] = [
    (r"shopify|stripe transfer", "Shopify Sales"),
    (r"amazon settlement|amzn settle", "Amazon Sales"),
    (r"facebk|google ads|klaviyo|tiktok|pinterest|influencer", "Advertising & Marketing"),
    (r"shipbob|usps|easypost|fedex|shipstation|flexport", "Shipping & Fulfillment"),
    (r"mfg|alibaba|packaging|ingredient|shenzhen", "Cost of Goods Sold"),
    (r"gsuite|qbooks|slack|figma|aws|notion|vercel|web services", "Software & SaaS"),
    (r"gusto pay|net direct dep", "Payroll Expense"),
    (r"gusto tax|irs 941", "Payroll Taxes"),
    (r"law|deloitte|upwork|atlas|cfo", "Professional Services"),
    (r"wework|staples|comcast|verizon|uber", "Office & Admin"),
    (r"transfer to savings|transfer from savings", "Internal Transfer"),
    (r"owner|distribution", "Owner Draw / Capital"),
]


def _mock_classify(description: str) -> tuple[str, float]:
    d = description.lower()
    for pattern, cat in _MOCK_RULES:
        if re.search(pattern, d):
            return cat, 0.9
    return "Needs Review", 0.2


def _rag_vote(retrieved: list[tuple] | None) -> tuple[str, float] | None:
    """Majority vote over retrieved (description, category, score) neighbours.

    This is how the offline mock *uses* the RAG memory: when the keyword rules
    abstain, near-identical past transactions resolve the label. Confidence
    scales with how strongly the neighbours agree.
    """
    if not retrieved:
        return None
    votes: dict[str, float] = {}
    for item in retrieved:
        cat, score = item[1], (item[2] if len(item) > 2 else 0.0)
        votes[cat] = votes.get(cat, 0.0) + max(score, 0.0)
    if not votes:
        return None
    best = max(votes, key=votes.get)
    share = votes[best] / sum(votes.values())
    return best, round(0.6 + 0.35 * share, 2)


def _stable_vector(text: str, dim: int = 256) -> list[float]:
    """Deterministic pseudo-embedding from a hash; good enough for the mock RAG."""
    out = []
    for i in range(dim):
        h = hashlib.sha256(f"{i}:{text}".encode()).digest()
        out.append((int.from_bytes(h[:4], "big") / 2**32) * 2 - 1)
    return out


# --- Public API ---------------------------------------------------------------

def classify(prompt: str, description: str, allowed: list[str],
             retrieved: list[tuple] | None = None) -> dict:
    """Return {category, confidence, rationale}. Pure judgment, no arithmetic.

    `retrieved` is the RAG neighbour list. The real LLM already sees it inside
    `prompt`; the offline mock consumes it here via a nearest-neighbour vote so
    the retrieval ablation is meaningful with no API key.
    """
    if USING_MOCK:
        cat, conf = _mock_classify(description)
        if conf < 0.5:  # keyword rules abstained -> let RAG memory try
            voted = _rag_vote(retrieved)
            if voted:
                return {"category": voted[0], "confidence": voted[1],
                        "rationale": "resolved by similar past transactions (mock RAG vote)"}
        return {"category": cat, "confidence": conf, "rationale": "keyword baseline (mock model)"}

    resp = _client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": description},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    cat = data.get("category", "Needs Review")
    if cat not in allowed and cat != "Needs Review":
        cat = "Needs Review"  # never let the model invent a category off-chart
    return {
        "category": cat,
        "confidence": float(data.get("confidence", 0.0)),
        "rationale": str(data.get("rationale", "")),
    }


def embed(texts: list[str]) -> list[list[float]]:
    if USING_MOCK:
        return [_stable_vector(t) for t in texts]
    resp = _client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]
