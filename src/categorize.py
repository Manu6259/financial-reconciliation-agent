"""Categorization agent: the *judgment* half of the system.

Given a raw bank-feed description, decide which chart-of-accounts category it
belongs to, with a confidence and a short rationale. The agent may retrieve
similar past-labelled transactions (RAG memory) to ground its decision, and may
abstain with "Needs Review" rather than guess.

What this module never does: arithmetic. It does not sum, net, or reconcile
anything. Numbers belong to reconcile.py.
"""

from __future__ import annotations

import csv
import os

from model import classify
from rag import Example, MemoryIndex, Retrieved
from schema import CATEGORY_NAMES, UNKNOWN

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

_BASE_PROMPT = """You are a bookkeeping assistant for a consumer brand. You map a single raw \
bank-transaction description to exactly one account category.

You are ONLY classifying. Do not compute, sum, or invent any numbers.

Allowed categories (use these names exactly):
{categories}

If the description is genuinely ambiguous or matches nothing, return \
"{unknown}" with low confidence — a flagged unknown is far cheaper than a \
confident wrong answer.

Respond as JSON: {{"category": "...", "confidence": 0.0-1.0, "rationale": "one short sentence"}}"""


def _build_prompt(retrieved: list[Retrieved], policy: list[dict]) -> str:
    base = _BASE_PROMPT.format(categories="\n".join(f"- {c}" for c in CATEGORY_NAMES), unknown=UNKNOWN)
    if policy:
        rules = "\n".join(f"- [{d['doc_id']}] {d['text']}" for d in policy)
        base += ("\n\nRelevant accounting policy (apply these rules; cite the doc_id you rely on):\n"
                 + rules)
    if retrieved:
        lines = "\n".join(f'- "{r.description}" -> {r.category}' for r in retrieved)
        base += ("\n\nFor reference, similar past transactions were categorized "
                 "(hint, not a rule):\n" + lines)
    return base


def _policy_basis(policy: list[dict], predicted: str) -> dict | None:
    """The KB passage that supports the decision: top one matching the predicted
    category, else the top retrieved passage. Makes the call auditable."""
    if not policy:
        return None
    for d in policy:
        if d.get("category") == predicted:
            return {"doc_id": d["doc_id"], "title": d["title"], "score": d.get("score")}
    top = policy[0]
    return {"doc_id": top["doc_id"], "title": top["title"], "score": top.get("score")}


def categorize_one(description: str, memory: MemoryIndex | None = None,
                   kb=None, k: int = 5) -> dict:
    retrieved = memory.retrieve(description, k=k) if memory else []
    policy = kb.retrieve(description, k=4) if kb else []
    prompt = _build_prompt(retrieved, policy)
    hints = [(r.description, r.category, round(r.score, 3)) for r in retrieved]
    result = classify(prompt, description, CATEGORY_NAMES, retrieved=hints)
    result["retrieved"] = hints
    result["policy_basis"] = _policy_basis(policy, result["category"])
    return result


def load_bank_feed() -> list[dict]:
    path = os.path.join(DATA_DIR, "bank_feed.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_memory_from_golden(holdout_ids: set[str]) -> MemoryIndex:
    """Build RAG memory from labelled history, EXCLUDING the rows we'll test on.

    This avoids the cardinal eval sin of letting the answer leak into retrieval.
    """
    feed = {r["txn_id"]: r["description"] for r in load_bank_feed()}
    examples: list[Example] = []
    with open(os.path.join(DATA_DIR, "golden_categories.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["txn_id"] in holdout_ids:
                continue
            desc = feed.get(row["txn_id"])
            if desc:
                examples.append(Example(desc, row["true_category"]))
    return MemoryIndex().build(examples)
