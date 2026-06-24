"""Quantify the knowledge-base RAG lift.

Compares categorization on the held-out test set in two conditions:
  - no RAG  : the LLM sees only the category list.
  - KB RAG  : the LLM also sees the retrieved accounting-policy passages and
              cites the rule it relied on.

Reports accuracy for each, the lift, and citation coverage (how often a decision
is backed by a policy passage matching the chosen category). Graded against the
same held-out ground truth, with no answer leakage.

Run: python src/evaluate_kb.py
"""

from __future__ import annotations

import json
import os

from categorize import categorize_one, load_bank_feed
from evaluate import _golden_categories, _is_test
from model import USING_MOCK
from policy_rag import KnowledgeBaseIndex

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _run(test_rows, golden, kb) -> dict:
    correct = cited = cited_correct = 0
    for r in test_rows:
        truth = golden[r["txn_id"]]
        res = categorize_one(r["description"], kb=kb)
        ok = res["category"] == truth
        correct += ok
        if res.get("policy_basis"):
            cited += 1
            cited_correct += ok
    n = len(test_rows)
    return {
        "n": n,
        "accuracy": round(correct / n, 3) if n else 0.0,
        "citation_coverage": round(cited / n, 3) if n else 0.0,
        "accuracy_when_cited": round(cited_correct / cited, 3) if cited else None,
    }


def main() -> None:
    golden = _golden_categories()
    test_rows = [r for r in load_bank_feed() if _is_test(r["txn_id"])]
    kb = KnowledgeBaseIndex()
    print(f"Model: {'MOCK' if USING_MOCK else os.getenv('APP_LLM_MODEL')}")
    print(f"Test set: {len(test_rows)} held-out transactions | KB: {len(kb)} passages\n")

    print("Categorization — no RAG ...")
    off = _run(test_rows, golden, kb=None)
    print("Categorization — KB RAG ...")
    on = _run(test_rows, golden, kb=kb)

    lift = round(on["accuracy"] - off["accuracy"], 3)
    report = {"no_rag": off, "kb_rag": on, "lift": lift}
    with open(os.path.join(DATA_DIR, "kb_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n================ KB RAG RESULTS ================")
    print(f"Accuracy   no-RAG: {off['accuracy']:.1%}   KB-RAG: {on['accuracy']:.1%}   (lift {lift:+.1%})")
    print(f"Citation coverage: {on['citation_coverage']:.1%}  "
          f"accuracy when cited: "
          f"{on['accuracy_when_cited'] if on['accuracy_when_cited'] is None else format(on['accuracy_when_cited'], '.1%')}")
    print("===============================================")
    print("\nReport -> data/kb_metrics.json")


if __name__ == "__main__":
    main()
