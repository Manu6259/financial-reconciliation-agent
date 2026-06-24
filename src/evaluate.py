"""Evaluation harness — the part that turns "looks plausible" into "is measured".

In finance, a demo that runs is worthless without an answer to "how often is it
wrong, and what does it do when it's unsure?" This harness answers both, against
ground truth the model never sees.

It reports three things:
  1. Categorization accuracy, with a RAG ablation (memory ON vs OFF) so the
     design choice is proven, not asserted.
  2. Confidence-routed human-in-the-loop: at a confidence threshold, how much can
     we auto-post, and how accurate is that auto-posted slice? (Coverage vs
     precision is the real production lever.)
  3. Reconciliation accuracy: did the deterministic engine match each deposit to
     the correct payout?

Run: python src/evaluate.py
"""

from __future__ import annotations

import csv
import json
import os
import zlib
from collections import defaultdict

from categorize import build_memory_from_golden, categorize_one, load_bank_feed
from model import USING_MOCK
from reconcile import reconcile, summarize

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TEST_FRACTION = 0.35
AUTO_APPROVE_THRESHOLD = 0.75


def _golden_categories() -> dict[str, str]:
    with open(os.path.join(DATA_DIR, "golden_categories.csv"), newline="") as f:
        return {r["txn_id"]: r["true_category"] for r in csv.DictReader(f)}


def _is_test(txn_id: str) -> bool:
    """Deterministic, stable train/test split from the id hash."""
    return (zlib.crc32(txn_id.encode()) % 100) < int(TEST_FRACTION * 100)


def _run_categorization(test_rows, golden, memory) -> dict:
    correct = abstained = 0
    auto_n = auto_correct = 0
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for r in test_rows:
        truth = golden[r["txn_id"]]
        res = categorize_one(r["description"], memory=memory)
        pred, conf = res["category"], res["confidence"]
        if pred == "Needs Review":
            abstained += 1
        elif pred == truth:
            correct += 1
        confusion[(truth, pred)] += 1
        if conf >= AUTO_APPROVE_THRESHOLD and pred != "Needs Review":
            auto_n += 1
            if pred == truth:
                auto_correct += 1
    n = len(test_rows)
    return {
        "n": n,
        "accuracy": round(correct / n, 3) if n else 0.0,
        "abstention_rate": round(abstained / n, 3) if n else 0.0,
        "auto_approve_coverage": round(auto_n / n, 3) if n else 0.0,
        "auto_approve_accuracy": round(auto_correct / auto_n, 3) if auto_n else None,
        "_confusion": {f"{t} -> {p}": c for (t, p), c in confusion.items() if t != p},
    }


def _eval_reconciliation() -> dict:
    with open(os.path.join(DATA_DIR, "golden_reconciliation.csv"), newline="") as f:
        golden = {r["txn_id"]: r["payout_id"] for r in csv.DictReader(f)}
    matches = reconcile()
    considered = correct = 0
    for m in matches:
        if m.txn_id in golden:
            considered += 1
            if m.payout_id == golden[m.txn_id]:
                correct += 1
    return {
        "engine_summary": summarize(matches),
        "deposits_with_truth": considered,
        "correct_payout_matches": correct,
        "match_accuracy": round(correct / considered, 3) if considered else 0.0,
    }


def main() -> None:
    golden = _golden_categories()
    feed = load_bank_feed()
    test_rows = [r for r in feed if _is_test(r["txn_id"])]
    test_ids = {r["txn_id"] for r in test_rows}

    print(f"Model: {'MOCK (offline baseline)' if USING_MOCK else os.getenv('APP_LLM_MODEL')}")
    print(f"Test set: {len(test_rows)} held-out transactions\n")

    # RAG memory is built only from NON-test rows (no answer leakage).
    memory = build_memory_from_golden(holdout_ids=test_ids)
    print(f"RAG memory: {len(memory)} labelled examples\n")

    print("Categorization — RAG memory OFF ...")
    no_rag = _run_categorization(test_rows, golden, memory=None)
    print("Categorization — RAG memory ON  ...")
    with_rag = _run_categorization(test_rows, golden, memory=memory)

    recon = _eval_reconciliation()

    report = {
        "model": "mock" if USING_MOCK else os.getenv("APP_LLM_MODEL"),
        "categorization": {"rag_off": no_rag, "rag_on": with_rag},
        "reconciliation": recon,
    }
    with open(os.path.join(DATA_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    lift = round(with_rag["accuracy"] - no_rag["accuracy"], 3)
    print("\n================ RESULTS ================")
    print(f"Categorization accuracy   RAG off: {no_rag['accuracy']:.1%}   "
          f"RAG on: {with_rag['accuracy']:.1%}   (lift {lift:+.1%})")
    print(f"Auto-approve coverage (conf>={AUTO_APPROVE_THRESHOLD}): {with_rag['auto_approve_coverage']:.1%}  "
          f"accuracy on that slice: "
          f"{with_rag['auto_approve_accuracy'] if with_rag['auto_approve_accuracy'] is None else format(with_rag['auto_approve_accuracy'], '.1%')}")
    print(f"Reconciliation match accuracy: {recon['match_accuracy']:.1%} "
          f"on {recon['deposits_with_truth']} deposits")
    print(f"Auto-matched by engine: {recon['engine_summary']['auto_matched_pct']:.1f}%  "
          f"(rest flagged, not guessed)")
    print("=========================================")
    print("\nFull report written to data/metrics.json")


if __name__ == "__main__":
    main()
