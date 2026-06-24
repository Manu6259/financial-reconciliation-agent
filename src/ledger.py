"""The posted ledger — structured financials the agent's tools query.

After transactions are categorized and deposits reconciled, the result is a
clean ledger: each transaction with its account and income-statement section.
The agent answers numeric questions by querying THIS structured ledger with
deterministic aggregation — never by asking the LLM to add things up. Same
principle as the rest of the system: the model decides *what* to look up; code
produces the *numbers*.

The ledger is built from the bank feed joined to the posted categories. Amounts
are summed in Python; the agent only ever reads the totals.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

from schema import CATEGORIES, COGS, OPEX, REVENUE, income_statement_section

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load_categories() -> dict[str, str]:
    path = os.path.join(DATA_DIR, "posted_categories.csv")
    if not os.path.exists(path):  # fall back to the reconciled/posted truth
        path = os.path.join(DATA_DIR, "golden_categories.csv")
    with open(path, newline="") as f:
        return {r["txn_id"]: r["true_category"] if "true_category" in r else r["category"]
                for r in csv.DictReader(f)}


def _money(s: str) -> float:
    return float(s.replace(",", "").replace("$", "")) if s else 0.0


def load_ledger() -> list[dict]:
    cats = _load_categories()
    rows = []
    with open(os.path.join(DATA_DIR, "bank_feed.csv"), newline="") as f:
        for r in csv.DictReader(f):
            cat = cats.get(r["txn_id"], "Needs Review")
            rows.append({
                "txn_id": r["txn_id"], "date": r["date"], "month": _month(r["date"]),
                "description": r["description"], "amount": _money(r["amount"]),
                "category": cat, "section": income_statement_section(cat),
            })
    return rows


def _month(date_str: str) -> str:
    # handles MM/DD/YYYY and YYYY-MM-DD
    if "/" in date_str:
        m, _, y = date_str.split("/")
        return f"{y}-{int(m):02d}"
    return date_str[:7]


# --- Aggregation tools (deterministic) ---------------------------------------

def revenue_by_channel(month: str | None = None) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for r in load_ledger():
        if r["section"] != REVENUE:
            continue
        if month and r["month"] != month:
            continue
        out[r["category"]] += r["amount"]
    return {k: round(v, 2) for k, v in sorted(out.items())}


def total_by_category(category: str, month: str | None = None) -> float:
    total = sum(r["amount"] for r in load_ledger()
                if r["category"].lower() == category.lower()
                and (not month or r["month"] == month))
    return round(total, 2)


def pnl_summary(month: str | None = None) -> dict[str, float]:
    rows = [r for r in load_ledger() if not month or r["month"] == month]
    rev = sum(r["amount"] for r in rows if r["section"] == REVENUE)
    cogs = sum(r["amount"] for r in rows if r["section"] == COGS)
    opex = sum(r["amount"] for r in rows if r["section"] == OPEX)
    return {
        "revenue": round(rev, 2),
        "cogs": round(cogs, 2),
        "gross_profit": round(rev + cogs, 2),       # cogs is negative (outflow)
        "operating_expense": round(opex, 2),
        "operating_income": round(rev + cogs + opex, 2),
    }


def months_available() -> list[str]:
    return sorted({r["month"] for r in load_ledger()})


if __name__ == "__main__":
    import json
    print("Months:", months_available())
    print("Revenue by channel:", json.dumps(revenue_by_channel(), indent=2))
    print("P&L:", json.dumps(pnl_summary(), indent=2))
