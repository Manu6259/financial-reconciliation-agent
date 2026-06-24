"""Streamlit dashboard — makes the system tangible in 30 seconds.

Three views that mirror how a finance operator would actually use this:
  - Categorization: every transaction, its predicted category, confidence, and
    the similar past transactions that informed it (the RAG memory, made visible).
  - Reconciliation: each deposit matched to its payout, with penny-level
    discrepancies and the audit trail.
  - Review queue: exactly the items the system was NOT confident enough to
    auto-post — the human-in-the-loop surface.

Run:  streamlit run src/app.py
By default it uses the offline mock model so it's instant and free; set an
OPENAI_API_KEY in .env to run the real agent.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

import agent as ops_agent
from categorize import build_memory_from_golden, categorize_one, load_bank_feed
from model import USING_MOCK
from policy_rag import KnowledgeBaseIndex
from reconcile import reconcile, summarize
from schema import income_statement_section

AUTO_APPROVE = 0.75

st.set_page_config(page_title="AI Reconciliation Agent", layout="wide")


@st.cache_data(show_spinner="Categorizing transactions…")
def categorize_all() -> pd.DataFrame:
    feed = load_bank_feed()
    memory = build_memory_from_golden(holdout_ids=set())  # demo: use all history
    kb = KnowledgeBaseIndex()                             # RAG over accounting policy
    rows = []
    for r in feed:
        res = categorize_one(r["description"], memory=memory, kb=kb)
        pb = res.get("policy_basis")
        rows.append({
            "txn_id": r["txn_id"], "date": r["date"], "description": r["description"],
            "amount": r["amount"], "category": res["category"],
            "confidence": round(res["confidence"], 2),
            "section": income_statement_section(res["category"]),
            "auto_post": res["confidence"] >= AUTO_APPROVE and res["category"] != "Needs Review",
            "cited_rule": f'{pb["doc_id"]}: {pb["title"]}' if pb else "—",
            "rationale": res["rationale"],
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Reconciling deposits…")
def reconcile_all() -> tuple[pd.DataFrame, dict]:
    ms = reconcile()
    df = pd.DataFrame([{
        "txn_id": m.txn_id, "deposit_amount": m.deposit_amount, "payout_id": m.payout_id,
        "expected_net": m.expected_net, "discrepancy": m.discrepancy,
        "status": m.status, "note": m.note,
    } for m in ms])
    return df, summarize(ms)


st.title("AI Reconciliation & Categorization Agent")
mode = "🔌 Offline mock model" if USING_MOCK else f"🤖 {os.getenv('APP_LLM_MODEL')}"
st.caption(f"{mode}  ·  The LLM judges & plans; all arithmetic & matching is deterministic code.")

cat_df = categorize_all()
rec_df, rec_summary = reconcile_all()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📂 Categorization", "🔗 Reconciliation", "🙋 Review Queue", "📊 Income Statement", "🤖 Ask the Agent"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", len(cat_df))
    c2.metric("Auto-postable", f"{cat_df['auto_post'].mean():.0%}")
    c3.metric("Need review", int((~cat_df["auto_post"]).sum()))
    st.caption("`cited_rule` shows the accounting-policy passage (from the 148-passage KB) "
               "the RAG layer retrieved to justify each categorization — every decision is auditable.")
    st.dataframe(cat_df, use_container_width=True, hide_index=True)

with tab2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Deposits", rec_summary["deposits_examined"])
    c2.metric("Auto-matched", f"{rec_summary['auto_matched_pct']:.0f}%")
    c3.metric("Reserve / short held", f"${rec_summary['reserve_or_short_held']:,.2f}")
    st.caption("Discrepancy is `deposit − expected_net`, computed to the penny. "
               "Non-zero rows are flagged, never silently accepted.")
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

with tab3:
    review = cat_df[~cat_df["auto_post"]]
    unmatched = rec_df[rec_df["status"] != "matched"]
    st.subheader(f"Low-confidence categorizations ({len(review)})")
    st.caption("The system declined to auto-post these — a cheap 'I'm not sure' beats an expensive wrong post.")
    st.dataframe(review[["txn_id", "description", "category", "confidence", "rationale"]],
                 use_container_width=True, hide_index=True)
    st.subheader(f"Deposits needing a human look ({len(unmatched)})")
    st.dataframe(unmatched, use_container_width=True, hide_index=True)

with tab4:
    st.caption("Rolled up from auto-categorized transactions. Numbers are summed in code, not by the model.")
    signed = cat_df.copy()
    signed["amount_num"] = signed["amount"].str.replace(",", "", regex=False).astype(float)
    pnl = signed.groupby("section")["amount_num"].sum().reindex(
        ["Revenue", "Cost of Goods Sold", "Operating Expense", "Other / Below the line"]).fillna(0)
    st.dataframe(pnl.rename("amount").reset_index().rename(columns={"section": "P&L section"}),
                 use_container_width=True, hide_index=True)
    st.metric("Operating income (rough)", f"${(pnl.get('Revenue', 0) + pnl.get('Cost of Goods Sold', 0) + pnl.get('Operating Expense', 0)):,.2f}")

with tab5:
    st.caption("A multi-step agent: it plans which tools to call (ledger lookups, policy search), "
               "runs them, and composes the answer. Numbers come from tools — never invented by the LLM.")
    q = st.text_input("Ask about the business's finances",
                      "What was my revenue by channel, and is operating income positive or negative?")
    if st.button("Ask the agent") and q:
        with st.spinner("Agent planning & calling tools…"):
            out = ops_agent.run(q)
        st.markdown("**Answer**")
        st.write(out["answer"])
        if out["trace"]:
            st.markdown("**Tool calls the agent made**")
            for t in out["trace"]:
                st.code(f"{t['tool']}({t['args']}) → {t['result']}", language="json")
