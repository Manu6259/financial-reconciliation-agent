"""A financial-operations agent — multi-step planning over real tools.

This is the agentic layer. Given a natural-language question about the business's
finances, the agent decides which tools to call (it may call several), executes
them, reasons over the results, and composes an answer. The tools return exact
numbers from the structured ledger and cited rules from the policy KB — the LLM
plans and explains, but never invents a figure.

Example: "What was my Amazon revenue, and is my operating income positive?"
  → the agent calls get_revenue_by_channel AND get_pnl_summary, then answers from
    both. A single-shot prompt can't do that; planning across tools is the point.

Tools:
  - get_pnl_summary(month?)        — income-statement rollup from the ledger
  - get_revenue_by_channel(month?) — channel-level revenue
  - get_total_by_category(category, month?)
  - search_accounting_policy(query)— RAG over the accounting KB (returns cited rules)
  - list_months()

Runs on OpenAI tool-calling; falls back to a deterministic keyword plan offline.
"""

from __future__ import annotations

import json
import os

import ledger
from model import USING_MOCK
from policy_rag import KnowledgeBaseIndex

try:
    from model import _client  # reuse the configured OpenAI client
except Exception:
    _client = None

_kb: KnowledgeBaseIndex | None = None


def _policy(query: str) -> list[dict]:
    global _kb
    if _kb is None:
        _kb = KnowledgeBaseIndex()
    return [{"doc_id": d["doc_id"], "title": d["title"], "text": d["text"][:240]}
            for d in _kb.retrieve(query, k=3)]


# name -> (python impl, OpenAI schema)
TOOLS_IMPL = {
    "get_pnl_summary": lambda month=None: ledger.pnl_summary(month),
    "get_revenue_by_channel": lambda month=None: ledger.revenue_by_channel(month),
    "get_total_by_category": lambda category, month=None: ledger.total_by_category(category, month),
    "search_accounting_policy": lambda query: _policy(query),
    "list_months": lambda: ledger.months_available(),
}

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_pnl_summary",
        "description": "Income-statement rollup: revenue, COGS, gross profit, operating expense, operating income. Optional month 'YYYY-MM'.",
        "parameters": {"type": "object", "properties": {"month": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "get_revenue_by_channel",
        "description": "Revenue broken down by sales channel (Shopify, Amazon, …). Optional month 'YYYY-MM'.",
        "parameters": {"type": "object", "properties": {"month": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "get_total_by_category",
        "description": "Total amount posted to a chart-of-accounts category (e.g. 'Advertising & Marketing').",
        "parameters": {"type": "object",
                       "properties": {"category": {"type": "string"}, "month": {"type": "string"}},
                       "required": ["category"]}}},
    {"type": "function", "function": {
        "name": "search_accounting_policy",
        "description": "Retrieve accounting-policy rules from the knowledge base to explain a treatment. Returns cited rules.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_months", "description": "List the months that have ledger data.",
        "parameters": {"type": "object", "properties": {}}}},
]

_SYSTEM = ("You are a financial-operations assistant for a consumer brand. Answer "
           "questions by calling the provided tools — they return exact numbers from "
           "the posted ledger and cited accounting rules. Never compute or guess a "
           "figure yourself; read it from a tool result. When you state a number, say "
           "where it came from. Be concise.")


def run(question: str, max_steps: int = 5) -> dict:
    """Return {answer, trace} where trace lists the tool calls the agent made."""
    if USING_MOCK or _client is None:
        return _mock_run(question)

    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": question}]
    trace = []
    for _ in range(max_steps):
        resp = _client.chat.completions.create(
            model=os.getenv("APP_LLM_MODEL", "gpt-4o-mini"),
            temperature=0, messages=messages, tools=TOOLS_SCHEMA)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"answer": msg.content or "", "trace": trace}
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = TOOLS_IMPL[call.function.name](**args)
            trace.append({"tool": call.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result)})
    return {"answer": "(stopped: max planning steps reached)", "trace": trace}


def _mock_run(question: str) -> dict:
    """Deterministic offline plan: keyword → one or two tools + templated answer."""
    q = question.lower()
    trace = []
    if "revenue" in q or "channel" in q or "amazon" in q or "shopify" in q:
        res = ledger.revenue_by_channel()
        trace.append({"tool": "get_revenue_by_channel", "args": {}, "result": res})
    if "margin" in q or "income" in q or "profit" in q or "p&l" in q or "pnl" in q or not trace:
        res = ledger.pnl_summary()
        trace.append({"tool": "get_pnl_summary", "args": {}, "result": res})
    answer = "(offline mock plan) Pulled from the ledger: " + "; ".join(
        f"{t['tool']} → {t['result']}" for t in trace)
    return {"answer": answer, "trace": trace}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else \
        "What was my revenue by channel, and is operating income positive or negative?"
    out = run(q)
    print("Q:", q, "\n")
    for t in out["trace"]:
        print(f"  🔧 {t['tool']}({t['args']}) → {json.dumps(t['result'])[:120]}")
    print("\nA:", out["answer"])
