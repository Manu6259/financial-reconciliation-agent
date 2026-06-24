# Case study: an AI system you can trust with the books

*A reconciliation & categorization agent for messy consumer-brand finance.*

This shows how I approach AI on top of real, messy financial data — accounting
exports, payout feeds, payroll — where the output has to be not just plausible
but *correct and auditable*. It runs end-to-end with no API key (offline mock)
and with a real LLM when you add one.

---

## The problem, in one paragraph

A consumer brand sells across its own site, Amazon, and wholesale. Money arrives
in lumps: one bank line might be "200 orders, net of fees, from two days ago,"
labelled `SHOPIFY PAYMENTS 0001 EDI PYMNT`. Someone has to (a) decide what every
transaction *is*, and (b) prove each deposit equals what the channel said it
owed, net of fees, refunds, and reserves. It's tedious, unforgiving, and the
data fights you: gross-vs-net mismatches, settlement lag, Amazon holding a
reserve, three payout files with three different schemas, an accounting export
full of `$1,234.56` strings and an "Ask My Accountant" bucket.

## The one design decision that matters

**The LLM is only allowed to judge. It is never allowed to do arithmetic.**

- *Judgment* (which category? which payout does this deposit belong to?) is
  fuzzy, language-shaped work — the LLM is good at it, and when it's wrong the
  evaluation catches it.
- *Arithmetic* (net = gross − fees − refunds; does the deposit match to the
  penny?) is done by plain, deterministic Python that produces an audit trail.
  A hallucinated number here wouldn't be caught by an eval — it would silently
  mis-state the P&L. That risk is unacceptable, so the model never touches it.

This split is the whole architecture, and it's the thing I'd defend in finance
AI generally: **don't ask the model to be a calculator; ask it to be a
classifier, and verify everything downstream.**

## What I built

A handful of small modules (see `README.md` for the map): a generator that
produces *deliberately* messy data plus held-out ground truth; a **148-passage
accounting knowledge base generated from the same source as the data**; a
categorization agent grounded by **RAG over that KB** (and over similar past
transactions), citing the policy rule it applied; a deterministic reconciliation
engine; evaluation harnesses; and a Streamlit dashboard. No fine-tuning —
intentionally. The senior move is showing you *don't* need to train a model; you
need good structure, retrieval, guardrails, and evaluation.

**Why the KB is generated from the data's own source:** the knowledge base and
the transactions both import the same vendor list, channels, and chart of
accounts, so the KB can never drift from the data it describes — every vendor
that can appear in a bank feed has matching policy passages. That alignment is
deliberate; a RAG corpus disconnected from the data is a common failure I wanted
to avoid.

## Grappling with the mess (the part that signals real experience)

Every failure mode below is reproduced in the synthetic data and handled
explicitly — this is the table I'd want a reviewer to read:

- **Gross vs. net** — payouts report gross; the bank sees net. The engine
  recomputes and verifies to the penny.
- **Settlement lag** — deposits land days later, so matching is over a date
  window, not an exact date.
- **Amazon reserve** — deposits sometimes come up short; the engine flags the
  exact shortfall as `partial_reserve` instead of forcing a wrong match. (It
  surfaced **$1,780.73** held back across the quarter in this run.)
- **Schema drift & unit traps** — three payout files, three schemas, Stripe in
  cents; all normalised at one boundary.
- **Dirty accounting export** — money-as-strings, messy account names, blank
  rows; parsed robustly, ambiguous accounts routed to review.
- **Ambiguity** — `AMZN MKTP US` (a purchase) vs `AMAZON SETTLEMENT` (a payout):
  a KB edge-case rule disambiguates by memo + direction; truly unresolvable cases
  abstain (`Needs Review`) rather than guess.

## Results (measured against held-out ground truth)

The eval grades on transactions the model never sees, and retrieval is built only
from non-test rows — no answer leakage. With **gpt-4o-mini** on 69 held-out
transactions:

| Metric | Result |
|---|---|
| Categorization accuracy, **no RAG** | 53.6% |
| Categorization accuracy, **KB RAG** | **100%** (**+46.4% lift from retrieval**) |
| Citation coverage / accuracy-when-cited | **100% / 100%** |
| Auto-post coverage at confidence ≥ 0.75 | ~100%, **100% accurate** on that slice |
| Reconciliation match accuracy | **100%** (37/37 deposits → correct payout) |
| Engine auto-match rate | 89.2% matched; the rest **flagged, never guessed** |

The **+46% lift is the real story**: alone, the LLM decodes cryptic memos like
`FACEBK *7H2K9` or `SHENZHEN MFG CO` only about half the time; given the retrieved
policy rule it's reliable — and every decision is **auditable**, citing the rule
(`Advertising, per kb-0054`). The reconciliation stays at 100% because it's
deterministic — the LLM is never in the numeric path.

**Honest caveat:** the KB is drawn from the same vendor distribution as the data,
so retrieval usually finds a near-exact rule → near-perfect accuracy. On genuinely
novel vendors it would be lower; the feedback loop (human corrections becoming new
KB entries) is how that gap closes. I'd rather state this than oversell the 100%.

## What's deliberately missing, and why

- **No fine-tuning** — unnecessary and a worse signal than disciplined prompting
  + retrieval + eval.
- **AR/AP and inventory** aren't built — they extend the same pattern (match an
  invoice to a payment; reconcile a 3PL inventory snapshot to recorded COGS) and
  I scoped to four sources deeply rather than seven shallowly.
- **The RAG layer is a clean v1.** Retrieval is isolated behind a small interface
  so it can be hardened — retrieval eval (recall@k/MRR), hybrid lexical+vector
  search, reranking, an ANN index, and a feedback loop where human corrections
  become new KB entries. I built exactly that hardening in the companion
  **filings-intelligence** project (hybrid retrieval + a measured retrieval eval +
  structured routing); the same upgrades apply here.

## What this demonstrates

A complete slice of real-world financial operations: a P&L spanning DTC / Amazon
/ wholesale, a bank feed reconciled against multiple processors, structured
handling of messy accounting exports, a multi-step tool-using agent, and — most
importantly — an **evaluation-first** posture, because in finance the expensive
failure isn't an outage, it's a confident wrong number. The same approach scales
directly to larger charts of accounts, more channels, and AR/AP and inventory.
