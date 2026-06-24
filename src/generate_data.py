"""Synthetic-but-realistic financial data generator.

This is the heart of the case study. Real financial data is messy in *specific*
ways, and the credibility of the whole project rests on reproducing those exact
failure modes rather than clean toy data. Every bit of mess injected here is
labelled in the README's "failure modes" table.

Sources produced (mirroring common real-world financial data types):
  - bank_feed.csv            a consumer brand's operating-account transactions
  - shopify_payouts.csv      channel revenue, netted of fees/refunds
  - amazon_payouts.csv       channel revenue, different schema, settlement reserve
  - stripe_payouts.csv       DTC card processor payouts
  - quickbooks_pl_export.csv  a messy accounting export (the kind you actually get)
  - payroll_register.csv     a payroll run (gross vs net vs employer taxes)

Ground truth produced (never shown to the model; used only by the eval harness):
  - golden_categories.csv        bank txn_id -> true category
  - golden_reconciliation.csv    bank deposit txn_id -> payout_id

Deterministic: a fixed seed makes every run identical and reviewable.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from datetime import date, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEED = 42
START = date(2025, 1, 1)
DAYS = 90  # one quarter

# --- Vendor / memo dictionaries: the cryptic strings a bank feed actually shows -
# Maps a true category to the kind of raw description that lands in the feed.
EXPENSE_VENDORS: dict[str, list[str]] = {
    "Software & SaaS": [
        "GOOGLE *GSUITE_acme", "INTUIT *QBOOKS", "SLACK T0288", "FIGMA MONTHLY",
        "AMAZON WEB SERVICES AWS", "NOTION LABS INC", "VERCEL INC",
    ],
    "Advertising & Marketing": [
        "FACEBK *7H2K9", "GOOGLE ADS 8842", "KLAVIYO INC", "TIKTOK ADS",
        "PINTEREST ADS 22", "INFLUENCER PAYOUT GRIN",
    ],
    "Shipping & Fulfillment": [
        "SHIPBOB INC", "USPS PB 8000", "EASYPOST", "FEDEX 7729", "SHIPSTATION",
        "FLEXPORT 3PL INV",
    ],
    "Cost of Goods Sold": [
        "SHENZHEN MFG CO", "ALIBABA *RAWMAT", "PACKAGING SUPPLY CO",
        "CONTRACT MFG ACH", "INGREDIENT SUPPLIER LLC",
    ],
    "Office & Admin": [
        "WEWORK MEMBERSHIP", "STAPLES 00471", "COMCAST BUSINESS", "VERIZON WRLS",
        "UBER *TRIP", "AMZN MKTP US*2H",  # intentionally ambiguous vs Amazon Sales
    ],
    "Professional Services": [
        "GUSTO LAW RETAINER", "DELOITTE TAX SVCS", "UPWORK *CONTRACTOR",
        "STRIPE ATLAS", "FRACTIONAL CFO LLC",
    ],
}

# Channels and their economics (fee rate, refund rate, settlement lag in days).
@dataclass
class Channel:
    name: str
    category: str
    fee_rate: float
    refund_rate: float
    lag_days: int
    bank_memo: str


CHANNELS = [
    Channel("Shopify", "Shopify Sales", 0.029, 0.04, 2, "SHOPIFY PAYMENTS"),
    Channel("Amazon", "Amazon Sales", 0.15, 0.06, 7, "AMAZON SETTLEMENT"),
    Channel("Stripe", "Shopify Sales", 0.029, 0.03, 2, "STRIPE TRANSFER"),
]


@dataclass
class Txn:
    txn_id: str
    date: date
    description: str
    amount: float  # signed: + deposit, - withdrawal
    true_category: str
    matched_payout_id: str | None = None


@dataclass
class Payout:
    payout_id: str
    channel: str
    payout_date: date  # when it hits the bank
    gross: float
    fees: float
    refunds: float
    order_count: int

    @property
    def net(self) -> float:
        return round(self.gross - self.fees - self.refunds, 2)


def _money(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 2)


def _d(offset: int) -> date:
    return START + timedelta(days=offset)


def build() -> dict:
    random.seed(SEED)
    txns: list[Txn] = []
    payouts: list[Payout] = []
    n = 0

    def next_id(prefix: str) -> str:
        nonlocal n
        n += 1
        return f"{prefix}{n:04d}"

    # --- Channel revenue -> payouts -> bank deposits --------------------------
    # Each channel batches many orders into a periodic payout. The payout NETS
    # fees and refunds (gross-vs-net mess) and lands in the bank a few days later
    # (settlement-lag mess). The bank only ever sees the net number + a memo.
    for ch in CHANNELS:
        day = 1
        while day < DAYS:
            order_count = random.randint(20, 120)
            gross = round(sum(_money(15, 180) for _ in range(order_count)), 2)
            fees = round(gross * ch.fee_rate, 2)
            refunds = round(gross * ch.refund_rate * random.uniform(0.3, 1.4), 2)
            pid = next_id("PO")
            settle_day = day + ch.lag_days
            if settle_day >= DAYS:
                break
            payouts.append(Payout(pid, ch.name, _d(settle_day), gross, fees, refunds, order_count))

            net = round(gross - fees - refunds, 2)
            # Amazon holds a rolling reserve: the bank deposit is sometimes a bit
            # less than `net`, with the remainder released later. This breaks
            # naive exact-amount matching and is a real Amazon behaviour.
            deposit = net
            if ch.name == "Amazon" and random.random() < 0.4:
                deposit = round(net * random.uniform(0.85, 0.95), 2)
            txns.append(Txn(
                txn_id=next_id("BT"),
                date=_d(settle_day),
                description=f"{ch.bank_memo} {pid[-4:]} EDI PYMNT",
                amount=deposit,
                true_category=ch.category,
                matched_payout_id=pid,
            ))
            day += random.randint(5, 9)  # roughly weekly payouts

    # --- Expenses (bank withdrawals) -----------------------------------------
    for _ in range(140):
        cat = random.choice(list(EXPENSE_VENDORS.keys()))
        memo = random.choice(EXPENSE_VENDORS[cat])
        amount = -_money(20, 4200)
        txns.append(Txn(
            txn_id=next_id("BT"),
            date=_d(random.randint(0, DAYS - 1)),
            description=memo,
            amount=amount,
            true_category=cat,
        ))

    # --- Payroll: shows up in the bank as a few large round-ish ACH debits ----
    payroll_rows = []
    for run_idx, pay_day in enumerate([14, 28, 42, 56, 70, 84]):
        employees = 8
        gross = round(sum(_money(2800, 7200) for _ in range(employees)), 2)
        taxes = round(gross * 0.0765, 2)  # employer FICA, simplified
        net = round(gross * 0.78, 2)      # after employee withholdings
        # Two separate bank debits: net pay to employees, taxes to the agency.
        txns.append(Txn(next_id("BT"), _d(pay_day), "GUSTO PAY xxNET DIRECT DEP",
                        -net, "Payroll Expense"))
        txns.append(Txn(next_id("BT"), _d(pay_day), "GUSTO TAX xxIRS 941",
                        -taxes, "Payroll Taxes"))
        payroll_rows.append({
            "run_id": f"RUN{run_idx + 1:02d}", "pay_date": _d(pay_day).isoformat(),
            "employees": employees, "gross_pay": gross,
            "employer_taxes": taxes, "net_pay": net,
        })

    # --- A couple of internal transfers and an owner draw (not on the P&L) -----
    txns.append(Txn(next_id("BT"), _d(30), "ONLINE TRANSFER TO SAVINGS xxxx8841",
                    -25000.0, "Internal Transfer"))
    txns.append(Txn(next_id("BT"), _d(60), "ONLINE TRANSFER FROM SAVINGS xxxx8841",
                    25000.0, "Internal Transfer"))
    txns.append(Txn(next_id("BT"), _d(75), "OWNER DISTRIBUTION ACH", -12000.0,
                    "Owner Draw / Capital"))

    txns.sort(key=lambda t: (t.date, t.txn_id))
    return {"txns": txns, "payouts": payouts, "payroll": payroll_rows}


# --- Writers: each emits the *format* (and mess) of its real-world source -----

def _running_balance(txns: list[Txn]) -> list[float]:
    bal, out = 50000.0, []
    for t in txns:
        bal = round(bal + t.amount, 2)
        out.append(bal)
    return out


def write_all(world: dict) -> None:
    import csv
    os.makedirs(DATA_DIR, exist_ok=True)
    txns: list[Txn] = world["txns"]
    payouts: list[Payout] = world["payouts"]

    # Bank feed: messy date formats + signed amounts as strings with commas.
    balances = _running_balance(txns)
    with open(os.path.join(DATA_DIR, "bank_feed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["txn_id", "date", "description", "amount", "balance"])
        for t, bal in zip(txns, balances):
            # Mix MM/DD/YYYY and YYYY-MM-DD to force real date parsing.
            ds = t.date.strftime("%m/%d/%Y") if t.txn_id[-1] in "02468" else t.date.isoformat()
            amt = f"{t.amount:,.2f}" if t.amount < 0 else f"{t.amount:,.2f}"
            w.writerow([t.txn_id, ds, t.description, amt, f"{bal:,.2f}"])

    # Per-channel payout files, each with a *different* schema on purpose.
    by_channel: dict[str, list[Payout]] = {}
    for p in payouts:
        by_channel.setdefault(p.channel, []).append(p)

    # Shopify: clean-ish, explicit net column.
    with open(os.path.join(DATA_DIR, "shopify_payouts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["payout_id", "payout_date", "gross_sales", "processing_fees", "refunds", "net_payout", "orders"])
        for p in by_channel.get("Shopify", []):
            w.writerow([p.payout_id, p.payout_date.isoformat(), p.gross, p.fees, p.refunds, p.net, p.order_count])

    # Amazon: different column names, NO net column (you must compute it), and a
    # "reserve_held" concept implied by mismatched deposits.
    with open(os.path.join(DATA_DIR, "amazon_payouts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["settlement_id", "date_initiated", "product_sales", "selling_fees", "refunded_amount", "units"])
        for p in by_channel.get("Amazon", []):
            w.writerow([p.payout_id, p.payout_date.isoformat(), p.gross, p.fees, p.refunds, p.order_count])

    # Stripe: yet another schema; amounts in cents (classic Stripe gotcha).
    with open(os.path.join(DATA_DIR, "stripe_payouts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "arrival_date", "amount_cents", "fee_cents", "refund_cents", "charge_count"])
        for p in by_channel.get("Stripe", []):
            w.writerow([p.payout_id, p.payout_date.isoformat(), int(p.gross * 100), int(p.fees * 100), int(p.refunds * 100), p.order_count])

    # QuickBooks-style P&L export: the messy accounting export. Inconsistent
    # account names, amounts as "$1,234.56" strings, blank subtotal rows.
    with open(os.path.join(DATA_DIR, "quickbooks_pl_export.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Account", "Type", "Amount"])
        rows = [
            ("Sales - Shopify", "Income", 184203.55),
            ("Amazon  Income", "Income", 96120.10),        # double space, vague name
            ("Sales:Wholesale", "Income", 22150.00),        # colon hierarchy
            ("Refunds/Returns", "Income", -8840.21),        # contra-revenue as negative income
            ("", "", None),                                  # blank subtotal row
            ("COGS", "Cost of Goods Sold", 71200.42),
            ("Merchant Fees", "Expense", 9123.88),          # processing fees mislabeled type
            ("Payroll", "Expense", 142000.00),
            ("Payroll Tax Expense", "Expense", 10863.00),
            ("Advertising & Promo", "Expense", 53400.75),
            ("Software Subscriptions", "Expense", 7820.40),
            ("Ask My Accountant", "Expense", 1450.00),      # the real-world "dunno" bucket
        ]
        for acct, typ, amt in rows:
            amt_str = "" if amt is None else (f"-${abs(amt):,.2f}" if amt < 0 else f"${amt:,.2f}")
            w.writerow([acct, typ, amt_str])

    # Payroll register.
    with open(os.path.join(DATA_DIR, "payroll_register.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "pay_date", "employees", "gross_pay", "employer_taxes", "net_pay"])
        for r in world["payroll"]:
            w.writerow([r["run_id"], r["pay_date"], r["employees"], r["gross_pay"], r["employer_taxes"], r["net_pay"]])

    # --- Ground truth (held out from the model) -------------------------------
    with open(os.path.join(DATA_DIR, "golden_categories.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["txn_id", "true_category"])
        for t in txns:
            w.writerow([t.txn_id, t.true_category])

    with open(os.path.join(DATA_DIR, "golden_reconciliation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["txn_id", "payout_id"])
        for t in txns:
            if t.matched_payout_id:
                w.writerow([t.txn_id, t.matched_payout_id])

    summary = {
        "transactions": len(txns),
        "payouts": len(payouts),
        "deposits_to_reconcile": sum(1 for t in txns if t.matched_payout_id),
        "categories_present": sorted({t.true_category for t in txns}),
    }
    with open(os.path.join(DATA_DIR, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Generated data:\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    write_all(build())
