"""Shared chart of accounts and category definitions.

One source of truth for the categories the agent is allowed to assign and the
reconciliation engine understands. Keeping this in one place is deliberate: in a
real finance system the chart of accounts is the contract everything else binds
to, and letting it drift is how you get silently mis-stated books.
"""

from __future__ import annotations

# --- Categories (a deliberately small, realistic chart of accounts) -----------
# Each maps to where it lands on a simple income statement.

REVENUE = "Revenue"
COGS = "Cost of Goods Sold"
OPEX = "Operating Expense"
OTHER = "Other / Below the line"

CATEGORIES: dict[str, str] = {
    # Revenue (channel-level)
    "Shopify Sales": REVENUE,
    "Amazon Sales": REVENUE,
    "Retail / Wholesale Sales": REVENUE,
    # Contra-revenue
    "Refunds & Chargebacks": REVENUE,
    # COGS
    "Cost of Goods Sold": COGS,
    "Shipping & Fulfillment": COGS,
    "Payment Processing Fees": COGS,
    # Operating expenses
    "Advertising & Marketing": OPEX,
    "Payroll Expense": OPEX,
    "Payroll Taxes": OPEX,
    "Software & SaaS": OPEX,
    "Office & Admin": OPEX,
    "Professional Services": OPEX,
    # Not on the P&L — money movement only
    "Internal Transfer": OTHER,
    "Owner Draw / Capital": OTHER,
}

CATEGORY_NAMES: list[str] = list(CATEGORIES.keys())

# The agent is allowed to abstain. This is a feature, not a failure: a confident
# "I don't know" routed to a human is far cheaper than a confident wrong answer.
UNKNOWN = "Needs Review"


def income_statement_section(category: str) -> str:
    """Where a category rolls up on the P&L."""
    return CATEGORIES.get(category, OTHER)
