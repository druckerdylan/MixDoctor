"""A hard ceiling on what the AI layer can cost.

`/engineer` is unauthenticated and costs real money per call, so a public
deployment with it switched on bills its owner for every visitor and every
refresh. That is the whole reason the layer currently ships off. This module is
what makes turning it on a bounded decision rather than an open tab.

Three properties it has to have, and the order matters:

1. **Refuse before spending, not after.** The check runs ahead of the API call.
   A budget that only notices it is over once the money is gone is not a budget.
2. **Survive a restart and a second worker.** Spend lives in the database, not
   in a process variable — Railway runs more than one worker and restarts on
   every deploy, and an in-memory counter would reset the ceiling both times.
3. **Degrade, never fail.** Out of budget is a 503 with a plain sentence, and
   the report the user already has is complete without it.

Nothing here is billing-grade accounting. It reads Anthropic's reported token
usage and applies the published per-token rates, which is close enough to keep
a hobby deployment from a surprise, and is not a substitute for the spend limit
in the Anthropic console. Set that too — this cannot catch a runaway that never
reaches our own bookkeeping.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from sqlalchemy import Column, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Session

from database import Base

logger = logging.getLogger(__name__)

__all__ = [
    "AISpend",
    "budget_usd",
    "budget_period",
    "check_budget",
    "record_spend",
    "remaining_usd",
    "estimate_cost_usd",
    "BudgetState",
]


# Claude Opus 5, USD per million tokens. Cache reads bill at a tenth of input.
# If these drift, the cap drifts with them — they are the only place rates live.
_INPUT_PER_MTOK = 5.00
_OUTPUT_PER_MTOK = 25.00
_CACHE_READ_PER_MTOK = 0.50

# What one report costs, for the pre-flight check. Measured on this codebase:
# ~10.2k input (mostly cached after the first call) and ~14.1k output at
# effort=high. Deliberately rounded up — reserving slightly too much is a
# smaller error than letting a call through that takes the account over.
TYPICAL_REPORT_USD = 0.45


class AISpend(Base):
    """One row per completed consult. The ledger the cap reads."""

    __tablename__ = "ai_spend"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    input_tokens = Column(Integer, default=0)
    cached_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    # Coarse only. Enough to spot one address eating the day's budget; never a
    # full address, because a spend ledger is not a reason to retain user IPs.
    client_hint = Column(String(64), default="")


def estimate_cost_usd(input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    """USD for one call from its reported token usage."""
    fresh = max(0, int(input_tokens))
    cached = max(0, int(cached_tokens))
    out = max(0, int(output_tokens))
    return (
        fresh / 1e6 * _INPUT_PER_MTOK
        + cached / 1e6 * _CACHE_READ_PER_MTOK
        + out / 1e6 * _OUTPUT_PER_MTOK
    )


def budget_usd() -> float:
    """The ceiling. 0 (the default) means the layer is not budgeted to run."""
    raw = os.environ.get("MIXDOCTOR_AI_BUDGET_USD", "0").strip()
    try:
        value = float(raw)
    except ValueError:
        logger.warning("budget: MIXDOCTOR_AI_BUDGET_USD=%r is not a number; treating as 0", raw)
        return 0.0
    return max(0.0, value)


def budget_period() -> str:
    """`day` or `month`. Anything else is read as `day`, the safer of the two."""
    value = os.environ.get("MIXDOCTOR_AI_BUDGET_PERIOD", "day").strip().lower()
    return "month" if value in {"month", "monthly"} else "day"


def _window_start(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if budget_period() == "month":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def spent_usd(db: Session, now: Optional[datetime] = None) -> float:
    """What the current window has already cost."""
    start = _window_start(now)
    total = db.query(func.coalesce(func.sum(AISpend.cost_usd), 0.0)).filter(
        AISpend.created_at >= start
    ).scalar()
    return float(total or 0.0)


def remaining_usd(db: Session, now: Optional[datetime] = None) -> float:
    return max(0.0, budget_usd() - spent_usd(db, now))


class BudgetState:
    """What the caller needs to decide, and what the UI needs to say."""

    def __init__(self, allowed: bool, reason: str, remaining: float, budget: float,
                 reports_left: int) -> None:
        self.allowed = allowed
        self.reason = reason
        self.remaining = remaining
        self.budget = budget
        self.reports_left = reports_left


def check_budget(db: Session, now: Optional[datetime] = None) -> BudgetState:
    """May we spend on one more report?

    Reserves `TYPICAL_REPORT_USD` rather than asking whether any money is left:
    a budget with three cents remaining cannot pay for a forty-cent report, and
    starting one anyway is how a cap gets exceeded on its final call.
    """
    budget = budget_usd()
    if budget <= 0.0:
        return BudgetState(False, "no-budget", 0.0, 0.0, 0)

    left = remaining_usd(db, now)
    reports_left = int(left // TYPICAL_REPORT_USD)

    if left < TYPICAL_REPORT_USD:
        return BudgetState(False, "exhausted", left, budget, 0)
    return BudgetState(True, "ok", left, budget, reports_left)


def record_spend(
    db: Session,
    *,
    input_tokens: int = 0,
    cached_tokens: int = 0,
    output_tokens: int = 0,
    client_hint: str = "",
) -> float:
    """Write one call to the ledger. Returns its cost.

    Never raises. A failed write must not turn a successful report into an
    error for the user — it costs us accuracy on the cap, which is recoverable,
    where raising costs them the thing they waited for.
    """
    cost = estimate_cost_usd(input_tokens, cached_tokens, output_tokens)
    try:
        db.add(AISpend(
            input_tokens=int(input_tokens),
            cached_tokens=int(cached_tokens),
            output_tokens=int(output_tokens),
            cost_usd=cost,
            client_hint=(client_hint or "")[:64],
        ))
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("budget: could not record %.4f USD of spend", cost)
    return cost


def prune(db: Session, keep_days: int = 120) -> int:
    """Drop ledger rows older than the longest window we ever read."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    try:
        removed = db.query(AISpend).filter(AISpend.created_at < cutoff).delete()
        db.commit()
        return int(removed or 0)
    except Exception:
        db.rollback()
        logger.exception("budget: prune failed")
        return 0
