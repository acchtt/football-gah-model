from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SETTLED_CATEGORIES = {
    "full_win",
    "half_win",
    "push",
    "half_loss",
    "full_loss",
}


@dataclass(frozen=True)
class ProspectiveReadiness:
    status: str
    settled_snapshots: int
    observation_span_days: int
    snapshots_remaining: int
    days_remaining: int

    @property
    def ready(self) -> bool:
        return self.status == "READY"


@dataclass(frozen=True)
class ProspectiveReviewSummary:
    settled_snapshots: int
    observation_span_days: int
    mean_captured_return: float | None
    same_line_clv_n: int
    same_line_clv_coverage: float
    mean_same_line_clv: float | None
    positive_same_line_clv_rate: float | None
    line_move_n: int
    mean_line_move: float | None


def _normalized(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    required = {
        "snapshot_id",
        "captured_at",
        "outcome_status",
        "settlement",
        "net_return",
        "captured_line",
        "closing_line",
        "market_probability",
        "closing_market_probability",
    }
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(
            f"Missing required prospective review columns: {sorted(missing)}"
        )

    out["captured_at"] = pd.to_datetime(
        out["captured_at"], utc=True, errors="raise"
    )
    if out["snapshot_id"].astype(str).duplicated().any():
        raise ValueError("snapshot_id values must be unique.")
    return out


def _settled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = _normalized(frame)
    status = out["outcome_status"].fillna("").astype(str).str.lower()
    settlement = out["settlement"].fillna("").astype(str)
    return out[
        (status == "settled") & settlement.isin(SETTLED_CATEGORIES)
    ].copy()


def prospective_readiness(
    frame: pd.DataFrame,
    *,
    minimum_settled: int = 100,
    minimum_days: int = 30,
) -> ProspectiveReadiness:
    if minimum_settled < 1:
        raise ValueError("minimum_settled must be at least 1.")
    if minimum_days < 0:
        raise ValueError("minimum_days cannot be negative.")

    out = _normalized(frame)
    settled = _settled_rows(out)

    if len(out) < 2:
        span_days = 0
    else:
        span = out["captured_at"].max() - out["captured_at"].min()
        span_days = int(span.total_seconds() // 86400)

    settled_n = int(len(settled))
    snapshots_remaining = max(0, minimum_settled - settled_n)
    days_remaining = max(0, minimum_days - span_days)
    status = (
        "READY"
        if snapshots_remaining == 0 and days_remaining == 0
        else "WAIT"
    )

    return ProspectiveReadiness(
        status=status,
        settled_snapshots=settled_n,
        observation_span_days=span_days,
        snapshots_remaining=snapshots_remaining,
        days_remaining=days_remaining,
    )


def prospective_review(
    frame: pd.DataFrame,
    *,
    minimum_settled: int = 100,
    minimum_days: int = 30,
) -> ProspectiveReviewSummary:
    readiness = prospective_readiness(
        frame,
        minimum_settled=minimum_settled,
        minimum_days=minimum_days,
    )
    if not readiness.ready:
        raise ValueError(
            "Formal prospective review is locked until both frozen readiness "
            "conditions are met."
        )

    settled = _settled_rows(frame)

    returns = pd.to_numeric(settled["net_return"], errors="coerce").dropna()
    mean_return = float(returns.mean()) if len(returns) else None

    cap_line = pd.to_numeric(settled["captured_line"], errors="coerce")
    close_line = pd.to_numeric(settled["closing_line"], errors="coerce")
    cap_prob = pd.to_numeric(settled["market_probability"], errors="coerce")
    close_prob = pd.to_numeric(
        settled["closing_market_probability"], errors="coerce"
    )

    same_line = (
        cap_line.notna()
        & close_line.notna()
        & ((cap_line - close_line).abs() <= 1e-9)
        & cap_prob.notna()
        & close_prob.notna()
    )
    clv = (close_prob[same_line] - cap_prob[same_line]).dropna()

    line_move_mask = cap_line.notna() & close_line.notna()
    line_moves = (close_line[line_move_mask] - cap_line[line_move_mask]).dropna()

    settled_n = len(settled)
    return ProspectiveReviewSummary(
        settled_snapshots=int(settled_n),
        observation_span_days=readiness.observation_span_days,
        mean_captured_return=mean_return,
        same_line_clv_n=int(len(clv)),
        same_line_clv_coverage=(
            float(len(clv) / settled_n) if settled_n else 0.0
        ),
        mean_same_line_clv=float(clv.mean()) if len(clv) else None,
        positive_same_line_clv_rate=(
            float((clv > 0).mean()) if len(clv) else None
        ),
        line_move_n=int(len(line_moves)),
        mean_line_move=float(line_moves.mean()) if len(line_moves) else None,
    )
