from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


SETTLED = {"full_win", "half_win", "push", "half_loss", "full_loss"}


@dataclass(frozen=True)
class AuditSummary:
    observations: int
    entered: int
    entry_conversion: float
    roi: float | None
    same_line_clv_n: int
    mean_same_line_clv: float | None
    positive_same_line_clv_rate: float | None
    protected_entries: int
    decay_entries: int
    not_entered: int


def _same_line(a: float | int | None, b: float | int | None) -> bool:
    if a is None or b is None:
        return False
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= 1e-9


def classify_row(row: pd.Series) -> str:
    entered = bool(row.get("entered", False))
    settlement = row.get("settlement")
    if not entered:
        return "NOT_ENTERED"
    if settlement not in SETTLED:
        return "UNSETTLED"
    if settlement in {"full_win", "half_win"}:
        return "ENTERED_WIN"
    if settlement == "push":
        return "ENTERED_PUSH"
    return "ENTERED_LOSS"


def enrich_audit_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    required = {
        "snapshot_id",
        "match_id",
        "captured_line",
        "market_probability",
        "gah_probability",
        "entered",
        "entry_type",
        "settlement",
        "net_return",
    }
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Missing required audit columns: {sorted(missing)}")

    out["gah_edge"] = (
        pd.to_numeric(out["gah_probability"], errors="raise")
        - pd.to_numeric(out["market_probability"], errors="raise")
    )
    out["diagnostic_bucket"] = out.apply(classify_row, axis=1)

    clv_values: list[float | None] = []
    line_moves: list[float | None] = []
    for _, row in out.iterrows():
        close_line = row.get("closing_line")
        cap_line = row.get("captured_line")
        close_prob = row.get("closing_market_probability")
        market_prob = row.get("market_probability")

        if _same_line(cap_line, close_line) and close_prob is not None and not pd.isna(close_prob):
            clv_values.append(float(close_prob) - float(market_prob))
        else:
            clv_values.append(None)

        if close_line is None or pd.isna(close_line):
            line_moves.append(None)
        else:
            line_moves.append(float(close_line) - float(cap_line))

    out["same_line_probability_clv"] = clv_values
    out["line_move"] = line_moves
    return out


def summarize_audit(frame: pd.DataFrame) -> AuditSummary:
    out = enrich_audit_rows(frame)
    entered = out[out["entered"].astype(bool)].copy()

    roi: float | None = None
    settled_entered = entered[entered["settlement"].isin(SETTLED)]
    if len(settled_entered):
        roi = float(pd.to_numeric(settled_entered["net_return"], errors="raise").mean())

    clv = pd.to_numeric(
        out["same_line_probability_clv"], errors="coerce"
    ).dropna()
    mean_clv = float(clv.mean()) if len(clv) else None
    positive_clv_rate = float((clv > 0).mean()) if len(clv) else None

    entry_type = entered["entry_type"].fillna("").astype(str)
    protected = int((entry_type == "protected").sum())
    decay = int((entry_type == "decay").sum())

    observations = len(out)
    entered_n = len(entered)
    return AuditSummary(
        observations=observations,
        entered=entered_n,
        entry_conversion=(entered_n / observations) if observations else 0.0,
        roi=roi,
        same_line_clv_n=int(len(clv)),
        mean_same_line_clv=mean_clv,
        positive_same_line_clv_rate=positive_clv_rate,
        protected_entries=protected,
        decay_entries=decay,
        not_entered=int(observations - entered_n),
    )


def grouped_audit(
    frame: pd.DataFrame,
    group_columns: Iterable[str] = ("competition", "entry_type"),
) -> pd.DataFrame:
    out = enrich_audit_rows(frame)
    groups = [c for c in group_columns if c in out.columns]
    if not groups:
        raise ValueError("No requested group columns are present.")

    rows = []
    grouper = groups[0] if len(groups) == 1 else groups
    for keys, part in out.groupby(grouper, dropna=False):
        if len(groups) == 1:
            keys = (keys,)
        s = summarize_audit(part)
        row = {col: key for col, key in zip(groups, keys)}
        row.update(s.__dict__)
        rows.append(row)
    return pd.DataFrame(rows)
