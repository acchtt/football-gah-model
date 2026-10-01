from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .airtable_state import AirtableState
from .lifecycle import LifecycleState
from .quality_control import (
    AIRTABLE_COLUMN_ALIASES,
    OperationalHealthReport,
    operational_health,
)
from .prospective_review import ProspectiveReadiness


REPORT_SCHEMA_VERSION = "v2.13"

_CANONICAL_EMPTY_COLUMNS = (
    "snapshot_id",
    "match_id",
    "kickoff",
    "captured_at",
    "outcome_status",
    "settlement",
    "net_return",
    "captured_line",
    "closing_line",
    "market_probability",
    "closing_market_probability",
    "execution_state",
    "entry_type",
    "entry_at",
    "entry_line",
    "entry_odds",
    "settlement_id",
    "last_lifecycle_event_id",
    "competition",
)


@dataclass(frozen=True)
class OperatorReport:
    generated_at: datetime
    data_status: str
    snapshot_count: int
    match_count: int
    lifecycle_event_count: int
    pending_snapshots: int
    settled_snapshots: int
    entered_snapshots: int
    passed_snapshots: int
    waiting_snapshots: int
    lifecycle_state_counts: dict[str, int]
    health: OperationalHealthReport
    readiness: ProspectiveReadiness | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "data_status": self.data_status,
            "counts": {
                "snapshots": self.snapshot_count,
                "matches": self.match_count,
                "lifecycle_events": self.lifecycle_event_count,
                "pending_snapshots": self.pending_snapshots,
                "settled_snapshots": self.settled_snapshots,
                "entered_snapshots": self.entered_snapshots,
                "passed_snapshots": self.passed_snapshots,
                "waiting_snapshots": self.waiting_snapshots,
            },
            "lifecycle_state_counts": dict(self.lifecycle_state_counts),
            "health": self.health.to_dict(),
            "readiness": (
                None if self.readiness is None else asdict(self.readiness)
            ),
        }

    def compact_text(self) -> str:
        health = (
            f"health={self.health.status} "
            f"errors={self.health.errors} "
            f"warnings={self.health.warnings}"
        )
        if self.readiness is None:
            gate = "gate=UNAVAILABLE"
        else:
            gate = (
                f"gate={self.readiness.status} "
                f"settled={self.readiness.settled_snapshots}/100 "
                f"span={self.readiness.observation_span_days}/30d"
            )
        return (
            f"GAH {REPORT_SCHEMA_VERSION} | "
            f"data={self.data_status} "
            f"snapshots={self.snapshot_count} "
            f"pending={self.pending_snapshots} "
            f"entered={self.entered_snapshots} "
            f"settled={self.settled_snapshots} | "
            f"{health} | {gate}"
        )


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware.")


def _normalized_frame(state: AirtableState) -> pd.DataFrame:
    frame = state.prospective_frame.copy()
    rename = {
        source: target
        for source, target in AIRTABLE_COLUMN_ALIASES.items()
        if source in frame.columns and target not in frame.columns
    }
    frame = frame.rename(columns=rename)

    # Airtable omits blank fields from returned records. Normalize the
    # canonical prospective schema for every state, not only an empty table,
    # so an all-pending forward sample still exposes closing/settlement columns
    # as blank and the frozen gate can report WAIT instead of UNAVAILABLE.
    for column in _CANONICAL_EMPTY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    return frame


def _text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.strip().str.lower()


def _snapshot_counts(frame: pd.DataFrame) -> dict[str, int]:
    outcome = _text_series(frame, "outcome_status")
    execution = _text_series(frame, "execution_state")
    entry_type = _text_series(frame, "entry_type")

    settled = int((outcome == "settled").sum())
    pending = int((outcome != "settled").sum())
    entered_mask = execution.isin(
        ["enter_protected", "enter_decay"]
    ) | entry_type.isin(["protected", "decay"])
    passed = int((execution == "pass").sum())
    waiting = int((execution == "wait").sum())

    return {
        "pending": pending,
        "settled": settled,
        "entered": int(entered_mask.sum()),
        "passed": passed,
        "waiting": waiting,
    }


def _lifecycle_counts(state: AirtableState) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for ledger in state.ledgers:
        counts[ledger.current_state.value] += 1
    return {
        state.value: int(counts.get(state.value, 0))
        for state in LifecycleState
    }


def build_operator_report(
    state: AirtableState,
    *,
    generated_at: datetime | None = None,
    stale_grace_hours: float = 12.0,
) -> OperatorReport:
    if generated_at is None:
        generated_at = datetime.now(timezone.utc)
    _aware(generated_at, "generated_at")

    frame = _normalized_frame(state)
    health = operational_health(
        frame,
        ledgers=state.ledgers,
        as_of=generated_at,
        stale_grace_hours=stale_grace_hours,
    )
    counts = _snapshot_counts(frame)
    data_status = "EMPTY" if len(frame) == 0 else "ACTIVE"

    return OperatorReport(
        generated_at=generated_at,
        data_status=data_status,
        snapshot_count=int(len(frame)),
        match_count=state.match_count,
        lifecycle_event_count=state.lifecycle_event_count,
        pending_snapshots=counts["pending"],
        settled_snapshots=counts["settled"],
        entered_snapshots=counts["entered"],
        passed_snapshots=counts["passed"],
        waiting_snapshots=counts["waiting"],
        lifecycle_state_counts=_lifecycle_counts(state),
        health=health,
        readiness=health.readiness,
    )
