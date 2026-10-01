from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable

import pandas as pd

from .board_capacity import BoardCandidate, plan_board
from .lifecycle import DecisionLedger, LifecycleState
from .prospective_review import ProspectiveReadiness, prospective_readiness


AIRTABLE_COLUMN_ALIASES = {
    "Snapshot ID": "snapshot_id",
    "Match ID": "match_id",
    "Kickoff": "kickoff",
    "Captured At": "captured_at",
    "Outcome Status": "outcome_status",
    "Settlement": "settlement",
    "Net Return": "net_return",
    "AH Line": "captured_line",
    "Closing AH Line": "closing_line",
    "Market Probability": "market_probability",
    "Closing Market Probability": "closing_market_probability",
    "Execution State": "execution_state",
    "Entry Type": "entry_type",
    "Entry At": "entry_at",
    "Entry Line": "entry_line",
    "Entry Odds": "entry_odds",
    "Settlement ID": "settlement_id",
    "Last Lifecycle Event ID": "last_lifecycle_event_id",
    "Competition": "competition",
}


class QASeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True)
class QAIssue:
    severity: QASeverity
    code: str
    message: str
    match_id: str | None = None
    snapshot_id: str | None = None


@dataclass(frozen=True)
class OperationalHealthReport:
    status: str
    errors: int
    warnings: int
    infos: int
    issues: tuple[QAIssue, ...]
    readiness: ProspectiveReadiness | None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "infos": self.infos,
            "readiness": (
                None
                if self.readiness is None
                else asdict(self.readiness)
            ),
            "issues": [
                {
                    **asdict(issue),
                    "severity": issue.severity.value,
                }
                for issue in self.issues
            ],
        }


def normalize_prospective_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rename = {
        source: target
        for source, target in AIRTABLE_COLUMN_ALIASES.items()
        if target not in out.columns and source in out.columns
    }
    return out.rename(columns=rename)


def _blank(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def _text(value) -> str | None:
    return None if _blank(value) else str(value)


def _issue(
    issues: list[QAIssue],
    severity: QASeverity,
    code: str,
    message: str,
    *,
    match_id=None,
    snapshot_id=None,
) -> None:
    issues.append(
        QAIssue(
            severity=severity,
            code=code,
            message=message,
            match_id=_text(match_id),
            snapshot_id=_text(snapshot_id),
        )
    )


def _row_identity(row: pd.Series) -> tuple[str | None, str | None]:
    return _text(row.get("match_id")), _text(row.get("snapshot_id"))


def _check_snapshot_rows(
    frame: pd.DataFrame,
    issues: list[QAIssue],
    *,
    as_of: datetime,
    stale_grace_hours: float,
) -> None:
    required = {"snapshot_id", "match_id", "captured_at", "outcome_status"}
    for name in sorted(required.difference(frame.columns)):
        _issue(
            issues,
            QASeverity.ERROR,
            "MISSING_COLUMN",
            f"Prospective data is missing required column: {name}.",
        )

    if "snapshot_id" in frame.columns:
        nonblank = frame["snapshot_id"].map(lambda v: not _blank(v))
        duplicated = frame.loc[nonblank, "snapshot_id"].astype(str).duplicated(
            keep=False
        )
        duplicate_values = sorted(
            set(
                frame.loc[nonblank, "snapshot_id"]
                .astype(str)[duplicated]
                .tolist()
            )
        )
        for snapshot_id in duplicate_values:
            _issue(
                issues,
                QASeverity.ERROR,
                "DUPLICATE_SNAPSHOT_ID",
                "Snapshot ID appears more than once.",
                snapshot_id=snapshot_id,
            )

    if "settlement_id" in frame.columns:
        nonblank = frame["settlement_id"].map(lambda v: not _blank(v))
        ids = frame.loc[nonblank, "settlement_id"].astype(str)
        for settlement_id in sorted(set(ids[ids.duplicated(keep=False)])):
            _issue(
                issues,
                QASeverity.ERROR,
                "DUPLICATE_SETTLEMENT_ID",
                f"Settlement ID appears more than once: {settlement_id}.",
            )

    for _, row in frame.iterrows():
        match_id, snapshot_id = _row_identity(row)
        if "snapshot_id" in frame.columns and snapshot_id is None:
            _issue(
                issues,
                QASeverity.ERROR,
                "MISSING_SNAPSHOT_ID",
                "Prospective row has no Snapshot ID.",
                match_id=match_id,
            )
        if "match_id" in frame.columns and match_id is None:
            _issue(
                issues,
                QASeverity.ERROR,
                "MISSING_MATCH_ID",
                "Prospective row has no Match ID.",
                snapshot_id=snapshot_id,
            )

        execution_state = (
            (_text(row.get("execution_state")) or "").upper()
            if "execution_state" in frame.columns
            else ""
        )
        entry_type = (
            (_text(row.get("entry_type")) or "").lower()
            if "entry_type" in frame.columns
            else ""
        )
        entered = execution_state in {
            "ENTER_PROTECTED",
            "ENTER_DECAY",
        } or entry_type in {"protected", "decay"}

        if entered:
            for name in ("entry_at", "entry_line", "entry_odds", "entry_type"):
                if name not in frame.columns or _blank(row.get(name)):
                    _issue(
                        issues,
                        QASeverity.ERROR,
                        "ENTERED_WITHOUT_ENTRY_DETAIL",
                        f"Entered snapshot is missing {name}.",
                        match_id=match_id,
                        snapshot_id=snapshot_id,
                    )
        if execution_state == "PASS" and entry_type in {"protected", "decay"}:
            _issue(
                issues,
                QASeverity.ERROR,
                "PASS_WITH_ENTRY",
                "Execution state PASS conflicts with an entry type.",
                match_id=match_id,
                snapshot_id=snapshot_id,
            )

        outcome_status = (
            (_text(row.get("outcome_status")) or "").lower()
            if "outcome_status" in frame.columns
            else ""
        )
        settled = outcome_status == "settled"
        if settled:
            for name in ("settlement", "net_return"):
                if name not in frame.columns or _blank(row.get(name)):
                    _issue(
                        issues,
                        QASeverity.ERROR,
                        "SETTLED_WITHOUT_RESULT",
                        f"Settled snapshot is missing {name}.",
                        match_id=match_id,
                        snapshot_id=snapshot_id,
                    )
            for name in ("closing_line", "closing_market_probability"):
                if name not in frame.columns or _blank(row.get(name)):
                    _issue(
                        issues,
                        QASeverity.WARNING,
                        "MISSING_CLOSING_DATA",
                        f"Settled snapshot is missing {name}; CLV coverage is incomplete.",
                        match_id=match_id,
                        snapshot_id=snapshot_id,
                    )

        if outcome_status and outcome_status != "settled":
            captured = None
            kickoff = None
            if "captured_at" in frame.columns and not _blank(row.get("captured_at")):
                captured = pd.to_datetime(row.get("captured_at"), utc=True)
            if "kickoff" in frame.columns and not _blank(row.get("kickoff")):
                kickoff = pd.to_datetime(row.get("kickoff"), utc=True)
            anchor = kickoff if kickoff is not None else captured
            if anchor is not None:
                stale_at = anchor.to_pydatetime() + timedelta(
                    hours=float(stale_grace_hours)
                )
                if as_of > stale_at:
                    _issue(
                        issues,
                        QASeverity.WARNING,
                        "STALE_PENDING_SNAPSHOT",
                        "Prospective snapshot remains pending past the stale threshold.",
                        match_id=match_id,
                        snapshot_id=snapshot_id,
                    )


def _check_ledgers(
    ledgers: Iterable[DecisionLedger],
    frame: pd.DataFrame,
    issues: list[QAIssue],
    *,
    as_of: datetime,
    stale_grace_hours: float,
) -> None:
    rows = list(ledgers)
    seen_ledgers: set[str] = set()
    snapshot_match_ids = (
        set(
            frame["match_id"]
            .dropna()
            .astype(str)
            .loc[lambda s: s.str.strip() != ""]
        )
        if "match_id" in frame.columns
        else set()
    )

    for ledger in rows:
        if ledger.match_id in seen_ledgers:
            _issue(
                issues,
                QASeverity.ERROR,
                "DUPLICATE_LEDGER",
                "More than one decision ledger exists for the match.",
                match_id=ledger.match_id,
            )
        seen_ledgers.add(ledger.match_id)

        try:
            ledger.validate()
        except ValueError as exc:
            _issue(
                issues,
                QASeverity.ERROR,
                "INVALID_LIFECYCLE",
                str(exc),
                match_id=ledger.match_id,
            )

        event_keys = [
            (
                event.state.value,
                event.at.isoformat(),
                event.reason,
                event.source,
            )
            for event in ledger.events
        ]
        if len(event_keys) != len(set(event_keys)):
            _issue(
                issues,
                QASeverity.ERROR,
                "DUPLICATE_LIFECYCLE_EVENT",
                "Exact duplicate lifecycle event detected.",
                match_id=ledger.match_id,
            )

        states = {event.state for event in ledger.events}
        snapshot_required = bool(
            states.intersection(
                {
                    LifecycleState.SNAPSHOT_CAPTURED,
                    LifecycleState.WAITING_ENTRY,
                    LifecycleState.ENTERED_PROTECTED,
                    LifecycleState.ENTERED_DECAY,
                    LifecycleState.LIVE_REPRICED,
                    LifecycleState.SETTLED,
                }
            )
        )
        if snapshot_required and ledger.match_id not in snapshot_match_ids:
            _issue(
                issues,
                QASeverity.ERROR,
                "LIFECYCLE_WITHOUT_SNAPSHOT",
                "Lifecycle progressed beyond board selection without a prospective snapshot.",
                match_id=ledger.match_id,
            )

        if (
            ledger.current_state == LifecycleState.BOARD_SELECTED
            and ledger.events
            and as_of
            > ledger.events[-1].at + timedelta(hours=float(stale_grace_hours))
        ):
            _issue(
                issues,
                QASeverity.WARNING,
                "STALE_BOARD_SELECTED",
                "Board-selected match has not reached snapshot capture.",
                match_id=ledger.match_id,
            )

        if ledger.current_state == LifecycleState.SETTLED:
            settled_for_match = False
            if {
                "match_id",
                "outcome_status",
            }.issubset(frame.columns):
                part = frame[frame["match_id"].astype(str) == ledger.match_id]
                settled_for_match = any(
                    str(v).lower() == "settled"
                    for v in part["outcome_status"].dropna()
                )
            if not settled_for_match:
                _issue(
                    issues,
                    QASeverity.ERROR,
                    "LIFECYCLE_SETTLED_WITHOUT_OUTCOME",
                    "Lifecycle is SETTLED but prospective outcome is not settled.",
                    match_id=ledger.match_id,
                )


def _check_capacity(
    selected_board_candidates: Iterable[BoardCandidate],
    issues: list[QAIssue],
    *,
    max_matches: int,
    max_concurrent: int,
) -> None:
    selected = list(selected_board_candidates)
    if not selected:
        return
    try:
        replay = plan_board(
            selected,
            max_matches=max_matches,
            max_concurrent=max_concurrent,
        )
    except ValueError as exc:
        _issue(
            issues,
            QASeverity.ERROR,
            "INVALID_BOARD_CAPACITY_INPUT",
            str(exc),
        )
        return

    for decision in replay.rejected:
        _issue(
            issues,
            QASeverity.ERROR,
            "BOARD_CAPACITY_VIOLATION",
            (
                "A match marked operationally selected would violate "
                f"v2.5 capacity: {decision.reason}"
            ),
            match_id=decision.candidate.match_id,
        )


def _readiness(
    frame: pd.DataFrame,
    issues: list[QAIssue],
) -> ProspectiveReadiness | None:
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
    missing = sorted(required.difference(frame.columns))
    if missing:
        _issue(
            issues,
            QASeverity.WARNING,
            "READINESS_UNAVAILABLE",
            "v2.1 readiness cannot be computed; missing columns: "
            + ", ".join(missing),
        )
        return None

    try:
        readiness = prospective_readiness(frame)
    except (ValueError, TypeError) as exc:
        _issue(
            issues,
            QASeverity.WARNING,
            "READINESS_UNAVAILABLE",
            f"v2.1 readiness could not be computed: {exc}",
        )
        return None

    _issue(
        issues,
        QASeverity.INFO,
        "PROSPECTIVE_GATE_PROGRESS",
        (
            f"v2.1 gate={readiness.status}; "
            f"settled={readiness.settled_snapshots}/100; "
            f"span={readiness.observation_span_days}/30 days; "
            f"remaining={readiness.snapshots_remaining} snapshots, "
            f"{readiness.days_remaining} days."
        ),
    )
    return readiness


def operational_health(
    prospective: pd.DataFrame,
    *,
    ledgers: Iterable[DecisionLedger] = (),
    selected_board_candidates: Iterable[BoardCandidate] = (),
    max_matches: int = 8,
    max_concurrent: int = 3,
    as_of: datetime | None = None,
    stale_grace_hours: float = 12.0,
) -> OperationalHealthReport:
    if stale_grace_hours < 0:
        raise ValueError("stale_grace_hours cannot be negative.")
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware.")

    frame = normalize_prospective_columns(prospective)
    issues: list[QAIssue] = []
    _check_snapshot_rows(
        frame,
        issues,
        as_of=as_of,
        stale_grace_hours=stale_grace_hours,
    )
    _check_ledgers(
        ledgers,
        frame,
        issues,
        as_of=as_of,
        stale_grace_hours=stale_grace_hours,
    )
    _check_capacity(
        selected_board_candidates,
        issues,
        max_matches=max_matches,
        max_concurrent=max_concurrent,
    )
    readiness = _readiness(frame, issues)

    errors = sum(i.severity == QASeverity.ERROR for i in issues)
    warnings = sum(i.severity == QASeverity.WARNING for i in issues)
    infos = sum(i.severity == QASeverity.INFO for i in issues)
    status = "ERROR" if errors else ("WARN" if warnings else "OK")
    return OperationalHealthReport(
        status=status,
        errors=errors,
        warnings=warnings,
        infos=infos,
        issues=tuple(issues),
        readiness=readiness,
    )
