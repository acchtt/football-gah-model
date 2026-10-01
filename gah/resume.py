from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .airtable_bridge import (
    AirtableBridgeClient,
    BridgeOperation,
    BridgeOperationKind,
    PROSPECTIVE_AH_TABLE_ID,
    execution_operations,
    lifecycle_event_id,
    lifecycle_operation,
    live_operations,
    operation_manifest,
    prospective_outcome_operation,
    settlement_operations,
)
from .airtable_state import AirtableState
from .execution import ExecutionDecision, ExecutionPlan, ExecutionState, evaluate_execution
from .lifecycle import DecisionLedger, LifecycleEvent, LifecycleState, make_event
from .live import LiveAHQuote, LiveState, reprice_live_ah
from .prospective import ProspectiveAHOutcome, airtable_outcome_fields


RESUME_SCHEMA_VERSION = "v2.12"


@dataclass(frozen=True)
class ResumeResult:
    snapshot_id: str
    match_id: str
    previous_state: LifecycleState
    current_state: LifecycleState
    operations: tuple[BridgeOperation, ...]
    status: str
    execution: ExecutionDecision | None = None
    live_quote: LiveAHQuote | None = None

    def apply(
        self,
        client: AirtableBridgeClient,
    ) -> tuple[Any, ...]:
        return tuple(client.apply(op) for op in self.operations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESUME_SCHEMA_VERSION,
            "snapshot_id": self.snapshot_id,
            "match_id": self.match_id,
            "previous_state": self.previous_state.value,
            "current_state": self.current_state.value,
            "status": self.status,
            "operations": [
                operation_manifest(op) for op in self.operations
            ],
            "execution": (
                None
                if self.execution is None
                else {
                    "state": self.execution.state.value,
                    "active_minimum_line": self.execution.active_minimum_line,
                    "active_minimum_odds": self.execution.active_minimum_odds,
                    "minutes_to_kickoff": self.execution.minutes_to_kickoff,
                    "line_ok": self.execution.line_ok,
                    "odds_ok": self.execution.odds_ok,
                    "reason": self.execution.reason,
                }
            ),
            "live_quote": (
                None
                if self.live_quote is None
                else {
                    "side": self.live_quote.side,
                    "line": self.live_quote.line,
                    "effective_win_probability": (
                        self.live_quote.effective_win_probability
                    ),
                    "fair_decimal_odds": self.live_quote.fair_decimal_odds,
                    "expected_value": self.live_quote.expected_value,
                }
            ),
        }


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware.")


def _snapshot_context(
    state: AirtableState,
    snapshot_id: str,
) -> tuple[dict[str, Any], str, DecisionLedger]:
    row = state.snapshot_row(snapshot_id)
    match_id = str(row.get("Match ID", "")).strip()
    if not match_id:
        raise ValueError("Snapshot is missing Match ID.")
    ledger = state.ledger_for(match_id)
    return row, match_id, ledger


def _event_exists(
    ledger: DecisionLedger,
    event: LifecycleEvent,
) -> int | None:
    for index, existing in enumerate(ledger.events):
        if existing == event:
            return index
    return None


def _last_event_id(
    row: dict[str, Any],
) -> str:
    value = row.get("Last Lifecycle Event ID")
    if value is None:
        return ""
    return str(value).strip()


def _filter_retry_operations(
    *,
    row: dict[str, Any],
    ledger: DecisionLedger,
    event: LifecycleEvent,
    match_id: str,
    snapshot_id: str,
    operations: tuple[BridgeOperation, ...],
) -> tuple[tuple[BridgeOperation, ...], str]:
    existing_index = _event_exists(ledger, event)
    if existing_index is None:
        return operations, "APPEND"

    expected_event_id = lifecycle_event_id(
        match_id,
        event,
        snapshot_id=snapshot_id,
        payload=_event_payload_from_operations(operations),
    )

    if existing_index != len(ledger.events) - 1:
        return (), "ALREADY_RECORDED_PAST"

    if _last_event_id(row) == expected_event_id:
        return (), "ALREADY_RECORDED"

    # The lifecycle append reached Airtable but the snapshot mutation did not.
    # Re-emit only Prospective AH UPDATE_EXISTING operations and never append
    # a duplicate lifecycle record.
    repair = tuple(
        op
        for op in operations
        if (
            op.table == PROSPECTIVE_AH_TABLE_ID
            and op.kind == BridgeOperationKind.UPDATE_EXISTING
        )
    )
    return repair, "REPAIR_SNAPSHOT_LINK"


def _event_payload_from_operations(
    operations: tuple[BridgeOperation, ...],
) -> dict[str, Any]:
    for op in operations:
        if op.key_field != "Event ID":
            continue
        raw = op.fields.get("Payload JSON", "{}")
        import json

        parsed = json.loads(str(raw))
        if not isinstance(parsed, dict):
            raise ValueError("Lifecycle Payload JSON must decode to an object.")
        return parsed
    return {}


def _prospective_settled(row: dict[str, Any]) -> bool:
    return str(row.get("Outcome Status", "")).strip().lower() == "settled"


def _validate_outcome_against_snapshot(
    row: dict[str, Any],
    outcome: ProspectiveAHOutcome,
) -> None:
    match_id = str(row.get("Match ID", "")).strip()
    side = str(row.get("Side", "")).strip().lower()
    if outcome.fixture_key != match_id:
        raise ValueError("Outcome fixture_key does not match snapshot Match ID.")
    if outcome.captured_side != side:
        raise ValueError("Outcome captured_side does not match snapshot Side.")
    if abs(float(outcome.captured_line) - float(row["AH Line"])) > 1e-9:
        raise ValueError("Outcome captured_line does not match snapshot AH Line.")
    if (
        abs(
            float(outcome.captured_market_probability)
            - float(row["Market Probability"])
        )
        > 1e-9
    ):
        raise ValueError(
            "Outcome captured_market_probability does not match snapshot."
        )


_EXECUTION_TO_LIFECYCLE = {
    ExecutionState.WAIT: LifecycleState.WAITING_ENTRY,
    ExecutionState.ENTER_PROTECTED: LifecycleState.ENTERED_PROTECTED,
    ExecutionState.ENTER_DECAY: LifecycleState.ENTERED_DECAY,
    ExecutionState.PASS: LifecycleState.PASSED,
}


def resume_execution(
    state: AirtableState,
    *,
    snapshot_id: str,
    at: datetime,
    source: str,
    plan: ExecutionPlan,
    current_line: float,
    current_odds: float,
    minutes_to_kickoff: int,
) -> ResumeResult:
    _aware(at, "at")
    row, match_id, ledger = _snapshot_context(state, snapshot_id)
    if _prospective_settled(row):
        raise ValueError("Cannot execute a settled prospective snapshot.")

    decision = evaluate_execution(
        plan=plan,
        current_line=current_line,
        current_odds=current_odds,
        minutes_to_kickoff=minutes_to_kickoff,
    )
    lifecycle_state = _EXECUTION_TO_LIFECYCLE[decision.state]
    event = make_event(
        lifecycle_state,
        at=at,
        reason=decision.reason,
        source=source,
    )
    operations = execution_operations(
        match_id=match_id,
        snapshot_id=snapshot_id,
        event=event,
        execution_state=decision.state.value,
        current_line=current_line,
        current_odds=current_odds,
        active_minimum_line=decision.active_minimum_line,
        active_minimum_odds=decision.active_minimum_odds,
        minutes_to_kickoff=decision.minutes_to_kickoff,
    ).operations

    existing_index = _event_exists(ledger, event)
    if existing_index is None:
        updated_ledger = ledger.append(event)
        filtered, status = operations, "APPEND"
    else:
        updated_ledger = ledger
        filtered, status = _filter_retry_operations(
            row=row,
            ledger=ledger,
            event=event,
            match_id=match_id,
            snapshot_id=snapshot_id,
            operations=operations,
        )

    return ResumeResult(
        snapshot_id=snapshot_id,
        match_id=match_id,
        previous_state=ledger.current_state,
        current_state=updated_ledger.current_state,
        operations=filtered,
        status=status,
        execution=decision,
    )


def resume_live(
    state: AirtableState,
    *,
    snapshot_id: str,
    at: datetime,
    source: str,
    live_state: LiveState,
    line: float,
    decimal_odds: float | None = None,
    change_type: str | None = None,
    reason: str = "Live AH re-priced from explicit football state.",
) -> ResumeResult:
    _aware(at, "at")
    row, match_id, ledger = _snapshot_context(state, snapshot_id)
    if _prospective_settled(row):
        raise ValueError("Cannot live re-price a settled prospective snapshot.")
    side = str(row.get("Side", "")).strip().lower()
    if side not in {"home", "away"}:
        raise ValueError("Snapshot Side must be home or away.")

    quote = reprice_live_ah(
        live_state,
        side=side,
        line=line,
        decimal_odds=decimal_odds,
    )
    event = make_event(
        LifecycleState.LIVE_REPRICED,
        at=at,
        reason=reason,
        source=source,
    )
    operations = live_operations(
        match_id=match_id,
        snapshot_id=snapshot_id,
        event=event,
        minute=live_state.minute,
        home_score=live_state.home_score,
        away_score=live_state.away_score,
        side=side,
        line=line,
        effective_win_probability=quote.effective_win_probability,
        fair_decimal_odds=quote.fair_decimal_odds,
        expected_value=quote.expected_value,
        change_type=change_type,
    ).operations

    existing_index = _event_exists(ledger, event)
    if existing_index is None:
        updated_ledger = ledger.append(event)
        filtered, status = operations, "APPEND"
    else:
        updated_ledger = ledger
        filtered, status = _filter_retry_operations(
            row=row,
            ledger=ledger,
            event=event,
            match_id=match_id,
            snapshot_id=snapshot_id,
            operations=operations,
        )

    return ResumeResult(
        snapshot_id=snapshot_id,
        match_id=match_id,
        previous_state=ledger.current_state,
        current_state=updated_ledger.current_state,
        operations=filtered,
        status=status,
        live_quote=quote,
    )


def resume_outcome(
    state: AirtableState,
    *,
    snapshot_id: str,
    at: datetime,
    source: str,
    outcome: ProspectiveAHOutcome,
    closing_at: str | None = None,
    reason: str = "Prospective AH outcome recorded.",
) -> ResumeResult:
    _aware(at, "at")
    row, match_id, ledger = _snapshot_context(state, snapshot_id)
    _validate_outcome_against_snapshot(row, outcome)
    fields = airtable_outcome_fields(outcome, closing_at=closing_at)

    expected = prospective_outcome_operation(
        snapshot_id=snapshot_id,
        outcome_fields=fields,
    )
    existing_settlement_id = str(row.get("Settlement ID", "")).strip()
    expected_settlement_id = str(expected.fields["Settlement ID"])

    entered = any(
        event.state in {
            LifecycleState.ENTERED_PROTECTED,
            LifecycleState.ENTERED_DECAY,
        }
        for event in ledger.events
    )

    if not entered:
        if ledger.current_state != LifecycleState.PASSED:
            if (
                _prospective_settled(row)
                and existing_settlement_id == expected_settlement_id
            ):
                return ResumeResult(
                    snapshot_id=snapshot_id,
                    match_id=match_id,
                    previous_state=ledger.current_state,
                    current_state=ledger.current_state,
                    operations=(),
                    status="ALREADY_RECORDED",
                )
            raise ValueError(
                "Non-entered outcome requires an explicit PASS lifecycle."
            )
        if (
            _prospective_settled(row)
            and existing_settlement_id == expected_settlement_id
        ):
            return ResumeResult(
                snapshot_id=snapshot_id,
                match_id=match_id,
                previous_state=ledger.current_state,
                current_state=ledger.current_state,
                operations=(),
                status="ALREADY_RECORDED",
            )
        return ResumeResult(
            snapshot_id=snapshot_id,
            match_id=match_id,
            previous_state=ledger.current_state,
            current_state=ledger.current_state,
            operations=(expected,),
            status="APPEND_PROSPECTIVE_OUTCOME",
        )

    event = make_event(
        LifecycleState.SETTLED,
        at=at,
        reason=reason,
        source=source,
    )
    operations = settlement_operations(
        match_id=match_id,
        snapshot_id=snapshot_id,
        event=event,
        outcome_fields=fields,
    ).operations

    existing_index = _event_exists(ledger, event)
    if existing_index is None:
        updated_ledger = ledger.append(event)
        filtered, status = operations, "APPEND"
    else:
        updated_ledger = ledger
        filtered, status = _filter_retry_operations(
            row=row,
            ledger=ledger,
            event=event,
            match_id=match_id,
            snapshot_id=snapshot_id,
            operations=operations,
        )

    return ResumeResult(
        snapshot_id=snapshot_id,
        match_id=match_id,
        previous_state=ledger.current_state,
        current_state=updated_ledger.current_state,
        operations=filtered,
        status=status,
    )


def resume_audit(
    state: AirtableState,
    *,
    snapshot_id: str,
    at: datetime,
    source: str,
    reason: str = "Operational audit completed.",
) -> ResumeResult:
    _aware(at, "at")
    row, match_id, ledger = _snapshot_context(state, snapshot_id)
    event = make_event(
        LifecycleState.AUDITED,
        at=at,
        reason=reason,
        source=source,
    )

    event_op = lifecycle_operation(
        match_id,
        event,
        snapshot_id=snapshot_id,
        payload={"kind": "audit_complete"},
    )
    snapshot_update = BridgeOperation(
        table=PROSPECTIVE_AH_TABLE_ID,
        key_field="Snapshot ID",
        key_value=snapshot_id,
        fields={
            "Snapshot ID": snapshot_id,
            "Last Lifecycle Event ID": event_op.key_value,
        },
        kind=BridgeOperationKind.UPDATE_EXISTING,
    )
    operations = (snapshot_update, event_op)

    existing_index = _event_exists(ledger, event)
    if existing_index is None:
        updated_ledger = ledger.append(event)
        filtered, status = operations, "APPEND"
    else:
        updated_ledger = ledger
        filtered, status = _filter_retry_operations(
            row=row,
            ledger=ledger,
            event=event,
            match_id=match_id,
            snapshot_id=snapshot_id,
            operations=operations,
        )

    return ResumeResult(
        snapshot_id=snapshot_id,
        match_id=match_id,
        previous_state=ledger.current_state,
        current_state=updated_ledger.current_state,
        operations=filtered,
        status=status,
    )
