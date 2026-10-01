from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from .airtable_bridge import (
    AirtableBridgeClient,
    BridgeOperation,
    execution_operations,
    lifecycle_operation,
    live_operations,
    prospective_outcome_operation,
    settlement_operations,
    snapshot_operation,
)
from .board_capacity import (
    BoardCandidate,
    BoardDecision,
    BoardPlan,
    BoardStatus,
    plan_board,
)
from .execution import (
    ExecutionDecision,
    ExecutionPlan,
    ExecutionState,
    evaluate_execution,
)
from .lifecycle import DecisionLedger, LifecycleState, make_event
from .live import LiveAHQuote, LiveState, reprice_live_ah
from .prospective import (
    ProspectiveAHOutcome,
    ProspectiveAHSnapshot,
    airtable_outcome_fields,
    airtable_snapshot_fields,
    settlement_net_return,
    validate_snapshot,
)


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware.")


def _parse_snapshot_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _aware(parsed, "snapshot_time_utc")
    return parsed


@dataclass(frozen=True)
class LiveSessionObservation:
    at: datetime
    state: LiveState
    side: str
    line: float
    decimal_odds: float | None = None
    change_type: str | None = None
    reason: str = "Live AH re-priced from explicit football state."
    source: str = "session"

    def validate(self) -> None:
        _aware(self.at, "live.at")
        self.state.validate()
        if self.side not in {"home", "away"}:
            raise ValueError("live.side must be home or away.")
        if self.decimal_odds is not None and self.decimal_odds <= 1.0:
            raise ValueError("live decimal_odds must be greater than 1.0.")
        if not self.reason.strip() or not self.source.strip():
            raise ValueError("live reason/source are required.")


@dataclass(frozen=True)
class OutcomeObservation:
    at: datetime
    outcome: ProspectiveAHOutcome
    closing_at: str | None = None
    reason: str = "Prospective AH outcome recorded."
    source: str = "session"

    def validate(self) -> None:
        _aware(self.at, "outcome.at")
        if not self.reason.strip() or not self.source.strip():
            raise ValueError("outcome reason/source are required.")


@dataclass(frozen=True)
class BoardSessionCandidate:
    board: BoardCandidate
    snapshot_id: str
    snapshot: ProspectiveAHSnapshot
    source: str
    board_selected_at: datetime
    execution_at: datetime
    execution_plan: ExecutionPlan
    current_line: float
    current_odds: float
    minutes_to_kickoff: int
    live: LiveSessionObservation | None = None
    outcome: OutcomeObservation | None = None

    def validate(self) -> None:
        self.board.validate()
        validate_snapshot(self.snapshot)
        self.execution_plan.validate()
        _aware(self.board_selected_at, "board_selected_at")
        _aware(self.execution_at, "execution_at")
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id is required.")
        if not self.source.strip():
            raise ValueError("source is required.")
        if self.board.match_id != self.snapshot.fixture_key:
            raise ValueError("board.match_id must equal snapshot.fixture_key.")
        if self.execution_plan.side != self.snapshot.side:
            raise ValueError(
                "execution_plan.side must equal prospective snapshot side."
            )
        snap_at = _parse_snapshot_time(self.snapshot.snapshot_time_utc)
        if self.board_selected_at > snap_at:
            raise ValueError("Board selection cannot occur after snapshot capture.")
        if snap_at > self.execution_at:
            raise ValueError("Execution observation cannot precede snapshot capture.")
        if self.live is not None:
            self.live.validate()
            if self.live.side != self.snapshot.side:
                raise ValueError("live.side must equal prospective snapshot side.")
            if self.live.at < self.execution_at:
                raise ValueError("Live repricing cannot precede execution observation.")
        if self.outcome is not None:
            self.outcome.validate()
            out = self.outcome.outcome
            if out.fixture_key != self.board.match_id:
                raise ValueError("Outcome fixture_key must equal board match_id.")
            if out.captured_side != self.snapshot.side:
                raise ValueError("Outcome captured_side must equal snapshot side.")
            if abs(float(out.captured_line) - float(self.snapshot.line)) > 1e-9:
                raise ValueError("Outcome captured_line must equal snapshot line.")
            if (
                abs(
                    float(out.captured_market_probability)
                    - float(self.snapshot.market_probability)
                )
                > 1e-9
            ):
                raise ValueError(
                    "Outcome captured market probability must equal snapshot."
                )
            latest = self.live.at if self.live is not None else self.execution_at
            if self.outcome.at < latest:
                raise ValueError("Outcome cannot precede prior session observations.")


@dataclass(frozen=True)
class SessionMatchResult:
    board_decision: BoardDecision
    ledger: DecisionLedger
    operations: tuple[BridgeOperation, ...]
    execution: ExecutionDecision | None
    live_quote: LiveAHQuote | None
    audit_row: dict[str, Any] | None


@dataclass(frozen=True)
class BoardSessionResult:
    board_plan: BoardPlan
    matches: tuple[SessionMatchResult, ...]

    @property
    def operations(self) -> tuple[BridgeOperation, ...]:
        return tuple(
            op
            for match in self.matches
            for op in match.operations
        )

    @property
    def audit_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            match.audit_row
            for match in self.matches
            if match.audit_row is not None
        )

    def apply(self, client: AirtableBridgeClient) -> tuple[Any, ...]:
        # Operations are deliberately sequential. Snapshot create/update,
        # execution, live and outcome may all target the same canonical row.
        return tuple(client.apply(op) for op in self.operations)


_EXECUTION_TO_LIFECYCLE = {
    ExecutionState.WAIT: LifecycleState.WAITING_ENTRY,
    ExecutionState.ENTER_PROTECTED: LifecycleState.ENTERED_PROTECTED,
    ExecutionState.ENTER_DECAY: LifecycleState.ENTERED_DECAY,
    ExecutionState.PASS: LifecycleState.PASSED,
}


def _audit_row(
    candidate: BoardSessionCandidate,
    decision: ExecutionDecision,
) -> dict[str, Any]:
    entered = decision.state in {
        ExecutionState.ENTER_PROTECTED,
        ExecutionState.ENTER_DECAY,
    }
    entry_type = None
    if decision.state == ExecutionState.ENTER_PROTECTED:
        entry_type = "protected"
    elif decision.state == ExecutionState.ENTER_DECAY:
        entry_type = "decay"

    outcome = candidate.outcome.outcome if candidate.outcome else None
    settlement = outcome.settlement_category if outcome else None
    net_return = None
    if entered and outcome is not None:
        net_return = settlement_net_return(
            outcome.settlement_category,
            candidate.current_odds,
        )

    return {
        "snapshot_id": candidate.snapshot_id,
        "match_id": candidate.board.match_id,
        "competition": candidate.snapshot.competition,
        "captured_line": candidate.snapshot.line,
        "market_probability": candidate.snapshot.market_probability,
        "gah_probability": candidate.snapshot.gah_probability,
        "entered": entered,
        "entry_type": entry_type,
        "entry_line": candidate.current_line if entered else None,
        "entry_odds": candidate.current_odds if entered else None,
        "settlement": settlement,
        "net_return": net_return,
        "closing_line": None if outcome is None else outcome.closing_line,
        "closing_market_probability": (
            None if outcome is None else outcome.closing_market_probability
        ),
    }


def _selected_match(
    candidate: BoardSessionCandidate,
    board_decision: BoardDecision,
) -> SessionMatchResult:
    candidate.validate()
    ledger = DecisionLedger(candidate.board.match_id, ())
    operations: list[BridgeOperation] = []

    board_event = make_event(
        LifecycleState.BOARD_SELECTED,
        at=candidate.board_selected_at,
        reason="Selected by explicit v2.5 board-capacity plan.",
        source=candidate.source,
    )
    ledger = ledger.append(board_event)
    operations.append(
        lifecycle_operation(candidate.board.match_id, board_event)
    )

    snapshot_at = _parse_snapshot_time(candidate.snapshot.snapshot_time_utc)
    snapshot_event = make_event(
        LifecycleState.SNAPSHOT_CAPTURED,
        at=snapshot_at,
        reason="Forward-only prospective AH snapshot captured.",
        source=candidate.source,
    )
    ledger = ledger.append(snapshot_event)
    operations.append(
        snapshot_operation(
            airtable_snapshot_fields(
                candidate.snapshot,
                snapshot_id=candidate.snapshot_id,
                kickoff=candidate.board.kickoff.isoformat(),
                source=candidate.source,
            )
        )
    )
    operations.append(
        lifecycle_operation(
            candidate.board.match_id,
            snapshot_event,
            snapshot_id=candidate.snapshot_id,
            payload={"kind": "prospective_snapshot"},
        )
    )

    execution = evaluate_execution(
        plan=candidate.execution_plan,
        current_line=candidate.current_line,
        current_odds=candidate.current_odds,
        minutes_to_kickoff=candidate.minutes_to_kickoff,
    )
    lifecycle_state = _EXECUTION_TO_LIFECYCLE[execution.state]
    execution_event = make_event(
        lifecycle_state,
        at=candidate.execution_at,
        reason=execution.reason,
        source=candidate.source,
    )
    ledger = ledger.append(execution_event)
    operations.extend(
        execution_operations(
            match_id=candidate.board.match_id,
            snapshot_id=candidate.snapshot_id,
            event=execution_event,
            execution_state=execution.state.value,
            current_line=candidate.current_line,
            current_odds=candidate.current_odds,
            active_minimum_line=execution.active_minimum_line,
            active_minimum_odds=execution.active_minimum_odds,
            minutes_to_kickoff=execution.minutes_to_kickoff,
        ).operations
    )

    entered = execution.state in {
        ExecutionState.ENTER_PROTECTED,
        ExecutionState.ENTER_DECAY,
    }
    live_quote = None
    if candidate.live is not None:
        if execution.state == ExecutionState.PASS:
            raise ValueError("A passed execution cannot later be live re-priced.")
        live = candidate.live
        live_quote = reprice_live_ah(
            live.state,
            side=live.side,
            line=live.line,
            decimal_odds=live.decimal_odds,
        )
        live_event = make_event(
            LifecycleState.LIVE_REPRICED,
            at=live.at,
            reason=live.reason,
            source=live.source,
        )
        ledger = ledger.append(live_event)
        operations.extend(
            live_operations(
                match_id=candidate.board.match_id,
                snapshot_id=candidate.snapshot_id,
                event=live_event,
                minute=live.state.minute,
                home_score=live.state.home_score,
                away_score=live.state.away_score,
                side=live.side,
                line=live.line,
                effective_win_probability=live_quote.effective_win_probability,
                fair_decimal_odds=live_quote.fair_decimal_odds,
                expected_value=live_quote.expected_value,
                change_type=live.change_type,
            ).operations
        )

    if candidate.outcome is not None:
        obs = candidate.outcome
        fields = airtable_outcome_fields(
            obs.outcome,
            closing_at=obs.closing_at,
        )
        if entered:
            settled_event = make_event(
                LifecycleState.SETTLED,
                at=obs.at,
                reason=obs.reason,
                source=obs.source,
            )
            ledger = ledger.append(settled_event)
            operations.extend(
                settlement_operations(
                    match_id=candidate.board.match_id,
                    snapshot_id=candidate.snapshot_id,
                    event=settled_event,
                    outcome_fields=fields,
                ).operations
            )
        else:
            if execution.state != ExecutionState.PASS:
                raise ValueError(
                    "Non-entered outcome requires an explicit PASS decision "
                    "before the outcome is attached."
                )
            operations.append(
                prospective_outcome_operation(
                    snapshot_id=candidate.snapshot_id,
                    outcome_fields=fields,
                )
            )

    ledger.validate()
    return SessionMatchResult(
        board_decision=board_decision,
        ledger=ledger,
        operations=tuple(operations),
        execution=execution,
        live_quote=live_quote,
        audit_row=_audit_row(candidate, execution),
    )


def orchestrate_board_session(
    candidates: Iterable[BoardSessionCandidate],
    *,
    max_matches: int = 8,
    max_concurrent: int = 3,
) -> BoardSessionResult:
    items = list(candidates)
    by_id: dict[str, BoardSessionCandidate] = {}
    for candidate in items:
        candidate.validate()
        match_id = candidate.board.match_id
        if match_id in by_id:
            raise ValueError(f"Duplicate session match_id: {match_id}")
        by_id[match_id] = candidate

    board = plan_board(
        [candidate.board for candidate in items],
        max_matches=max_matches,
        max_concurrent=max_concurrent,
    )

    results: list[SessionMatchResult] = []
    for decision in board.decisions:
        candidate = by_id[decision.candidate.match_id]
        if decision.status == BoardStatus.SELECTED:
            results.append(_selected_match(candidate, decision))
        else:
            results.append(
                SessionMatchResult(
                    board_decision=decision,
                    ledger=DecisionLedger(candidate.board.match_id, ()),
                    operations=(),
                    execution=None,
                    live_quote=None,
                    audit_row=None,
                )
            )

    return BoardSessionResult(board, tuple(results))
