from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class LifecycleState(str, Enum):
    IDENTIFIED = "IDENTIFIED"
    BOARD_SELECTED = "BOARD_SELECTED"
    SNAPSHOT_CAPTURED = "SNAPSHOT_CAPTURED"
    WAITING_ENTRY = "WAITING_ENTRY"
    ENTERED_PROTECTED = "ENTERED_PROTECTED"
    ENTERED_DECAY = "ENTERED_DECAY"
    PASSED = "PASSED"
    LIVE_REPRICED = "LIVE_REPRICED"
    SETTLED = "SETTLED"
    AUDITED = "AUDITED"


@dataclass(frozen=True)
class LifecycleEvent:
    state: LifecycleState
    at: datetime
    reason: str
    source: str

    def validate(self) -> None:
        if self.at.tzinfo is None:
            raise ValueError("Lifecycle event timestamp must be timezone-aware.")
        if not self.reason.strip():
            raise ValueError("Lifecycle event reason is required.")
        if not self.source.strip():
            raise ValueError("Lifecycle event source is required.")


@dataclass(frozen=True)
class DecisionLedger:
    match_id: str
    events: tuple[LifecycleEvent, ...]

    @property
    def current_state(self) -> LifecycleState:
        if not self.events:
            return LifecycleState.IDENTIFIED
        return self.events[-1].state

    def validate(self) -> None:
        if not self.match_id.strip():
            raise ValueError("match_id is required.")
        if not self.events:
            return

        previous_time = None
        previous_state = LifecycleState.IDENTIFIED
        for event in self.events:
            event.validate()
            if previous_time is not None and event.at < previous_time:
                raise ValueError("Lifecycle events must be chronological.")
            _validate_transition(previous_state, event.state)
            previous_time = event.at
            previous_state = event.state

    def append(self, event: LifecycleEvent) -> "DecisionLedger":
        self.validate()
        event.validate()
        if self.events and event.at < self.events[-1].at:
            raise ValueError("Lifecycle events must be chronological.")
        _validate_transition(self.current_state, event.state)
        return DecisionLedger(
            match_id=self.match_id,
            events=self.events + (event,),
        )


_ALLOWED: dict[LifecycleState, set[LifecycleState]] = {
    LifecycleState.IDENTIFIED: {
        LifecycleState.BOARD_SELECTED,
        LifecycleState.PASSED,
    },
    LifecycleState.BOARD_SELECTED: {
        LifecycleState.SNAPSHOT_CAPTURED,
        LifecycleState.PASSED,
    },
    LifecycleState.SNAPSHOT_CAPTURED: {
        LifecycleState.WAITING_ENTRY,
        LifecycleState.ENTERED_PROTECTED,
        LifecycleState.ENTERED_DECAY,
        LifecycleState.PASSED,
        LifecycleState.LIVE_REPRICED,
    },
    LifecycleState.WAITING_ENTRY: {
        LifecycleState.WAITING_ENTRY,
        LifecycleState.ENTERED_PROTECTED,
        LifecycleState.ENTERED_DECAY,
        LifecycleState.PASSED,
        LifecycleState.LIVE_REPRICED,
    },
    LifecycleState.ENTERED_PROTECTED: {
        LifecycleState.LIVE_REPRICED,
        LifecycleState.SETTLED,
    },
    LifecycleState.ENTERED_DECAY: {
        LifecycleState.LIVE_REPRICED,
        LifecycleState.SETTLED,
    },
    LifecycleState.LIVE_REPRICED: {
        LifecycleState.LIVE_REPRICED,
        LifecycleState.WAITING_ENTRY,
        LifecycleState.ENTERED_PROTECTED,
        LifecycleState.ENTERED_DECAY,
        LifecycleState.PASSED,
        LifecycleState.SETTLED,
    },
    LifecycleState.PASSED: {
        LifecycleState.AUDITED,
    },
    LifecycleState.SETTLED: {
        LifecycleState.AUDITED,
    },
    LifecycleState.AUDITED: set(),
}


def _validate_transition(
    previous: LifecycleState,
    current: LifecycleState,
) -> None:
    if current not in _ALLOWED[previous]:
        raise ValueError(
            f"Invalid lifecycle transition: {previous.value} -> {current.value}"
        )


def make_event(
    state: LifecycleState | str,
    *,
    at: datetime,
    reason: str,
    source: str,
) -> LifecycleEvent:
    if isinstance(state, str):
        state = LifecycleState(state)
    return LifecycleEvent(state=state, at=at, reason=reason, source=source)
