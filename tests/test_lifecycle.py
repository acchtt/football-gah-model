from datetime import datetime, timedelta, timezone

import pytest

from gah.lifecycle import (
    DecisionLedger,
    LifecycleState,
    make_event,
)


T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def event(state, minutes, reason="test"):
    return make_event(
        state,
        at=T0 + timedelta(minutes=minutes),
        reason=reason,
        source="unit-test",
    )


def test_normal_protected_entry_lifecycle():
    ledger = DecisionLedger("m1", ())
    for state, minute in [
        (LifecycleState.BOARD_SELECTED, 0),
        (LifecycleState.SNAPSHOT_CAPTURED, 1),
        (LifecycleState.WAITING_ENTRY, 2),
        (LifecycleState.ENTERED_PROTECTED, 3),
        (LifecycleState.SETTLED, 120),
        (LifecycleState.AUDITED, 121),
    ]:
        ledger = ledger.append(event(state, minute))
    ledger.validate()
    assert ledger.current_state == LifecycleState.AUDITED


def test_decay_entry_lifecycle():
    ledger = DecisionLedger("m2", ())
    ledger = ledger.append(event("BOARD_SELECTED", 0))
    ledger = ledger.append(event("SNAPSHOT_CAPTURED", 1))
    ledger = ledger.append(event("WAITING_ENTRY", 2))
    ledger = ledger.append(event("ENTERED_DECAY", 30))
    ledger = ledger.append(event("SETTLED", 150))
    assert ledger.current_state == LifecycleState.SETTLED


def test_passed_match_cannot_become_entry():
    ledger = DecisionLedger("m3", ())
    ledger = ledger.append(event("BOARD_SELECTED", 0))
    ledger = ledger.append(event("PASSED", 5))
    with pytest.raises(ValueError):
        ledger.append(event("ENTERED_PROTECTED", 6))


def test_live_repricing_requires_snapshot():
    ledger = DecisionLedger("m4", ())
    ledger = ledger.append(event("BOARD_SELECTED", 0))
    with pytest.raises(ValueError):
        ledger.append(event("LIVE_REPRICED", 5))


def test_live_repricing_can_precede_entry():
    ledger = DecisionLedger("m5", ())
    ledger = ledger.append(event("BOARD_SELECTED", 0))
    ledger = ledger.append(event("SNAPSHOT_CAPTURED", 1))
    ledger = ledger.append(event("LIVE_REPRICED", 50))
    ledger = ledger.append(event("ENTERED_DECAY", 51))
    assert ledger.current_state == LifecycleState.ENTERED_DECAY


def test_audited_is_terminal():
    ledger = DecisionLedger("m6", ())
    ledger = ledger.append(event("PASSED", 0))
    ledger = ledger.append(event("AUDITED", 1))
    with pytest.raises(ValueError):
        ledger.append(event("BOARD_SELECTED", 2))


def test_events_must_be_chronological():
    ledger = DecisionLedger("m7", ())
    ledger = ledger.append(event("BOARD_SELECTED", 5))
    with pytest.raises(ValueError):
        ledger.append(event("SNAPSHOT_CAPTURED", 4))
