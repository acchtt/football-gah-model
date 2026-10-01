import pytest

from gah.execution import (
    DecayStep,
    ExecutionPlan,
    ExecutionState,
    evaluate_execution,
)


def plan() -> ExecutionPlan:
    return ExecutionPlan(
        side="home",
        protected_line=-0.50,
        protected_minimum_odds=1.90,
        expiry_minutes_to_kickoff=0,
        decay_steps=(
            DecayStep(
                activates_at_minutes_to_kickoff=60,
                minimum_line=-0.75,
                minimum_odds=1.85,
            ),
            DecayStep(
                activates_at_minutes_to_kickoff=15,
                minimum_line=-1.00,
                minimum_odds=1.80,
            ),
        ),
    )


def test_protected_entry_before_decay():
    d = evaluate_execution(
        plan=plan(),
        current_line=-0.50,
        current_odds=1.92,
        minutes_to_kickoff=120,
    )
    assert d.state == ExecutionState.ENTER_PROTECTED


def test_wait_before_decay_if_target_missing():
    d = evaluate_execution(
        plan=plan(),
        current_line=-0.75,
        current_odds=1.92,
        minutes_to_kickoff=120,
    )
    assert d.state == ExecutionState.WAIT
    assert not d.line_ok


def test_first_decay_allows_predeclared_fallback():
    d = evaluate_execution(
        plan=plan(),
        current_line=-0.75,
        current_odds=1.86,
        minutes_to_kickoff=45,
    )
    assert d.state == ExecutionState.ENTER_DECAY
    assert d.active_minimum_line == -0.75


def test_second_decay_is_used_near_kickoff():
    d = evaluate_execution(
        plan=plan(),
        current_line=-1.00,
        current_odds=1.81,
        minutes_to_kickoff=10,
    )
    assert d.state == ExecutionState.ENTER_DECAY
    assert d.active_minimum_line == -1.00


def test_better_selected_side_line_is_numerically_larger():
    d = evaluate_execution(
        plan=plan(),
        current_line=-0.25,
        current_odds=1.90,
        minutes_to_kickoff=120,
    )
    assert d.state == ExecutionState.ENTER_PROTECTED


def test_expired_plan_passes():
    p = ExecutionPlan(
        side="away",
        protected_line=0.50,
        protected_minimum_odds=1.90,
        expiry_minutes_to_kickoff=5,
    )
    d = evaluate_execution(
        plan=p,
        current_line=0.75,
        current_odds=2.00,
        minutes_to_kickoff=4,
    )
    assert d.state == ExecutionState.PASS


def test_decay_cannot_get_stricter():
    p = ExecutionPlan(
        side="home",
        protected_line=-0.50,
        protected_minimum_odds=1.90,
        decay_steps=(
            DecayStep(60, -0.25, 1.85),
        ),
    )
    with pytest.raises(ValueError):
        p.validate()
