from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionState(str, Enum):
    WAIT = "WAIT"
    ENTER_PROTECTED = "ENTER_PROTECTED"
    ENTER_DECAY = "ENTER_DECAY"
    PASS = "PASS"


@dataclass(frozen=True)
class DecayStep:
    activates_at_minutes_to_kickoff: int
    minimum_line: float
    minimum_odds: float


@dataclass(frozen=True)
class ExecutionPlan:
    side: str
    protected_line: float
    protected_minimum_odds: float
    expiry_minutes_to_kickoff: int = 0
    decay_steps: tuple[DecayStep, ...] = ()

    def validate(self) -> None:
        if self.side not in {"home", "away"}:
            raise ValueError("side must be 'home' or 'away'.")
        if self.protected_minimum_odds <= 1.0:
            raise ValueError("protected_minimum_odds must be greater than 1.0.")
        if self.expiry_minutes_to_kickoff < 0:
            raise ValueError("expiry_minutes_to_kickoff cannot be negative.")

        previous_activation = None
        previous_line = self.protected_line
        previous_odds = self.protected_minimum_odds

        for step in self.decay_steps:
            if step.activates_at_minutes_to_kickoff < self.expiry_minutes_to_kickoff:
                raise ValueError("Decay step cannot activate after plan expiry.")
            if step.minimum_odds <= 1.0:
                raise ValueError("Decay minimum odds must be greater than 1.0.")
            if previous_activation is not None:
                if step.activates_at_minutes_to_kickoff >= previous_activation:
                    raise ValueError(
                        "Decay steps must progress toward kickoff in descending "
                        "minutes-to-kickoff order."
                    )
            if step.minimum_line > previous_line:
                raise ValueError(
                    "Decay may not demand a better line than the previous stage."
                )
            if step.minimum_odds > previous_odds:
                raise ValueError(
                    "Decay may not demand higher odds than the previous stage."
                )

            previous_activation = step.activates_at_minutes_to_kickoff
            previous_line = step.minimum_line
            previous_odds = step.minimum_odds


@dataclass(frozen=True)
class ExecutionDecision:
    state: ExecutionState
    active_minimum_line: float
    active_minimum_odds: float
    minutes_to_kickoff: int
    line_ok: bool
    odds_ok: bool
    reason: str


def _active_thresholds(
    plan: ExecutionPlan,
    minutes_to_kickoff: int,
) -> tuple[float, float, bool]:
    line = plan.protected_line
    odds = plan.protected_minimum_odds
    decayed = False

    for step in plan.decay_steps:
        if minutes_to_kickoff <= step.activates_at_minutes_to_kickoff:
            line = step.minimum_line
            odds = step.minimum_odds
            decayed = True

    return float(line), float(odds), decayed


def evaluate_execution(
    *,
    plan: ExecutionPlan,
    current_line: float,
    current_odds: float,
    minutes_to_kickoff: int,
) -> ExecutionDecision:
    plan.validate()

    if current_odds <= 1.0:
        raise ValueError("current_odds must be greater than 1.0.")

    minutes = int(minutes_to_kickoff)
    if minutes < plan.expiry_minutes_to_kickoff:
        return ExecutionDecision(
            state=ExecutionState.PASS,
            active_minimum_line=float(plan.protected_line),
            active_minimum_odds=float(plan.protected_minimum_odds),
            minutes_to_kickoff=minutes,
            line_ok=False,
            odds_ok=False,
            reason="Execution plan expired; do not chase after the cutoff.",
        )

    minimum_line, minimum_odds, decayed = _active_thresholds(plan, minutes)
    line_ok = float(current_line) >= minimum_line
    odds_ok = float(current_odds) >= minimum_odds

    if line_ok and odds_ok:
        return ExecutionDecision(
            state=(
                ExecutionState.ENTER_DECAY
                if decayed
                else ExecutionState.ENTER_PROTECTED
            ),
            active_minimum_line=minimum_line,
            active_minimum_odds=minimum_odds,
            minutes_to_kickoff=minutes,
            line_ok=True,
            odds_ok=True,
            reason=(
                "Current line and odds meet the active decay threshold."
                if decayed
                else "Current line and odds meet the protected entry."
            ),
        )

    missing = []
    if not line_ok:
        missing.append("line")
    if not odds_ok:
        missing.append("odds")

    return ExecutionDecision(
        state=ExecutionState.WAIT,
        active_minimum_line=minimum_line,
        active_minimum_odds=minimum_odds,
        minutes_to_kickoff=minutes,
        line_ok=line_ok,
        odds_ok=odds_ok,
        reason=f"Wait: active {' and '.join(missing)} threshold not met.",
    )
