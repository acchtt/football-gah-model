from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class BoardStatus(str, Enum):
    SELECTED = "SELECTED"
    REJECTED_TOTAL_CAPACITY = "REJECTED_TOTAL_CAPACITY"
    REJECTED_CONCURRENCY = "REJECTED_CONCURRENCY"


@dataclass(frozen=True)
class BoardCandidate:
    match_id: str
    kickoff: datetime
    priority: int
    monitor_minutes_before: int = 60
    monitor_minutes_after: int = 120

    def validate(self) -> None:
        if not self.match_id.strip():
            raise ValueError("match_id is required.")
        if self.kickoff.tzinfo is None:
            raise ValueError("kickoff must be timezone-aware.")
        if self.monitor_minutes_before < 0:
            raise ValueError("monitor_minutes_before cannot be negative.")
        if self.monitor_minutes_after < 0:
            raise ValueError("monitor_minutes_after cannot be negative.")

    @property
    def watch_start(self) -> datetime:
        self.validate()
        return self.kickoff - timedelta(minutes=self.monitor_minutes_before)

    @property
    def watch_end(self) -> datetime:
        self.validate()
        return self.kickoff + timedelta(minutes=self.monitor_minutes_after)


@dataclass(frozen=True)
class BoardDecision:
    candidate: BoardCandidate
    status: BoardStatus
    reason: str


@dataclass(frozen=True)
class BoardPlan:
    decisions: tuple[BoardDecision, ...]

    @property
    def selected(self) -> tuple[BoardCandidate, ...]:
        return tuple(
            d.candidate
            for d in self.decisions
            if d.status == BoardStatus.SELECTED
        )

    @property
    def rejected(self) -> tuple[BoardDecision, ...]:
        return tuple(
            d for d in self.decisions if d.status != BoardStatus.SELECTED
        )


def _max_concurrency(candidates: list[BoardCandidate]) -> int:
    events: list[tuple[datetime, int]] = []
    for c in candidates:
        # End events sort before start events at the same instant.
        events.append((c.watch_start, 1))
        events.append((c.watch_end, -1))

    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def plan_board(
    candidates: list[BoardCandidate] | tuple[BoardCandidate, ...],
    *,
    max_matches: int = 8,
    max_concurrent: int = 3,
) -> BoardPlan:
    if max_matches < 1:
        raise ValueError("max_matches must be at least 1.")
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be at least 1.")

    items = list(candidates)
    seen: set[str] = set()
    for candidate in items:
        candidate.validate()
        if candidate.match_id in seen:
            raise ValueError(f"Duplicate match_id: {candidate.match_id}")
        seen.add(candidate.match_id)

    ordered = sorted(
        items,
        key=lambda c: (-int(c.priority), c.kickoff, c.match_id),
    )

    selected: list[BoardCandidate] = []
    decisions: list[BoardDecision] = []

    for candidate in ordered:
        if len(selected) >= max_matches:
            decisions.append(
                BoardDecision(
                    candidate,
                    BoardStatus.REJECTED_TOTAL_CAPACITY,
                    "Board total-match capacity already reached.",
                )
            )
            continue

        proposed = selected + [candidate]
        if _max_concurrency(proposed) > max_concurrent:
            decisions.append(
                BoardDecision(
                    candidate,
                    BoardStatus.REJECTED_CONCURRENCY,
                    "Adding this monitoring window would exceed concurrent capacity.",
                )
            )
            continue

        selected.append(candidate)
        decisions.append(
            BoardDecision(
                candidate,
                BoardStatus.SELECTED,
                "Accepted within total and concurrent monitoring capacity.",
            )
        )

    # Present the final plan chronologically while preserving every decision.
    decisions.sort(key=lambda d: (d.candidate.kickoff, d.candidate.match_id))
    return BoardPlan(tuple(decisions))
