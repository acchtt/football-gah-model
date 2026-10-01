from datetime import datetime, timezone

import pytest

from gah.board_capacity import BoardCandidate, BoardStatus, plan_board


def dt(hour: int, minute: int = 0):
    return datetime(2026, 10, 1, hour, minute, tzinfo=timezone.utc)


def candidate(match_id, kickoff, priority=1, before=30, after=90):
    return BoardCandidate(
        match_id=match_id,
        kickoff=kickoff,
        priority=priority,
        monitor_minutes_before=before,
        monitor_minutes_after=after,
    )


def test_higher_priority_wins_overlap_capacity():
    board = plan_board(
        [
            candidate("low", dt(12), priority=1),
            candidate("high", dt(12, 5), priority=3),
        ],
        max_matches=8,
        max_concurrent=1,
    )
    status = {d.candidate.match_id: d.status for d in board.decisions}
    assert status["high"] == BoardStatus.SELECTED
    assert status["low"] == BoardStatus.REJECTED_CONCURRENCY


def test_total_capacity_is_enforced():
    board = plan_board(
        [
            candidate("a", dt(10), priority=3, before=0, after=10),
            candidate("b", dt(11), priority=2, before=0, after=10),
            candidate("c", dt(12), priority=1, before=0, after=10),
        ],
        max_matches=2,
        max_concurrent=3,
    )
    assert len(board.selected) == 2
    status = {d.candidate.match_id: d.status for d in board.decisions}
    assert status["c"] == BoardStatus.REJECTED_TOTAL_CAPACITY


def test_non_overlapping_matches_fit_single_monitor():
    board = plan_board(
        [
            candidate("a", dt(10), before=0, after=30),
            candidate("b", dt(11), before=0, after=30),
        ],
        max_matches=8,
        max_concurrent=1,
    )
    assert len(board.selected) == 2


def test_touching_windows_do_not_overlap():
    board = plan_board(
        [
            candidate("a", dt(10), before=0, after=60),
            candidate("b", dt(11), before=0, after=60),
        ],
        max_matches=8,
        max_concurrent=1,
    )
    assert len(board.selected) == 2


def test_duplicate_match_id_rejected():
    with pytest.raises(ValueError):
        plan_board(
            [candidate("a", dt(10)), candidate("a", dt(12))],
            max_matches=8,
            max_concurrent=2,
        )


def test_rejected_matches_remain_in_plan():
    board = plan_board(
        [
            candidate("a", dt(10), priority=2),
            candidate("b", dt(10), priority=1),
        ],
        max_matches=8,
        max_concurrent=1,
    )
    assert len(board.decisions) == 2
    assert len(board.rejected) == 1
