from datetime import datetime, timezone

import numpy as np

from gah.prospective import (
    ProspectiveAHOutcome,
    ProspectiveAHSnapshot,
    utc_timestamp,
    validate_snapshot,
)


def test_snapshot_metrics():
    snap = ProspectiveAHSnapshot(
        fixture_key="2026-10-01|A|B",
        snapshot_time_utc="2026-10-01T10:00:00+00:00",
        competition="EPL",
        home_team="A",
        away_team="B",
        side="home",
        line=-0.5,
        decimal_odds=1.95,
        market_probability=0.51,
        gah_probability=0.58,
    )
    validate_snapshot(snap)
    assert np.isclose(snap.edge, 0.07)
    assert np.isclose(snap.abs_edge, 0.07)
    assert np.isclose(snap.model_ev, 0.131)


def test_same_line_clv_only_when_comparable():
    same = ProspectiveAHOutcome(
        fixture_key="x",
        captured_side="home",
        captured_line=-0.5,
        closing_line=-0.5,
        captured_market_probability=0.51,
        closing_market_probability=0.54,
        settlement_category="full_win",
        realized_net_return=0.95,
    )
    moved = ProspectiveAHOutcome(
        fixture_key="x",
        captured_side="home",
        captured_line=-0.5,
        closing_line=-0.75,
        captured_market_probability=0.51,
        closing_market_probability=0.54,
        settlement_category="full_win",
        realized_net_return=0.95,
    )
    assert np.isclose(same.same_line_probability_clv, 0.03)
    assert moved.same_line_probability_clv is None
    assert np.isclose(moved.line_move, -0.25)


def test_timestamp_requires_timezone():
    aware = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
    assert utc_timestamp(aware).endswith("+00:00")

    try:
        utc_timestamp(datetime(2026, 10, 1, 10, 0))
    except ValueError as exc:
        assert "timezone-aware" in str(exc)
    else:
        raise AssertionError("Expected naive datetime to be rejected")
