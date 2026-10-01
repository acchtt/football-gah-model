from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from gah.prospective_review import (
    prospective_readiness,
    prospective_review,
)


T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def frame(n=100, span_days=30):
    rows = []
    for i in range(n):
        day = 0 if n == 1 else round(i * span_days / (n - 1))
        line = -0.5 if i % 2 == 0 else 0.75
        same = i % 3 != 0
        rows.append(
            {
                "snapshot_id": f"s{i}",
                "captured_at": (T0 + timedelta(days=day)).isoformat(),
                "outcome_status": "Settled",
                "settlement": "full_win" if i % 2 == 0 else "full_loss",
                "net_return": 0.9 if i % 2 == 0 else -1.0,
                "captured_line": line,
                "closing_line": line if same else line - 0.25,
                "market_probability": 0.50,
                "closing_market_probability": 0.52 if same else 0.53,
            }
        )
    return pd.DataFrame(rows)


def test_gate_waits_for_snapshot_count():
    r = prospective_readiness(frame(n=99, span_days=30))
    assert r.status == "WAIT"
    assert r.snapshots_remaining == 1
    assert r.days_remaining == 0


def test_gate_waits_for_calendar_span():
    r = prospective_readiness(frame(n=100, span_days=29))
    assert r.status == "WAIT"
    assert r.snapshots_remaining == 0
    assert r.days_remaining == 1


def test_gate_opens_only_when_both_conditions_met():
    r = prospective_readiness(frame(n=100, span_days=30))
    assert r.ready
    assert r.settled_snapshots == 100
    assert r.observation_span_days >= 30


def test_formal_review_locked_before_ready():
    with pytest.raises(ValueError):
        prospective_review(frame(n=80, span_days=40))


def test_ready_review_reports_clv_coverage_without_filling_missing():
    f = frame(n=100, span_days=30)
    s = prospective_review(f)
    assert s.settled_snapshots == 100
    assert 0 < s.same_line_clv_n < 100
    assert np.isclose(
        s.same_line_clv_coverage,
        s.same_line_clv_n / s.settled_snapshots,
    )
    assert np.isclose(s.mean_same_line_clv, 0.02)
    assert np.isclose(s.positive_same_line_clv_rate, 1.0)


def test_duplicate_snapshot_ids_rejected():
    f = frame(n=100, span_days=30)
    f.loc[1, "snapshot_id"] = f.loc[0, "snapshot_id"]
    with pytest.raises(ValueError):
        prospective_readiness(f)
