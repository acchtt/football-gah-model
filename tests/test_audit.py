import numpy as np
import pandas as pd

from gah.audit import enrich_audit_rows, grouped_audit, summarize_audit


def sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "snapshot_id": "a",
                "match_id": "m1",
                "competition": "EPL",
                "captured_line": -0.5,
                "captured_odds": 1.90,
                "market_probability": 0.50,
                "gah_probability": 0.56,
                "execution_state": "ENTER_PROTECTED",
                "entry_type": "protected",
                "entered": True,
                "closing_line": -0.5,
                "closing_market_probability": 0.53,
                "settlement": "full_win",
                "net_return": 0.90,
            },
            {
                "snapshot_id": "b",
                "match_id": "m2",
                "competition": "EPL",
                "captured_line": 0.75,
                "captured_odds": 1.88,
                "market_probability": 0.51,
                "gah_probability": 0.55,
                "execution_state": "ENTER_DECAY",
                "entry_type": "decay",
                "entered": True,
                "closing_line": 0.50,
                "closing_market_probability": 0.54,
                "settlement": "full_loss",
                "net_return": -1.0,
            },
            {
                "snapshot_id": "c",
                "match_id": "m3",
                "competition": "SERIE_A",
                "captured_line": -0.25,
                "captured_odds": 1.95,
                "market_probability": 0.49,
                "gah_probability": 0.54,
                "execution_state": "PASS",
                "entry_type": "",
                "entered": False,
                "closing_line": -0.25,
                "closing_market_probability": 0.48,
                "settlement": None,
                "net_return": np.nan,
            },
        ]
    )


def test_summary_separates_entries_and_clv():
    s = summarize_audit(sample())
    assert s.observations == 3
    assert s.entered == 2
    assert np.isclose(s.roi, -0.05)
    assert s.same_line_clv_n == 2
    assert np.isclose(s.mean_same_line_clv, 0.01)
    assert np.isclose(s.positive_same_line_clv_rate, 0.5)
    assert s.protected_entries == 1
    assert s.decay_entries == 1


def test_line_move_does_not_fake_same_line_clv():
    out = enrich_audit_rows(sample())
    assert np.isnan(out.loc[1, "same_line_probability_clv"])
    assert np.isclose(out.loc[1, "line_move"], -0.25)


def test_diagnostic_buckets_are_descriptive():
    out = enrich_audit_rows(sample())
    assert list(out["diagnostic_bucket"]) == [
        "ENTERED_WIN",
        "ENTERED_LOSS",
        "NOT_ENTERED",
    ]


def test_grouped_audit():
    g = grouped_audit(sample(), ("competition",))
    assert set(g["competition"]) == {"EPL", "SERIE_A"}
