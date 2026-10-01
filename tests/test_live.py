import numpy as np
import pytest

from gah.live import (
    LiveAdjustment,
    LiveChangeType,
    LiveState,
    classify_live_change,
    conditional_final_score_matrix,
    reprice_live_ah,
)


def base_state(**kwargs) -> LiveState:
    values = {
        "minute": 0.0,
        "home_score": 0,
        "away_score": 0,
        "pre_match_home_xg": 1.6,
        "pre_match_away_xg": 1.1,
    }
    values.update(kwargs)
    return LiveState(**values)


def test_full_time_distribution_is_current_score_point_mass():
    state = base_state(minute=90.0, home_score=2, away_score=1)
    matrix = conditional_final_score_matrix(state)
    assert np.isclose(matrix.sum(), 1.0)
    assert np.isclose(matrix[2, 1], 1.0)


def test_time_decay_reduces_remaining_xg():
    start = base_state(minute=0)
    late = base_state(minute=75)
    assert late.remaining_home_xg < start.remaining_home_xg
    assert late.remaining_away_xg < start.remaining_away_xg


def test_one_goal_lead_late_improves_home_draw_no_bet_probability():
    level = base_state(minute=80, home_score=0, away_score=0)
    leading = base_state(minute=80, home_score=1, away_score=0)
    p_level = reprice_live_ah(level, side="home", line=0.0)
    p_lead = reprice_live_ah(leading, side="home", line=0.0)
    assert p_lead.effective_win_probability > p_level.effective_win_probability


def test_quote_returns_fair_odds_and_ev():
    quote = reprice_live_ah(
        base_state(minute=60, home_score=1, away_score=0),
        side="home",
        line=-0.5,
        decimal_odds=1.95,
    )
    assert 0.0 < quote.effective_win_probability < 1.0
    assert quote.fair_decimal_odds > 1.0
    assert quote.expected_value is not None


def test_material_adjustment_requires_reason():
    state = base_state(
        adjustment=LiveAdjustment(
            home_rate_multiplier=0.8,
            material_events=("red_card_home",),
        )
    )
    with pytest.raises(ValueError):
        state.validate()


def test_documented_adjustment_changes_remaining_rate():
    neutral = base_state(minute=45)
    adjusted = base_state(
        minute=45,
        adjustment=LiveAdjustment(
            home_rate_multiplier=0.75,
            away_rate_multiplier=1.15,
            material_events=("red_card_home",),
            reason="Observed home red card; explicit analyst adjustment.",
        ),
    )
    assert adjusted.remaining_home_xg < neutral.remaining_home_xg
    assert adjusted.remaining_away_xg > neutral.remaining_away_xg


def test_market_move_alone_does_not_create_football_state_change():
    state = base_state(minute=30)
    assert (
        classify_live_change(state, state, market_changed=True)
        == LiveChangeType.MARKET_ONLY
    )


def test_score_or_clock_change_is_match_state_change():
    old = base_state(minute=30, home_score=0)
    new = base_state(minute=31, home_score=1)
    assert classify_live_change(old, new) == LiveChangeType.MATCH_STATE


def test_adjustment_change_is_material_event():
    old = base_state(minute=30)
    new = base_state(
        minute=30,
        adjustment=LiveAdjustment(
            away_rate_multiplier=0.8,
            material_events=("key_sub_away",),
            reason="Away striker removed.",
        ),
    )
    assert classify_live_change(old, new) == LiveChangeType.MATERIAL_EVENT
