import numpy as np

from gah.markets import price_total, price_handicap, expected_value_at_odds


def one_score(hg, ag, max_goals=5):
    m = np.zeros((max_goals + 1, max_goals + 1))
    m[hg, ag] = 1.0
    return m


def test_total_over_225_on_exact_two_goals_is_half_loss():
    p = price_total(one_score(1, 1), 2.25, "over")
    assert p["half_loss"] == 1.0


def test_total_over_275_on_exact_three_goals_is_half_win():
    p = price_total(one_score(2, 1), 2.75, "over")
    assert p["half_win"] == 1.0


def test_home_minus_025_draw_is_half_loss():
    p = price_handicap(one_score(1, 1), -0.25, "home")
    assert p["half_loss"] == 1.0


def test_home_plus_025_draw_is_half_win():
    p = price_handicap(one_score(1, 1), 0.25, "home")
    assert p["half_win"] == 1.0


def test_home_minus_10_one_goal_win_is_push():
    p = price_handicap(one_score(2, 1), -1.0, "home")
    assert p["push"] == 1.0


def test_fair_odds_full_binary():
    m = np.zeros((3, 3))
    m[1, 0] = 0.6
    m[0, 1] = 0.4
    p = price_handicap(m, -0.5, "home")
    assert abs(p["fair_decimal_odds"] - (1 / 0.6)) < 1e-9
    assert abs(expected_value_at_odds(p, p["fair_decimal_odds"])) < 1e-9
