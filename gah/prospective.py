from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from .market_aware import devig_two_way


@dataclass(frozen=True)
class ProspectiveAHSnapshot:
    fixture_key: str
    snapshot_time_utc: str
    competition: str
    home_team: str
    away_team: str
    side: str
    line: float
    decimal_odds: float
    market_probability: float
    gah_probability: float

    @property
    def edge(self) -> float:
        return float(self.gah_probability) - float(self.market_probability)

    @property
    def abs_edge(self) -> float:
        return abs(self.edge)

    @property
    def model_ev(self) -> float:
        return float(self.gah_probability) * float(self.decimal_odds) - 1.0

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "edge": self.edge,
                "abs_edge": self.abs_edge,
                "model_ev": self.model_ev,
            }
        )
        return record


@dataclass(frozen=True)
class ProspectiveAHOutcome:
    fixture_key: str
    captured_side: str
    captured_line: float
    closing_line: float | None
    captured_market_probability: float
    closing_market_probability: float | None
    settlement_category: str
    realized_net_return: float

    @property
    def same_line_probability_clv(self) -> float | None:
        if self.closing_line is None or self.closing_market_probability is None:
            return None
        if abs(float(self.closing_line) - float(self.captured_line)) > 1e-9:
            return None
        return float(self.closing_market_probability) - float(
            self.captured_market_probability
        )

    @property
    def line_move(self) -> float | None:
        if self.closing_line is None:
            return None
        return float(self.closing_line) - float(self.captured_line)


def utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Prospective timestamps must be timezone-aware.")
    return value.astimezone().isoformat()


def validate_snapshot(snapshot: ProspectiveAHSnapshot) -> None:
    if snapshot.side not in {"home", "away"}:
        raise ValueError("side must be 'home' or 'away'.")
    if snapshot.decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be greater than 1.0.")
    for name, value in {
        "market_probability": snapshot.market_probability,
        "gah_probability": snapshot.gah_probability,
    }.items():
        if not 0.0 < float(value) < 1.0:
            raise ValueError(f"{name} must be strictly between 0 and 1.")


def snapshot_from_two_way_market(
    *,
    fixture_key: str,
    snapshot_time_utc: str,
    competition: str,
    home_team: str,
    away_team: str,
    side: str,
    line: float,
    selected_odds: float,
    opposing_odds: float,
    gah_probability: float,
) -> ProspectiveAHSnapshot:
    market_probability, _ = devig_two_way(selected_odds, opposing_odds)
    snapshot = ProspectiveAHSnapshot(
        fixture_key=fixture_key,
        snapshot_time_utc=snapshot_time_utc,
        competition=competition,
        home_team=home_team,
        away_team=away_team,
        side=side,
        line=float(line),
        decimal_odds=float(selected_odds),
        market_probability=float(market_probability),
        gah_probability=float(gah_probability),
    )
    validate_snapshot(snapshot)
    return snapshot


def settlement_net_return(category: str, decimal_odds: float) -> float:
    odds = float(decimal_odds)
    if odds <= 1.0:
        raise ValueError("decimal_odds must be greater than 1.0.")
    mapping = {
        "full_win": odds - 1.0,
        "half_win": 0.5 * (odds - 1.0),
        "push": 0.0,
        "half_loss": -0.5,
        "full_loss": -1.0,
    }
    if category not in mapping:
        raise ValueError(f"Unknown settlement category: {category}")
    return float(mapping[category])


def airtable_snapshot_fields(
    snapshot: ProspectiveAHSnapshot,
    *,
    snapshot_id: str,
    kickoff: str,
    source: str,
    protocol_version: str = "v2.1",
) -> dict[str, Any]:
    validate_snapshot(snapshot)
    return {
        "Snapshot ID": snapshot_id,
        "Protocol Version": protocol_version,
        "Match ID": snapshot.fixture_key,
        "Kickoff": kickoff,
        "Captured At": snapshot.snapshot_time_utc,
        "Competition": snapshot.competition,
        "Home Team": snapshot.home_team,
        "Away Team": snapshot.away_team,
        "Side": snapshot.side,
        "AH Line": snapshot.line,
        "Odds": snapshot.decimal_odds,
        "Bookmaker / Source": source,
        "Market Probability": snapshot.market_probability,
        "GAH Probability": snapshot.gah_probability,
        "GAH Edge": snapshot.edge,
        "Absolute Edge": snapshot.abs_edge,
        "Model EV": snapshot.model_ev,
        "Outcome Status": "Pending",
    }


def airtable_outcome_fields(
    outcome: ProspectiveAHOutcome,
    *,
    closing_at: str | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "Outcome Status": "Settled",
        "Closing AH Line": outcome.closing_line,
        "Closing Market Probability": outcome.closing_market_probability,
        "Same-Line Probability CLV": outcome.same_line_probability_clv,
        "Line Move": outcome.line_move,
        "Settlement": outcome.settlement_category,
        "Net Return": outcome.realized_net_return,
    }
    if closing_at is not None:
        fields["Closing At"] = closing_at
    return {k: v for k, v in fields.items() if v is not None}
