from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


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
