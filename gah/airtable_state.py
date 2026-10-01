from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Iterable, Mapping

import pandas as pd

from .airtable_bridge import (
    AirtableBridgeClient,
    LIFECYCLE_EVENTS_TABLE_ID,
    PROSPECTIVE_AH_TABLE_ID,
)
from .lifecycle import DecisionLedger, LifecycleEvent, LifecycleState


STATE_SCHEMA_VERSION = "v2.11"


def _aware_datetime(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{field} is required.")
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware.")
    return result


def _fields(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("fields", {})
    if not isinstance(value, Mapping):
        raise ValueError("Airtable record fields must be an object.")
    return dict(value)


@dataclass(frozen=True)
class AirtableState:
    prospective_records: tuple[dict[str, Any], ...]
    lifecycle_records: tuple[dict[str, Any], ...]
    prospective_frame: pd.DataFrame
    ledgers: tuple[DecisionLedger, ...]

    @property
    def snapshot_count(self) -> int:
        return int(len(self.prospective_frame))

    @property
    def lifecycle_event_count(self) -> int:
        return int(
            sum(len(ledger.events) for ledger in self.ledgers)
        )

    @property
    def match_count(self) -> int:
        ids: set[str] = set()
        if "Match ID" in self.prospective_frame.columns:
            ids.update(
                value
                for value in self.prospective_frame["Match ID"]
                .dropna()
                .astype(str)
                .tolist()
                if value.strip()
            )
        ids.update(ledger.match_id for ledger in self.ledgers)
        return len(ids)

    def ledger_for(self, match_id: str) -> DecisionLedger:
        matches = [
            ledger for ledger in self.ledgers
            if ledger.match_id == match_id
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Expected one ledger for {match_id}; found {len(matches)}."
            )
        return matches[0]

    def snapshot_row(self, snapshot_id: str) -> dict[str, Any]:
        if "Snapshot ID" not in self.prospective_frame.columns:
            raise KeyError("Snapshot ID column is unavailable.")
        part = self.prospective_frame[
            self.prospective_frame["Snapshot ID"].astype(str)
            == str(snapshot_id)
        ]
        if len(part) != 1:
            raise KeyError(
                f"Expected one snapshot {snapshot_id}; found {len(part)}."
            )
        return part.iloc[0].to_dict()

    def summary(self) -> dict[str, Any]:
        pending = 0
        settled = 0
        if "Outcome Status" in self.prospective_frame.columns:
            status = (
                self.prospective_frame["Outcome Status"]
                .fillna("")
                .astype(str)
                .str.lower()
            )
            settled = int((status == "settled").sum())
            pending = int((status != "settled").sum())
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "snapshots": self.snapshot_count,
            "matches": self.match_count,
            "lifecycle_events": self.lifecycle_event_count,
            "pending_snapshots": pending,
            "settled_snapshots": settled,
        }

    def to_export(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "prospective_records": list(self.prospective_records),
            "lifecycle_records": list(self.lifecycle_records),
            "summary": self.summary(),
        }


def prospective_frame_from_records(
    records: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        fields = _fields(record)
        record_id = record.get("id")
        if record_id not in (None, ""):
            fields["Airtable Record ID"] = str(record_id)
        rows.append(fields)

    frame = pd.DataFrame(rows)
    if "Snapshot ID" in frame.columns:
        ids = frame["Snapshot ID"].fillna("").astype(str)
        present = ids.str.strip() != ""
        duplicates = sorted(
            set(ids[present & ids.duplicated(keep=False)].tolist())
        )
        if duplicates:
            raise ValueError(
                "Duplicate Snapshot ID values in Airtable state: "
                + ", ".join(duplicates)
            )
        frame = frame.sort_values(
            ["Snapshot ID"],
            kind="stable",
        ).reset_index(drop=True)
    return frame


def lifecycle_ledgers_from_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[DecisionLedger, ...]:
    by_match: dict[str, list[tuple[str, LifecycleEvent]]] = {}
    seen_event_ids: set[str] = set()

    for record in records:
        fields = _fields(record)
        event_id = str(fields.get("Event ID", "")).strip()
        match_id = str(fields.get("Match ID", "")).strip()
        state = str(fields.get("State", "")).strip()
        reason = str(fields.get("Reason", "")).strip()
        source = str(fields.get("Source", "")).strip()

        if not event_id:
            raise ValueError("Lifecycle record is missing Event ID.")
        if event_id in seen_event_ids:
            raise ValueError(f"Duplicate lifecycle Event ID: {event_id}")
        seen_event_ids.add(event_id)
        if not match_id:
            raise ValueError(
                f"Lifecycle event {event_id} is missing Match ID."
            )
        event = LifecycleEvent(
            state=LifecycleState(state),
            at=_aware_datetime(
                fields.get("Event At"),
                field="Event At",
            ),
            reason=reason,
            source=source,
        )
        event.validate()
        by_match.setdefault(match_id, []).append((event_id, event))

    ledgers: list[DecisionLedger] = []
    for match_id, rows in sorted(by_match.items()):
        ordered = sorted(
            rows,
            key=lambda item: (
                item[1].at,
                item[0],
            ),
        )
        ledger = DecisionLedger(
            match_id=match_id,
            events=tuple(event for _, event in ordered),
        )
        ledger.validate()
        ledgers.append(ledger)
    return tuple(ledgers)


def state_from_records(
    prospective_records: Iterable[Mapping[str, Any]],
    lifecycle_records: Iterable[Mapping[str, Any]],
) -> AirtableState:
    prospective = tuple(dict(record) for record in prospective_records)
    lifecycle = tuple(dict(record) for record in lifecycle_records)
    return AirtableState(
        prospective_records=prospective,
        lifecycle_records=lifecycle,
        prospective_frame=prospective_frame_from_records(prospective),
        ledgers=lifecycle_ledgers_from_records(lifecycle),
    )


def load_airtable_state(
    client: AirtableBridgeClient,
) -> AirtableState:
    return state_from_records(
        client.list_records(PROSPECTIVE_AH_TABLE_ID),
        client.list_records(LIFECYCLE_EVENTS_TABLE_ID),
    )


def state_from_export(payload: Mapping[str, Any]) -> AirtableState:
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version and schema_version != STATE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Airtable state export version: "
            f"{schema_version}"
        )
    prospective = payload.get("prospective_records", [])
    lifecycle = payload.get("lifecycle_records", [])
    if not isinstance(prospective, list) or not isinstance(lifecycle, list):
        raise ValueError(
            "State export prospective_records/lifecycle_records "
            "must be arrays."
        )
    return state_from_records(prospective, lifecycle)


def load_state_export(path: str) -> AirtableState:
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("State export must be a JSON object.")
    return state_from_export(payload)
