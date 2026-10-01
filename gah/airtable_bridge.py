from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from .lifecycle import LifecycleEvent, LifecycleState

AIRTABLE_BASE_ID = "app2wBjpvj69bCoMS"
PROSPECTIVE_AH_TABLE_ID = "tblk9kLKRqrnBOQ2k"
LIFECYCLE_EVENTS_TABLE_ID = "tblF4rvY4jQPgb3an"
LIFECYCLE_EVENTS_TABLE = LIFECYCLE_EVENTS_TABLE_ID
BRIDGE_SCHEMA_VERSION = "v2.8"


class BridgeOperationKind(str, Enum):
    UPSERT = "UPSERT"
    UPDATE_EXISTING = "UPDATE_EXISTING"


@dataclass(frozen=True)
class BridgeOperation:
    table: str
    key_field: str
    key_value: str
    fields: dict[str, Any]
    kind: BridgeOperationKind = BridgeOperationKind.UPSERT

    def validate(self) -> None:
        if not self.table.strip():
            raise ValueError("table is required.")
        if not self.key_field.strip():
            raise ValueError("key_field is required.")
        if not self.key_value.strip():
            raise ValueError("key_value is required.")
        if self.fields.get(self.key_field) != self.key_value:
            raise ValueError("fields must contain the operation key field/value.")
        canonical_json(self.fields)


@dataclass(frozen=True)
class BridgeBatch:
    operations: tuple[BridgeOperation, ...]

    def normalized(self) -> "BridgeBatch":
        return BridgeBatch(tuple(deduplicate_operations(self.operations)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("Bridge timestamps must be timezone-aware.")
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(k): _jsonable(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Bridge payloads cannot contain NaN or infinity.")
        return value
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def stable_record_id(prefix: str, *parts: Any) -> str:
    if not prefix or not prefix.strip():
        raise ValueError("prefix is required.")
    digest = hashlib.sha256(
        canonical_json(parts).encode("utf-8")
    ).hexdigest()[:20]
    return f"{prefix}-{digest}"


def lifecycle_event_id(
    match_id: str,
    event: LifecycleEvent,
    *,
    snapshot_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> str:
    event.validate()
    if not match_id.strip():
        raise ValueError("match_id is required.")
    return stable_record_id(
        "v28evt",
        match_id,
        snapshot_id or "",
        event.state.value,
        event.at,
        event.reason,
        event.source,
        dict(payload or {}),
    )


def settlement_id(snapshot_id: str, fields: Mapping[str, Any]) -> str:
    if not snapshot_id.strip():
        raise ValueError("snapshot_id is required.")
    settlement = fields.get("Settlement")
    net_return = fields.get("Net Return")
    closing_line = fields.get("Closing AH Line")
    closing_probability = fields.get("Closing Market Probability")
    if not settlement:
        raise ValueError("Settlement is required.")
    if net_return is None:
        raise ValueError("Net Return is required.")
    return stable_record_id(
        "v28set",
        snapshot_id,
        settlement,
        net_return,
        closing_line,
        closing_probability,
    )


def snapshot_operation(fields: Mapping[str, Any]) -> BridgeOperation:
    row = dict(fields)
    required = {"Snapshot ID", "Match ID", "Captured At", "AH Line", "Odds"}
    missing = sorted(
        name for name in required if row.get(name) in {None, ""}
    )
    if missing:
        raise ValueError(f"Missing snapshot fields: {missing}")
    if row.get("Outcome Status") not in {None, "Pending"}:
        raise ValueError("A new snapshot operation cannot start as settled.")
    row.setdefault("Outcome Status", "Pending")
    row.setdefault("Bridge Schema Version", BRIDGE_SCHEMA_VERSION)
    op = BridgeOperation(
        table=PROSPECTIVE_AH_TABLE_ID,
        key_field="Snapshot ID",
        key_value=str(row["Snapshot ID"]),
        fields=row,
        kind=BridgeOperationKind.UPSERT,
    )
    op.validate()
    return op


def lifecycle_operation(
    match_id: str,
    event: LifecycleEvent,
    *,
    snapshot_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> BridgeOperation:
    event.validate()
    body = dict(payload or {})
    event_id = lifecycle_event_id(
        match_id,
        event,
        snapshot_id=snapshot_id,
        payload=body,
    )
    fields: dict[str, Any] = {
        "Event ID": event_id,
        "Match ID": match_id,
        "State": event.state.value,
        "Event At": event.at.isoformat(),
        "Reason": event.reason,
        "Source": event.source,
        "Payload JSON": canonical_json(body),
        "Schema Version": BRIDGE_SCHEMA_VERSION,
    }
    if snapshot_id:
        fields["Snapshot ID"] = snapshot_id
    op = BridgeOperation(
        table=LIFECYCLE_EVENTS_TABLE,
        key_field="Event ID",
        key_value=event_id,
        fields=fields,
        kind=BridgeOperationKind.UPSERT,
    )
    op.validate()
    return op


def execution_operation(
    *,
    match_id: str,
    snapshot_id: str,
    event: LifecycleEvent,
    state: str,
    current_line: float,
    current_odds: float,
    active_minimum_line: float,
    active_minimum_odds: float,
    minutes_to_kickoff: int,
) -> BridgeOperation:
    if event.state.value != state:
        raise ValueError(
            "Execution lifecycle state must match the execution state."
        )
    if state not in {
        "WAITING_ENTRY",
        "ENTERED_PROTECTED",
        "ENTERED_DECAY",
        "PASSED",
    }:
        raise ValueError("Unsupported execution lifecycle state.")
    return lifecycle_operation(
        match_id,
        event,
        snapshot_id=snapshot_id,
        payload={
            "kind": "execution",
            "current_line": float(current_line),
            "current_odds": float(current_odds),
            "active_minimum_line": float(active_minimum_line),
            "active_minimum_odds": float(active_minimum_odds),
            "minutes_to_kickoff": int(minutes_to_kickoff),
        },
    )


def live_operation(
    *,
    match_id: str,
    snapshot_id: str,
    event: LifecycleEvent,
    minute: float,
    home_score: int,
    away_score: int,
    side: str,
    line: float,
    effective_win_probability: float,
    fair_decimal_odds: float,
    expected_value: float | None = None,
    change_type: str | None = None,
) -> BridgeOperation:
    if event.state != LifecycleState.LIVE_REPRICED:
        raise ValueError(
            "Live operation requires a LIVE_REPRICED lifecycle event."
        )
    payload: dict[str, Any] = {
        "kind": "live_reprice",
        "minute": float(minute),
        "home_score": int(home_score),
        "away_score": int(away_score),
        "side": side,
        "line": float(line),
        "effective_win_probability": float(effective_win_probability),
        "fair_decimal_odds": float(fair_decimal_odds),
    }
    if expected_value is not None:
        payload["expected_value"] = float(expected_value)
    if change_type is not None:
        payload["change_type"] = str(change_type)
    return lifecycle_operation(
        match_id,
        event,
        snapshot_id=snapshot_id,
        payload=payload,
    )


def prospective_outcome_operation(
    *,
    snapshot_id: str,
    outcome_fields: Mapping[str, Any],
) -> BridgeOperation:
    """Settle the forward-only prospective observation.

    This is deliberately separate from the decision lifecycle: a captured
    snapshot still needs an outcome for v2.1 review even when execution later
    PASSes and therefore must not transition to lifecycle SETTLED.
    """
    fields = dict(outcome_fields)
    if fields.get("Outcome Status") != "Settled":
        raise ValueError("Outcome Status must be Settled.")
    sid = settlement_id(snapshot_id, fields)
    update_fields = dict(fields)
    update_fields["Snapshot ID"] = snapshot_id
    update_fields["Settlement ID"] = sid
    update_fields["Bridge Schema Version"] = BRIDGE_SCHEMA_VERSION
    update = BridgeOperation(
        table=PROSPECTIVE_AH_TABLE_ID,
        key_field="Snapshot ID",
        key_value=snapshot_id,
        fields=update_fields,
        kind=BridgeOperationKind.UPDATE_EXISTING,
    )
    update.validate()
    return update


def settlement_operations(
    *,
    match_id: str,
    snapshot_id: str,
    event: LifecycleEvent,
    outcome_fields: Mapping[str, Any],
) -> BridgeBatch:
    """Settle an actually-entered decision and link it to its lifecycle."""
    if event.state != LifecycleState.SETTLED:
        raise ValueError("Settlement requires a SETTLED lifecycle event.")
    update = prospective_outcome_operation(
        snapshot_id=snapshot_id,
        outcome_fields=outcome_fields,
    )
    sid = str(update.fields["Settlement ID"])
    lifecycle = lifecycle_operation(
        match_id,
        event,
        snapshot_id=snapshot_id,
        payload={
            "kind": "settlement",
            "settlement_id": sid,
            "settlement": update.fields["Settlement"],
            "net_return": update.fields["Net Return"],
        },
    )
    linked_fields = dict(update.fields)
    linked_fields["Last Lifecycle Event ID"] = lifecycle.key_value
    update = BridgeOperation(
        table=update.table,
        key_field=update.key_field,
        key_value=update.key_value,
        fields=linked_fields,
        kind=update.kind,
    )
    update.validate()
    lifecycle.validate()
    return BridgeBatch((update, lifecycle)).normalized()


def deduplicate_operations(
    operations: Iterable[BridgeOperation],
) -> list[BridgeOperation]:
    unique: dict[tuple[str, str, str], BridgeOperation] = {}
    order: list[tuple[str, str, str]] = []
    for op in operations:
        op.validate()
        key = (op.table, op.key_field, op.key_value)
        previous = unique.get(key)
        if previous is None:
            unique[key] = op
            order.append(key)
            continue
        if (
            previous.kind != op.kind
            or canonical_json(previous.fields)
            != canonical_json(op.fields)
        ):
            raise ValueError(
                "Conflicting duplicate bridge operation for "
                f"{op.table} {op.key_value}."
            )
    return [unique[key] for key in order]


def operation_manifest(operation: BridgeOperation) -> dict[str, Any]:
    operation.validate()
    return {
        "table": operation.table,
        "key_field": operation.key_field,
        "key_value": operation.key_value,
        "kind": operation.kind.value,
        "fields": _jsonable(operation.fields),
    }


@dataclass(frozen=True)
class ApplyResult:
    action: str
    record_id: str


class AirtableBridgeClient:
    """Minimal Airtable REST adapter for deterministic bridge operations.

    Credentials are supplied at runtime only. UPSERT is intentionally
    insert-only/idempotent: an existing key with different fields is a conflict
    rather than a silent overwrite. Mutations of an existing Prospective AH row
    must use UPDATE_EXISTING.
    """

    def __init__(
        self,
        token: str,
        *,
        base_id: str = AIRTABLE_BASE_ID,
        timeout: float = 15.0,
    ):
        if not token.strip():
            raise ValueError("Airtable token is required at runtime.")
        self.token = token
        self.base_id = base_id
        self.timeout = float(timeout)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "AirtableBridgeClient":
        import os

        values = os.environ if env is None else env
        token = values.get("AIRTABLE_TOKEN", "")
        if not token:
            raise ValueError("AIRTABLE_TOKEN is not set.")
        return cls(token)

    def _request(
        self,
        method: str,
        table: str,
        *,
        query: Mapping[str, str] | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from urllib.parse import quote, urlencode
        from urllib.request import Request, urlopen

        url = (
            "https://api.airtable.com/v0/"
            f"{quote(self.base_id)}/{quote(table, safe='')}"
        )
        if query:
            url += "?" + urlencode(dict(query))
        data = (
            None
            if payload is None
            else canonical_json(payload).encode("utf-8")
        )
        req = Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _formula_string(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _find(
        self,
        op: BridgeOperation,
    ) -> dict[str, Any] | None:
        formula = (
            "{" + op.key_field + "}='"
            + self._formula_string(op.key_value)
            + "'"
        )
        data = self._request(
            "GET",
            op.table,
            query={"filterByFormula": formula, "maxRecords": "2"},
        )
        records = list(data.get("records", []))
        if len(records) > 1:
            raise ValueError(
                "Duplicate Airtable records already exist for "
                f"{op.key_field}={op.key_value}."
            )
        return records[0] if records else None

    @staticmethod
    def _same_subset(
        existing: Mapping[str, Any],
        desired: Mapping[str, Any],
    ) -> bool:
        return all(
            existing.get(k) == _jsonable(v)
            for k, v in desired.items()
        )

    def apply(
        self,
        operation: BridgeOperation,
    ) -> ApplyResult:
        operation.validate()
        existing = self._find(operation)
        if existing is None:
            if operation.kind == BridgeOperationKind.UPDATE_EXISTING:
                raise ValueError(
                    "Refusing to create missing record for UPDATE_EXISTING "
                    f"{operation.key_value}."
                )
            created = self._request(
                "POST",
                operation.table,
                payload={"fields": _jsonable(operation.fields)},
            )
            return ApplyResult("created", str(created["id"]))

        record_id = str(existing["id"])
        existing_fields = dict(existing.get("fields", {}))
        if operation.kind == BridgeOperationKind.UPSERT:
            if self._same_subset(
                existing_fields,
                operation.fields,
            ):
                return ApplyResult("noop", record_id)
            raise ValueError(
                "Conflicting existing immutable record for "
                f"{operation.key_field}={operation.key_value}."
            )

        for protected in (
            "Settlement ID",
            "Entry Type",
            "Entry At",
            "Entry Line",
            "Entry Odds",
        ):
            old = existing_fields.get(protected)
            new = operation.fields.get(protected)
            if (
                old not in (None, "")
                and new not in (None, "")
                and old != _jsonable(new)
            ):
                raise ValueError(
                    f"Conflicting existing {protected} "
                    f"for {operation.key_value}."
                )

        if self._same_subset(
            existing_fields,
            operation.fields,
        ):
            return ApplyResult("noop", record_id)
        updated = self._request(
            "PATCH",
            f"{operation.table}/{record_id}",
            payload={"fields": _jsonable(operation.fields)},
        )
        return ApplyResult("updated", str(updated["id"]))

    def apply_batch(
        self,
        batch: BridgeBatch,
    ) -> tuple[ApplyResult, ...]:
        normalized = batch.normalized()
        return tuple(
            self.apply(op)
            for op in normalized.operations
        )


def execution_operations(
    *,
    match_id: str,
    snapshot_id: str,
    event: LifecycleEvent,
    execution_state: str,
    current_line: float,
    current_odds: float,
    active_minimum_line: float,
    active_minimum_odds: float,
    minutes_to_kickoff: int,
) -> BridgeBatch:
    lifecycle_state = {
        "WAIT": LifecycleState.WAITING_ENTRY,
        "ENTER_PROTECTED": LifecycleState.ENTERED_PROTECTED,
        "ENTER_DECAY": LifecycleState.ENTERED_DECAY,
        "PASS": LifecycleState.PASSED,
    }.get(execution_state)
    if lifecycle_state is None:
        raise ValueError("Unknown execution_state.")
    if event.state != lifecycle_state:
        raise ValueError(
            "Execution event state does not match execution_state."
        )

    lifecycle = lifecycle_operation(
        match_id,
        event,
        snapshot_id=snapshot_id,
        payload={
            "kind": "execution",
            "execution_state": execution_state,
            "current_line": float(current_line),
            "current_odds": float(current_odds),
            "active_minimum_line": float(active_minimum_line),
            "active_minimum_odds": float(active_minimum_odds),
            "minutes_to_kickoff": int(minutes_to_kickoff),
        },
    )
    fields: dict[str, Any] = {
        "Snapshot ID": snapshot_id,
        "Execution State": execution_state,
        "Last Lifecycle Event ID": lifecycle.key_value,
        "Bridge Schema Version": BRIDGE_SCHEMA_VERSION,
    }
    if execution_state in {
        "ENTER_PROTECTED",
        "ENTER_DECAY",
    }:
        fields.update(
            {
                "Entry Type": (
                    "protected"
                    if execution_state == "ENTER_PROTECTED"
                    else "decay"
                ),
                "Entry At": event.at.isoformat(),
                "Entry Line": float(current_line),
                "Entry Odds": float(current_odds),
            }
        )
    update = BridgeOperation(
        table=PROSPECTIVE_AH_TABLE_ID,
        key_field="Snapshot ID",
        key_value=snapshot_id,
        fields=fields,
        kind=BridgeOperationKind.UPDATE_EXISTING,
    )
    update.validate()
    return BridgeBatch((update, lifecycle)).normalized()


def live_operations(
    *,
    match_id: str,
    snapshot_id: str,
    event: LifecycleEvent,
    minute: float,
    home_score: int,
    away_score: int,
    side: str,
    line: float,
    effective_win_probability: float,
    fair_decimal_odds: float,
    expected_value: float | None = None,
    change_type: str | None = None,
) -> BridgeBatch:
    lifecycle = live_operation(
        match_id=match_id,
        snapshot_id=snapshot_id,
        event=event,
        minute=minute,
        home_score=home_score,
        away_score=away_score,
        side=side,
        line=line,
        effective_win_probability=effective_win_probability,
        fair_decimal_odds=fair_decimal_odds,
        expected_value=expected_value,
        change_type=change_type,
    )
    update = BridgeOperation(
        table=PROSPECTIVE_AH_TABLE_ID,
        key_field="Snapshot ID",
        key_value=snapshot_id,
        fields={
            "Snapshot ID": snapshot_id,
            "Last Lifecycle Event ID": lifecycle.key_value,
            "Bridge Schema Version": BRIDGE_SCHEMA_VERSION,
        },
        kind=BridgeOperationKind.UPDATE_EXISTING,
    )
    update.validate()
    return BridgeBatch((update, lifecycle)).normalized()


def operation_from_manifest(
    row: Mapping[str, Any],
) -> BridgeOperation:
    op = BridgeOperation(
        table=str(row["table"]),
        key_field=str(row["key_field"]),
        key_value=str(row["key_value"]),
        fields=dict(row["fields"]),
        kind=BridgeOperationKind(
            str(row.get("kind", "UPSERT"))
        ),
    )
    op.validate()
    return op
