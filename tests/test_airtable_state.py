from datetime import datetime, timezone

import pytest

from gah.airtable_bridge import AirtableBridgeClient
from gah.airtable_state import (
    lifecycle_ledgers_from_records,
    prospective_frame_from_records,
    state_from_export,
    state_from_records,
)


def prospective(snapshot_id="s1", match_id="m1", status="Pending"):
    return {
        "id": f"rec-{snapshot_id}",
        "fields": {
            "Snapshot ID": snapshot_id,
            "Match ID": match_id,
            "Captured At": "2026-10-01T10:00:00+00:00",
            "Outcome Status": status,
        },
    }


def lifecycle(
    event_id,
    state,
    at,
    *,
    match_id="m1",
):
    return {
        "id": f"rec-{event_id}",
        "fields": {
            "Event ID": event_id,
            "Match ID": match_id,
            "State": state,
            "Event At": at,
            "Reason": state.lower(),
            "Source": "unit-test",
        },
    }


def test_prospective_frame_is_deterministic_and_keeps_record_id():
    frame = prospective_frame_from_records(
        [
            prospective("s2", "m2"),
            prospective("s1", "m1"),
        ]
    )
    assert frame["Snapshot ID"].tolist() == ["s1", "s2"]
    assert frame.iloc[0]["Airtable Record ID"] == "rec-s1"


def test_duplicate_snapshot_id_is_rejected():
    with pytest.raises(ValueError, match="Duplicate Snapshot ID"):
        prospective_frame_from_records(
            [
                prospective("s1", "m1"),
                prospective("s1", "m2"),
            ]
        )


def test_lifecycle_events_reconstruct_valid_ledger_in_time_order():
    records = [
        lifecycle(
            "e2",
            "SNAPSHOT_CAPTURED",
            "2026-10-01T10:05:00+00:00",
        ),
        lifecycle(
            "e1",
            "BOARD_SELECTED",
            "2026-10-01T10:00:00+00:00",
        ),
        lifecycle(
            "e3",
            "WAITING_ENTRY",
            "2026-10-01T10:10:00+00:00",
        ),
    ]
    ledgers = lifecycle_ledgers_from_records(records)
    assert len(ledgers) == 1
    assert ledgers[0].current_state.value == "WAITING_ENTRY"
    assert [
        event.state.value for event in ledgers[0].events
    ] == [
        "BOARD_SELECTED",
        "SNAPSHOT_CAPTURED",
        "WAITING_ENTRY",
    ]


def test_duplicate_event_id_is_rejected():
    rows = [
        lifecycle(
            "e1",
            "BOARD_SELECTED",
            "2026-10-01T10:00:00+00:00",
        ),
        lifecycle(
            "e1",
            "SNAPSHOT_CAPTURED",
            "2026-10-01T10:05:00+00:00",
        ),
    ]
    with pytest.raises(ValueError, match="Duplicate lifecycle Event ID"):
        lifecycle_ledgers_from_records(rows)


def test_invalid_reconstructed_transition_is_rejected():
    rows = [
        lifecycle(
            "e1",
            "PASSED",
            "2026-10-01T10:00:00+00:00",
        ),
        lifecycle(
            "e2",
            "ENTERED_PROTECTED",
            "2026-10-01T10:05:00+00:00",
        ),
    ]
    with pytest.raises(ValueError, match="Invalid lifecycle transition"):
        lifecycle_ledgers_from_records(rows)


def test_state_summary_counts_pending_and_settled():
    state = state_from_records(
        [
            prospective("s1", "m1", "Pending"),
            prospective("s2", "m2", "Settled"),
        ],
        [
            lifecycle(
                "e1",
                "BOARD_SELECTED",
                "2026-10-01T10:00:00+00:00",
                match_id="m1",
            )
        ],
    )
    assert state.summary() == {
        "schema_version": "v2.11",
        "snapshots": 2,
        "matches": 2,
        "lifecycle_events": 1,
        "pending_snapshots": 1,
        "settled_snapshots": 1,
    }


def test_export_round_trip():
    state = state_from_records(
        [prospective()],
        [
            lifecycle(
                "e1",
                "BOARD_SELECTED",
                "2026-10-01T10:00:00+00:00",
            )
        ],
    )
    rebuilt = state_from_export(state.to_export())
    assert rebuilt.summary() == state.summary()


class FakeClient(AirtableBridgeClient):
    def __init__(self):
        super().__init__("token")
        self.pages = [
            {
                "records": [{"id": "r1", "fields": {"x": 1}}],
                "offset": "next",
            },
            {
                "records": [{"id": "r2", "fields": {"x": 2}}],
            },
        ]
        self.queries = []

    def _request(self, method, table, *, query=None, payload=None):
        assert method == "GET"
        self.queries.append(dict(query or {}))
        return self.pages.pop(0)


def test_airtable_client_list_records_paginates():
    client = FakeClient()
    records = client.list_records("table")
    assert [row["id"] for row in records] == ["r1", "r2"]
    assert client.queries == [
        {"pageSize": "100"},
        {"pageSize": "100", "offset": "next"},
    ]


def test_airtable_client_page_size_validation():
    client = FakeClient()
    with pytest.raises(ValueError, match="page_size"):
        client.list_records("table", page_size=101)
