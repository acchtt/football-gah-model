# Airtable integration

Base: Football GAH
Base ID: app2wBjpvj69bCoMS

Tables:
- Matches — tblLMsnOdSHJikGEq
- Predictions — tblGp5dTVOv0pFsoo
- Markets — tblFSVsGajJ7uSUp9
- Model Runs — tbl4tHyy0NkZ7dEXk
- Prospective AH — tblk9kLKRqrnBOQ2k
- Lifecycle Events — tblF4rvY4jQPgb3an

The Airtable base is intentionally separate from the existing football-model logs.
It is the operational datastore for GAH forecasts, market snapshots, results, and backtest metadata.

Do not commit Airtable tokens. Use an environment variable such as AIRTABLE_TOKEN.

## GAH v2.1 prospective AH

The `Prospective AH` table is forward-only. Do not backfill historical selections into it.

Operational flow:
1. Use `prospective_cli.py snapshot` to de-vig the selected/opposing AH prices and produce Airtable-ready fields.
2. Create one Airtable record with those fields.
3. After close/settlement, use `prospective_cli.py outcome` and update that same record.
4. Same-line probability CLV is only populated when captured and closing AH lines are identical; line moves are stored separately.

The first formal v2.1 review is not allowed until at least 100 settled snapshots and at least 30 calendar days have elapsed.


## GAH v2.8+ operational bridge

`Prospective AH` remains the canonical forward-only snapshot/outcome table.
The bridge adds execution/entry linkage fields there and writes the v2.6
decision history to the append-only `Lifecycle Events` table.

Operational identifiers are deterministic. Identical retries are idempotent;
conflicting duplicate snapshot, event, entry, or settlement facts must fail
rather than silently overwrite the canonical record.

The v2.9 board/session orchestrator applies snapshot -> execution -> optional
live -> outcome mutations sequentially. The v2.10 quality-control layer can
read Airtable-style field names directly and reports datastore/lifecycle
integrity separately from any forecasting or betting decision.

Credentials remain runtime-only through `AIRTABLE_TOKEN`.
