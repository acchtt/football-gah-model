# Airtable integration

Base: Football GAH
Base ID: app2wBjpvj69bCoMS

Tables:
- Matches — tblLMsnOdSHJikGEq
- Predictions — tblGp5dTVOv0pFsoo
- Markets — tblFSVsGajJ7uSUp9
- Model Runs — tbl4tHyy0NkZ7dEXk

The Airtable base is intentionally separate from the existing football-model logs.
It is the operational datastore for GAH forecasts, market snapshots, results, and backtest metadata.

Do not commit Airtable tokens. Use an environment variable such as AIRTABLE_TOKEN.
