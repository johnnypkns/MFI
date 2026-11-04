# Madrid Franchise Intelligence

Decision support for one question: *given a franchise, where in Madrid should I open to maximise risk-adjusted return on invested capital?*

Status: **Milestone 1 done + dashboard v0.1 (site model, Claude layer wired)** — local PostGIS database with genuine Madrid boundaries, population, income and commercial premises, reproducible ingestion, 17 automated checks.

## Dashboard

```bash
python -m mfi.model.export   # freeze DB -> data/app/ (only after a new ingest)
streamlit run app.py         # opens http://localhost:8501
```

Type a franchise → red = recommended vacant units, black = existing stores. Pages: **Analysis**, **Investment report** (Claude research → Apify rents → P&L Monte Carlo → Claude memo, plus a Python audit trail), **Methodology** (data, scraping, formulas, simulations, limits). Tabs: Claude analyst brief, Claude franchise
research (needs `ANTHROPIC_API_KEY` in `.streamlit/secrets.toml`), model diagnostics, store-count curve, sources.
The app reads only `data/app/` (~11 MB), so it deploys to Streamlit Community Cloud without Postgres.

Model code: `mfi/model/site_model.py` (method in its docstring). Claude layer: `mfi/reasoning.py`.

## Run it

Setup: [`docs/MAC_SETUP.md`](docs/MAC_SETUP.md). Then:

```bash
source .venv/bin/activate
python -m mfi.cli all        # init + ingest + validate (idempotent: unchanged files are skipped)
python -m mfi.cli export     # GeoPackage for QGIS
python -m pytest -q
```

Re-run `ingest` any time. A new Padrón month or a new daily premises snapshot is stored **alongside** the old one (history preserved); `v_*` views always read the latest.

## Layout

```
mfi/
  sources.py        registry: every dataset, URL rule, licence, resolution, confidence
  fetch.py          download -> data/raw/<source>/<date>/ + sha256 + source_files row
  db.py             connection, schema apply, research_runs context
  ingest/           one loader per source (boundaries, padron, income, premises)
  validate.py       QC checks -> validation_results
  cli.py            entry point
sql/
  001_schema.sql    tables (EPSG:25830 everywhere)
  002_views.sql     latest-snapshot views + v_section_profile
tests/              parser unit tests
docs/               setup, data sources, quirks
data/raw/           immutable downloads (git-ignored)
```

## Database (M1)

| Table | Rows (2026-11-04) | Notes |
|---|---|---|
| `sources`, `source_files`, `research_runs` | — | provenance: who, what URL, when, sha256, which run |
| `districts` / `neighbourhoods` / `census_sections` | 21 / 131 / 2,462 | polygons |
| `demographics` | 715,709 | section × age × sex × nationality; 3,535,865 residents |
| `income_statistics` | 152,040 | INE ADRH 2015–2024, section/district/city |
| `premises` | 203,784 | 139,802 open; point + `geom_source` |
| `premise_activities` | 225,875 | epígrafe codes (e.g. 561004 bar-restaurante) |
| `validation_results` | — | every check, every run |

Key view: **`v_section_profile`** — one row per section: area, population, age bands, density, income, open premises, geometry. The base layer for M2–M4.

## Rules the code enforces

- No fabricated data. Missing = NULL, logged. Fallbacks are labelled (`geom_source`).
- Nothing overwritten. Every fact row carries its `file_id`.
- Same bytes never loaded twice (sha256).
- A failing check exits non-zero. Don't build on red.

See [`TASKS.md`](TASKS.md) for done / next.
