# Tasks

## Done — Milestone 1: Madrid data foundation (2026-11-04)

- [x] Environment inspected (macOS arm64, 164 GB free; Madrid + INE portals reachable)
- [x] Exact datasets identified (`docs/DATA_SOURCES.md`)
- [x] Project structure, requirements, schema (EPSG:25830), provenance tables
- [x] Ingestion: districts, barrios, sections, padrón, INE income, premises, activities
- [x] History-preserving, idempotent loads (sha256 dedupe, latest-views)
- [x] 17 validation checks, all passing; 4 parser unit tests
- [x] `v_section_profile` + GeoPackage export for QGIS
- [ ] **You:** Mac setup + first local run (`docs/MAC_SETUP.md`) → confirm `17 checks, 0 failed`
- [ ] **You:** `git init` + first commit

## Done — Dashboard v0.1 (2026-11-04)

- [x] `app.py` Streamlit dashboard: franchise input, Madrid map (red = recommended, black = existing), KPIs, site table, tabs
- [x] Site model v0.1: brand matching, category inference, logistic suitability + spatial CV, Huff demand calibrated on the
      brand's own network, greedy cannibalisation-aware selection, store-count curve
- [x] Claude layer wired (`mfi/reasoning.py`): cited franchise research with web search + analyst brief (needs API key)
- [x] Tested: McDonald's 56 stores AUC 0.73 · Burger King 92/0.65 · Starbucks 36/0.79 · KFC 27/0.78 · Goiko 24/0.78
- [ ] **You:** pay Apify invoices → Idealista rent + Google Maps reviews
- [ ] **You:** Anthropic API key with a monthly cap → `.streamlit/secrets.toml`

## Done — v0.2 reasoning layer (2026-11-04)

- [x] 3-page app: **Analysis** (map + Monte Carlo stability/alternatives) · **Investment report** · **Methodology**
- [x] Robustness Monte Carlo (`mfi/model/robustness.py`): 200 draws = bootstrap × demand-parameter uncertainty → stability, demand P10–P90
- [x] P&L Monte Carlo (`mfi/model/pnl.py`): 10k draws/site, triangular inputs with source + label, refuses to run on gaps; tornado sensitivity
- [x] Apify rent scraper (`mfi/scrape.py`) — **untested live: account blocked by unpaid invoices**
- [x] Claude: cited franchise research incl. P&L input ranges; evidence-only investment memo (`mfi/reasoning.py`)
- [x] Python-generated audit trail ("How we got here"), evidence pack (`mfi/report.py`)
- [ ] First live run with API key + Apify token → tighten Idealista field parsing on real response

## Done — Layers 1a, 2, 4 + money-first ranking (2026-11-04)

- [x] Brand registry: 1,353 Madrid chains, sector, franchisability (`mfi/brands.py`) · Brand directory page
- [x] Store inventory: census + OpenStreetMap merge, offices/warehouses/airport excluded (`mfi/stores.py`)
- [x] Competitors = activity class + known chains in it (fixes Dia / 'Mi Alcampo'); reliability gate on AUC
- [x] Layer 2 demand (`mfi/demand.py`): consumer + office premises, industry excluded, calibrated on 31 pedestrian
      counters — rank correlation 0.71 (held-out 0.62) vs 0.29 residents-only, 0.53 old proxy; metro/rail, offices features
- [x] Layer 4 listings (`mfi/listings.py`): 2,127 clean Idealista units (8 Apify pulls, 2,501 unique raw)
- [x] Money-first ranking (`mfi/model/rank.py`, `mfi/pipeline.py`): research → filter → score → P&L MC per listing →
      greedy rank by P(payback ≤ target) with cannibalisation; one click in the app
- [x] Claude research cached per brand (`data/research/`): Telepizza (full economics), Tierra Burrito Bar (eligibility unclear),
      Goiko (not franchised)

## Next — model v0.3 (in this order)

1. **OSM + Google Maps store cross-check** — the census misses stores registered under company names (McDonald's shows
   56; "15+ more supported" is partly this gap). Fixes the store-count curve.
2. **Google review counts** as observed demand → calibrate Huff on real visits instead of the even-catchment rule.
3. **Survival model** on the 2014–2026 premises history (502 snapshots): P(still open at 3 yrs) per location + category.
4. **Backtest**: train on 2019, predict 2019→2024 openings. Show hit rate in the dashboard.
5. **Idealista rent** per candidate (€/m², size, traspaso) → replace 'Cerrado' census units with real listings.
6. **P&L Monte Carlo**: Claude-researched AUV/fees/royalties + rent + Madrid hostelería wage agreement → P(payback < 4y), P10/P50/P90.
7. Walking-network distances (OSMnx); deploy to Streamlit Cloud.

## Earlier backlog — Milestone 2/3: competitors + market

Order matters: cheapest, highest-signal first.

1. **OSM layer via Overpass** (free, rate-limited → one city-wide pull per category, cached in `data/raw/osm/`): cafés, fast food, restaurants, gyms, offices, universities, schools, malls, metro/cercanías/bus stops. New tables: `businesses`, `competitors`.
2. **Cross-validate the premises census against OSM** — the register says "Abierto"; OSM + later spot checks say whether it trades. Confidence score per competitor.
3. **Epígrafe → franchise category map** (e.g. 561004/CAFETERIA → coffee). Config file, not code.
4. **Barrio and district rollups** + first neighbourhood shortlist per category (Module 2).
5. **Walking catchments** (500 m / 1 km / 2 km). Start with straight-line buffers (instant, PostGIS), then network isochrones via OSMnx on the OSM walk graph — still free/local.

## Later

- M1 franchise intelligence + eligibility gate (McDonald's, Starbucks, Oakberry — check who actually sells individual franchises in Spain *before* any location work)
- M5 property listings (manual Idealista import, CSV template, geocode)
- M6 financial model (bear/base/bull, sensitivities, then Monte Carlo)
- M7 independent validation stage, M8 reports, Streamlit dashboard
- INE household composition / employment by section; Madrid "Panel de indicadores" (300087) for barrio socio-economics
