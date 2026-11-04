# Mac setup (one time, ~15 min, €0)

Everything here is free and local. No accounts, no cloud.

## 1. PostgreSQL + PostGIS → Postgres.app

Postgres.app bundles PostGIS, needs no Homebrew and runs as a menu-bar app.

1. Download from https://postgresapp.com → drag to Applications → open → **Initialize**.
2. Put its command-line tools on your PATH (Terminal):

```bash
sudo mkdir -p /etc/paths.d && echo /Applications/Postgres.app/Contents/Versions/latest/bin | sudo tee /etc/paths.d/postgresapp
```

3. Close and reopen Terminal, then:

```bash
createdb mfi
psql -d mfi -c "CREATE EXTENSION postgis; SELECT postgis_version();"
```

The code's default connection is `postgresql://localhost/mfi` (your Mac user, no password) — matches Postgres.app out of the box. Different setup? `export DATABASE_URL=postgresql://user:pass@host/db`.

## 2. Python environment

Needs Python 3.11+. Check: `python3 --version`. If older, install 3.12 from https://www.python.org/downloads/macos/.

```bash
cd ~/Desktop/"Claude Cowork"/madrid-franchise-intel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Build the database

```bash
python -m mfi.cli all        # schema + download (~500 MB) + load + 17 checks. ~2 min.
python -m pytest -q          # parser unit tests
python -m mfi.cli export     # data/exports/section_profile.gpkg -> drag into QGIS
```

Expected last line: `17 checks, 0 failed`.

## 4. Git

```bash
git init && git add . && git commit -m "Milestone 1: Madrid data foundation"
```

## 5. QGIS (optional now, useful from M3)

Free: https://qgis.org. Layer → Add Layer → PostGIS → new connection: host `localhost`, database `mfi`. Or just open the exported `.gpkg`.

## Note on location

The project sits in your Cowork folder so Claude can work on it. If Desktop syncs to iCloud, `data/raw/` (~500 MB, grows with each snapshot) will sync too. Fix: move the project to `~/Projects/` and reconnect that folder in Cowork.
