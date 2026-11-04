"""Milestone 1 entry point.

    python -m mfi.cli init       # create schema + views
    python -m mfi.cli ingest     # download + load every M1 source (idempotent)
    python -m mfi.cli validate   # run QC checks, exit 1 on any fail
    python -m mfi.cli all        # init + ingest + validate
    python -m mfi.cli export     # section profile -> data/exports/section_profile.gpkg (QGIS)
"""
import sys
import time

from . import db, validate
from .ingest import boundaries, income, padron, premises

STEPS = [
    ("districts", boundaries.load_districts),
    ("barrios", boundaries.load_barrios),
    ("census sections", boundaries.load_sections),
    ("padrón", padron.load),
    ("income (INE ADRH)", income.load),
    ("premises", premises.load_locales),
    ("activities", premises.load_actividades),
]


def cmd_init():
    db.apply_sql()
    print("schema ok")


def cmd_ingest():
    with db.run("ingest", {"steps": [s for s, _ in STEPS]}) as run_id:
        for name, fn in STEPS:
            t = time.time()
            file_id, n = fn(run_id)
            note = f"{n:,} rows" if n else "unchanged since last run, skipped"
            print(f"  {name:<20} file {file_id:<4} {note:<36} {time.time() - t:5.1f}s")


def cmd_validate():
    with db.run("validate") as run_id:
        rows = validate.run_all(run_id)
    icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
    for name, status, observed in rows:
        print(f"  {icon[status]}  {name:<34} {observed}")
    fails = sum(s == "fail" for _, s, _ in rows)
    print(f"\n{len(rows)} checks, {fails} failed (run {run_id})")
    return 1 if fails else 0


def cmd_export():
    """Write the section profile to a GeoPackage you can drag into QGIS."""
    import geopandas as gpd
    out = db.ROOT / "data" / "exports" / "section_profile.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_postgis("SELECT * FROM v_section_profile", db.engine(), geom_col="geom")
    gdf.to_file(out, layer="section_profile", driver="GPKG")
    print(f"  {len(gdf):,} sections -> {out}")


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd == "export":
        cmd_export()
        return 0
    if cmd in ("init", "all"):
        cmd_init()
    if cmd in ("ingest", "all"):
        cmd_ingest()
    if cmd in ("validate", "all"):
        return cmd_validate()
    if cmd not in ("init", "ingest", "validate", "all"):
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
