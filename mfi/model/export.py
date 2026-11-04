"""Freeze the database into portable files the dashboard reads (no Postgres needed to share it).

    python -m mfi.model.export   ->  data/app/premises.parquet, sections.parquet, districts.geojson, meta.json
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "data" / "app"


def norm_name(s):
    """'McDonald´s', 'MC DONALD S' -> 'MCDONALDS'. Used for brand matching."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    return re.sub(r"[^A-Z0-9]", "", str(s).upper().replace("Ñ", "N")) or None


def main():
    import geopandas as gpd  # heavy deps only needed when exporting from the DB
    from ..db import engine
    APP.mkdir(parents=True, exist_ok=True)
    eng = engine()

    prem = pd.read_sql("""
        SELECT p.id_local, p.status, p.access_type, p.trade_name, p.street_type, p.street_name, p.street_number,
               p.postal_code, p.district_code, p.barrio_code, p.geom_source,
               ST_X(p.geom) x, ST_Y(p.geom) y,
               ST_X(ST_Transform(p.geom,4326)) lon, ST_Y(ST_Transform(p.geom,4326)) lat,
               s.section_code AS section_spatial,
               a.epigrafes, a.divisions
        FROM v_premises p
        LEFT JOIN LATERAL (SELECT section_code FROM v_census_sections s WHERE ST_Contains(s.geom, p.geom) LIMIT 1) s ON true
        LEFT JOIN (SELECT id_local, string_agg(DISTINCT epigrafe_id, ',') epigrafes, string_agg(DISTINCT division_id, ',') divisions
                   FROM v_premise_activities GROUP BY id_local) a USING (id_local)
        WHERE p.geom IS NOT NULL""", eng)
    prem["name_norm"] = prem["trade_name"].map(norm_name)
    prem.to_parquet(APP / "premises.parquet", index=False)

    sec = pd.read_sql("""
        SELECT c.section_code, c.district_code, c.barrio_code,
               ST_X(ST_PointOnSurface(c.geom)) x, ST_Y(ST_PointOnSurface(c.geom)) y,
               p.pop_total, p.pop_15_24, p.pop_25_44, p.pop_45_64, p.pop_65_plus, p.pop_0_14, p.mean_age,
               i.net_income_per_person, i.year AS income_year, p.reference_date AS padron_date
        FROM v_census_sections c
        LEFT JOIN v_section_population p USING (section_code)
        LEFT JOIN v_section_income i USING (section_code)""", eng)
    sec.to_parquet(APP / "sections.parquet", index=False)

    d = gpd.read_postgis("SELECT district_code, name, ST_Transform(ST_SimplifyPreserveTopology(geom, 15), 4326) geom FROM v_districts",
                         eng, geom_col="geom")
    d.to_file(APP / "districts.geojson", driver="GeoJSON")
    b = pd.read_sql("SELECT barrio_code, district_code, name FROM v_neighbourhoods", eng)
    b.to_parquet(APP / "barrios.parquet", index=False)
    names = dict(zip(d.district_code, d.name))

    files = pd.read_sql("SELECT source_key, url, retrieved_at, data_reference_date, sha256 FROM v_latest_file", eng)
    meta = {"districts": {int(k): v for k, v in names.items()},
            "sources": json.loads(files.to_json(orient="records", date_format="iso"))}
    (APP / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"premises {len(prem):,} | sections {len(sec):,} | districts {len(d)} -> {APP}")


if __name__ == "__main__":
    main()
