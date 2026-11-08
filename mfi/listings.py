"""Layer 4 — what is actually for rent: every commercial unit listed on Idealista in Madrid (via Apify).

    python -m mfi.listings   ->  data/app/listings.parquet

Cleaning rules (each counted and reported, nothing silently dropped):
  - commercial premises only (Idealista 'industrialPremise' = warehouses/naves -> out)
  - inside Madrid municipality
  - plausible rent: €200-€100k/month, €3-€200/m²/month (catches '25 €/month' typos and per-m² prices entered as totals)
  - plausible size: 20-3,000 m²
  - de-duplicated by Idealista property code (several pulls overlap)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from .model.export import APP, ROOT

TO_UTM = Transformer.from_crs(4326, 25830, always_xy=True)
FLOOR = {"bj": "ground", "en": "mezzanine", "ss": "basement", "st": "basement", "-1": "basement", "-2": "basement"}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_raw(raw_dir=None, files=None):
    raw_dir = Path(raw_dir or ROOT / "data" / "raw" / "apify" / "listings")
    rows = []
    for f in (files or sorted(raw_dir.glob("*.json"))):
        f = Path(f)
        for it in json.loads(f.read_text()):
            dt, pi = it.get("detailedType") or {}, it.get("priceInfo") or {}
            if isinstance(dt, str):
                dt = {}
            rows.append({
                "code": str(it.get("propertyCode")), "url": it.get("url"), "address": it.get("address"),
                "neighbourhood": it.get("neighborhood"), "district_name": it.get("district"),
                "lat": _num(it.get("latitude")), "lon": _num(it.get("longitude")),
                "rent_month": _num(it.get("price")), "m2": _num(it.get("size")),
                "eur_m2_listed": _num(it.get("priceByArea")),
                "floor_raw": it.get("floor"), "subtype": dt.get("subTypology"),
                "traspaso": bool(dt.get("transfer")),
                "traspaso_eur": _num((pi.get("transferPrice") or {}).get("amount")) if isinstance(pi, dict) else np.nan,
                "status": it.get("status"), "first_listed": it.get("firstActivationDate"),
                "description": (it.get("description") or "")[:600], "source_file": f.name,
            })
    return pd.DataFrame(rows)


def clean(df, municipality_shape=None):
    log = {"raw": len(df)}
    df = df.drop_duplicates("code").copy()
    log["after_dedupe"] = len(df)
    df = df[df["subtype"].fillna("commercialProperty") == "commercialProperty"]
    log["commercial_only"] = len(df)
    df["eur_m2"] = df["rent_month"] / df["m2"]
    ok = df["rent_month"].between(200, 100_000) & df["m2"].between(20, 3000) & df["eur_m2"].between(3, 200)
    df = df[ok & df["lat"].notna()]
    log["plausible_rent_size"] = len(df)
    df["x"], df["y"] = TO_UTM.transform(df["lon"].values, df["lat"].values)
    if municipality_shape is not None:
        from shapely.geometry import Point
        df = df[[municipality_shape.contains(Point(x, y)) for x, y in zip(df["x"], df["y"])]]
    log["inside_madrid"] = len(df)
    fl = df["floor_raw"].astype(str)
    df["floor"] = fl.map(FLOOR).fillna(pd.Series(np.where(fl.str.fullmatch(r"\d+"), "upper", "unknown"), index=fl.index))
    df["traspaso_eur"] = df["traspaso_eur"].where(df["traspaso"], 0.0).fillna(0.0)
    return df.reset_index(drop=True), log


def main():
    import geopandas as gpd
    from shapely.ops import unary_union
    d = gpd.read_file(APP / "districts.geojson").to_crs(25830)
    df, log = clean(load_raw(), unary_union(d.geometry).buffer(50))
    df.to_parquet(APP / "listings.parquet", index=False)
    (APP / "listings_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1))
    print(df[["rent_month", "m2", "eur_m2", "traspaso_eur"]].describe().round(1).to_string())
    print(df["floor"].value_counts().to_dict())


if __name__ == "__main__":
    main()
