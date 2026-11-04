"""Layer 1 — where stores actually are.

Merges independent sources into one store inventory per brand, with a confidence score:
  census  Ayuntamiento premises census (open premises, matched by trade name)
  osm     OpenStreetMap (brand / brand:wikidata / name tags)
  gmaps   Google Maps via Apify (added in the next step)

Two records from different sources are the same store if they are within MERGE_M metres.
n_sources = how many independent sources saw it. A store seen by 2+ sources is near-certain;
a store seen by one source is kept but flagged.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from .model.export import ROOT, norm_name
from .model.site_model import match_brand

MERGE_M = 120          # malls: census uses the mall's point, OSM the unit's — they can sit ~100 m apart
TO_UTM = Transformer.from_crs(4326, 25830, always_xy=True)
TO_WGS = Transformer.from_crs(25830, 4326, always_xy=True)


def load_osm(path=None):
    slim = ROOT / "data" / "app" / "osm_food.parquet"
    if path is None and slim.exists():
        df = pd.read_parquet(slim)
        df["x"], df["y"] = TO_UTM.transform(df["lon"].values, df["lat"].values)
        return df, str(slim)
    path = path or sorted((ROOT / "data" / "raw" / "osm").glob("*/food_amenities.json"))[-1]
    rows = []
    for e in json.loads(Path(path).read_text())["elements"]:
        t = e.get("tags", {})
        lat, lon = (e.get("lat"), e.get("lon")) if "lat" in e else ((e.get("center") or {}).get("lat"), (e.get("center") or {}).get("lon"))
        if lat is None:
            continue
        rows.append({"osm_id": f"{e['type']}/{e['id']}", "lat": lat, "lon": lon, "amenity": t.get("amenity"),
                     "name": t.get("name"), "brand": t.get("brand"), "wikidata": t.get("brand:wikidata"),
                     "cuisine": t.get("cuisine"), "street": t.get("addr:street"), "housenumber": t.get("addr:housenumber")})
    df = pd.DataFrame(rows)
    df["x"], df["y"] = TO_UTM.transform(df["lon"].values, df["lat"].values)
    df["brand_norm"] = df["brand"].map(norm_name)
    df["name_norm"] = df["name"].map(norm_name)
    return df, str(path)


def osm_brand(osm, query, wikidata=None):
    q = norm_name(query)
    hit = (osm["brand_norm"] == q) | osm["name_norm"].fillna("").str.startswith(q)
    if wikidata:
        hit |= osm["wikidata"].fillna("").str.contains(wikidata)
    return osm[hit].copy()


def merge(sources: dict):
    """sources: {name: DataFrame with x, y, lon, lat, label}. Greedy proximity clustering across sources."""
    allp = pd.concat([df.assign(source=k)[["x", "y", "lon", "lat", "label", "source"]] for k, df in sources.items()],
                     ignore_index=True)
    if allp.empty:
        return allp
    tree = cKDTree(allp[["x", "y"]].values)
    cluster = -np.ones(len(allp), int)
    cid = 0
    for i in range(len(allp)):
        if cluster[i] >= 0:
            continue
        nb = [j for j in tree.query_ball_point(allp.loc[i, ["x", "y"]].values, MERGE_M) if cluster[j] < 0]
        cluster[nb] = cid
        cid += 1
    allp["store_id"] = cluster
    g = allp.groupby("store_id")
    out = g.agg(x=("x", "mean"), y=("y", "mean"), lon=("lon", "mean"), lat=("lat", "mean"),
                sources=("source", lambda s: ",".join(sorted(set(s)))),
                label=("label", lambda s: " | ".join(dict.fromkeys(v for v in s if v))))
    out["n_sources"] = out["sources"].str.count(",") + 1
    return out.reset_index()


def inventory(data, query, osm=None, wikidata=None):
    census, _ = match_brand(data.premises, query)
    census = census.assign(label=(census["trade_name"].fillna("") + " — " + census["street_name"].fillna("").str.title()
                                  + " " + census["street_number"].fillna("")))
    osm = osm if osm is not None else load_osm()[0]
    o = osm_brand(osm, query, wikidata).assign(label=lambda d: d["name"].fillna("") + " — " +
                                               d["street"].fillna("") + " " + d["housenumber"].fillna(""))
    return merge({"census": census, "osm": o}), census, o


def missing_from_census(inv):
    """Stores other sources saw that the census did not — to feed site_model.run(extra_stores=...)."""
    return inv[~inv["sources"].str.contains("census")].copy()
