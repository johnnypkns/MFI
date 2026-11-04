"""Apify scrapers (scraping of Idealista + Google Maps approved by the founder 2026-11-04; see docs/DATA_SOURCES.md).

    rents_near(lat, lon)        Idealista commercial premises FOR RENT within a radius -> €/m²/month distribution
    places_near(query, lat, lon) Google Maps places (rating, review count) — demand proxy for competitors/brand stores

Uses Apify's run-sync REST endpoint (no SDK). Needs APIFY_TOKEN (env or Streamlit secret).
Every raw response is archived to data/raw/apify/<actor>/<timestamp>.json for provenance.
STATUS: Idealista parser verified against a live run (2026-11-04). Google Maps parser not yet run live.
"""
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import requests

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IDEALISTA = "igolaizola~idealista-scraper"
GMAPS = "compass~crawler-google-places"
RAW = ROOT / "data" / "raw" / "apify"


def _token():
    t = os.environ.get("APIFY_TOKEN")
    if not t:
        try:
            import streamlit as st
            t = st.secrets.get("APIFY_TOKEN")
        except Exception:
            t = None
    return t or None


def available():
    return _token() is not None


def _run(actor, payload, timeout=240):
    tok = _token()
    if not tok:
        raise RuntimeError("APIFY_TOKEN not set")
    r = requests.post(f"https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items",
                      params={"token": tok, "timeout": timeout}, json=payload, timeout=timeout + 30)
    if r.status_code >= 400:
        raise RuntimeError(f"Apify {actor} failed {r.status_code}: {r.text[:300]}")
    items = r.json()
    out = RAW / actor.replace("~", "__")
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"{stamp}.json").write_text(json.dumps({"input": payload, "items": items}, ensure_ascii=False))
    return items, stamp


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def rents_near(lat, lon, radius_km=1.0, max_items=40, min_m2=None):
    payload = {"operation": "rent", "propertyType": "premises", "country": "es",
               "latitude": lat, "longitude": lon, "distanceKm": max(1, round(radius_km)), "maxItems": max_items}
    if min_m2:
        payload["minSize"] = str(int(min_m2))
    items, stamp = _run(IDEALISTA, payload)
    rows = []
    for it in items:  # field names verified against a live run on 2026-11-04
        price, size = _num(it.get("price")), _num(it.get("size"))
        dist = _num(it.get("distance"))
        if not (price and size and size > 0) or (dist is not None and dist > radius_km * 1000):
            continue
        dt, pi = it.get("detailedType") or {}, it.get("priceInfo") or {}
        rows.append({"price_month": price, "m2": size, "eur_m2": _num(it.get("priceByArea")) or price / size,
                     "address": it.get("address"), "neighbourhood": it.get("neighborhood"), "url": it.get("url"),
                     "lat": _num(it.get("latitude")), "lon": _num(it.get("longitude")), "distance_m": dist,
                     "floor": it.get("floor"), "traspaso": bool(dt.get("transfer")),
                     "traspaso_eur": _num((pi.get("transferPrice") or {}).get("amount"))})
    eur = np.array([r["eur_m2"] for r in rows])
    stats = ({"n": len(rows), "p25": float(np.percentile(eur, 25)), "p50": float(np.percentile(eur, 50)),
              "p75": float(np.percentile(eur, 75))} if len(rows) >= 3 else {"n": len(rows)})
    return {"listings": rows, "stats": stats, "retrieved": stamp, "actor": IDEALISTA, "radius_km": radius_km}


def places_near(query, lat, lon, max_places=60):
    payload = {"searchStringsArray": [query], "customGeolocation": {"type": "Point", "coordinates": [lon, lat], "radiusKm": 2},
               "maxCrawledPlacesPerSearch": max_places, "language": "es", "skipClosedPlaces": True}
    items, stamp = _run(GMAPS, payload)
    rows = [{"name": it.get("title"), "rating": _num(it.get("totalScore")), "reviews": _num(it.get("reviewsCount")),
             "lat": _num((it.get("location") or {}).get("lat")), "lon": _num((it.get("location") or {}).get("lng")),
             "category": it.get("categoryName"), "url": it.get("url")} for it in items]
    return {"places": rows, "retrieved": stamp, "actor": GMAPS}
