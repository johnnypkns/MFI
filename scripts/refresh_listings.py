"""Weekly refresh of Madrid commercial rentals from Idealista (Apify) -> data/app/listings.parquet.

    APIFY_TOKEN=... python scripts/refresh_listings.py

Run by .github/workflows/refresh-listings.yml every Monday night. One Apify run, rent · premises · Madrid,
unlimited items (the actor's iterator continues past Idealista's 2,500-result window). Only this week's
pull is used. Safety: if the new pull is under 60% of the current listing count, nothing is replaced
(a half-failed scrape must not shrink the candidate set) and the job exits with an error.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mfi.listings import APP, clean, load_raw  # noqa: E402

ACTOR = "igolaizola~idealista-scraper"
INPUT = {"operation": "rent", "propertyType": "premises", "country": "es", "location": "Madrid",
         "maxItems": 0, "sortBy": "mostRecent", "fetchDetails": False, "fetchStats": False}
API = "https://api.apify.com/v2"
MIN_KEEP = 0.60


def main():
    tok = os.environ.get("APIFY_TOKEN")
    if not tok:
        sys.exit("APIFY_TOKEN not set")
    r = requests.post(f"{API}/acts/{ACTOR}/runs", params={"token": tok}, json=INPUT, timeout=60)
    r.raise_for_status()
    run = r.json()["data"]
    print("run", run["id"])
    t0 = time.time()
    while run["status"] in ("READY", "RUNNING"):
        if time.time() - t0 > 3 * 3600:
            sys.exit("Apify run took over 3 hours — aborted, listings unchanged")
        time.sleep(30)
        run = requests.get(f"{API}/actor-runs/{run['id']}", params={"token": tok}, timeout=60).json()["data"]
    if run["status"] != "SUCCEEDED":
        sys.exit(f"Apify run {run['status']} — listings unchanged")
    items, offset = [], 0
    while True:
        page = requests.get(f"{API}/datasets/{run['defaultDatasetId']}/items",
                            params={"token": tok, "offset": offset, "limit": 1000, "clean": "true"}, timeout=120).json()
        items += page
        offset += len(page)
        if len(page) < 1000:
            break
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw = ROOT / "data" / "raw" / "apify" / "listings_weekly"
    raw.mkdir(parents=True, exist_ok=True)
    f = raw / f"{stamp}.json"
    f.write_text(json.dumps(items, ensure_ascii=False))

    import geopandas as gpd
    from shapely.ops import unary_union
    d = gpd.read_file(APP / "districts.geojson").to_crs(25830)
    df, log = clean(load_raw(files=[f]), unary_union(d.geometry).buffer(50))
    cur = APP / "listings.parquet"
    if cur.exists():
        import pandas as pd
        old = len(pd.read_parquet(cur))
        if len(df) < MIN_KEEP * old:
            sys.exit(f"New pull has {len(df)} listings vs {old} now (< {MIN_KEEP:.0%}) — listings unchanged")
    df.to_parquet(cur, index=False)
    log.update(scraped_at=stamp, apify_run=run["id"], usage_usd=run.get("usageTotalUsd"))
    (APP / "listings_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
