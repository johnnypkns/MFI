"""One-click analysis: research -> listings -> score -> P&L Monte Carlo -> rank. Used by the app and scripts."""
import json
import re

import pandas as pd

from .model.export import APP, ROOT, norm_name
from .model.rank import rank
from .model.site_model import run
from .stores import inventory, load_osm, missing_from_census

ECON_KEYS = ["auv_eur", "cogs_pct", "labour_pct", "royalty_pct", "marketing_pct", "other_opex_pct", "investment_eur"]
CACHE = ROOT / "data" / "research"
DEFAULT_M2 = (60, 400)


def cached_research(brand):
    f = CACHE / f"{norm_name(brand)}.json"
    return json.loads(f.read_text()) if f.exists() else None


def save_research(brand, profile):
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{norm_name(brand)}.json").write_text(json.dumps(profile, default=str, ensure_ascii=False, indent=1))


def econ_from_profile(profile, overrides=None):
    """Triangular (low, mode, high) per P&L input from Claude's research, user overrides on top.
    Returns econ (or None), the missing inputs, the rent model and a suggested unit-size range."""
    d = (profile or {}).get("data") or {}
    src = dict(d.get("pnl_inputs") or {})
    src.update(overrides or {})
    econ, missing = {}, []
    for k in ECON_KEYS + ["occupancy_pct"]:
        v = src.get(k) or {}
        if v.get("included_in"):
            econ[k] = (0.0, 0.0, 0.0)
        elif v.get("mode") is not None:
            lo, mo, hi = v.get("low"), v["mode"], v.get("high")
            lo = mo if lo is None else lo
            hi = mo if hi is None else hi
            econ[k] = tuple(sorted((float(lo), float(mo), float(hi))))
            econ[k] = (econ[k][0], float(mo), econ[k][2])
        elif k in ECON_KEYS:
            missing.append(k)
    rent_model = d.get("rent_model") if d.get("rent_model") in ("market_lease", "franchisor_property") else "market_lease"
    if rent_model == "market_lease":
        econ.pop("occupancy_pct", None)
    elif "occupancy_pct" not in econ:
        missing.append("occupancy_pct")
    m2 = (d.get("min_floor_area_m2") or {}).get("value") or ((src.get("unit_m2") or {}).get("mode"))
    m2_range = (int(m2 * 0.8), int(max(m2 * 2.2, m2 + 150))) if m2 else None
    return (econ if not missing else None), missing, rent_model, m2_range


def analyse(data, brand, n=5, listings=None, profile=None, m2_range=None, floors=("ground", "mezzanine", "unknown"),
            payback_target=4.0, overrides=None, osm=None):
    reg = pd.read_parquet(APP / "brands.parquet")
    row = reg[reg["brand"] == brand]
    extra = None
    if len(row) and row.iloc[0]["sector"] == "Food & drink":
        inv, _, _ = inventory(data, brand, osm if osm is not None else load_osm()[0])
        extra = missing_from_census(inv)
    R = run(data, brand, n, extra_stores=extra)
    econ, missing, rent_model, m2_suggest = econ_from_profile(profile, overrides)
    m2_range = m2_range or m2_suggest or DEFAULT_M2
    listings = listings if listings is not None else pd.read_parquet(APP / "listings.parquet")
    ranked = rank(data, R, listings, econ=econ, n=n, m2_range=m2_range, floors=floors, payback_target=payback_target)
    return {"R": R, "ranked": ranked, "econ": econ, "missing": missing, "rent_model": rent_model,
            "m2_range": m2_range, "profile": profile}
