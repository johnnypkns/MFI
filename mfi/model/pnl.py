"""Unit-level P&L Monte Carlo for each recommended site.

Revenue(site, draw) = AUV x demand_ratio(site, draw)
    AUV          average annual revenue of a median store of the brand (researched, cited)
    demand_ratio site's captured demand / median existing store's, drawn from the robustness simulation
                 -> ASSUMPTION: revenue scales proportionally with captured demand (elasticity 1)
Annual EBITDA = Revenue x (1 - COGS% - labour% - royalty% - marketing% - other%) - rent x m2 x 12
Payback (yrs) = initial investment / EBITDA   (pre-tax, unlevered; infinite if EBITDA <= 0)
ROIC          = EBITDA / initial investment

Rent model: 'market_lease' (franchisee leases: rent = €/m² x m² x 12, from Idealista) or 'franchisor_property'
(franchisor owns/leases the site and charges occupancy as % of sales — McDonald's Spain works this way).

Cost lines must be mutually exclusive. If a source bundles labour into 'other opex', labour is entered as 0 with
the note 'included in other_opex_pct' — never counted twice.

Every input is a triangular distribution (low, mode, high) carrying its own source and label
('verified', 'third_party_estimate', 'listing_data', 'user_assumption'). If a required input is missing the
simulation does not run — it never fills gaps with invented numbers.
"""
import numpy as np
import pandas as pd

REQUIRED = {
    "auv_eur": "Average revenue per store (€/yr)",
    "rent_eur_m2_month": "Rent (€/m²/month)",
    "unit_m2": "Unit size (m²)",
    "cogs_pct": "Food & packaging cost (% sales)",
    "labour_pct": "Labour incl. social security (% sales)",
    "royalty_pct": "Royalty (% sales)",
    "marketing_pct": "Marketing fund (% sales)",
    "other_opex_pct": "Other opex: utilities, insurance, maintenance (% sales)",
    "investment_eur": "Initial investment (€)",
    "occupancy_pct": "Occupancy paid to franchisor (% sales) — only if franchisor provides the site",
}
PCT = [k for k in REQUIRED if k.endswith("_pct") and k != "occupancy_pct"]
MARKET_ONLY = {"rent_eur_m2_month", "unit_m2"}       # needed when the franchisee leases on the open market
FRANCHISOR_ONLY = {"occupancy_pct"}                  # needed when the franchisor provides the property


def empty_inputs():
    return pd.DataFrame([{"key": k, "parameter": v, "low": None, "mode": None, "high": None,
                          "source": "", "label": ""} for k, v in REQUIRED.items()])


def missing(inputs: pd.DataFrame, rent_model="market_lease"):
    skip = FRANCHISOR_ONLY if rent_model == "market_lease" else MARKET_ONLY
    bad = []
    for _, r in inputs.iterrows():
        if r["key"] in skip:
            continue
        vals = [r["low"], r["mode"], r["high"]]
        if any(v is None or pd.isna(v) for v in vals) or not (r["low"] <= r["mode"] <= r["high"]):
            bad.append(r["parameter"])
    return bad


def _tri(rng, lo, mo, hi, n):
    return np.full(n, mo) if hi == lo else rng.triangular(lo, mo, hi, n)


def simulate(sites: pd.DataFrame, ratio_draws: np.ndarray, inputs: pd.DataFrame, N=10_000, seed=3,
             payback_target_yrs=4.0, site_rents: dict | None = None, rent_model="market_lease"):
    """site_rents: {rank: {"p25","p50","p75"}} from the Idealista scrape. When present for a site it replaces the
    city-level rent input for that site (triangular p25/p50/p75).
    rent_model: 'market_lease' (rent = €/m² x m²) or 'franchisor_property' (occupancy = % of sales, e.g. McDonald's)."""
    gaps = missing(inputs, rent_model)
    if gaps:
        return {"ok": False, "missing": gaps}
    rng = np.random.default_rng(seed)
    skip = FRANCHISOR_ONLY if rent_model == "market_lease" else MARKET_ONLY
    p = {r.key: (float(r.low), float(r["mode"]), float(r.high)) for _, r in inputs.iterrows() if r.key not in skip}
    draws = {k: _tri(rng, *v, N) for k, v in p.items()}
    for k in PCT:
        draws[k] = draws[k] / 100.0
    var_pct = sum(draws[k] for k in PCT)

    rows, sens = [], []
    for i, (_, s) in enumerate(sites.iterrows()):
        ratio = rng.choice(ratio_draws[:, i], N)
        sr = (site_rents or {}).get(int(s["rank"])) if rent_model == "market_lease" else None
        if rent_model == "market_lease":
            rent_m2 = _tri(rng, sr["p25"], sr["p50"], sr["p75"], N) if sr and "p50" in sr else draws["rent_eur_m2_month"]
            rent = rent_m2 * draws["unit_m2"] * 12
        else:
            rent_m2 = None
            rent = draws["auv_eur"] * ratio * draws["occupancy_pct"] / 100.0
        rev = draws["auv_eur"] * ratio
        ebitda = rev * (1 - var_pct) - rent
        payback = np.where(ebitda > 0, draws["investment_eur"] / np.maximum(ebitda, 1e-9), np.inf)
        breakeven_rev = rent / np.maximum(1 - var_pct, 1e-9)
        rows.append({
            "rank": int(s["rank"]), "address": s["address"],
            "rent_source": ("franchisor occupancy % of sales" if rent_model != "market_lease"
                            else "Idealista listings near site" if sr and "p50" in sr else "city-level input"),
            "revenue_p10": np.percentile(rev, 10), "revenue_p50": np.percentile(rev, 50), "revenue_p90": np.percentile(rev, 90),
            "ebitda_p10": np.percentile(ebitda, 10), "ebitda_p50": np.percentile(ebitda, 50), "ebitda_p90": np.percentile(ebitda, 90),
            "margin_p50": np.percentile(ebitda / rev, 50),
            "p_loss": float((ebitda <= 0).mean()),
            "payback_p50_yrs": float(np.percentile(payback, 50)),
            f"p_payback_le_{payback_target_yrs:g}y": float((payback <= payback_target_yrs).mean()),
            "roic_p50": float(np.percentile(ebitda / draws["investment_eur"], 50)),
            "breakeven_revenue_p50": float(np.percentile(breakeven_rev, 50)),
            "rent_share_of_revenue_p50": float(np.percentile(rent / rev, 50)),
        })
        # sensitivity: rank correlation of EBITDA with each uncertain input (tornado)
        drivers = {**{k: draws[k] for k in REQUIRED if k in draws}, "demand_ratio": ratio}
        if rent_m2 is not None:
            drivers["rent_eur_m2_month"] = rent_m2
        rk = pd.Series(ebitda).rank().values
        for k, v in drivers.items():
            if np.ptp(v) > 0:
                sens.append({"rank": int(s["rank"]), "driver": REQUIRED.get(k, "Site demand (Monte Carlo)"),
                             "spearman": float(np.corrcoef(rk, pd.Series(v).rank().values)[0, 1])})
    return {"ok": True, "N": N, "summary": pd.DataFrame(rows), "sensitivity": pd.DataFrame(sens),
            "payback_target_yrs": payback_target_yrs, "inputs": inputs, "rent_model": rent_model}
