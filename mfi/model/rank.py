"""Money-first ranking: score EVERY real listing, simulate its P&L, rank by risk-adjusted return.

Order (the fix John asked for):
  1. franchise profile (Claude research, cached)  -> unit size range, investment, AUV, cost ratios, rent model
  2. candidate listings = real Idealista units: commercial, size within the brand's range, ground floor or
     mezzanine (configurable), not in an industrial zone, not on top of an existing store of the brand
  3. score each: suitability (location model) + the customers the unit would serve itself (Huff), relative to a
     median existing store of the brand, capped at the brand's 90th-percentile store; own stores nearby share demand
  4. P&L Monte Carlo per listing with ITS OWN rent, m² and traspaso
  5. rank, greedily: pick the best, add it to the brand's network, re-score the rest (cannibalisation), repeat

Ranking metric
  with brand economics : P(payback <= target years), ties broken by median EBITDA
  without them         : captured demand per € of annual rent (customers per euro), within the suitability gate
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .site_model import INDUSTRIAL_SHARE_MAX, get_surface, transform

MODEL_ERROR_SIGMA = 0.20      # lognormal noise on revenue: model error around the demand estimate (assumption, shown)
RENT_OUTLIER = (0.35, 3.0)    # listing €/m² vs the median of comparable listings within 1 km
CAP_QUANTILE = 0.90           # a new unit can't capture more than the brand's 90th-percentile existing store


def candidates(data, R, listings, m2_range=(60, 600), floors=("ground", "mezzanine", "unknown"), min_dist_own_m=150):
    df = listings[listings["m2"].between(*m2_range) & listings["floor"].isin(floors)].copy()
    log = {"listings": len(listings), "size_and_floor": len(df)}
    xy = df[["x", "y"]].values
    df = df[get_surface(data).industrial_share(xy, 200) <= INDUSTRIAL_SHARE_MAX]
    log["not_industrial"] = len(df)
    # local-market sanity: compare €/m² with comparable listings (same size band) within 1 km
    allx = listings[["x", "y"]].values
    tree = cKDTree(allx)
    med = []
    for (x, y) in df[["x", "y"]].values:
        nb = tree.query_ball_point([x, y], 1000)
        med.append(np.median(listings["eur_m2"].values[nb]) if len(nb) >= 5 else np.nan)
    df["local_eur_m2"] = med
    rel = df["eur_m2"] / df["local_eur_m2"]
    df = df[rel.isna() | rel.between(*RENT_OUTLIER)]
    log["rent_consistent_with_area"] = len(df)
    own = cKDTree(R.ctx["stores_xy"]).query(df[["x", "y"]].values)[0]
    df = df[own >= min_dist_own_m].copy()
    df["dist_nearest_brand_m"] = own[own >= min_dist_own_m]
    log["not_on_own_store"] = len(df)
    return df, log


def score(R, df):
    c = R.ctx
    feat = c["fs"].compute(df[["x", "y"]].values)
    df = pd.concat([df.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)
    df["suitability"] = c["model"].predict_proba(transform(feat))[:, 1]
    df["passes_gate"] = df["suitability"] >= c["gate"]
    return df


def _own(huff, cur, w):
    """Customers each candidate would serve itself if opened next to the current network (cannibalisation:
    nearby own stores take part of the same demand)."""
    return np.array([huff.own_capture(cur + w[:, j], w[:, j]) for j in range(w.shape[1])])


def _pnl(rev_mult, rent_year, inv, econ, rng, N):
    tri = lambda k: rng.triangular(*econ[k], N) if econ[k][0] < econ[k][2] else np.full(N, econ[k][1])
    auv = tri("auv_eur")
    var = (tri("cogs_pct") + tri("labour_pct") + tri("royalty_pct") + tri("marketing_pct") + tri("other_opex_pct")) / 100
    rev = auv * rev_mult * rng.lognormal(0, MODEL_ERROR_SIGMA, N)
    if "occupancy_pct" in econ:                       # franchisor provides the property: rent = % of sales
        rent_year = rev * tri("occupancy_pct") / 100
    ebitda = rev * (1 - var) - rent_year
    invest = tri("investment_eur") + inv
    payback = np.where(ebitda > 0, invest / np.maximum(ebitda, 1e-9), np.inf)
    return rev, ebitda, payback, invest


def rank(data, R, listings, econ=None, n=5, m2_range=(60, 600), floors=("ground", "mezzanine", "unknown"),
         payback_target=4.0, pool=120, N=4000, seed=5):
    """econ: {key: (low, mode, high)} for auv_eur, cogs_pct, labour_pct, royalty_pct, marketing_pct, other_opex_pct,
    investment_eur. None -> rank by demand per euro of rent."""
    rng = np.random.default_rng(seed)
    c = R.ctx
    cand, log = candidates(data, R, listings, m2_range, floors)
    if cand.empty:
        return {"ok": False, "reason": "No listing matches the size/floor filter.", "log": log}
    df = score(R, cand)
    log["pass_suitability_gate"] = int(df["passes_gate"].sum())
    df = df[df["passes_gate"]].copy()
    if df.empty:
        return {"ok": False, "reason": "No listing passes the suitability gate.", "log": log}

    huff, cur = c["huff"], c["brand_w_sum"].copy()
    med = float(np.median(c["store_capture"]))                         # customers a median existing store serves itself
    cap = float(np.quantile(c["store_capture"], CAP_QUANTILE) / med)   # the brand's own busiest stores
    W = huff.weights(df[["x", "y"]].values)
    df["demand_ratio"] = np.minimum(_own(huff, cur, W) / med, cap)
    df["rent_year"] = df["rent_month"] * 12
    df["demand_per_k_rent"] = df["demand_ratio"] * med / (df["rent_year"] / 1000)
    # pre-screen the pool for the expensive steps
    key = "demand_ratio" if econ else "demand_per_k_rent"
    df = df.nlargest(pool, key).reset_index(drop=True)
    W = huff.weights(df[["x", "y"]].values)

    picks, alive = [], np.ones(len(df), bool)
    for r in range(1, n + 1):
        ratio = np.minimum(_own(huff, cur, W) / med, cap)
        if econ:
            stats = []
            for j in range(len(df)):
                if not alive[j]:
                    stats.append((-1, -np.inf))
                    continue
                _, e, pb, _ = _pnl(max(ratio[j], 0), df.at[j, "rent_year"], df.at[j, "traspaso_eur"], econ, rng, N)
                stats.append(((pb <= payback_target).mean(), np.median(e)))
            order = sorted(range(len(df)), key=lambda j: stats[j], reverse=True)
        else:
            val = np.where(alive, ratio * med / (df["rent_year"].values / 1000), -np.inf)
            order = list(np.argsort(-val))
        j = order[0]
        if not alive[j]:
            break
        row = df.iloc[j].to_dict()
        row.update(rank=r, demand_ratio=float(ratio[j]))
        if econ:
            rev, e, pb, inv = _pnl(max(ratio[j], 0), row["rent_year"], row["traspaso_eur"], econ, np.random.default_rng(seed + r), 10_000)
            row.update(revenue_p50=np.median(rev), ebitda_p10=np.percentile(e, 10), ebitda_p50=np.median(e),
                       ebitda_p90=np.percentile(e, 90), p_loss=(e <= 0).mean(), payback_p50=np.median(pb),
                       p_payback=(pb <= payback_target).mean(), invest_p50=np.median(inv),
                       rent_share=(row["rent_year"] / np.median(rev)) if "occupancy_pct" not in econ else econ["occupancy_pct"][1] / 100,
                       roic_p50=np.median(e / inv))
        picks.append(row)
        cur = cur + W[:, j]                                  # cannibalisation: next pick competes with this one
        alive &= np.hypot(df["x"].values - df.at[j, "x"], df["y"].values - df.at[j, "y"]) >= 400
    out = pd.DataFrame(picks)
    return {"ok": True, "sites": out, "log": log, "metric": "P(payback ≤ %.0fy), then median EBITDA" % payback_target
            if econ else "captured demand per €1k of annual rent", "pool": len(df), "model_error_sigma": MODEL_ERROR_SIGMA,
            "demand_cap": cap}
