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

from . import evaluate as EV
from .site_model import FEATURES, INDUSTRIAL_SHARE_MAX, get_surface, transform

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


def _pnl(rev_mult, rent_year, inv, econ, rng, N, sigma=MODEL_ERROR_SIGMA):
    tri = lambda k: rng.triangular(*econ[k], N) if econ[k][0] < econ[k][2] else np.full(N, econ[k][1])
    auv = tri("auv_eur")
    var = (tri("cogs_pct") + tri("labour_pct") + tri("royalty_pct") + tri("marketing_pct") + tri("other_opex_pct")) / 100
    rev = auv * rev_mult * rng.lognormal(0, sigma, N)
    if "occupancy_pct" in econ:                       # franchisor provides the property: rent = % of sales
        rent_year = rev * tri("occupancy_pct") / 100
    ebitda = rev * (1 - var) - rent_year
    invest = tri("investment_eur") + inv
    payback = np.where(ebitda > 0, invest / np.maximum(ebitda, 1e-9), np.inf)
    return rev, ebitda, payback, invest


def rank(data, R, listings, econ=None, n=5, m2_range=(60, 600), floors=("ground", "mezzanine", "unknown"),
         payback_target=4.0, pool=120, N=4000, seed=5, eval_params=None, food=False, closure=None):
    """econ: {key: (low, mode, high)} for auv_eur, cogs_pct, labour_pct, royalty_pct, marketing_pct, other_opex_pct,
    investment_eur. None -> rank by demand per euro of rent. Every unit in the pool goes through the 10-stage
    evaluation (evaluate.py); ranking = committee decision first (APPROVE > CONDITIONAL > REJECT), then money."""
    P = {**EV.DEFAULTS, **(eval_params or {})}
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
    key = "demand_ratio" if econ else "demand_per_k_rent"
    df = df.nlargest(pool, key).reset_index(drop=True)
    W = huff.weights(df[["x", "y"]].values)

    # ---- static parts of the 10 stages (don't change as picks are added)
    df["district"] = df["district_name"]
    head = EV.market_headroom(data, R)
    sec = data.sections.dropna(subset=["pop_total"])            # district = that of the nearest census section
    dc = pd.Series(sec["district_code"].astype(int).values[cKDTree(sec[["x", "y"]].values).query(df[["x", "y"]].values)[1]])
    df["district_code_est"] = dc.values
    df["district_headroom"] = dc.map(head["headroom"]).values
    df["text"] = df["description"].map(EV.parse_listing)
    ens = EV.Ensemble(R, P)
    Xc = transform(df[FEATURES]).values
    ref = EV.brand_reference(R)
    df["field"] = [EV.field_checks(r, ref, r["text"]) for r in df.to_dict("records")]
    if closure and closure.get("ok"):
        pc = closure["model"].predict_proba(transform(df[FEATURES]))[:, 1]
        df["closure_p"] = pc
        df["closure_rel"] = pc / max(closure["score_quantiles"][1], 1e-6)   # vs the median chain unit in the class
        df["closure_valid"] = (closure.get("auc") or 0) >= EV.CLOSURE_MIN_AUC
    Wnet = huff.weights(c["stores_xy"])

    picks, alive, first_round = [], np.ones(len(df), bool), None
    for r in range(1, n + 1):
        hr = np.minimum(_own(huff, cur, W) / med, cap)
        F = ens.predict(Xc, np.maximum(hr, 1e-3))
        T = EV.transfer(huff, cur, Wnet, W)                     # existing (+picked) stores x candidates
        tmax = T.max(0) if T.shape[0] else np.zeros(len(df))
        evals = []
        for j in range(len(df)):
            if not alive[j]:
                evals.append(None)
                continue
            row = df.iloc[j].to_dict()
            row.update({k: F.iloc[j][k] for k in F.columns}, transfer_max=float(tmax[j]), demand_ratio=float(F.iloc[j]["f_blend"]),
                       huff_ratio=float(hr[j]))
            if econ:
                sig = float(np.hypot(MODEL_ERROR_SIGMA, F.iloc[j]["f_spread"]))
                rev, e, pb, inv = _pnl(row["f_blend"], row["rent_year"], row["traspaso_eur"], econ, rng, N, sig)
                rs = (row["rent_year"] / np.median(rev)) if "occupancy_pct" not in econ else econ["occupancy_pct"][1] / 100
                row.update(p_payback=float((pb <= payback_target).mean()), ebitda_p50=float(np.median(e)), p_loss=float((e <= 0).mean()),
                           rent_share=float(rs), sales_to_investment=float(np.median(rev) / np.median(inv)), sigma=sig)
            dec, reasons, stages = EV.committee(row, P, food, payback_target, bool(econ))
            row.update(decision=dec, reasons=reasons, stages=stages)
            evals.append(row)
        if r == 1:
            first_round = pd.Series([e["decision"] for e in evals if e]).value_counts().to_dict()
        if econ:
            keyf = lambda e: (EV.STATUS_ORDER[e["decision"]], -e["p_payback"], -e["ebitda_p50"])
        else:
            keyf = lambda e: (EV.STATUS_ORDER[e["decision"]], -e["demand_ratio"] * med / (e["rent_year"] / 1000))
        live = [j for j in range(len(df)) if evals[j] is not None]
        if not live:
            break
        j = min(live, key=lambda j: keyf(evals[j]))
        row = evals[j]
        row["rank"] = r
        if econ:
            sig = row["sigma"]
            rev, e, pb, inv = _pnl(row["f_blend"], row["rent_year"], row["traspaso_eur"], econ, np.random.default_rng(seed + r), 10_000, sig)
            row.update(revenue_p10=np.percentile(rev, 10), revenue_p50=np.median(rev), revenue_p90=np.percentile(rev, 90),
                       ebitda_p10=np.percentile(e, 10), ebitda_p50=np.median(e), ebitda_p90=np.percentile(e, 90),
                       p_loss=(e <= 0).mean(), payback_p50=np.median(pb), p_payback=(pb <= payback_target).mean(),
                       invest_p50=np.median(inv), roic_p50=np.median(e / inv))
        row["analogs"] = [int(i) for i in row.pop("analog_idx")]
        picks.append(row)
        cur = cur + W[:, j]                                  # cannibalisation: next pick competes with this one
        Wnet = np.hstack([Wnet, W[:, j:j + 1]])
        alive &= np.hypot(df["x"].values - df.at[j, "x"], df["y"].values - df.at[j, "y"]) >= 400
    out = pd.DataFrame(picks)
    return {"ok": True, "sites": out, "log": log, "metric": ("committee decision, then P(payback ≤ %.0fy), then median EBITDA" % payback_target)
            if econ else "committee decision, then captured demand per €1k of annual rent", "pool": len(df),
            "model_error_sigma": MODEL_ERROR_SIGMA, "demand_cap": cap, "decisions": first_round or {},
            "ensemble": ens.loo, "headroom": head, "closure": {k: v for k, v in (closure or {}).items() if k != "model"},
            "eval_params": P}
