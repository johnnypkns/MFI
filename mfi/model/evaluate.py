"""The franchisor's 10-stage site evaluation, applied to every candidate unit.

  1 Market planning      district headroom: demand vs the brand's best-penetrated districts
  2 Territory            minimum distance to the brand's existing stores (hard gate)
  3 Site criteria        franchisor size range, floor, and what the listing text says (smoke outlet, corner, frontage)
  4 Submission package   the dossier: catchment, footfall, competitors, rent (assembled, shown on the unit page)
  5 Sales forecast       ensemble of three methods: Huff gravity, analog (k nearest existing stores), regression
  6 Sales transfer       % of customers taken from each existing store (encroachment), Huff before/after
  7 Field check          data stand-ins for the site visit: corner, transit, footfall, competitors, traffic generators
  8 Underwriting         occupancy ratio, sales-to-investment, P(payback ≤ target), P(loss)
  9 Committee            APPROVE / CONDITIONAL / REJECT with the reasons
 10 Licensing & survival smoke-outlet and licence risk; closure risk from a model trained on open vs closed chain units

Every threshold lives in DEFAULTS and is editable in Settings. Sources for the benchmarks are in the methodology page.
"""
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .site_model import FEATURES, transform

DEFAULTS = {
    "territory_m": 250,            # 2: no unit closer than this to an existing store of the brand
    "transfer_flag": 0.10,         # 6: >10% of an existing store's customers -> needs franchisee consent
    "transfer_reject": 0.25,       # 6: >25% -> encroachment, rejected
    "occupancy_max": 0.10,         # 8: rent ≤ 10% of sales (industry rule of thumb 6–10%)
    "sales_to_investment_min": 1.5,  # 8: first-year sales ≥ 1.5× total investment
    "p_payback_min": 0.50,         # 8: at least even odds of paying back within the target
    "p_loss_max": 0.20,            # 8: at most 20% chance of losing money at EBITDA level
    "field_min": 0.5,              # 7: pass at least half the field checks
    "w_huff": 0.5, "w_analog": 0.25, "w_regression": 0.25,   # 5: ensemble weights (assumption)
    "k_analog": 7,
}
CLOSURE_MIN_AUC = 0.65           # 10: closure risk only affects the decision if the closure model is this accurate
STATUS_ORDER = {"APPROVE": 0, "CONDITIONAL": 1, "REJECT": 2}

# ------------------------------------------------------------------ 3 + 10: what the listing text says
PATTERNS = {
    "smoke_none": r"sin (salida de humos|chimenea|extracci[oó]n)|no (tiene|dispone de|hay) (salida de humos|chimenea)",
    "smoke_outlet": r"salida de humos|extracci[oó]n de humos|chimenea|extractor",
    "corner": r"esquina|chafl[aá]n",
    "shop_window": r"escaparate|cristalera|luna[s]? a (la )?calle",
    "prior_food": r"(antiguo|anterior|actualmente|funcionando como|ha sido)\s+(un |una )?(bar|restaurante|cafeter|pizzer|hamburgues|"
                  r"hosteler)|cocina (montada|industrial|equipada)|licencia de (bar|restauraci|hosteler|cafeter)",
    "licence": r"con licencia|licencia (en vigor|de actividad|concedida)",
    "needs_works": r"para reformar|a reformar|en bruto|necesita reforma",
    "mall": r"centro comercial",
}
FRONTAGE = re.compile(r"(\d{1,2}(?:[.,]\d)?)\s*(?:m|metros|mts)\.?\s*(?:lineales\s*)?de\s*fachada|fachada\s*de\s*(\d{1,2}(?:[.,]\d)?)\s*(?:m|metros)")


def parse_listing(text):
    t = (text or "").lower()
    out = {k: bool(re.search(p, t)) for k, p in PATTERNS.items()}
    if out["smoke_none"]:
        out["smoke_outlet"] = False
    m = FRONTAGE.search(t)
    out["frontage_m"] = float((m.group(1) or m.group(2)).replace(",", ".")) if m else None
    return out


# ------------------------------------------------------------------ 1: market headroom per district
def market_headroom(data, R):
    """Penetration = brand stores per 10k demand units (residents + weighted daytime activity) in each district.
    Benchmark = the 75th percentile of districts where the brand trades. Headroom = units the district could still take."""
    c = R.ctx
    huff = c["huff"]
    dist_of_sec = pd.Series(data.sections.dropna(subset=["pop_total"])["district_code"].astype(int).values, name="district")
    demand = pd.Series(huff.pop).groupby(dist_of_sec).sum()
    st = pd.Series(R.stores["district_code"].values).value_counts()
    df = pd.DataFrame({"demand": demand}).fillna(0)
    df["stores"] = st.reindex(df.index).fillna(0)
    df["per_10k"] = df["stores"] / (df["demand"] / 1e4).clip(lower=1e-9)
    bench = float(df.loc[df["stores"] > 0, "per_10k"].quantile(0.75)) if (df["stores"] > 0).any() else 0.0
    df["headroom"] = bench * df["demand"] / 1e4 - df["stores"]
    df.attrs["benchmark_per_10k"] = bench
    return df



# ------------------------------------------------------------------ 5: forecast ensemble
class Ensemble:
    """Three estimates of the customers a unit serves, each relative to the brand's median existing store:
       a) Huff gravity: the unit's own capture given competitors and the brand's network (computed in rank.py)
       b) Analog: similarity-weighted average of the k existing stores most like this site (feature distance)
       c) Regression: ridge regression of log(store capture) on the site features, fitted on existing stores
    Blend = weighted geometric mean; the disagreement between methods widens the forecast range."""

    def __init__(self, R, P):
        c = R.ctx
        self.P = P
        self.Xs = c["store_X"].values
        med = float(np.median(c["store_capture"]))
        self.ys = np.log(np.maximum(c["store_capture"] / med, 1e-3))
        self.scaler = StandardScaler().fit(self.Xs)
        self.Z = self.scaler.transform(self.Xs)
        self.tree = cKDTree(self.Z)
        self.k = int(min(P["k_analog"], len(self.Xs)))
        self.reg = Ridge(alpha=3.0).fit(self.Z, self.ys)
        self.loo = self._loo_errors()

    def _loo_errors(self):
        """Leave-one-store-out error of analog and regression against each store's Huff capture."""
        n = len(self.Z)
        ea, er = [], []
        for i in range(n):
            m = np.arange(n) != i
            d = np.linalg.norm(self.Z[m] - self.Z[i], axis=1)
            nn = np.argsort(d)[: self.k]
            w = 1 / (d[nn] + 0.25)
            ea.append((w * self.ys[m][nn]).sum() / w.sum() - self.ys[i])
            er.append(Ridge(alpha=3.0).fit(self.Z[m], self.ys[m]).predict(self.Z[i:i + 1])[0] - self.ys[i])
        rmse = lambda e: float(np.sqrt(np.mean(np.square(e)))) if len(e) else None
        return {"analog_rmse_log": rmse(ea), "regression_rmse_log": rmse(er),
                "analog_mae_pct": float(np.mean(np.abs(np.expm1(ea)))) if ea else None,
                "regression_mae_pct": float(np.mean(np.abs(np.expm1(er)))) if er else None, "n": n}

    def predict(self, X, huff_ratio):
        Z = self.scaler.transform(np.asarray(X))
        d, nn = self.tree.query(Z, k=self.k)
        d, nn = np.atleast_2d(d), np.atleast_2d(nn)
        w = 1 / (d + 0.25)
        analog = np.exp((w * self.ys[nn]).sum(1) / w.sum(1))
        regr = np.exp(self.reg.predict(Z))
        h = np.maximum(np.asarray(huff_ratio, float), 1e-3)
        P = self.P
        logs = np.vstack([np.log(h), np.log(analog), np.log(regr)])
        wts = np.array([P["w_huff"], P["w_analog"], P["w_regression"]])[:, None]
        blend = np.exp((wts * logs).sum(0) / wts.sum())
        spread = logs.std(0)                      # method disagreement (log scale)
        similarity = 1 / (1 + d.mean(1))          # how close the nearest analogs are (1 = identical)
        return pd.DataFrame({"f_huff": h, "f_analog": analog, "f_regression": regr, "f_blend": blend,
                             "f_spread": spread, "analog_similarity": similarity, "analog_idx": list(nn)})


# ------------------------------------------------------------------ 6: sales transfer (encroachment)
def transfer(huff, cur, Wstores, Wc):
    """Share of each existing store's customers the candidate(s) would take. Returns (stores x candidates)."""
    pop, base = huff.pop[:, None], huff.base[:, None]
    before = Wstores.T @ (huff.pop / (cur + huff.base))                      # J
    after = Wstores.T @ (pop / (cur[:, None] + Wc + base))                   # J x C
    return 1 - after / np.maximum(before[:, None], 1e-9)


# ------------------------------------------------------------------ 10: closure risk (backtest proxy)
def closure_model(data, R, registry_members, min_n=150):
    """Logistic model of P(closed) for CHAIN units in the brand's activity class, from the census status
    (Cerrado with the chain's sign still registered vs Abierto). Validated by spatial CV across districts.
    It is a cross-sectional proxy for failure (the census has no dates), not a time-to-closure model."""
    prem = data.premises
    cls = (R.category or "")[:4]
    if not cls:
        return None
    chain = prem["name_norm"].isin(set(registry_members["name_norm"]))
    in_cls = prem["epigrafes"].fillna("").str.split(",").map(lambda l: any(c.startswith(cls) for c in l))
    u = prem[chain & in_cls & prem["status"].isin(["Abierto", "Cerrado"]) & prem["access_type"].isin(["Puerta Calle", "Agrupado"])]
    y = (u["status"] == "Cerrado").astype(int).values
    if len(u) < min_n or y.sum() < 20 or (1 - y).sum() < 20:
        return {"ok": False, "n": int(len(u)), "closed": int(y.sum()), "class": cls}
    X = transform(R.ctx["fs"].compute(u[["x", "y"]].values))
    groups = u["district_code"].fillna(-1).values
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))
    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        if 0 < y[te].sum() < len(te):
            m.fit(X.iloc[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    m.fit(X, y)
    base = m.predict_proba(X)[:, 1]
    coef = dict(zip(FEATURES, m[-1].coef_[0]))
    return {"ok": True, "model": m, "auc": float(np.mean(aucs)) if aucs else None, "n": int(len(u)),
            "closed": int(y.sum()), "closed_rate": float(y.mean()), "class": cls,
            "score_quantiles": np.quantile(base, [0.25, 0.5, 0.75]).tolist(), "coef": coef}


# ------------------------------------------------------------------ 7: field check
def field_checks(row, ref, text_flags):
    """Data stand-ins for what a real-estate manager checks on site. ref = brand's existing-store percentiles."""
    checks = [
        ("Street-level unit", row.get("floor") in ("ground", "unknown"), "ground floor / street access"),
        ("Transit within 300 m", row.get("transit_300", 0) >= 1, f"{int(row.get('transit_300', 0))} metro/train stations"),
        ("Footfall ≥ brand's weaker stores", row.get("footfall", 0) >= ref["footfall_p25"],
         f"footfall index {row.get('footfall', 0):,.0f} vs brand p25 {ref['footfall_p25']:,.0f}"),
        ("Traffic generators nearby", row.get("food_300", 0) >= ref["food_p25"],
         f"{int(row.get('food_300', 0))} food & drink outlets within 300 m (brand p25 {ref['food_p25']:.0f})"),
        ("Competition no denser than brand norm", row.get("comp_500", 0) <= ref["comp_p75"],
         f"{int(row.get('comp_500', 0))} direct competitors within 500 m (brand p75 {ref['comp_p75']:.0f})"),
        ("Corner or shop window", text_flags.get("corner") or text_flags.get("shop_window"),
         "listing mentions corner/shop window" if (text_flags.get("corner") or text_flags.get("shop_window")) else "not mentioned in listing"),
    ]
    return [{"check": c, "ok": bool(ok), "detail": d} for c, ok, d in checks]


def brand_reference(R):
    f = R.ctx["fs"].compute(R.ctx["stores_xy"])
    return {"footfall_p25": float(f["footfall"].quantile(0.25)), "food_p25": float(f["food_300"].quantile(0.25)),
            "comp_p75": float(f["comp_500"].quantile(0.75)), "pop_1000_p25": float(f["pop_1000"].quantile(0.25))}


# ------------------------------------------------------------------ 9: committee
def committee(row, P, food, payback_target, has_econ):
    """Build the 10-stage trail for one unit and decide. Each stage: pass / flag / fail / info."""
    S = []
    add = lambda n, name, status, detail: S.append({"stage": n, "name": name, "status": status, "detail": detail})

    h = row.get("district_headroom")
    if h is None or np.isnan(h):
        add(1, "Market planning", "info", "district headroom not computed")
    elif h >= 0.5:
        add(1, "Market planning", "pass", f"{row['district']}: room for ~{h:.1f} more units at the brand's best-district penetration")
    else:
        add(1, "Market planning", "flag", f"{row['district']}: already at or above the brand's best-district penetration (headroom {h:.1f})")

    d = row.get("dist_nearest_brand_m", np.inf)
    add(2, "Territory", "pass" if d >= P["territory_m"] else "fail",
        f"nearest existing store {d:,.0f} m (protected radius {P['territory_m']} m)")

    t = row.get("text", {})
    crit = f"{row['m2']:.0f} m², floor: {row.get('floor')}"
    if t.get("frontage_m"):
        crit += f", {t['frontage_m']:.0f} m frontage"
    add(3, "Site criteria", "pass", crit + " — within the franchisor's size range")

    add(4, "Submission package", "info",
        f"{row.get('pop_1000', 0):,.0f} residents within 1 km · {int(row.get('comp_500', 0))} competitors within 500 m · "
        f"€{row['rent_month']:,.0f}/month")

    sp = row.get("f_spread", 0)
    add(5, "Sales forecast", "flag" if sp > 0.35 else "pass",
        f"{row['f_blend']:.2f}× a median store (Huff {row['f_huff']:.2f}, analog {row['f_analog']:.2f}, regression {row['f_regression']:.2f})"
        + (" — methods disagree, wide range" if sp > 0.35 else ""))

    tr = row.get("transfer_max", 0)
    st = "fail" if tr > P["transfer_reject"] else "flag" if tr > P["transfer_flag"] else "pass"
    add(6, "Sales transfer", st, f"takes up to {tr:.0%} of an existing store's customers"
        + ("" if st == "pass" else " — needs the affected franchisee's consent" if st == "flag" else " — encroachment"))

    fc = row.get("field", [])
    share = np.mean([c["ok"] for c in fc]) if fc else 0
    add(7, "Field check", "pass" if share >= P["field_min"] else "flag",
        f"{sum(c['ok'] for c in fc)}/{len(fc)} checks passed — verify on site")

    if has_econ:
        gates = [("rent ≤ %.0f%% of sales" % (P["occupancy_max"] * 100), row["rent_share"] <= P["occupancy_max"], f"{row['rent_share']:.1%}"),
                 ("sales ≥ %.1f× investment" % P["sales_to_investment_min"], row["sales_to_investment"] >= P["sales_to_investment_min"],
                  f"{row['sales_to_investment']:.2f}×"),
                 ("P(payback ≤ %.0fy) ≥ %.0f%%" % (payback_target, P["p_payback_min"] * 100), row["p_payback"] >= P["p_payback_min"],
                  f"{row['p_payback']:.0%}"),
                 ("P(loss) ≤ %.0f%%" % (P["p_loss_max"] * 100), row["p_loss"] <= P["p_loss_max"], f"{row['p_loss']:.0%}")]
        failed = [g for g in gates if not g[1]]
        hard = [g for g in failed if g[0].startswith("P(")]
        add(8, "Underwriting", "fail" if hard else "flag" if failed else "pass",
            " · ".join(f"{'✓' if ok else '✗'} {n} ({v})" for n, ok, v in gates))
        row["gates"] = [{"gate": n, "ok": bool(ok), "value": v} for n, ok, v in gates]
    else:
        add(8, "Underwriting", "flag", "brand economics not verified — cannot underwrite")

    lic = []
    lstatus = "pass"
    if food:
        if t.get("smoke_none"):
            lic.append("listing says NO smoke outlet — Madrid kitchens need a roof-level outlet; usually needs the building's consent")
            lstatus = "fail"
        elif t.get("smoke_outlet"):
            lic.append("smoke outlet mentioned")
        else:
            lic.append("smoke outlet not mentioned — ask before visiting")
            lstatus = "flag"
        if t.get("prior_food"):
            lic.append("previously food & drink (licence change usually simpler)")
    if row.get("traspaso_eur", 0) > 0:
        lic.append(f"traspaso €{row['traspaso_eur']:,.0f} (included in investment)")
    if t.get("needs_works"):
        lic.append("needs works")
        lstatus = "flag" if lstatus == "pass" else lstatus
    cr = row.get("closure_rel")
    if cr is not None and np.isfinite(cr):
        valid = bool(row.get("closure_valid"))
        lic.append(f"closure risk {cr:.1f}× the category median" + ("" if valid else " (closure model too weak to act on)"))
        if valid and cr >= 1.5 and lstatus == "pass":
            lstatus = "flag"
    add(10, "Licensing & survival", lstatus, "; ".join(lic) if lic else "no issues found in the listing")

    fails = [s for s in S if s["status"] == "fail"]
    flags = [s for s in S if s["status"] == "flag"]
    decision = "REJECT" if fails else "CONDITIONAL" if flags else "APPROVE"
    reasons = [f"Stage {s['stage']} {s['name']}: {s['detail']}" for s in (fails or flags)]
    S.insert(8, {"stage": 9, "name": "Committee", "status": {"APPROVE": "pass", "CONDITIONAL": "flag", "REJECT": "fail"}[decision],
                 "detail": decision + (" — " + str(len(fails or flags)) + (" blocking issue(s)" if fails else " condition(s)") if (fails or flags) else "")})
    return decision, reasons, S
