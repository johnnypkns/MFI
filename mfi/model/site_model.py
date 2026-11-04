"""Site-selection model v0.1 — runs from the frozen parquet files, no database needed.

Pipeline for one brand:
  1. find the brand's existing Madrid stores in the premises census (fuzzy name match)
  2. infer its category from those stores' registered activities -> competitor set
  3. describe every location with the same features (population, income, age, commercial intensity, competition)
  4. SUITABILITY: logistic regression, brand stores vs background premises ("where does this brand choose to be?")
     validated with spatial cross-validation (whole districts held out) -> AUC
  5. DEMAND: Huff gravity model over the 2,462 census sections -> residents captured by each store,
     accounting for competitors and the brand's own stores (cannibalisation)
  6. SELECTION: greedy optimisation over vacant street-level units that pass the suitability gate,
     adding the site with the largest *incremental* captured demand each round
     demand = residents + k x open premises (daytime activity); k and the distance decay are
     CALIBRATED on the brand's own stores (most even catchments across its existing network), not assumed
  7. HOW MANY: keep adding while the next store captures >= 50% of what a typical existing store captures

Every tunable number is in PARAMS and reported in the output — nothing hidden.
"""
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .export import APP, norm_name

PARAMS = dict(
    huff_decay_m=700,          # distance at which a store's pull falls to 1/e. ~9-min walk. ASSUMPTION until calibrated on review counts.
    huff_cutoff_m=3000,        # beyond this a store captures nobody
    outside_option_m=1200,     # 'no fast food nearby' alternative, expressed as an equivalent distance
    background_ratio=30,       # background points per brand store
    gate_quantile=0.25,        # candidate must be at least as suitable as the bottom-quartile existing store
    candidate_pool=800,        # top-suitability candidates kept for the optimiser
    min_spacing_m=600,         # recommended sites at least this far apart
    saturation_ratio=0.5,      # stop when the next store captures < 50% of a median existing store
    max_sites=15,
    seed=7,
)

FEATURES = ["pop_500", "pop_1000", "young_share_1000", "income_1000", "footfall", "office_500", "transit_300",
            "food_300", "comp_500"]
FEATURE_LABELS = {
    "pop_500": "Residents within 500 m", "pop_1000": "Residents within 1 km",
    "young_share_1000": "Share aged 15–44 within 1 km", "income_1000": "Net income/person within 1 km (€)",
    "footfall": "Footfall index (calibrated on pedestrian counters)", "office_500": "Offices within 500 m (daytime workers)",
    "transit_300": "Metro/rail stations within 300 m", "food_300": "Open food & drink within 300 m",
    "comp_500": "Same-category competitors within 500 m",
}
INDUSTRIAL_SHARE_MAX = 0.5      # candidate sites in industrial/wholesale zones are excluded


@dataclass
class Data:
    premises: pd.DataFrame
    sections: pd.DataFrame
    meta: dict
    barrios: pd.DataFrame
    generators: pd.DataFrame | None = None
    demand_cal: dict | None = None
    _surface: object = None

    def surface(self):
        """Layer 2 demand surface (built once)."""
        if self._surface is None:
            from ..demand import DemandSurface, TO_UTM
            g = self.generators.copy()
            g["x"], g["y"] = TO_UTM.transform(g["lon"].values, g["lat"].values)
            self._surface = DemandSurface(self.premises, self.sections, g)
        return self._surface


def load_data(app_dir=APP) -> Data:
    gen = pd.read_parquet(app_dir / "generators.parquet") if (app_dir / "generators.parquet").exists() else None
    cal = json.loads((app_dir / "demand.json").read_text()) if (app_dir / "demand.json").exists() else None
    return Data(pd.read_parquet(app_dir / "premises.parquet"), pd.read_parquet(app_dir / "sections.parquet"),
                json.loads((app_dir / "meta.json").read_text()), pd.read_parquet(app_dir / "barrios.parquet"), gen, cal)


# ------------------------------------------------------------------ brand + category
STORE_CODE_PREFIXES = ("56", "4724", "4729", "4711")   # food service + food retail (Telepizza delivery shops are 472407)
NOT_A_STORE = r"OFICINA|CORPORATIV|ALMAC[EÉ]N|OBRADOR CENTRAL|CENTRAL DE|SEDE"
SPECIAL_VENUE_BARRIOS = {212}                            # Aeropuerto: travellers, no residential catchment


NON_TRADING_CODES = ("000000", "821001", "")         # no activity / back-office call centre


def is_store(df):
    """A matched premises counts as a store if it isn't an office/warehouse and holds a trading licence.
    Food brands (most premises in division 56) must hold a food licence specifically."""
    codes = df["epigrafes"].fillna("").str.split(",")
    office = df["trade_name"].fillna("").str.upper().str.contains(NOT_A_STORE, regex=True)
    divs = codes.explode().str[:2]
    divs = divs[~divs.isin(["00", "82", ""])]
    food_brand = len(divs) and divs.value_counts().index[0] == "56"
    if food_brand:
        ok = codes.map(lambda l: any(c.startswith(STORE_CODE_PREFIXES) for c in l))
    else:
        ok = codes.map(lambda l: any(c not in NON_TRADING_CODES for c in l))
    return ok & ~office


def _registry():
    try:
        return pd.read_parquet(APP / "brands.parquet"), pd.read_parquet(APP / "brand_members.parquet")
    except Exception:
        return None, None


def match_brand(prem: pd.DataFrame, query: str, threshold=0.86, stores_only=True):
    q = norm_name(query) or ""
    reg, mem = _registry()
    if reg is not None:
        hit = reg[(reg["brand_key"] == q) | (reg["brand"].map(norm_name) == q)]
        if len(hit):
            names = set(mem.loc[mem["brand_id"] == hit.iloc[0]["brand_id"], "name_norm"])
            return _finish(prem, names, stores_only)
    names = prem["name_norm"].dropna().unique()
    hits = [n for n in names if (n == q) or len(q) >= 3 and (n.startswith(q) or (q in n and len(n) <= len(q) + 8)
                                                  or SequenceMatcher(None, q, n).ratio() >= threshold)]
    return _finish(prem, set(hits), stores_only)


def _finish(prem, hits, stores_only):
    stores = prem[prem["name_norm"].isin(hits) & (prem["status"] == "Abierto")].copy()
    if stores_only:
        stores = stores[is_store(stores)]
    # one physical store can appear twice (shop + its terrace/kiosk unit): collapse points < 30 m apart
    if len(stores) > 1:
        tree = cKDTree(stores[["x", "y"]].values)
        keep, seen = [], set()
        for i, nb in enumerate(tree.query_ball_point(stores[["x", "y"]].values, 30)):
            if i in seen:
                continue
            keep.append(i)
            seen.update(nb)
        stores = stores.iloc[keep]
    matched = (prem[prem["name_norm"].isin(hits)].groupby("trade_name").size().sort_values(ascending=False))
    return stores, matched


def brand_category(stores: pd.DataFrame):
    """Most common activity code among the brand's own stores (food brands: food-service code)."""
    codes = stores["epigrafes"].dropna().str.split(",").explode()
    codes = codes[~codes.isin(NON_TRADING_CODES)]
    food = codes[codes.str.startswith("56")]
    if len(food) >= 0.5 * max(len(stores), 1):
        codes = food
    return codes.value_counts().index[0] if len(codes) else None


def competitor_mask(op, category):
    """A competitor is any open premises in the same activity CLASS (first 4 digits: supermarkets 471101 and
    hypermarkets 471102 compete), or any premises of a known chain whose main activity is in that class —
    which catches e.g. 'Mi Alcampo' registered under fresh-food codes. Food & drink uses the exact code
    (fast food vs restaurant vs café are different formats; the class would lump every bar together)."""
    cls = category if category.startswith("56") else category[:4]   # food codes are already format-level
    by_code = op["epigrafes"].fillna("").str.split(",").map(lambda l: any(c.startswith(cls) for c in l))
    reg, mem = _registry()
    if reg is None:
        return by_code
    same_class = reg[reg["top_activity"].fillna("").str.startswith(cls)]["brand_id"]
    names = set(mem.loc[mem["brand_id"].isin(same_class), "name_norm"])
    return by_code | op["name_norm"].isin(names)


def is_open(p):
    return p["status"] == "Abierto"


def has_code(series, code):
    return series.fillna("").str.split(",").map(lambda l: code in l)


# ------------------------------------------------------------------ features
class FeatureSpace:
    def __init__(self, data: Data, brand_idx, category):
        p, s = data.premises, data.sections.dropna(subset=["pop_total"])
        self.sec_xy = s[["x", "y"]].values
        self.sec_pop = s["pop_total"].values.astype(float)
        self.sec_young = (s["pop_15_24"] + s["pop_25_44"]).values.astype(float)
        self.sec_inc = s["net_income_per_person"].fillna(s["net_income_per_person"].median()).values
        self.sec_tree = cKDTree(self.sec_xy)
        op = p[is_open(p)]
        self.surf = data.surface()
        self.cal = data.demand_cal
        # daytime activity per section = consumer-facing + office premises (industrial/wholesale excluded: Layer 2)
        sd = self.surf.sec.set_index("section_code")
        self.sec_activity = s["section_code"].map(sd["consumer"] + sd["office"]).fillna(0).values.astype(float)
        food = op[op["divisions"].fillna("").str.contains("56")]
        self.food_tree = cKDTree(food[["x", "y"]].values)
        comp = op[competitor_mask(op, category)] if category else op.iloc[:0]
        comp = comp[~comp.index.isin(brand_idx)]
        self.comp = comp
        self.comp_tree = cKDTree(comp[["x", "y"]].values) if len(comp) else None

    def _sum_within(self, xy, r, w):
        return np.array([w[i].sum() for i in self.sec_tree.query_ball_point(xy, r)])

    def compute(self, xy: np.ndarray) -> pd.DataFrame:
        idx1k = self.sec_tree.query_ball_point(xy, 1000)
        pop1k = np.array([self.sec_pop[i].sum() for i in idx1k])
        young1k = np.array([self.sec_young[i].sum() for i in idx1k])
        inc1k = np.array([(self.sec_inc[i] * self.sec_pop[i]).sum() / max(self.sec_pop[i].sum(), 1) if len(i) else np.nan
                          for i in idx1k])
        f = pd.DataFrame({
            "pop_500": self._sum_within(xy, 500, self.sec_pop),
            "pop_1000": pop1k,
            "young_share_1000": np.where(pop1k > 0, young1k / np.maximum(pop1k, 1), np.nan),
            "income_1000": inc1k,
            "footfall": self.surf.potential(xy, self.cal["a_consumer"], self.cal["b_office"], self.cal["lambda_m"],
                                            self.cal["c_station"]),
            "office_500": self.surf.count("office", xy, 500),
            "transit_300": self.surf.stations_within(xy, 300),
            "food_300": [len(i) for i in self.food_tree.query_ball_point(xy, 300)],
            "comp_500": [len(i) for i in self.comp_tree.query_ball_point(xy, 500)] if self.comp_tree else 0,
        })
        return f.fillna(f.median(numeric_only=True))


def transform(f):
    out = f.copy()
    for c in ["pop_500", "pop_1000", "footfall", "office_500", "transit_300", "food_300", "comp_500"]:
        out[c] = np.log1p(out[c])
    out["income_1000"] = np.log(out["income_1000"].clip(lower=1))
    return out[FEATURES]


# ------------------------------------------------------------------ demand calibration
DECAY_GRID = [300, 500, 700, 1000, 1500, 2000]
ACTIVITY_GRID = [0, 2, 5, 10, 20, 40]


def calibrate_demand(fs, store_xy, comp_xy, P):
    """Pick the distance decay and the weight of daytime activity from the brand's own network.

    Demand per section = residents + k * open premises (premises = jobs/visitors proxy).
    Premise: every existing store is a viable store, so the right demand definition is the one
    under which existing stores capture the most SIMILAR demand (lowest coefficient of variation
    of each store's leave-one-out captured demand in the Huff model). 30-cell grid search."""
    rows = []
    for lam in DECAY_GRID:
        Pl = {**P, "huff_decay_m": lam}
        for k in ACTIVITY_GRID:
            h = Huff(fs.sec_xy, fs.sec_pop + k * fs.sec_activity, comp_xy, Pl)
            w = h.weights(store_xy)
            tot = w.sum(1)
            base = h.captured(tot)
            loo = np.array([base - h.captured(tot - w[:, j]) for j in range(w.shape[1])])
            rows.append((lam, k, loo.std() / loo.mean()))
    grid = pd.DataFrame(rows, columns=["decay_m", "activity_weight", "cv_store_demand"])
    return grid.loc[grid["cv_store_demand"].idxmin()], grid


# ------------------------------------------------------------------ Huff gravity model
class Huff:
    def __init__(self, sec_xy, sec_pop, comp_xy, P=PARAMS):
        self.xy, self.pop, self.P = sec_xy, sec_pop, P
        self.base = self.weights(comp_xy).sum(1) + np.exp(-P["outside_option_m"] / P["huff_decay_m"])

    def weights(self, xy):
        if len(xy) == 0:
            return np.zeros((len(self.xy), 0))
        d = np.sqrt(((self.xy[:, None, :] - np.asarray(xy)[None, :, :]) ** 2).sum(-1))
        w = np.exp(-d / self.P["huff_decay_m"])
        w[d > self.P["huff_cutoff_m"]] = 0
        return w

    def own_capture(self, brand_w_sum, w):
        """Demand a single store with pull w captures itself, given the brand's total pull brand_w_sum
        (which may or may not already include w)."""
        return (self.pop * w / (brand_w_sum + self.base)).sum()

    def captured(self, brand_w_sum):
        """Residents captured by the brand given the summed pull of its stores per section."""
        return (self.pop * brand_w_sum / (brand_w_sum + self.base)).sum()


# ------------------------------------------------------------------ main entry
@dataclass
class Result:
    brand: str
    matched_names: pd.Series
    category: str | None
    stores: pd.DataFrame
    competitors: pd.DataFrame
    sites: pd.DataFrame
    curve: pd.DataFrame
    metrics: dict
    coefficients: pd.DataFrame
    params: dict = field(default_factory=lambda: dict(PARAMS))
    warnings: list = field(default_factory=list)


def run(data: Data, query: str, n_sites=5, P=PARAMS, extra_stores: pd.DataFrame | None = None) -> Result:
    """extra_stores: brand stores found by other sources (OSM, Google Maps) but missing from the census —
    DataFrame with x, y, lon, lat, label, sources. They join the brand's network and the training positives."""
    rng = np.random.default_rng(P["seed"])
    prem = data.premises
    stores, matched = match_brand(prem, query)
    warns = []
    if extra_stores is not None and len(extra_stores):
        near = cKDTree(prem[["x", "y"]].values).query(extra_stores[["x", "y"]].values)[1]
        ref = prem.iloc[near]
        add = pd.DataFrame({"x": extra_stores["x"].values, "y": extra_stores["y"].values,
                            "lon": extra_stores["lon"].values, "lat": extra_stores["lat"].values,
                            "district_code": ref["district_code"].values, "barrio_code": ref["barrio_code"].values,
                            "trade_name": extra_stores["label"].values, "street_type": "", "street_name": extra_stores["label"].values,
                            "street_number": "", "epigrafes": "561002", "status": "Abierto",
                            "source": extra_stores["sources"].values},
                           index=pd.RangeIndex(10**9, 10**9 + len(extra_stores)))
        stores = pd.concat([stores.assign(source="census"), add])
    if len(stores) < 3:
        raise ValueError(f"Found {len(stores)} open '{query}' premises in the Madrid census — too few to learn from. "
                         "v0.1 needs ≥3 existing stores; brands new to Madrid need the category model (next release).")
    if len(stores) < 15:
        warns.append(f"Only {len(stores)} stores to learn from: suitability scores are indicative, not reliable.")
    special = stores["barrio_code"].isin(SPECIAL_VENUE_BARRIOS)
    if special.any():
        warns.append(f"{int(special.sum())} store(s) at special venues (airport) excluded from learning and from the demand model.")
    category = brand_category(stores)
    all_stores = stores
    stores = stores[~special]
    fs = FeatureSpace(data, all_stores.index, category)

    # ---- training set: brand stores vs background premises (presence-background design)
    bg_pool = prem[prem["access_type"].isin(["Puerta Calle", "Agrupado"]) & ~prem.index.isin(stores.index)]
    near = cKDTree(stores[["x", "y"]].values).query(bg_pool[["x", "y"]].values)[0]
    bg_pool = bg_pool[near > 50]
    bg = bg_pool.sample(min(len(bg_pool), max(3000, P["background_ratio"] * len(stores))), random_state=P["seed"])
    train = pd.concat([stores.assign(is_brand=1), bg.assign(is_brand=0)])
    X = transform(fs.compute(train[["x", "y"]].values))
    y = train["is_brand"].values
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))

    # ---- spatial cross-validation: hold out whole districts
    groups = train["district_code"].fillna(-1).values
    k = min(5, len(np.unique(groups[y == 1])))
    aucs = []
    if k >= 3:
        for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
            if y[te].sum() == 0 or y[te].sum() == len(te):
                continue
            model.fit(X.iloc[tr], y[tr])
            aucs.append(roc_auc_score(y[te], model.predict_proba(X.iloc[te])[:, 1]))
    model.fit(X, y)
    coefs = pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in FEATURES],
                          "effect": model[-1].coef_[0]}).sort_values("effect", key=abs, ascending=False)
    auc = float(np.mean(aucs)) if aucs else None
    reliability = ("unreliable" if auc is None or auc < 0.60 else "weak" if auc < 0.70 else "usable" if auc < 0.80 else "good")
    if reliability == "unreliable":
        warns.append(f"UNRELIABLE — accuracy {auc if auc is None else round(auc, 2)}: the model cannot tell where this brand "
                     "chooses to open (barely better than chance). Sites below are exploratory only, not recommendations.")
    elif reliability == "weak":
        warns.append(f"Weak model — accuracy {auc:.2f}: treat the ranking as a shortlist to investigate, not an answer.")

    store_feat = fs.compute(stores[["x", "y"]].values)
    store_score = model.predict_proba(transform(store_feat))[:, 1]
    gate = float(np.quantile(store_score, P["gate_quantile"]))

    # ---- candidates: vacant street-level units ('Cerrado' = registered but not trading)
    cand = prem[(prem["status"] == "Cerrado") & prem["access_type"].isin(["Puerta Calle", "Agrupado"])
                & (prem["geom_source"] == "local") & ~prem["barrio_code"].isin(SPECIAL_VENUE_BARRIOS)].copy()
    ind = data.surface().industrial_share(cand[["x", "y"]].values, 200)
    cand = cand[ind <= INDUSTRIAL_SHARE_MAX].copy()
    cfeat = fs.compute(cand[["x", "y"]].values)
    cand["suitability"] = model.predict_proba(transform(cfeat))[:, 1]
    cand = pd.concat([cand, cfeat.set_index(cand.index)], axis=1)
    n_gate = int((cand["suitability"] >= gate).sum())
    pool = cand[cand["suitability"] >= gate].nlargest(P["candidate_pool"], "suitability")
    if pool.empty:
        raise ValueError("No vacant unit passes the suitability gate.")

    # ---- Huff demand + greedy selection
    comp_xy = fs.comp[["x", "y"]].values if len(fs.comp) else np.empty((0, 2))
    best, grid = calibrate_demand(fs, stores[["x", "y"]].values, comp_xy, P)
    P = {**P, "huff_decay_m": int(best.decay_m), "activity_weight": float(best.activity_weight)}
    demand = fs.sec_pop + P["activity_weight"] * fs.sec_activity
    huff = Huff(fs.sec_xy, demand, comp_xy, P)
    brand_w = huff.weights(stores[["x", "y"]].values)
    cur = brand_w.sum(1)
    base_captured = huff.captured(cur)
    # what a typical existing store adds (leave-one-out)
    loo = np.array([base_captured - huff.captured(cur - brand_w[:, j]) for j in range(brand_w.shape[1])])
    median_store = float(np.median(loo))

    cw = huff.weights(pool[["x", "y"]].values)                    # sections x candidates
    chosen, rows = [], []
    alive = np.ones(len(pool), bool)
    pxy = pool[["x", "y"]].values
    for rank in range(1, P["max_sites"] + 1):
        gains = np.array([huff.captured(cur + cw[:, j]) for j in range(cw.shape[1])]) - huff.captured(cur)
        gains[~alive] = -np.inf
        j = int(np.argmax(gains))
        if not np.isfinite(gains[j]) or gains[j] <= 0:
            break
        cur = cur + cw[:, j]
        chosen.append(j)
        alive &= np.hypot(*(pxy - pxy[j]).T) >= P["min_spacing_m"]
        r = pool.iloc[j]
        dist_brand = cKDTree(stores[["x", "y"]].values).query([[r.x, r.y]])[0][0]
        rows.append({**r.to_dict(), "rank": rank, "incremental_demand": gains[j],
                     "vs_median_store": gains[j] / median_store if median_store else np.nan,
                     "dist_nearest_brand_m": dist_brand})
    curve = pd.DataFrame(rows)[["rank", "incremental_demand", "vs_median_store"]]
    supported = int((curve["vs_median_store"] >= P["saturation_ratio"]).sum())

    sites = pd.DataFrame(rows).head(n_sites)
    names = {int(k): v for k, v in data.meta["districts"].items()}
    bnames = dict(zip(data.barrios.barrio_code, data.barrios.name))
    for df in (sites, stores):
        df["district"] = df["district_code"].map(names)
        df["barrio"] = df["barrio_code"].map(bnames)
        df["address"] = (df["street_type"].fillna("").str.title() + " " + df["street_name"].fillna("").str.title()
                         + " " + df["street_number"].fillna("")).str.strip()

    metrics = dict(
        stores=len(stores), category=category, competitors=len(fs.comp), reliability=reliability,
        auc_spatial_cv=auc, auc_folds=[round(a, 3) for a in aucs],
        background_n=len(bg), candidates_vacant=len(cand), candidates_pass_gate=n_gate,
        gate_score=gate, median_existing_store_demand=median_store,
        brand_captured_demand_now=base_captured, additional_stores_supported=supported,
        demand_decay_m=P["huff_decay_m"], demand_activity_weight=P["activity_weight"], demand_calibration_cv=float(best.cv_store_demand),
    )
    res = Result(query, matched, category, stores, fs.comp, sites, curve, metrics, coefs, dict(P), warns)
    res.calibration = grid
    # context kept for the Monte Carlo layer (robustness.py) — avoids recomputing features
    res.ctx = dict(fs=fs, X=X, y=y, groups=groups, cand=cand, cand_X=transform(cand[FEATURES]), stores_xy=stores[["x", "y"]].values,
                   store_X=transform(store_feat), comp_xy=comp_xy, pool_index=pool.index, median_store=median_store,
                   model=model, huff=huff, brand_w_sum=brand_w.sum(1), base_captured=base_captured, gate=gate,
                   store_capture=np.array([huff.own_capture(brand_w.sum(1), brand_w[:, j]) for j in range(brand_w.shape[1])]))
    return res
