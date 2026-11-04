"""Monte Carlo robustness: how much do the recommendations depend on the data we happened to have
and on the parameters we chose?

Each of B simulation draws:
  1. bootstrap-resamples the training set (brand stores AND background, with replacement) and refits the
     suitability model  -> sampling uncertainty in "where does this brand choose to be"
  2. draws the demand parameters (distance decay, daytime-activity weight) from the best-fitting cells of the
     calibration grid, weighted by fit  -> parameter uncertainty in the demand model
  3. re-runs the suitability gate + Huff greedy selection

Outputs per recommended site:
  p_selected      share of draws in which the optimiser picks a unit within `radius_m` of the site
  demand P10/P50/P90  the site's incremental demand vs a median existing store, across draws
  p_beats_median  share of draws in which the site out-captures a median existing store
The per-draw demand ratios feed the P&L Monte Carlo (pnl.py), so the two simulations are coupled.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .site_model import Huff


def _param_draws(grid, n_cells=5):
    top = grid.nsmallest(n_cells, "cv_store_demand").copy()
    w = np.exp(-(top["cv_store_demand"] - top["cv_store_demand"].min()) / 0.02)  # closer fit -> more weight
    top["w"] = w / w.sum()
    return top


def simulate(res, B=300, radius_m=400, seed=11, progress=None):
    c, P = res.ctx, res.params
    rng = np.random.default_rng(seed)
    X, y = c["X"].values, c["y"]
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    cand, cand_X, fs = c["cand"], c["cand_X"].values, c["fs"]
    cxy = cand[["x", "y"]].values
    sites = res.sites
    sxy = sites[["x", "y"]].values
    n = len(sites)
    cells = _param_draws(res.calibration)

    huffs = {}  # one Huff model per parameter cell (competitor pull is the expensive part)

    def huff_for(lam, k):
        key = (int(lam), float(k))
        if key not in huffs:
            Pk = {**P, "huff_decay_m": int(lam)}
            h = Huff(fs.sec_xy, fs.sec_pop + k * fs.sec_activity, c["comp_xy"], Pk)
            bw = h.weights(c["stores_xy"])
            cur = bw.sum(1)
            base = h.captured(cur)
            loo = np.array([base - h.captured(cur - bw[:, j]) for j in range(bw.shape[1])])
            huffs[key] = (h, cur, base, float(np.median(loo)), h.weights(sxy))
        return huffs[key]

    selected = np.zeros((B, n), bool)
    ratio = np.zeros((B, n))
    picks = []
    for b in range(B):
        bi = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        m.fit(X[bi], y[bi])
        gate = np.quantile(m.predict_proba(c["store_X"].values)[:, 1], P["gate_quantile"])
        suit = m.predict_proba(cand_X)[:, 1]
        ok = np.where(suit >= gate)[0]
        pool = ok[np.argsort(-suit[ok])[:400]]

        cell = cells.sample(1, weights="w", random_state=int(rng.integers(1 << 31))).iloc[0]
        h, cur0, base, med, site_w = huff_for(cell.decay_m, cell.activity_weight)
        # each recommended site evaluated alone against the existing network
        ratio[b] = [(h.captured(cur0 + site_w[:, i]) - base) / med for i in range(n)]

        cw = h.weights(cxy[pool])
        cur, alive, chosen = cur0.copy(), np.ones(len(pool), bool), []
        for _ in range(n):
            g = np.array([h.captured(cur + cw[:, j]) for j in range(len(pool))]) - h.captured(cur)
            g[~alive] = -np.inf
            j = int(np.argmax(g))
            if not np.isfinite(g[j]):
                break
            cur = cur + cw[:, j]
            chosen.append(pool[j])
            alive &= np.hypot(*(cxy[pool] - cxy[pool[j]]).T) >= P["min_spacing_m"]
        if chosen:
            d = np.hypot(sxy[:, None, 0] - cxy[chosen][None, :, 0], sxy[:, None, 1] - cxy[chosen][None, :, 1])
            selected[b] = (d <= radius_m).any(1)
            picks.extend(cand.index[chosen].tolist())
        if progress and b % 10 == 0:
            progress(b / B)

    summary = pd.DataFrame({
        "rank": sites["rank"].values,
        "address": sites["address"].values,
        "p_selected": selected.mean(0),
        "demand_p10": np.percentile(ratio, 10, axis=0),
        "demand_p50": np.percentile(ratio, 50, axis=0),
        "demand_p90": np.percentile(ratio, 90, axis=0),
        "p_beats_median": (ratio >= 1).mean(0),
    })
    freq = pd.Series(picks).value_counts().rename("times_picked").to_frame()
    alt = cand.loc[freq.index.intersection(cand.index), ["lon", "lat", "street_name", "street_number"]].join(freq)
    return dict(B=B, radius_m=radius_m, summary=summary, ratio_draws=ratio, param_cells=cells,
                alternatives=alt.sort_values("times_picked", ascending=False).head(25))
