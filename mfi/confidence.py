"""Explainable confidence score per recommended unit (0-100) + deterministic plain-English verdict.

Four visible components, each 0-1, weighted:
  model       35%  How well the location model explains where this brand opens (spatial-CV AUC: 0.5 -> 0, 0.85+ -> 1)
  inputs      25%  Share of the P&L inputs that are verified (official) rather than third-party estimates / assumptions
  stability   25%  Does the unit's demand hold when the demand model's parameters change? (5 best-fitting settings)
  listing     15%  Listing quality: exact address, rent consistent with comparable units within 1 km

Nothing here is a model output in itself; it summarises how much evidence stands behind the numbers.
"""
import numpy as np
import pandas as pd

from .model.site_model import Huff

WEIGHTS = {"model": 0.35, "inputs": 0.25, "stability": 0.25, "listing": 0.15}
LABELS = {"model": "Location model accuracy", "inputs": "Verified brand economics",
          "stability": "Demand stability", "listing": "Listing quality"}


def _model_part(auc):
    return float(np.clip(((auc or 0.5) - 0.5) / 0.35, 0, 1))


def _inputs_part(profile, econ, overrides):
    if not econ:
        return 0.0, "No brand economics — ranked by demand per € of rent."
    src = (((profile or {}).get("data") or {}).get("pnl_inputs")) or {}
    scores = []
    for k in econ:
        if k in (overrides or {}):
            scores.append(0.5)
            continue
        lab = str((src.get(k) or {}).get("label") or "").lower()
        scores.append(1.0 if lab.startswith("verified") else 0.5 if "verified" in lab else 0.25)
    v = float(np.mean(scores)) if scores else 0.0
    n_ver = sum(s == 1.0 for s in scores)
    return v, f"{n_ver} of {len(scores)} inputs verified from official sources; the rest are third-party estimates or your settings."


def stability(R, sites, n_cells=5):
    """CV of each unit's own captured demand (vs a median store) across the best-fitting demand settings."""
    c, P, fs = R.ctx, R.params, R.ctx["fs"]
    cells = R.calibration.nsmallest(n_cells, "cv_store_demand")
    xy = sites[["x", "y"]].values
    ratios = []
    for _, cell in cells.iterrows():
        h = Huff(fs.sec_xy, fs.sec_pop + cell.activity_weight * fs.sec_activity, c["comp_xy"],
                 {**P, "huff_decay_m": int(cell.decay_m)})
        bw = h.weights(c["stores_xy"])
        cur = bw.sum(1)
        med = np.median([h.own_capture(cur, bw[:, j]) for j in range(bw.shape[1])])
        w = h.weights(xy)
        ratios.append([h.own_capture(cur + w[:, j], w[:, j]) / med for j in range(w.shape[1])])
    r = np.array(ratios)
    cv = r.std(0) / np.maximum(r.mean(0), 1e-9)
    return cv, r


def _listing_part(row):
    addr = str(row.get("address") or "")
    precise = 1.0 if any(ch.isdigit() for ch in addr) else 0.6
    rel = row.get("eur_m2", np.nan) / row.get("local_eur_m2", np.nan) if row.get("local_eur_m2") else np.nan
    price = 1.0 if (np.isnan(rel) or 0.6 <= rel <= 1.6) else 0.6
    note = ("exact street address" if precise == 1 else "neighbourhood only, no street number") + \
           (f"; rent {rel:.0%} of comparable units nearby" if not np.isnan(rel) else "; too few comparable listings nearby")
    return (precise + price) / 2, note


def score_units(R, sites, profile, econ, overrides):
    cv, _ = stability(R, sites)
    m = _model_part(R.metrics.get("auc_spatial_cv"))
    inp, inp_note = _inputs_part(profile, econ, overrides)
    rows = []
    for i, (_, s) in enumerate(sites.iterrows()):
        st = float(1 - min(cv[i] / 0.5, 1))
        li, li_note = _listing_part(s)
        parts = {"model": m, "inputs": inp, "stability": st, "listing": li}
        notes = {"model": f"Spatial-CV AUC {R.metrics.get('auc_spatial_cv', 0):.2f} ({R.metrics.get('reliability')}).",
                 "inputs": inp_note,
                 "stability": f"Demand varies ±{cv[i]:.0%} across the 5 best-fitting demand settings.",
                 "listing": li_note.capitalize() + "."}
        total = round(100 * sum(WEIGHTS[k] * v for k, v in parts.items()))
        rows.append({"rank": int(s["rank"]), "confidence": total, "parts": parts, "notes": notes})
    return rows


def grade(score):
    return "High" if score >= 70 else "Medium" if score >= 50 else "Low"


def eur(v):
    return "—" if v is None or not np.isfinite(v) else (f"€{v/1e6:.2f}M" if abs(v) >= 1e6 else f"€{v/1e3:,.0f}k" if abs(v) >= 1e4 else f"€{v:,.0f}")


def verdict(brand, O, conf):
    """Plain-English verdict, built only from computed numbers and cited research."""
    R, K = O["R"], O["ranked"]
    prof = (O.get("profile") or {}).get("data") or {}
    lines = []
    elig = prof.get("individual_franchise_available_spain")
    if elig == "yes":
        lines.append(f"**{brand} sells individual franchises in Spain** (cited research).")
    elif elig == "no":
        lines.append(f"**{brand} does not sell individual franchises in Spain** — the units below show where its demand is, "
                     "not something you can open.")
    elif elig == "unclear":
        lines.append(f"**Whether {brand} franchises to individuals in Spain is unclear** — verify with the brand before anything else.")
    else:
        lines.append(f"**{brand}'s franchise terms haven't been researched yet.**")
    if O["rent_model"] == "franchisor_property":
        lines.append("The franchisor chooses and provides the premises, so these are demand evidence rather than leasable choices.")
    if K.get("ok") and len(K["sites"]):
        s = K["sites"].iloc[0]
        c = conf[0]["confidence"] if conf else None
        if O["econ"]:
            lines.append(f"Best unit: **{s['address']}** — {s['m2']:.0f} m² at {eur(s['rent_month'])}/month. "
                         f"**{s['p_payback']:.0%} chance of paying back within {O.get('payback', 4):.0f} years**; "
                         f"EBITDA {eur(s['ebitda_p10'])}–{eur(s['ebitda_p90'])} a year (middle case {eur(s['ebitda_p50'])}).")
        else:
            lines.append(f"Best unit by customers per euro of rent: **{s['address']}** — {s['m2']:.0f} m² at "
                         f"{eur(s['rent_month'])}/month, demand {s['demand_ratio']:.2f}× a typical {brand} store.")
        if c is not None:
            weakest = min(conf[0]["parts"], key=conf[0]["parts"].get)
            lines.append(f"Confidence **{c}/100 ({grade(c)})** — weakest link: {LABELS[weakest].lower()}.")
    return lines
