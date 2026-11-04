"""Seize — Site Selection Engine. Dashboard (3 pages: Analysis · Investment report · Methodology).

    streamlit run app.py

Reads only data/app/* (frozen by `python -m mfi.model.export`) — shareable without Postgres.
Optional secrets (.streamlit/secrets.toml): ANTHROPIC_API_KEY (Claude research + memo), APIFY_TOKEN (rent scrape).
"""
import json

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from mfi import pipeline, quota, reasoning, report, scrape
from mfi.model import pnl
from mfi.model.export import APP
from mfi.model.robustness import simulate as robustness
from mfi.model.site_model import FEATURE_LABELS, FEATURES, PARAMS, load_data, run
from mfi.stores import inventory, load_osm, missing_from_census

st.set_page_config(page_title="Seize · Site Selection Engine", page_icon="🔴", layout="wide")
RED, BLACK, GREY = [220, 38, 38], [17, 17, 17], [150, 150, 150]
QUICK = ["McDonald's", "Burger King", "Starbucks", "Five Guys", "Goiko", "KFC", "Taco Bell", "Oakberry"]
SS = st.session_state

st.markdown("""
<style>
.block-container {padding-top: 3rem; max-width: 1500px;}
.hero h1 {font-size: 2.2rem; letter-spacing: .22em; margin: 0; font-weight: 800; display:flex; align-items:center; gap:18px;}
.hero h1 .tag {font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size:.72rem; letter-spacing:.38em; color:#dc2626;
  font-weight:500; border:1px solid #dc2626; padding:4px 10px 4px 13px;}
.hero p {color: #6b7280; margin: .2rem 0 1rem 0;}
.legend span {display:inline-flex; align-items:center; margin-right:18px; font-size:.9rem; color:#374151;}
.dot {width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:6px;}
div[data-testid="stMetricValue"] {font-size: 1.7rem;}
.small {color:#6b7280; font-size:.82rem;}
.step {border-left: 3px solid #dc2626; padding: .35rem .8rem; margin: .35rem 0; background: #fafafa;}
.step b {color:#111;}
</style>""", unsafe_allow_html=True)


# ================================================================== PASSCODE GATE
def _passcode():
    import os
    v = os.environ.get("APP_PASSCODE")
    if not v:
        try:
            v = st.secrets.get("APP_PASSCODE")
        except Exception:
            v = None
    return v or None


def gate():
    """Server-side passcode check (st.secrets APP_PASSCODE). The code never reaches the browser."""
    import hmac
    import time
    if SS.get("authed"):
        return True
    st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], header[data-testid="stHeader"], footer {display:none !important;}
.stApp {background:#ffffff;
  background-image: linear-gradient(rgba(220,38,38,.06) 1px, transparent 1px), linear-gradient(90deg, rgba(220,38,38,.06) 1px, transparent 1px);
  background-size: 44px 44px;}
.block-container {max-width: 620px !important; padding-top: 15vh !important;}
.gate-title, .gate-h, .gate-sub, .gate-foot {white-space: nowrap;}
.gate-mark {width:54px; height:54px; border:2px solid #dc2626; transform: rotate(45deg); margin: 0 auto 34px auto; position:relative;}
.gate-mark:after {content:""; position:absolute; inset:12px; background:#dc2626; animation: pulse 2.4s ease-in-out infinite;}
@keyframes pulse {0%,100%{opacity:1} 50%{opacity:.25}}
.gate-title {font-family: ui-monospace, "SF Mono", Menlo, monospace; text-align:center; letter-spacing:.45em; font-size:.74rem;
  color:#dc2626; margin:0 0 6px 0; padding-left:.55em;}
.gate-h {text-align:center; font-weight:800; letter-spacing:.12em; font-size:1.75rem; line-height:1.25; color:#111; margin:0;}
.gate-sub {text-align:center; color:#9ca3af; font-family: ui-monospace, Menlo, monospace; font-size:.72rem; letter-spacing:.3em;
  margin: 10px 0 38px 0;}
.gate-line {height:1px; background:linear-gradient(90deg, transparent, #dc2626, transparent); margin: 0 0 30px 0; position:relative; overflow:hidden;}
.gate-line:after {content:""; position:absolute; top:0; left:-30%; width:30%; height:1px; background:#fff; animation: scan 2.8s linear infinite;}
@keyframes scan {to {left:130%}}
div[data-testid="stTextInput"] input {text-align:center; font-family: ui-monospace, Menlo, monospace; letter-spacing:.5em; font-size:1.1rem;
  border:none !important; border-bottom:2px solid #dc2626 !important; border-radius:0 !important; background:transparent !important; box-shadow:none !important;}
div[data-testid="stTextInput"] > div, div[data-testid="stTextInput"] div[data-baseweb="input"],
div[data-testid="stTextInput"] div[data-baseweb="base-input"] {border:none !important; background:transparent !important; box-shadow:none !important;}
div[data-testid="stTextInput"] {max-width: 380px; margin: 0 auto;}
div[data-testid="stTextInputRootElement"], div[data-testid="stTextInputRootElement"] * {background:transparent !important; border-color:transparent !important;}
div[data-testid="stTextInputRootElement"] {border-bottom:2px solid #dc2626 !important; border-radius:0 !important;}
p.gate-h {font-size:3.4rem !important; letter-spacing:.28em !important; padding-left:.28em; line-height:1.2 !important; margin-bottom:4px !important;}
p.gate-title {font-size:.74rem !important;} p.gate-sub, p.gate-foot {font-size:.7rem !important;}
div[data-testid="stFormSubmitButton"] {max-width: 380px; margin: 8px auto 0 auto;}
div[data-testid="stFormSubmitButton"] button, .stButton button {width:100%; border-radius:0; background:#dc2626; color:#fff; border:none; letter-spacing:.4em; font-family: ui-monospace, Menlo, monospace;
  font-size:.8rem; padding:.75rem 0; transition: all .2s;}
div[data-testid="stFormSubmitButton"] button:hover, .stButton button:hover {background:#111 !important; color:#fff !important;}
div[data-testid="stFormSubmitButton"] button p {letter-spacing:.4em; font-size:.8rem; color:#fff;}
.gate-foot {text-align:center; color:#d1d5db; font-family: ui-monospace, Menlo, monospace; font-size:.65rem; letter-spacing:.3em; margin-top:46px;}
</style>
<div class="gate-mark"></div>
<p class="gate-title">RESTRICTED ACCESS</p>
<p class="gate-h">SEIZE</p>
<p class="gate-sub">SITE SELECTION ENGINE</p>
<div class="gate-line"></div>
""", unsafe_allow_html=True)
    code = _passcode()
    if not code:
        st.error("No APP_PASSCODE configured on the server.")
        st.stop()
    wait = SS.get("lock_until", 0) - time.time()
    with st.form("gate", border=False):
        entered = st.text_input("Passcode", type="password", label_visibility="collapsed", placeholder="PASSCODE")
        ok = st.form_submit_button("ENTER", disabled=wait > 0, width="stretch")
    if wait > 0:
        st.markdown(f'<p class="gate-sub" style="color:#dc2626">LOCKED · RETRY IN {int(wait)+1}s</p>', unsafe_allow_html=True)
    if ok and entered:
        if hmac.compare_digest(entered.encode(), str(code).encode()):
            SS["authed"] = True
            SS.pop("fails", None)
            st.rerun()
        SS["fails"] = SS.get("fails", 0) + 1
        if SS["fails"] >= 5:
            SS["lock_until"], SS["fails"] = time.time() + 60, 0
        st.markdown('<p class="gate-sub" style="color:#dc2626">ACCESS DENIED</p>', unsafe_allow_html=True)
    st.markdown('<p class="gate-foot">© SEIZE · MADRID</p>', unsafe_allow_html=True)
    st.stop()


gate()


def hero(sub):
    st.markdown(f'<div class="hero"><h1>SEIZE<span class="tag">SITE SELECTION ENGINE</span></h1><p>{sub}</p></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ cached compute
@st.cache_resource(show_spinner="Loading Madrid data…")
def data():
    return load_data()


@st.cache_data(show_spinner=False)
def districts_geojson():
    return json.loads((APP / "districts.geojson").read_text())


@st.cache_data(show_spinner=False)
def registry():
    return pd.read_parquet(APP / "brands.parquet")


@st.cache_data(show_spinner=False)
def listings():
    return pd.read_parquet(APP / "listings.parquet")


@st.cache_resource(show_spinner=False)
def osm_points():
    return load_osm()[0]


@st.cache_resource(show_spinner=False)
def run_pipeline(brand, n, m2_range, payback, profile_json):
    profile = json.loads(profile_json) if profile_json else None
    return pipeline.analyse(data(), brand, n, listings(), profile, m2_range=m2_range, payback_target=payback, osm=osm_points())


def get_profile(brand, allow_research):
    prof = pipeline.cached_research(brand)
    if prof is None and allow_research and reasoning.available():
        if quota.allow("claude", SS):
            with st.spinner(f"Step 1/3 · Claude is researching {brand}'s franchise economics on the web (~1 min)…"):
                prof = reasoning.research_franchise(brand)
            if prof and prof.get("data"):
                pipeline.save_research(brand, prof)
        else:
            st.warning("Daily research limit reached — ranking without brand economics.")
    return prof


def eur(v):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"€{v:,.0f}"


# ================================================================== PAGE 1 — ANALYSIS
def page_analysis():
    hero("Pick a franchise. The engine researches its economics, screens every commercial unit for rent in Madrid, "
         "simulates each unit's P&L and ranks them by risk-adjusted return.")
    reg = registry()
    c0, c1, c2, c3 = st.columns([1.4, 3.2, 1.1, 1])
    sectors = ["All sectors"] + reg.groupby("sector")["n_stores"].sum().sort_values(ascending=False).index.tolist()
    sector = c0.selectbox("Sector", sectors, label_visibility="collapsed")
    opts = reg[reg["franchisable"] & (reg["n_stores"] >= 3)]
    if sector != "All sectors":
        opts = opts[opts["sector"] == sector]
    names = opts["brand"].tolist()
    label = dict(zip(opts["brand"], opts["brand"] + "  ·  " + opts["sector"] + "  ·  " + opts["n_stores"].astype(str) + " stores"))
    cur = SS.get("brand", "Tierra Burrito Bar")
    brand = c1.selectbox("Franchise", names, index=names.index(cur) if cur in names else 0,
                         format_func=lambda b: label.get(b, b), label_visibility="collapsed")
    n_sites = c2.selectbox("Sites", [3, 5, 10], index=[3, 5, 10].index(SS.get("n_sites", 5)),
                           label_visibility="collapsed", format_func=lambda n: f"Top {n} sites")
    go = c3.button("Analyse  →", type="primary", width="stretch")

    prof_cached = pipeline.cached_research(brand)
    _, _, _, m2_suggest = pipeline.econ_from_profile(prof_cached)
    with st.expander("Unit & return settings", expanded=False):
        a, b, c = st.columns(3)
        m2_default = tuple(SS.get("m2_range") or m2_suggest or pipeline.DEFAULT_M2)
        m2_range = a.slider("Unit size (m²)", 20, 1500, m2_default, step=10,
                            help="Prefilled from the franchise's published minimum floor area when Claude has researched it.")
        payback = b.selectbox("Payback target", [3.0, 4.0, 5.0, 7.0], index=1, format_func=lambda v: f"{v:.0f} years")
        research_ok = c.toggle("Research the franchise with Claude", value=True,
                               help="Needed to model money (revenue, costs, investment). Cached per brand after the first run.")

    if go or "out" not in SS or SS.get("brand") != brand:
        if go or "out" not in SS:
            prof = get_profile(brand, research_ok)
            with st.spinner("Step 2/3 · Scoring every listing: location model, demand & competition, cannibalisation…"):
                with st.spinner("Step 3/3 · Monte Carlo P&L for each candidate unit…"):
                    try:
                        out = run_pipeline(brand, n_sites, tuple(m2_range), payback,
                                           json.dumps(prof, default=str) if prof else "")
                        SS.update(out=out, brand=brand, n_sites=n_sites, m2_range=tuple(m2_range), payback=payback)
                        SS.pop("error", None)
                        SS.pop("memo", None)
                    except ValueError as e:
                        SS["error"], SS["brand"] = str(e), brand
                        SS.pop("out", None)
    if "error" in SS:
        st.error(SS["error"])
        st.stop()
    if "out" not in SS:
        st.info("Choose a brand and press **Analyse**.")
        st.stop()
    O = SS["out"]
    R, K, m = O["R"], O["ranked"], O["R"].metrics
    prof = (O.get("profile") or {}).get("data") or {}

    # ---- banners: eligibility, rent model, reliability, missing economics
    if prof:
        elig = prof.get("individual_franchise_available_spain")
        if elig and elig != "yes":
            st.error(f"**Franchise availability in Spain: {elig}.** {prof.get('eligibility_explanation', '')}")
        if O["rent_model"] == "franchisor_property":
            st.warning("**The franchisor selects and provides the premises** — these units show where demand is strongest; "
                       "rent is modelled as occupancy % of sales. " + (prof.get("rent_model_explanation") or ""))
    for w in R.warnings:
        (st.error if w.startswith("UNRELIABLE") else st.warning)(w)
    if not O["econ"]:
        miss = ", ".join(m.replace("_eur", "").replace("_pct", " %").replace("_", " ") for m in O["missing"])
        why = (f"Claude's research found no published figure for: **{miss}**" if O.get("profile")
               else "The franchise hasn't been researched yet")
        st.info(f"{why}. Without brand economics the units are ranked by *customers per euro of rent* instead of payback.")
    if not K.get("ok"):
        st.error(K.get("reason"))
        st.stop()
    S = K["sites"]

    k = st.columns(5)
    k[0].metric("Existing stores", m["stores"], help="Census + OpenStreetMap, merged and de-duplicated")
    k[1].metric("Units for rent screened", f"{K['log']['listings']:,}")
    k[2].metric("Pass all filters", f"{K['log']['pass_suitability_gate']:,}",
                help=" → ".join(f"{a.replace('_', ' ')}: {b:,}" for a, b in K["log"].items()))
    k[3].metric("Location model", f"{m['auc_spatial_cv']:.2f} AUC · {m['reliability']}",
                help="Spatial cross-validation (whole districts held out). 0.5 = chance, 1.0 = perfect.")
    if O["econ"]:
        k[4].metric(f"#1 · P(payback ≤ {SS.get('payback', 4.0):.0f} yrs)", f"{S['p_payback'].iloc[0]:.0%}")
    else:
        k[4].metric("Ranked by", "Demand / €", help="Captured customers per €1k of annual rent")

    show_cand = st.toggle("Show all units that passed the filters", value=False)
    show_comp = st.toggle("Show competitors", value=False)
    legend = ['<span><i class="dot" style="background:#dc2626"></i>Recommended unit (for rent)</span>',
              f'<span><i class="dot" style="background:#111"></i>Existing {R.brand}</span>']
    if show_cand:
        legend.append('<span><i class="dot" style="background:#fca5a5"></i>Other qualifying units</span>')
    if show_comp:
        legend.append('<span><i class="dot" style="background:#9ca3af"></i>Competitor</span>')
    st.markdown('<div class="legend">' + "".join(legend) + "</div>", unsafe_allow_html=True)

    sites = S.copy()
    sites["label"] = sites["rank"].astype(int).astype(str)
    fin = (" · EBITDA P50 " + sites["ebitda_p50"].map(eur) + " · P(payback) " + (sites["p_payback"] * 100).round(0).astype(int).astype(str) + "%") \
        if O["econ"] else ""
    sites["tip"] = ("<b>#" + sites["label"] + " " + sites["address"].fillna("") + "</b><br/>" + sites["m2"].round(0).astype(int).astype(str)
                    + " m² · " + sites["rent_month"].map(eur) + "/month" + np.where(sites["traspaso_eur"] > 0, " · traspaso " + sites["traspaso_eur"].map(eur), "")
                    + "<br/>Demand " + sites["demand_ratio"].round(2).astype(str) + "× a median store" + fin)
    stores = R.stores.copy()
    stores["tip"] = "<b>" + stores["trade_name"].fillna(R.brand).astype(str) + "</b>"
    layers = [pdk.Layer("GeoJsonLayer", districts_geojson(), stroked=True, filled=False,
                        get_line_color=[90, 90, 90, 110], line_width_min_pixels=1)]
    if show_comp:
        comp = R.competitors.copy()
        comp["tip"] = "<b>" + comp["trade_name"].fillna("Competitor").astype(str) + "</b>"
        layers.append(pdk.Layer("ScatterplotLayer", comp, get_position=["lon", "lat"], get_fill_color=GREY + [140],
                                get_radius=35, radius_min_pixels=2, pickable=True))
    if show_cand:
        pool = score_pool(R, K)
        layers.append(pdk.Layer("ScatterplotLayer", pool, get_position=["lon", "lat"], get_fill_color=[252, 165, 165, 170],
                                get_radius=45, radius_min_pixels=3, pickable=True))
    layers += [
        pdk.Layer("ScatterplotLayer", stores, get_position=["lon", "lat"], get_fill_color=BLACK + [230],
                  get_radius=70, radius_min_pixels=4, pickable=True),
        pdk.Layer("ScatterplotLayer", sites, get_position=["lon", "lat"], get_fill_color=RED + [235],
                  get_line_color=[255, 255, 255], line_width_min_pixels=2, stroked=True,
                  get_radius=160, radius_min_pixels=11, pickable=True),
        pdk.Layer("TextLayer", sites, get_position=["lon", "lat"], get_text="label", get_color=[255, 255, 255],
                  get_size=13, get_alignment_baseline="'center'", font_weight=800),
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, map_provider="carto", map_style="light",
                             initial_view_state=pdk.ViewState(latitude=40.43, longitude=-3.69, zoom=11.2),
                             tooltip={"html": "{tip}", "style": {"fontSize": "12px"}}), height=600)

    st.subheader("Recommended units — ranked by " + K["metric"])
    t = pd.DataFrame({"#": S["rank"].astype(int), "Unit": S["address"], "m²": S["m2"].round(0),
                      "Rent / month": S["rent_month"], "€/m²": S["eur_m2"].round(1), "Traspaso": S["traspaso_eur"],
                      "Demand vs median store": S["demand_ratio"], "Location fit": S["suitability"],
                      f"Nearest {R.brand} (m)": S["dist_nearest_brand_m"].round(0)})
    cfg = {"Rent / month": st.column_config.NumberColumn(format="€%d"), "Traspaso": st.column_config.NumberColumn(format="€%d"),
           "Demand vs median store": st.column_config.NumberColumn(format="%.2f×"),
           "Location fit": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)}
    if O["econ"]:
        t["Revenue P50"] = S["revenue_p50"]
        t["EBITDA P10 · P50 · P90"] = S["ebitda_p10"].map(eur) + " · " + S["ebitda_p50"].map(eur) + " · " + S["ebitda_p90"].map(eur)
        t["P(loss)"] = S["p_loss"]
        t["Payback P50 (yrs)"] = S["payback_p50"].round(1)
        t["P(payback ≤ target)"] = S["p_payback"]
        cfg.update({"Revenue P50": st.column_config.NumberColumn(format="€%d"),
                    "P(loss)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                    "P(payback ≤ target)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    t["Listing"] = S["url"]
    cfg["Listing"] = st.column_config.LinkColumn(display_text="Idealista ↗")
    st.dataframe(t, hide_index=True, width="stretch", column_config=cfg)
    st.caption(f"Live Idealista listings (via Apify). Revenue = brand average × demand ratio (capped at the brand's 90th-percentile "
               f"store, {K['demand_cap']:.2f}×) × model-error noise σ={K['model_error_sigma']:.0%}. Each next pick is re-scored "
               "after adding the previous ones to the network, so the list accounts for cannibalisation.")

    t1, t2, t3 = st.tabs(["Filter funnel", "Location model", "Demand calibration"])
    with t1:
        f = pd.DataFrame({"step": [s.replace("_", " ") for s in K["log"]], "units": list(K["log"].values())})
        st.altair_chart(alt.Chart(f).mark_bar(color="#dc2626").encode(
            x=alt.X("units:Q"), y=alt.Y("step:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=["step", "units"]).properties(height=36 * len(f)), width="stretch")
        st.caption(f"Size {O['m2_range'][0]}–{O['m2_range'][1]} m² · ground/mezzanine · not industrial · rent consistent with "
                   "comparable units within 1 km · ≥150 m from the brand's own stores · location fit ≥ the brand's bottom-quartile store.")
    with t2:
        a, b = st.columns([1.2, 1])
        coef = R.coefficients.copy()
        coef["direction"] = coef["effect"].map(lambda v: "more likely" if v > 0 else "less likely")
        a.markdown("**What predicts a store location** (standardised coefficients)")
        a.altair_chart(alt.Chart(coef).mark_bar().encode(
            x=alt.X("effect:Q", title="effect on log-odds (per std. dev.)"),
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["more likely", "less likely"], range=["#dc2626", "#6b7280"]),
                            legend=None)).properties(height=300), width="stretch")
        b.markdown("**Validation**")
        b.write(f"Spatial-CV AUC **{m['auc_spatial_cv']:.3f}** ({m['reliability']}) · folds {m['auc_folds']}")
        b.write(f"{m['stores']} stores vs {m['background_n']:,} background premises · activity `{m['category']}` · "
                f"{m['competitors']:,} competitors")
        b.write(f"Brand demand decay {m['demand_decay_m']} m · daytime-activity weight {m['demand_activity_weight']:.0f}")
    with t3:
        cal = data().demand_cal or {}
        st.markdown(f"Footfall is calibrated on the city's **{cal.get('n_sensors', '?')} permanent pedestrian counters** "
                    f"({cal.get('sensor_period', '')}).")
        cc = pd.DataFrame({"Demand measure": ["Residents only", "v0: residents + every premises", "Now: consumer + office premises, industry excluded"],
                           "Rank correlation with measured footfall": [cal.get("baseline_residents_only"), cal.get("baseline_v0_all_premises"),
                                                                       cal.get("spearman_in_sample")]})
        st.dataframe(cc, hide_index=True, width="stretch", column_config={"Rank correlation with measured footfall":
                     st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
        st.caption(f"Held-out check (each sensor predicted by a fit that never saw it): {cal.get('spearman_leave_one_out', 0):.2f}.")


def score_pool(R, K):
    """All units that passed the filters (for the optional map layer)."""
    from mfi.model.rank import candidates, score
    cand, _ = candidates(data(), R, listings(), SS["m2_range"])
    df = score(R, cand)
    df = df[df["passes_gate"]].copy()
    df["tip"] = df["address"].fillna("") + " · " + df["m2"].round(0).astype(int).astype(str) + " m² · €" + df["rent_month"].round(0).astype(int).astype(str)
    return df[["lon", "lat", "tip"]]


# ================================================================== PAGE 2 — INVESTMENT REPORT
def page_report():
    hero("The reasoning behind the recommendation: franchise research, the data trail, the simulations and the memo.")
    if "out" not in SS:
        st.info("Run an analysis on the **Analysis** page first.")
        return
    O = SS["out"]
    R, K = O["R"], O["ranked"]
    st.markdown(f"### {R.brand}")
    ok_claude = reasoning.available()
    a, b = st.columns([1, 3])
    if a.button("Write investment memo", type="primary", disabled=not ok_claude or not K.get("ok")) and \
            (quota.allow("claude", SS) or st.error("Daily limit reached.")):
        with st.spinner("Claude is writing the memo from the evidence pack…"):
            SS["memo"] = reasoning.write_report(report.evidence_from_pipeline(O))
    if not ok_claude:
        b.info("Add `ANTHROPIC_API_KEY` to the app's secrets to enable Claude research and the memo.")

    t_memo, t_trail, t_fin, t_res = st.tabs(["Investment memo", "How we got here", "P&L Monte Carlo", "Franchise research"])
    with t_memo:
        if SS.get("memo"):
            st.markdown(SS["memo"])
        else:
            st.caption("Press **Write investment memo**. Claude may only use numbers from the evidence pack.")
    with t_trail:
        st.caption("Generated by Python from this exact run — not written by an AI.")
        for i, (step, detail) in enumerate(report.trail_from_pipeline(O), 1):
            st.markdown(f'<div class="step"><b>{i}. {step}</b><br>{detail}</div>', unsafe_allow_html=True)
    with t_fin:
        if not O["econ"]:
            st.warning("Not run — missing brand economics: " + ", ".join(O["missing"]))
        elif K.get("ok"):
            S = K["sites"].copy()
            S["site"] = "#" + S["rank"].astype(str) + " " + S["address"].fillna("").str.slice(0, 48)
            st.markdown("**EBITDA range per unit** (P10–P90, dot = P50) — 10,000 simulations each")
            st.altair_chart((alt.Chart(S).mark_rule(strokeWidth=3, color="#9ca3af").encode(
                x=alt.X("ebitda_p10:Q", title="€ / year"), x2="ebitda_p90:Q",
                y=alt.Y("site:N", sort=None, title=None, axis=alt.Axis(labelLimit=380)))
                + alt.Chart(S).mark_point(filled=True, size=90, color="#dc2626").encode(x="ebitda_p50:Q", y=alt.Y("site:N", sort=None))
                ).properties(height=44 * len(S)), width="stretch")
            econ = pd.DataFrame([{"input": k, "low": v[0], "mode": v[1], "high": v[2]} for k, v in O["econ"].items()])
            st.markdown("**Brand economics used** (triangular distributions, from Claude's cited research)")
            st.dataframe(econ, hide_index=True, width="stretch")
    with t_res:
        p = O.get("profile")
        if p:
            st.markdown(p.get("summary") or "")
            if p.get("citations"):
                st.markdown("**Sources:** " + " · ".join(f"[{c.get('title') or c['url']}]({c['url']})" for c in p["citations"][:20]))
            with st.expander("Structured data"):
                st.json(p.get("data") or {})
        else:
            st.caption("No research for this brand yet.")


# ================================================================== PAGE — BRAND DIRECTORY
def page_directory():
    hero("Every chain operating in Madrid, detected from the municipal premises census and OpenStreetMap.")
    reg = registry()
    k = st.columns(4)
    k[0].metric("Brands detected", f"{len(reg):,}")
    k[1].metric("Franchisable sectors", f"{int(reg['franchisable'].sum()):,}")
    k[2].metric("Food & drink brands", f"{int((reg['sector'] == 'Food & drink').sum()):,}")
    k[3].metric("Seen in OpenStreetMap too", f"{int((reg['n_osm'] > 0).sum()):,}")
    by = reg.groupby("sector").agg(brands=("brand_id", "count"), stores=("n_stores", "sum")).reset_index()
    st.altair_chart(alt.Chart(by).mark_bar(color="#dc2626").encode(
        x=alt.X("brands:Q", title="brands"), y=alt.Y("sector:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        tooltip=["sector", "brands", "stores"]).properties(height=30 * len(by)), width="stretch")
    f1, f2, f3 = st.columns([2, 1, 1])
    q = f1.text_input("Search", placeholder="Search a brand…", label_visibility="collapsed")
    sec = f2.selectbox("Sector", ["All"] + sorted(reg["sector"].unique()), label_visibility="collapsed")
    only_fr = f3.toggle("Franchisable only", value=True)
    t = reg.copy()
    if q:
        t = t[t["brand"].str.contains(q, case=False, regex=False)]
    if sec != "All":
        t = t[t["sector"] == sec]
    if only_fr:
        t = t[t["franchisable"]]
    st.dataframe(t[["brand", "sector", "subcategory", "n_stores", "n_osm", "n_districts", "name_variants", "not_franchisable_reason"]],
                 hide_index=True, width="stretch", height=520, column_config={
                     "brand": "Brand", "sector": "Sector", "subcategory": "Main activity", "n_stores": "Stores (census)",
                     "n_osm": "In OpenStreetMap", "n_districts": "Districts", "name_variants": "Name variants merged",
                     "not_franchisable_reason": "Why not franchisable"})
    st.caption("How it's built: names on open premises → generic names removed (activity words, sign types like "
               "'muestra opaca') → chains = ≥3 distinct locations → spelling variants merged → sector from licences → "
               "cross-checked with OpenStreetMap. Google Maps confirmation comes next.")


# ================================================================== PAGE 3 — METHODOLOGY
def page_methodology():
    hero("Exactly what runs behind every number: data, scraping, models, simulations, formulas and their limits.")
    D = data()
    cal = D.demand_cal or {}
    from mfi.model.rank import CAP_QUANTILE, MODEL_ERROR_SIGMA, RENT_OUTLIER

    st.markdown("## 1 · Pipeline (one click)")
    st.graphviz_chart("""
    digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f9fafb", color="#d1d5db", fontname="Helvetica", fontsize=11];
      edge [color="#9ca3af"];
      subgraph cluster_d { label="Data"; style=dashed; color="#d1d5db";
        census [label="Premises census\\n+ Padrón + INE income\\n(Ayto. Madrid, INE)"]; osm [label="OpenStreetMap\\nstores, metro, offices"];
        ped [label="31 pedestrian\\ncounters"]; idea [label="Idealista listings\\n(Apify)"]; web [label="Franchise economics\\n(Claude web search)"]; }
      reg [label="Brand registry\\n+ store inventory"]; dem [label="Demand surface\\n(calibrated on footfall)"];
      logit [label="Location model\\nlogistic + spatial CV"]; huff [label="Huff demand &\\ncompetition"];
      filt [label="Listing filters\\nsize · floor · zone · rent sanity"]; mc [label="P&L Monte Carlo\\nper listing", fillcolor="#fee2e2"];
      rank [label="Greedy ranking\\n(cannibalisation)", fillcolor="#fee2e2"]; memo [label="Investment memo\\n(Claude, evidence-only)", fillcolor="#111827", fontcolor="white"];
      census -> reg; osm -> reg; census -> dem; osm -> dem; ped -> dem; reg -> logit; dem -> logit; dem -> huff; reg -> huff;
      web -> filt; idea -> filt; filt -> logit; logit -> rank; huff -> mc; web -> mc; idea -> mc; mc -> rank; rank -> memo; web -> memo; }""")
    st.markdown("Order matters: **research → filter real listings → score → simulate the money → rank**. Money decides the ranking; "
                "it is not a grade applied afterwards.")

    st.markdown("## 2 · Data")
    src = pd.DataFrame(D.meta["sources"])
    labels = {"madrid_padron": ("Padrón municipal", "Residents by section × age × sex × nationality", "Census section"),
              "madrid_locales": ("Censo de Locales", "Every commercial premise: status, address, point", "Point"),
              "madrid_actividades": ("Censo de Locales — Actividades", "Licensed activity codes per premise", "Point"),
              "ine_adrh_madrid": ("INE ADRH (table 31097)", "Net income per person / household", "Census section"),
              "madrid_sections": ("Seccionado censal", "2,462 section polygons", "Census section"),
              "madrid_barrios": ("Barrios", "131 barrio polygons", "Barrio"),
              "madrid_districts": ("Distritos", "21 district polygons", "District")}
    src["Dataset"] = src.source_key.map(lambda k: labels.get(k, (k,))[0])
    src["What it gives"] = src.source_key.map(lambda k: labels.get(k, ("", "?"))[1])
    src["Resolution"] = src.source_key.map(lambda k: labels.get(k, ("", "", "?"))[2])
    src["Data date"] = src.data_reference_date.fillna(src.retrieved_at).astype(str).str[:10]
    extra = pd.DataFrame([
        {"Dataset": "Aforos de peatones (300321)", "What it gives": f"{cal.get('n_sensors', 31)} permanent pedestrian counters, hourly",
         "Resolution": "Point", "Data date": "2022–2024", "url": "https://datos.madrid.es/dataset/300321-0-aforos-peatones-bicicletas"},
        {"Dataset": "OpenStreetMap", "What it gives": "7,448 food places · metro/rail · bus · offices · universities · malls",
         "Resolution": "Point", "Data date": "2026-11-04", "url": "https://www.openstreetmap.org"},
        {"Dataset": "Idealista (via Apify)", "What it gives": f"{len(listings()):,} commercial units for rent (after cleaning)",
         "Resolution": "Point", "Data date": "2026-11-04", "url": "https://www.idealista.com"},
    ])
    st.dataframe(pd.concat([src[["Dataset", "What it gives", "Resolution", "Data date", "url"]], extra]), hide_index=True,
                 width="stretch", column_config={"url": st.column_config.LinkColumn("Source")})
    st.markdown("""
- **Licences:** Ayuntamiento de Madrid open data (CC BY 4.0), INE (attribution), OpenStreetMap (ODbL). Downloads are SHA-256
  fingerprinted and archived unchanged; re-runs keep history.
- **Quality checks:** 18 automated checks on every ingest. A failed check blocks the build.
- **Brand registry:** 1,353 chains detected from trade names on open premises — generic names ("BAR", sign types like
  "muestra opaca") removed, spelling variants merged, sector from licences, cross-checked with OpenStreetMap.
""")

    st.markdown("## 3 · Scraping")
    st.dataframe(pd.DataFrame([
        {"What": "Every commercial unit for rent in Madrid", "Site": "Idealista", "Tool": "Apify · igolaizola/idealista-scraper",
         "Query": "rent · premises · Madrid municipality · split into 6 size bands (2,500-result cap per search)",
         "Used for": "The candidate set: rent, m², floor, traspaso, link", "Listed cost": "~$0.001 / listing"},
        {"What": "Places + ratings + review counts", "Site": "Google Maps", "Tool": "Apify · compass/crawler-google-places",
         "Query": "citywide by category (next)", "Used for": "Store confirmation; reviews as observed demand",
         "Listed cost": "$0.004 / place"},
    ]), hide_index=True, width="stretch")
    st.markdown("Listing cleaning: commercial only (no warehouses) · €200–€100k/month · 20–3,000 m² · €3–200/m² · de-duplicated · "
                f"inside Madrid · €/m² within {RENT_OUTLIER[0]:.0%}–{RENT_OUTLIER[1]:.0f}× the median of comparable units within 1 km "
                "(catches typos like '€350/month for 100 m² on Castellana'). Raw responses archived with query and timestamp. "
                "Scraping Idealista and Google Maps is against their terms; accepted by the project owner for research use.")

    st.markdown("## 4 · Features (per location)")
    src_map = {"pop_500": "Padrón", "pop_1000": "Padrón", "young_share_1000": "Padrón", "income_1000": "INE ADRH",
               "footfall": "Census + pedestrian counters", "office_500": "Census (office licences)",
               "transit_300": "OpenStreetMap", "food_300": "Census", "comp_500": "Census + brand registry"}
    st.dataframe(pd.DataFrame({"Feature": [FEATURE_LABELS[f] for f in FEATURES],
                               "Transform": ["share" if f == "young_share_1000" else "log" if f == "income_1000" else "log(1+x)" for f in FEATURES],
                               "Source": [src_map[f] for f in FEATURES]}), hide_index=True, width="stretch")

    st.markdown("## 5 · Models & formulas")
    st.markdown("**5.1 Demand surface — calibrated on measured footfall.** Premises are classed by licence: *consumer* (shops, food, "
                "services, clinics, education), *office* (professional, public administration) and *industrial* (manufacturing, "
                "wholesale, logistics — excluded). For a point $p$:")
    st.latex(r"F(p)=\sum_i \big(R_i + a\,C_i + b\,O_i\big)\,e^{-d(p,i)/\lambda} + c\cdot\text{stations}_{300m}(p)")
    st.markdown(f"$a, b, \\lambda, c$ maximise the rank correlation between $F$ and the mean weekday count at the city's "
                f"{cal.get('n_sensors', 31)} pedestrian counters. Result: **ρ = {cal.get('spearman_in_sample', 0):.2f}** "
                f"(held-out sensors {cal.get('spearman_leave_one_out', 0):.2f}) vs residents only {cal.get('baseline_residents_only', 0):.2f} "
                f"and the old all-premises proxy {cal.get('baseline_v0_all_premises', 0):.2f}. λ = {cal.get('lambda_m')} m.")
    st.markdown("**5.2 Location model — where does this brand choose to be?** Logistic regression, brand stores vs background "
                "street-level premises, standardised features $z$, validated by **spatial cross-validation** (whole districts held out):")
    st.latex(r"P(\text{store}\mid \mathbf{x}) = \frac{1}{1+e^{-(\beta_0+\boldsymbol\beta^\top \mathbf z)}},\qquad "
             r"\text{AUC}=P\big(\hat p(\text{store}) > \hat p(\text{background})\big)")
    st.markdown("Reliability label: AUC < 0.60 **unreliable** (red banner) · 0.60–0.70 weak · 0.70–0.80 usable · > 0.80 good. "
                f"**Gate:** a listing must score at least the bottom-quartile existing store.")
    st.markdown("**5.3 Huff gravity model — who wins each section's demand.** Pull decays with distance; demand splits across the "
                "brand's stores $S$, competitors $C$ (same activity class or any known chain in it) and an outside option $w_0$:")
    st.latex(r"w_{ij}=e^{-d_{ij}/\lambda}\,\mathbb 1[d_{ij}\le 3\text{ km}],\qquad "
             r"\text{Own}_j=\sum_i D_i\,\frac{w_{ij}}{\sum_{s\in S\cup\{j\}}w_{is}+\sum_{c\in C}w_{ic}+w_0}")
    st.markdown("Brand demand $D_i = R_i + k\\,(C_i+O_i)$; $\\lambda$ and $k$ are fitted per brand so its existing stores have the "
                "most even catchments. **Demand ratio** = $\\text{Own}_j$ ÷ the median existing store's own capture, capped at the "
                f"brand's {CAP_QUANTILE:.0%}-percentile store (a new unit can't outperform the brand's own busiest stores).")

    st.markdown("## 6 · Simulations & ranking")
    st.markdown(f"**6.1 P&L Monte Carlo (N = 4,000 per candidate during ranking, 10,000 for the final picks).** Brand inputs are "
                "triangular distributions (low, mode, high) from Claude's cited research; rent, m² and traspaso are the listing's own:")
    st.latex(r"\text{Rev}=\text{AUV}\times r_j\times\varepsilon,\ \ \varepsilon\sim\text{LogNormal}(0,\sigma),\qquad "
             r"\text{EBITDA}=\text{Rev}\,(1-\textstyle\sum c)-\text{rent}_{\text{listing}}\times 12")
    st.latex(r"\text{Payback}=\frac{I_{\text{brand}}+\text{traspaso}}{\text{EBITDA}},\qquad \text{ROIC}=\frac{\text{EBITDA}}{I}")
    st.markdown(f"σ = {MODEL_ERROR_SIGMA:.0%} model error (assumption). Brands where the franchisor provides the premises use "
                "occupancy as % of sales instead of listing rent. Missing inputs → no simulation; nothing is invented.")
    st.markdown("**6.2 Ranking.** Primary: **P(payback ≤ target)**; tie-break: median EBITDA. Greedy: after each pick the unit joins "
                "the brand's network and every remaining candidate is re-scored, so the list accounts for cannibalisation; picks "
                "≥ 400 m apart. Without brand economics the metric is *customers per €1k of annual rent*.")

    st.markdown("## 7 · Claude's role")
    st.markdown("""
| Task | What Claude does | What it may not do |
|---|---|---|
| Franchise research (step 1, cached per brand) | Web search; eligibility in Spain, who provides premises, fees, investment, revenue, cost ratios, unit size — each cited and labelled | Invent a number; present a non-Spain figure as Spanish |
| Investment memo | Verdict and unit-by-unit reasoning from the evidence pack | Use any number not in the evidence pack; change model outputs |
""")
    st.markdown("## 8 · Limitations")
    st.markdown("""
- **Listings coverage:** ~2,500 units; Idealista caps results per search, so coverage may be incomplete.
- **Revenue:** brand average × modelled demand; no measured sales per store. Google review counts (next) will calibrate demand on real visits.
- **Footfall calibration:** 31 sensors, mostly central shopping streets.
- **Distance:** straight-line, not walking network.
- **Units:** as advertised; licence (e.g. hostelería smoke extraction), condition and frontage are not verified.
- **Not yet built:** survival model on 2014–2026 openings/closures; out-of-time backtest (train 2019 → predict 2024).
""")


pg = st.navigation([st.Page(page_analysis, title="Analysis", icon="🗺️", default=True),
                    st.Page(page_directory, title="Brand directory", icon="🏷️", url_path="brands"),
                    st.Page(page_report, title="Investment report", icon="📑", url_path="report"),
                    st.Page(page_methodology, title="Methodology", icon="🧮", url_path="methodology")])
pg.run()
