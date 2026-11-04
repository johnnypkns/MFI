"""Madrid Franchise Intelligence — dashboard (3 pages: Analysis · Investment report · Methodology).

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

from mfi import quota, reasoning, report, scrape
from mfi.model import pnl
from mfi.model.export import APP
from mfi.model.robustness import simulate as robustness
from mfi.model.site_model import FEATURE_LABELS, PARAMS, load_data, run

st.set_page_config(page_title="Madrid Franchise Intelligence", page_icon="📍", layout="wide")
RED, BLACK, GREY = [220, 38, 38], [17, 17, 17], [150, 150, 150]
QUICK = ["McDonald's", "Burger King", "Starbucks", "Five Guys", "Goiko", "KFC", "Taco Bell", "Oakberry"]
SS = st.session_state

st.markdown("""
<style>
.block-container {padding-top: 3rem; max-width: 1500px;}
.hero h1 {font-size: 2.0rem; letter-spacing: .04em; margin: 0; font-weight: 800;}
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
p.gate-h {font-size:2.1rem !important; line-height:1.2 !important; margin-bottom:4px !important;}
p.gate-title {font-size:.74rem !important;} p.gate-sub, p.gate-foot {font-size:.7rem !important;}
div[data-testid="stFormSubmitButton"] {max-width: 380px; margin: 8px auto 0 auto;}
div[data-testid="stFormSubmitButton"] button, .stButton button {width:100%; border-radius:0; background:#dc2626; color:#fff; border:none; letter-spacing:.4em; font-family: ui-monospace, Menlo, monospace;
  font-size:.8rem; padding:.75rem 0; transition: all .2s;}
div[data-testid="stFormSubmitButton"] button:hover, .stButton button:hover {background:#111 !important; color:#fff !important;}
div[data-testid="stFormSubmitButton"] button p {letter-spacing:.4em; font-size:.8rem; color:#fff;}
.gate-foot {text-align:center; color:#d1d5db; font-family: ui-monospace, Menlo, monospace; font-size:.65rem; letter-spacing:.3em; margin-top:46px;}
</style>
<div class="gate-mark"></div>
<p class="gate-title">MFI · RESTRICTED</p>
<p class="gate-h">MADRID FRANCHISE<br>INTELLIGENCE</p>
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
    st.markdown('<p class="gate-foot">© MFI · MADRID</p>', unsafe_allow_html=True)
    st.stop()


gate()


def hero(sub):
    st.markdown(f'<div class="hero"><h1>MADRID FRANCHISE INTELLIGENCE</h1><p>{sub}</p></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ cached compute
@st.cache_resource(show_spinner="Loading Madrid data…")
def data():
    return load_data()


@st.cache_data(show_spinner=False)
def districts_geojson():
    return json.loads((APP / "districts.geojson").read_text())


@st.cache_resource(show_spinner=False)
def analyse(brand, n):
    return run(data(), brand, n)


@st.cache_resource(show_spinner=False)
def robust(brand, n, B=200):
    return robustness(analyse(brand, n), B=B)


def reset_downstream():
    for k in ["profile", "rents", "pnl_inputs", "pnl_out", "memo"]:
        SS.pop(k, None)


# ================================================================== PAGE 1 — ANALYSIS
def page_analysis():
    hero("Where should this franchise open next in Madrid? Census, income and commerce data → spatial model → "
         "Monte Carlo → ranked sites.")
    c1, c2, c3 = st.columns([4, 1.2, 1])
    pick = st.pills("Quick picks", QUICK, default=None, label_visibility="collapsed")
    brand = c1.text_input("Franchise", value=pick or SS.get("brand", "McDonald's"), placeholder="Type a franchise name…",
                          label_visibility="collapsed")
    n_sites = c2.selectbox("Sites", [3, 5, 10], index=[3, 5, 10].index(SS.get("n_sites", 5)),
                           label_visibility="collapsed", format_func=lambda n: f"Top {n} sites")
    go = c3.button("Analyse  →", type="primary", width="stretch")
    if go or pick or "result" not in SS:
        try:
            with st.spinner(f"Running the site model for {brand}…"):
                R = analyse(brand, n_sites)
            with st.spinner(f"Monte Carlo: 200 simulations of the model…"):
                rob = robust(brand, n_sites)
            if SS.get("brand") != brand or SS.get("n_sites") != n_sites:
                reset_downstream()
            SS.update(result=R, rob=rob, brand=brand, n_sites=n_sites)
            SS.pop("error", None)
        except ValueError as e:
            SS["error"] = str(e)
            SS.pop("result", None)
    if "error" in SS:
        st.error(SS["error"])
        st.stop()

    R, rob, m = SS["result"], SS["rob"], SS["result"].metrics
    k = st.columns(5)
    k[0].metric("Existing stores (census)", m["stores"])
    k[1].metric("Recommended sites", len(R.sites))
    k[2].metric("Model accuracy", f"{m['auc_spatial_cv']:.2f} AUC" if m["auc_spatial_cv"] else "n/a",
                help="Spatial cross-validation: whole districts held out, then predicted. 0.5 = coin flip, 1.0 = perfect.")
    k[3].metric("More stores Madrid can absorb",
                f"{m['additional_stores_supported']}{'+' if m['additional_stores_supported'] >= R.params['max_sites'] else ''}",
                help="Stores whose captured demand is ≥50% of a typical existing store, accounting for cannibalisation.")
    k[4].metric("Same-category competitors", f"{m['competitors']:,}")
    for w in R.warnings:
        st.warning(w)

    show_comp = st.toggle("Show competitors", value=False)
    show_alt = st.toggle("Show Monte Carlo alternatives", value=False,
                         help="Units the optimiser picked most often across the 200 simulations.")
    st.markdown(f"""<div class="legend">
    <span><i class="dot" style="background:#dc2626"></i>Recommended site</span>
    <span><i class="dot" style="background:#111"></i>Existing {R.brand}</span>
    {'<span><i class="dot" style="background:#9ca3af"></i>Competitor</span>' if show_comp else ''}
    {'<span><i class="dot" style="background:#f59e0b"></i>Monte Carlo alternative</span>' if show_alt else ''}</div>""",
                unsafe_allow_html=True)

    sites = R.sites.merge(rob["summary"][["rank", "p_selected", "demand_p10", "demand_p90"]], on="rank")
    sites["label"] = sites["rank"].astype(int).astype(str)
    sites["tip"] = ("<b>#" + sites["label"] + " " + sites["address"] + "</b><br/>" + sites["barrio"].fillna("") + ", "
                    + sites["district"].fillna("") + "<br/>Suitability " + (sites["suitability"] * 100).round(0).astype(int).astype(str)
                    + "% · demand " + sites["vs_median_store"].round(2).astype(str) + "× median store<br/>Monte Carlo stability "
                    + (sites["p_selected"] * 100).round(0).astype(int).astype(str) + "%")
    stores = R.stores.copy()
    stores["tip"] = "<b>" + stores["trade_name"].fillna(R.brand) + "</b><br/>" + stores["address"]
    layers = [pdk.Layer("GeoJsonLayer", districts_geojson(), stroked=True, filled=False,
                        get_line_color=[90, 90, 90, 110], line_width_min_pixels=1)]
    if show_comp:
        comp = R.competitors.copy()
        comp["tip"] = "<b>" + comp["trade_name"].fillna("Competitor") + "</b>"
        layers.append(pdk.Layer("ScatterplotLayer", comp, get_position=["lon", "lat"], get_fill_color=GREY + [140],
                                get_radius=35, radius_min_pixels=2, pickable=True))
    if show_alt:
        alt_df = rob["alternatives"].copy()
        alt_df["tip"] = ("<b>" + alt_df["street_name"].str.title() + " " + alt_df["street_number"] + "</b><br/>picked in "
                         + alt_df["times_picked"].astype(str) + "/" + str(rob["B"]) + " simulations")
        layers.append(pdk.Layer("ScatterplotLayer", alt_df, get_position=["lon", "lat"], get_fill_color=[245, 158, 11, 210],
                                get_radius=110, radius_min_pixels=6, pickable=True))
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
                             tooltip={"html": "{tip}", "style": {"fontSize": "12px"}}), height=620)

    st.subheader("Recommended sites")
    tbl = pd.DataFrame({
        "#": sites["rank"].astype(int), "Address": sites["address"], "Barrio": sites["barrio"], "District": sites["district"],
        "Suitability": sites["suitability"], "Demand vs median store": sites["vs_median_store"],
        "Demand P10–P90": sites["demand_p10"].round(2).astype(str) + "–" + sites["demand_p90"].round(2).astype(str) + "×",
        "MC stability": sites["p_selected"],
        f"Nearest {R.brand} (m)": sites["dist_nearest_brand_m"].round(0), "Competitors ≤500 m": sites["comp_500"],
        "Income/person ≤1 km": sites["income_1000"].round(0)})
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "Suitability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "MC stability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1,
                                                        help="Share of 200 simulations that pick a unit within 400 m"),
        "Demand vs median store": st.column_config.NumberColumn(format="%.2f×"),
        "Income/person ≤1 km": st.column_config.NumberColumn(format="€%d")})
    st.caption("Candidate sites are vacant street-level units in the municipal census. Full reasoning, rents and "
               "financials → **Investment report**. How every number is made → **Methodology**.")

    t3, t4 = st.tabs(["Model diagnostics", "How many stores?"])
    with t3:
        a, b = st.columns([1.2, 1])
        coef = R.coefficients.copy()
        coef["direction"] = coef["effect"].map(lambda v: "more likely" if v > 0 else "less likely")
        a.markdown("**What predicts a store location** (standardised coefficients)")
        a.altair_chart(alt.Chart(coef).mark_bar().encode(
            x=alt.X("effect:Q", title="effect on log-odds (per std. dev.)"),
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["more likely", "less likely"], range=["#dc2626", "#6b7280"]),
                            legend=None)).properties(height=260), width="stretch")
        b.markdown("**Validation**")
        b.write(f"Spatial-CV AUC **{m['auc_spatial_cv']:.3f}** · folds {m['auc_folds']}")
        b.write(f"{m['stores']} stores vs {m['background_n']:,} background premises · category `{m['category']}`")
        b.write(f"Demand: decay **{m['demand_decay_m']} m**, activity weight **{m['demand_activity_weight']:.0f}** "
                f"(catchment CV {m['demand_calibration_cv']:.2f})")
        b.write(f"{m['candidates_vacant']:,} vacant units scored → {m['candidates_pass_gate']:,} pass the gate")
    with t4:
        cur = R.curve.copy()
        st.markdown("Each additional store captures less new demand. Stop below **50%** of a typical existing store.")
        st.altair_chart((alt.Chart(cur).mark_line(point=True, color="#dc2626").encode(
            x=alt.X("rank:O", title="n-th new store"), y=alt.Y("vs_median_store:Q", title="× median existing store"))
            + alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color="#111").encode(y="y:Q")
        ).properties(height=300), width="stretch")
        if m["additional_stores_supported"] >= R.params["max_sites"]:
            st.info("The curve never crosses 50% within 15 stores — partly because the census misses stores registered under "
                    "company names (OSM/Google cross-check pending).")


# ================================================================== PAGE 2 — INVESTMENT REPORT
def page_report():
    hero("The reasoning behind the recommendation: research, scraping, simulations and the investment memo.")
    if "result" not in SS:
        st.info("Run an analysis on the **Analysis** page first.")
        return
    R, rob = SS["result"], SS["rob"]
    st.markdown(f"### {R.brand} · {len(R.sites)} sites")

    ok_claude, ok_apify = reasoning.available(), scrape.available()
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(f"**1 · Franchise research**<br><span class='small'>Claude + web search {'✅' if ok_claude else '— needs ANTHROPIC_API_KEY'}</span>", unsafe_allow_html=True)
    s2.markdown(f"**2 · Rent scrape**<br><span class='small'>Apify → Idealista {'✅' if ok_apify else '— needs APIFY_TOKEN'}</span>", unsafe_allow_html=True)
    s3.markdown("**3 · P&L Monte Carlo**<br><span class='small'>10,000 draws per site</span>", unsafe_allow_html=True)
    s4.markdown(f"**4 · Investment memo**<br><span class='small'>Claude, evidence-only {'✅' if ok_claude else ''}</span>", unsafe_allow_html=True)

    b1, b2, b3, b4, b5 = st.columns(5)
    do_all = b5.button("Run everything ▶", type="primary", width="stretch")
    limit_msg = "Daily limit reached for this demo — try again tomorrow."
    if (b1.button("Research", disabled=not ok_claude, width="stretch") or (do_all and ok_claude)) and (quota.allow("claude", SS) or st.error(limit_msg)):
        with st.spinner("Claude is researching the franchise on the web (~1 min)…"):
            SS["profile"] = reasoning.research_franchise(R.brand)
            SS["pnl_inputs"] = report.pnl_inputs_from_research(SS["profile"], SS.get("pnl_inputs"))
    if (b2.button("Scrape rents", disabled=not ok_apify, width="stretch") or (do_all and ok_apify)) and (quota.allow("apify", SS) or st.error(limit_msg)):
        rents, prog = {}, st.progress(0.0, "Scraping Idealista listings near each site…")
        min_m2 = None
        prof = (SS.get("profile") or {}).get("data") or {}
        min_m2 = (prof.get("min_floor_area_m2") or {}).get("value")
        for i, r in enumerate(R.sites.itertuples()):
            try:
                rents[int(r.rank)] = scrape.rents_near(r.lat, r.lon, min_m2=min_m2)
            except Exception as e:
                st.warning(f"Site #{r.rank}: rent scrape failed — {e}")
            prog.progress((i + 1) / len(R.sites))
        SS["rents"] = rents
        if rents:
            med = [v["stats"]["p50"] for v in rents.values() if "p50" in v["stats"]]
            inp = SS.get("pnl_inputs", pnl.empty_inputs()).copy()
            if med and report.rent_model(SS.get("profile")) == "market_lease":
                inp.loc[inp.key == "rent_eur_m2_month", ["low", "mode", "high", "source", "label"]] = [
                    float(np.min(med)), float(np.median(med)), float(np.max(med)), "Idealista via Apify (site medians)", "listing_data"]
            SS["pnl_inputs"] = inp
    run_pnl = b3.button("Run P&L", width="stretch") or do_all
    run_memo = b4.button("Write memo", disabled=not ok_claude, width="stretch") or (do_all and ok_claude)

    prof = (SS.get("profile") or {}).get("data") or {}
    if prof:
        elig = prof.get("individual_franchise_available_spain")
        if elig != "yes":
            st.error(f"**Eligibility gate:** individual franchise in Spain = **{elig}**. {prof.get('eligibility_explanation', '')}")
        if prof.get("rent_model") == "franchisor_property":
            st.warning("**The franchisor chooses and provides the site.** You can't lease one of these units yourself — the map "
                       "shows where the brand's demand is strongest (useful to argue for a Madrid placement), and the P&L "
                       "charges occupancy as % of sales instead of market rent. " + (prof.get("rent_model_explanation") or ""))
    with st.expander("P&L inputs — every number has a source and a label", expanded=not ok_claude):
        st.caption("Prefilled by Claude's research and the rent scrape. You can override any cell; "
                   "set label to `user_assumption` when you do. Empty rows block the simulation — nothing is invented.")
        SS["pnl_inputs"] = st.data_editor(SS.get("pnl_inputs", pnl.empty_inputs()), hide_index=True, width="stretch",
                                          disabled=["key", "parameter"], key="pnl_editor",
                                          column_config={"key": None,
                                                         "label": st.column_config.SelectboxColumn(options=[
                                                             "verified", "third_party_estimate", "listing_data", "user_assumption", ""])})
    if run_pnl:
        site_rents = {k: v["stats"] for k, v in (SS.get("rents") or {}).items()}
        SS["pnl_out"] = pnl.simulate(R.sites, rob["ratio_draws"], SS["pnl_inputs"], site_rents=site_rents,
                                     rent_model=report.rent_model(SS.get("profile")))
    if run_memo and (quota.allow("claude", SS) or st.error(limit_msg)):
        with st.spinner("Claude is writing the investment memo from the evidence pack…"):
            ev = report.evidence_pack(R, rob, SS.get("pnl_out"), SS.get("rents"), SS.get("profile"))
            SS["memo"] = reasoning.write_report(ev)

    tab_memo, tab_trail, tab_sim, tab_fin, tab_market = st.tabs(
        ["Investment memo", "How we got here", "Robustness Monte Carlo", "P&L Monte Carlo", "Research & scraped data"])

    with tab_memo:
        if SS.get("memo"):
            st.markdown(SS["memo"])
            st.download_button("Download memo (.md)", SS["memo"], file_name=f"{R.brand}-madrid-memo.md")
        elif not ok_claude:
            st.info("The memo is written by Claude from the evidence pack. Add `ANTHROPIC_API_KEY` to "
                    "`.streamlit/secrets.toml` and restart. Until then, **How we got here** shows the full computed trail.")
        else:
            st.info("Press **Run everything** (or the steps one by one).")

    with tab_trail:
        st.caption("Generated by Python from this exact run — not written by an AI.")
        for i, (step, detail) in enumerate(report.audit_trail(R, rob, SS.get("pnl_out"), SS.get("rents"), SS.get("profile")), 1):
            st.markdown(f'<div class="step"><b>{i}. {step}</b><br>{detail}</div>', unsafe_allow_html=True)

    with tab_sim:
        s = rob["summary"].copy()
        st.markdown(f"**{rob['B']} simulations.** Each one re-samples the training data (bootstrap) and redraws the demand "
                    f"parameters from the 5 best-fitting calibration cells, then re-runs the optimiser.")
        a, b = st.columns(2)
        s["site"] = "#" + s["rank"].astype(str) + " " + s["address"]
        a.markdown("**Stability** — share of simulations picking a unit within 400 m")
        a.altair_chart(alt.Chart(s).mark_bar(color="#dc2626").encode(
            x=alt.X("p_selected:Q", axis=alt.Axis(format="%"), title=None, scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("site:N", sort=None, title=None, axis=alt.Axis(labelLimit=300))).properties(height=40 * len(s)),
            width="stretch")
        b.markdown("**Captured demand vs a median existing store** — P10 · P50 · P90")
        b.altair_chart((alt.Chart(s).mark_rule(strokeWidth=3, color="#9ca3af").encode(
            x=alt.X("demand_p10:Q", title="× median store"), x2="demand_p90:Q", y=alt.Y("site:N", sort=None, title=None,
                                                                                         axis=alt.Axis(labelLimit=300)))
            + alt.Chart(s).mark_point(filled=True, size=90, color="#dc2626").encode(x="demand_p50:Q", y=alt.Y("site:N", sort=None))
            + alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")).properties(height=40 * len(s)),
            width="stretch")
        st.markdown("**Units the optimiser picked most often** (show them on the map with the toggle on Analysis)")
        alt_t = rob["alternatives"].head(10).assign(address=lambda d: d.street_name.str.title() + " " + d.street_number,
                                                    share=lambda d: d.times_picked / rob["B"])
        st.dataframe(alt_t[["address", "times_picked", "share"]], hide_index=True,
                     column_config={"share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

    with tab_fin:
        po = SS.get("pnl_out")
        if not po:
            st.info("Not run yet. Fill or prefill the P&L inputs, then **Run P&L**.")
        elif not po["ok"]:
            st.warning("Not run — missing inputs: " + ", ".join(po["missing"]) + ". The simulation refuses to fill gaps.")
        else:
            f = po["summary"].copy()
            f["site"] = "#" + f["rank"].astype(str) + " " + f["address"]
            tgt = f"p_payback_le_{po['payback_target_yrs']:g}y"
            st.dataframe(pd.DataFrame({
                "Site": f["site"], "Revenue P50": f["revenue_p50"], "EBITDA P10": f["ebitda_p10"], "EBITDA P50": f["ebitda_p50"],
                "EBITDA P90": f["ebitda_p90"], "Margin P50": f["margin_p50"], "P(loss)": f["p_loss"],
                "Payback P50 (yrs)": f["payback_p50_yrs"], f"P(payback ≤ {po['payback_target_yrs']:g}y)": f[tgt],
                "ROIC P50": f["roic_p50"], "Rent source": f["rent_source"]}), hide_index=True, width="stretch",
                column_config={**{c: st.column_config.NumberColumn(format="€%d") for c in
                                  ["Revenue P50", "EBITDA P10", "EBITDA P50", "EBITDA P90"]},
                               "Margin P50": st.column_config.NumberColumn(format="percent"),
                               "ROIC P50": st.column_config.NumberColumn(format="percent"),
                               "P(loss)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                               f"P(payback ≤ {po['payback_target_yrs']:g}y)": st.column_config.ProgressColumn(
                                   format="percent", min_value=0, max_value=1)})
            a, b = st.columns(2)
            a.markdown("**EBITDA range** (P10–P90, dot = P50)")
            a.altair_chart((alt.Chart(f).mark_rule(strokeWidth=3, color="#9ca3af").encode(
                x=alt.X("ebitda_p10:Q", title="€ / year"), x2="ebitda_p90:Q", y=alt.Y("site:N", sort=None, title=None,
                                                                                    axis=alt.Axis(labelLimit=300)))
                + alt.Chart(f).mark_point(filled=True, size=90, color="#dc2626").encode(x="ebitda_p50:Q", y=alt.Y("site:N", sort=None))
                ).properties(height=40 * len(f)), width="stretch")
            pick_rank = b.selectbox("Sensitivity for site", f["rank"].tolist(), format_func=lambda r: f"#{r}")
            sens = po["sensitivity"][po["sensitivity"]["rank"] == pick_rank].copy()
            sens["abs"] = sens["spearman"].abs()
            b.markdown("**What drives EBITDA** (Spearman rank correlation, tornado)")
            b.altair_chart(alt.Chart(sens).mark_bar().encode(
                x=alt.X("spearman:Q", scale=alt.Scale(domain=[-1, 1]), title="correlation with EBITDA"),
                y=alt.Y("driver:N", sort=alt.EncodingSortField("abs", order="descending"), title=None, axis=alt.Axis(labelLimit=320)),
                color=alt.condition("datum.spearman > 0", alt.value("#dc2626"), alt.value("#6b7280"))).properties(height=300),
                width="stretch")

    with tab_market:
        p = SS.get("profile")
        st.markdown("#### Franchise research (Claude + web search)")
        if p:
            st.markdown(p["summary"])
            if p.get("citations"):
                st.markdown("**Sources:** " + " · ".join(f"[{c.get('title') or c['url']}]({c['url']})" for c in p["citations"][:15]))
            with st.expander("Structured data"):
                st.json(p.get("data") or {})
        else:
            st.caption("Not run.")
        st.markdown("#### Rent listings (Apify → Idealista)")
        rents = SS.get("rents")
        if rents:
            for k, v in rents.items():
                st.markdown(f"**Site #{k}** — {v['stats'].get('n', 0)} listings within {v['radius_km']} km · retrieved {v['retrieved']}"
                            + (f" · €/m²/mo P25 {v['stats']['p25']:.1f} · P50 {v['stats']['p50']:.1f} · P75 {v['stats']['p75']:.1f}"
                               if "p50" in v["stats"] else ""))
                if v["listings"]:
                    st.dataframe(pd.DataFrame(v["listings"])[["address", "m2", "price_month", "eur_m2", "distance_m", "traspaso_eur", "url"]], hide_index=True,
                                 column_config={"url": st.column_config.LinkColumn()})
        else:
            st.caption("Not run.")


# ================================================================== PAGE 3 — METHODOLOGY
def page_methodology():
    hero("Exactly what runs behind every number: data, scraping, models, simulations, formulas and their limits.")
    D = data()
    P = PARAMS

    st.markdown("## 1 · Pipeline")
    st.graphviz_chart("""
    digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f9fafb", color="#d1d5db", fontname="Helvetica", fontsize=11];
      edge [color="#9ca3af"];
      subgraph cluster_d { label="Data"; style=dashed; color="#d1d5db"; census [label="Premises census\\n(Ayto. Madrid)"]; padron [label="Padrón\\n(Ayto. Madrid)"]; ine [label="Income atlas\\n(INE)"]; geo [label="Boundaries\\n(Ayto. Madrid)"]; idea [label="Idealista rents\\n(Apify)"]; web [label="Franchise facts\\n(Claude web search)"]; }
      feat [label="Feature space\\n7 features / location"]; logit [label="Suitability model\\nlogistic regression\\n+ spatial CV"]; huff [label="Huff demand model\\ncalibrated"]; opt [label="Greedy optimiser\\n(cannibalisation)"];
      mc1 [label="Robustness\\nMonte Carlo", fillcolor="#fee2e2"]; mc2 [label="P&L\\nMonte Carlo", fillcolor="#fee2e2"]; memo [label="Investment memo\\n(Claude, evidence-only)", fillcolor="#111827", fontcolor="white"];
      census -> feat; padron -> feat; ine -> feat; geo -> feat; feat -> logit -> opt; feat -> huff -> opt; opt -> mc1 -> mc2; idea -> mc2; web -> mc2; mc2 -> memo; mc1 -> memo; web -> memo; }""")

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
    st.dataframe(src[["Dataset", "What it gives", "Resolution", "Data date", "url"]], hide_index=True, width="stretch",
                 column_config={"url": st.column_config.LinkColumn("Source")})
    st.markdown("""
- **Publisher / licence:** Ayuntamiento de Madrid open-data portal (CC BY 4.0) and INE (attribution). Downloaded by script,
  SHA-256 fingerprinted, archived unchanged; re-runs keep history instead of overwriting.
- **Quality checks:** 18 automated checks on every ingest (boundary counts, geometry validity, population totals, income units,
  geocoding coverage, code consistency). A failed check blocks the build.
- **Claude franchise research:** web search over official franchise sites, filings, Spanish franchise directories.
  Every figure carries a URL and a label — `verified` or `third_party_estimate`.
""")

    st.markdown("## 3 · Scraping")
    st.dataframe(pd.DataFrame([
        {"What": "Commercial premises for rent", "Site": "Idealista", "Tool": "Apify actor igolaizola/idealista-scraper",
         "Query": "rent · premises · within 1 km of each site · min size = brand's floor area", "Used for": "Rent €/m²/month per site (P25/P50/P75)",
         "Listed cost": "~$0.001 per listing"},
        {"What": "Places + ratings + review counts", "Site": "Google Maps", "Tool": "Apify actor compass/crawler-google-places",
         "Query": "brand + category within 2 km", "Used for": "Store cross-check; reviews as observed demand (v0.2)",
         "Listed cost": "$0.004 per place"},
    ]), hide_index=True, width="stretch")
    st.markdown("Raw responses are archived as JSON with the query and timestamp (`data/raw/apify/`). Scraping Idealista and "
                "Google Maps goes against their terms of use; the project owner has accepted that for research use.")

    st.markdown("## 4 · Features (per location)")
    st.dataframe(pd.DataFrame({"Feature": list(FEATURE_LABELS.values()),
                               "Transform": ["log(1+x)", "log(1+x)", "share", "log", "log(1+x)", "log(1+x)", "log(1+x)"],
                               "Source": ["Padrón", "Padrón", "Padrón", "INE ADRH", "Census of premises", "Census of premises",
                                          "Census of premises (brand's activity code)"]}), hide_index=True, width="stretch")
    st.caption("Section values are assigned to the section's representative point; radius sums use straight-line distance.")

    st.markdown("## 5 · Models & formulas")
    st.markdown("**5.1 Suitability — where does this brand choose to be?** Logistic regression, presence vs background "
                f"(brand stores vs ~{P['background_ratio']}× random street-level premises), standardised features $z$:")
    st.latex(r"P(\text{store}\mid \mathbf{x}) = \frac{1}{1+e^{-(\beta_0+\boldsymbol\beta^\top \mathbf z)}},\qquad z_j=\frac{f_j(x_j)-\mu_j}{\sigma_j}")
    st.markdown("Validated by **spatial cross-validation**: folds are whole districts, so the model is always scored on territory "
                "it never saw. Metric = **AUC**, the probability that a random real store scores above a random background point:")
    st.latex(r"\text{AUC}=P\big(\hat p(\text{store}) > \hat p(\text{background})\big)\quad(0.5=\text{chance},\;1=\text{perfect})")
    st.markdown(f"**Gate:** a candidate must score at least the {P['gate_quantile']:.0%} quantile of the brand's existing stores.")

    st.markdown("**5.2 Demand.** Each census section $i$ supplies residents $R_i$ plus daytime activity — open premises $A_i$ "
                "(a jobs/visitors proxy) weighted by $k$:")
    st.latex(r"D_i = R_i + k\,A_i")
    st.markdown("**5.3 Huff gravity model — who wins each section's demand.** Pull of store $j$ on section $i$ decays with distance; "
                "demand splits across the brand's stores $S$, competitors $C$ and an outside option $w_0$:")
    st.latex(r"w_{ij}=e^{-d_{ij}/\lambda}\,\mathbb 1[d_{ij}\le 3\text{ km}],\qquad "
             r"\text{Captured}(S)=\sum_i D_i\,\frac{\sum_{s\in S}w_{is}}{\sum_{s\in S}w_{is}+\sum_{c\in C}w_{ic}+w_0}")
    st.markdown("**5.4 Calibration — $\\lambda$ and $k$ are fitted, not assumed.** Every existing store is viable, so the right demand "
                "definition gives them the most even catchments. Grid search over "
                "λ ∈ {300…2000 m} × k ∈ {0…40}, minimising the coefficient of variation of each store's leave-one-out capture:")
    st.latex(r"(\lambda^*,k^*)=\arg\min_{\lambda,k}\ \mathrm{CV}\big(\{\text{Captured}(S)-\text{Captured}(S\setminus\{s\})\}_{s\in S}\big)")
    st.markdown("**5.5 Optimiser — which sites, and how many.** Greedy maximisation of *incremental* captured demand. Adding a site "
                "next to the brand's own store gains little, so cannibalisation is built in:")
    st.latex(r"j^*=\arg\max_{j}\ \text{Captured}(S\cup\{j\})-\text{Captured}(S),\qquad S\leftarrow S\cup\{j^*\}")
    st.markdown(f"Picks ≥{P['min_spacing_m']} m apart. **How many:** keep adding while the next store captures ≥"
                f"{P['saturation_ratio']:.0%} of a median existing store.")

    st.markdown("## 6 · Simulations")
    st.markdown("**6.1 Robustness Monte Carlo (B = 200).** Each draw: bootstrap-resample stores and background, refit the "
                "suitability model; draw (λ, k) from the 5 best calibration cells, weighted by fit; re-run the optimiser.")
    st.latex(r"\text{Stability}_j=\frac{1}{B}\sum_{b=1}^{B}\mathbb 1\big[\exists\,\text{pick within }400\text{ m of }j\text{ in draw }b\big]")
    st.markdown("Also reported: the site's demand ratio vs a median store at P10/P50/P90 across draws.")
    st.markdown("**6.2 P&L Monte Carlo (N = 10,000 per site).** Inputs are triangular distributions (low, mode, high), each with "
                "its source. Site demand is drawn from the robustness simulation, so model uncertainty flows into the money:")
    st.latex(r"\text{Revenue}=\text{AUV}\times r_{\text{site}},\qquad "
             r"\text{EBITDA}=\text{Revenue}\,(1-c_{\text{food}}-c_{\text{labour}}-c_{\text{royalty}}-c_{\text{mkt}}-c_{\text{other}})-\text{rent}\cdot m^2\cdot 12")
    st.latex(r"\text{Payback}=\frac{I}{\text{EBITDA}},\quad \text{ROIC}=\frac{\text{EBITDA}}{I},\quad "
             r"\text{Break-even revenue}=\frac{\text{rent}\cdot m^2\cdot 12}{1-\sum c}")
    st.markdown("Reported: P10/P50/P90 of revenue and EBITDA, P(loss), median payback, P(payback ≤ 4 yrs), ROIC. "
                "**Sensitivity:** Spearman rank correlation of each input with EBITDA (tornado chart). "
                "If any input is missing the simulation does not run.")

    st.markdown("## 7 · Claude's role")
    st.markdown("""
| Task | What Claude does | What it may not do |
|---|---|---|
| Franchise research | Web search; extracts fees, investment, royalties, revenue disclosures, Spain eligibility; cites each figure | Invent a number; present a non-Spain figure as Spanish |
| Investment memo | Writes the verdict and site-by-site reasoning from the evidence pack | Use any number not in the evidence pack; change model outputs |
""")
    st.markdown("## 8 · Limitations")
    st.markdown("""
- **Store coverage:** existing stores come from the census by trade name; stores filed under company names are missed (OSM/Google cross-check next).
- **Demand:** residents + a daytime-activity proxy. No measured footfall yet (Google review counts next).
- **Distance:** straight-line, not walking network.
- **Sites:** vacant census units; size and availability unverified unless a listing matches.
- **Rent model:** brands where the franchisor provides the property (e.g. McDonald's Spain) are modelled with occupancy as % of
  sales; the recommended units are then demand evidence, not leasable sites.
- **Money:** revenue assumed proportional to captured demand (elasticity 1); EBITDA pre-tax, unlevered.
- **Not yet built:** survival model on 2014–2026 openings/closures; out-of-time backtest (train 2019 → predict 2024).
""")


pg = st.navigation([st.Page(page_analysis, title="Analysis", icon="🗺️", default=True),
                    st.Page(page_report, title="Investment report", icon="📑", url_path="report"),
                    st.Page(page_methodology, title="Methodology", icon="🧮", url_path="methodology")])
pg.run()
