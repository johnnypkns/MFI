"""Seize — Site Selection Engine.

    streamlit run app.py

Pages (top bar): Analyse · Brands · Settings · Report · Methodology  (+ hidden Unit page)
Reads only data/app/* — shareable without Postgres. Secrets: ANTHROPIC_API_KEY, APIFY_TOKEN, APP_PASSCODE.
"""
import json
import time

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from mfi import confidence, pipeline, quota, reasoning, report
from mfi.model.export import APP, norm_name
from mfi.model.rank import CAP_QUANTILE, MODEL_ERROR_SIGMA, RENT_OUTLIER, _pnl, rank
from mfi.model.site_model import FEATURE_LABELS, FEATURES, PARAMS, demand_cal, load_data, run, transform
from mfi.stores import inventory, load_osm, missing_from_census

st.set_page_config(page_title="Seize · Site Selection Engine", page_icon="🔴", layout="wide", initial_sidebar_state="collapsed")
RED, WHITE, GREY = [239, 35, 60], [245, 245, 247], [120, 120, 130]
SS = st.session_state
DATA_VERSION = "2026-11-05b"   # bump when the data model changes: forces a fresh load after a redeploy

st.markdown("""
<style>
:root {--red:#ef233c; --bg:#08080a; --card:#121216; --line:#24242b; --mute:#8b8b95; --txt:#f5f5f7;}
.block-container {padding-top: 4.2rem; max-width: 1500px;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display:none;}
.hero {margin-bottom:.6rem;}
.hero h1 {font-size: 2.1rem; letter-spacing: .22em; margin: 0; font-weight: 800; display:flex; align-items:center; gap:18px; color:var(--txt);}
.hero h1 .tag {font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size:.68rem; letter-spacing:.38em; color:var(--red);
  font-weight:500; border:1px solid var(--red); padding:4px 10px 4px 13px;}
.hero p {color: var(--mute); margin: .35rem 0 0 0;}
.mono {font-family: ui-monospace, "SF Mono", Menlo, monospace; letter-spacing:.08em;}
.muted {color:var(--mute); font-size:.85rem;}
.verdict {border:1px solid var(--line); border-left:3px solid var(--red); background:var(--card); padding:1rem 1.2rem; margin:.6rem 0 1rem 0;}
.verdict p {margin:.25rem 0; color:var(--txt);}
.verdict .lbl {font-family: ui-monospace, Menlo, monospace; font-size:.68rem; letter-spacing:.35em; color:var(--red); margin-bottom:.4rem;}
.badge {display:inline-block; font-family: ui-monospace, Menlo, monospace; font-size:.68rem; letter-spacing:.08em; padding:2px 8px;
  border:1px solid var(--line); margin-right:6px; color:var(--mute);}
.badge.high {border-color:#16a34a; color:#4ade80;} .badge.med {border-color:#ca8a04; color:#facc15;} .badge.low {border-color:var(--red); color:#fb7185;}
.card {border:1px solid var(--line); background:var(--card); padding:.85rem 1rem; margin:0 0 .35rem 0; transition: border-color .2s, transform .2s;}
.card:hover {border-color: var(--red); transform: translateX(3px);}
.card .top {display:flex; justify-content:space-between; align-items:baseline; gap:12px;}
.card .rk {font-family: ui-monospace, Menlo, monospace; color:var(--red); font-size:1.1rem; font-weight:700; margin-right:10px;}
.card .addr {font-weight:600; color:var(--txt);}
.card .kpi {font-family: ui-monospace, Menlo, monospace; color:var(--txt); font-size:.95rem; white-space:nowrap;}
.card .sub {color:var(--mute); font-size:.83rem; margin-top:.2rem;}
.card .more {max-height:0; overflow:hidden; transition:max-height .35s ease; }
.card:hover .more {max-height:260px;}
.card .grid {display:grid; grid-template-columns: repeat(3, 1fr); gap:6px 14px; margin-top:.7rem; padding-top:.6rem; border-top:1px dashed var(--line);}
.card .grid div {font-size:.8rem; color:var(--mute);} .card .grid b {display:block; color:var(--txt); font-family: ui-monospace, Menlo, monospace; font-weight:500;}
.legend span {display:inline-flex; align-items:center; margin-right:18px; font-size:.85rem; color:var(--mute);}
.dot {width:11px; height:11px; border-radius:50%; display:inline-block; margin-right:6px;}
.step {border-left: 3px solid var(--red); padding: .4rem .8rem; margin: .35rem 0; background: var(--card);}
.part {display:flex; align-items:center; gap:10px; margin:.35rem 0;}
.part .bar {flex:0 0 160px; height:6px; background:var(--line);} .part .bar i {display:block; height:6px; background:var(--red);}
div[data-testid="stMetricValue"] {font-size: 1.55rem; font-family: ui-monospace, Menlo, monospace;}
.stButton button[kind="secondary"] {border-radius:0; border-color:var(--line); font-family: ui-monospace, Menlo, monospace; font-size:.75rem; letter-spacing:.12em;}
.stButton button[kind="primary"] {border-radius:0; letter-spacing:.2em; font-family: ui-monospace, Menlo, monospace;}
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
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], header[data-testid="stHeader"], footer, [data-testid="stToolbar"] {display:none !important;}
.stApp {background:#08080a;
  background-image: linear-gradient(rgba(239,35,60,.09) 1px, transparent 1px), linear-gradient(90deg, rgba(239,35,60,.09) 1px, transparent 1px);
  background-size: 44px 44px;}
.block-container {max-width: 620px !important; padding-top: 15vh !important;}
.gate-title, .gate-h, .gate-sub, .gate-foot {white-space: nowrap;}
.gate-mark {width:54px; height:54px; border:2px solid #ef233c; transform: rotate(45deg); margin: 0 auto 34px auto; position:relative;}
.gate-mark:after {content:""; position:absolute; inset:12px; background:#ef233c; animation: pulse 2.4s ease-in-out infinite;}
@keyframes pulse {0%,100%{opacity:1} 50%{opacity:.25}}
.gate-title {font-family: ui-monospace, "SF Mono", Menlo, monospace; text-align:center; letter-spacing:.45em; font-size:.74rem;
  color:#ef233c; margin:0 0 6px 0; padding-left:.55em;}
.gate-h {text-align:center; font-weight:800; letter-spacing:.12em; font-size:1.75rem; line-height:1.25; color:#f5f5f7; margin:0;}
.gate-sub {text-align:center; color:#8b8b95; font-family: ui-monospace, Menlo, monospace; font-size:.72rem; letter-spacing:.3em;
  margin: 10px 0 38px 0;}
.gate-line {height:1px; background:linear-gradient(90deg, transparent, #ef233c, transparent); margin: 0 0 30px 0; position:relative; overflow:hidden;}
.gate-line:after {content:""; position:absolute; top:0; left:-30%; width:30%; height:1px; background:#08080a; animation: scan 2.8s linear infinite;}
@keyframes scan {to {left:130%}}
div[data-testid="stTextInput"] input {color:#f5f5f7 !important; text-align:center; font-family: ui-monospace, Menlo, monospace; letter-spacing:.5em; font-size:1.1rem;
  border:none !important; border-bottom:2px solid #ef233c !important; border-radius:0 !important; background:transparent !important; box-shadow:none !important;}
div[data-testid="stTextInput"] > div, div[data-testid="stTextInput"] div[data-baseweb="input"],
div[data-testid="stTextInput"] div[data-baseweb="base-input"] {border:none !important; background:transparent !important; box-shadow:none !important;}
div[data-testid="stTextInput"] {max-width: 380px; margin: 0 auto;}
div[data-testid="stTextInputRootElement"], div[data-testid="stTextInputRootElement"] * {background:transparent !important; border-color:transparent !important;}
div[data-testid="stTextInputRootElement"] {border-bottom:2px solid #ef233c !important; border-radius:0 !important;}
p.gate-h {font-size:3.4rem !important; letter-spacing:.28em !important; padding-left:.28em; line-height:1.2 !important; margin-bottom:4px !important;}
p.gate-title {font-size:.74rem !important;} p.gate-sub, p.gate-foot {font-size:.7rem !important;}
div[data-testid="stFormSubmitButton"] {max-width: 380px; margin: 8px auto 0 auto;}
div[data-testid="stFormSubmitButton"] button, .stButton button {width:100%; border-radius:0; background:#ef233c; color:#fff; border:none; letter-spacing:.4em; font-family: ui-monospace, Menlo, monospace;
  font-size:.8rem; padding:.75rem 0; transition: all .2s;}
div[data-testid="stFormSubmitButton"] button:hover, .stButton button:hover {background:#f5f5f7 !important; color:#08080a !important;}
div[data-testid="stFormSubmitButton"] button p {letter-spacing:.4em; font-size:.8rem; color:#fff;}
.gate-foot {text-align:center; color:#3f3f46; font-family: ui-monospace, Menlo, monospace; font-size:.65rem; letter-spacing:.3em; margin-top:46px;}
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
        st.markdown(f'<p class="gate-sub" style="color:#ef233c">LOCKED · RETRY IN {int(wait)+1}s</p>', unsafe_allow_html=True)
    if ok and entered:
        if hmac.compare_digest(entered.encode(), str(code).encode()):
            SS["authed"] = True
            SS.pop("fails", None)
            st.rerun()
        SS["fails"] = SS.get("fails", 0) + 1
        if SS["fails"] >= 5:
            SS["lock_until"], SS["fails"] = time.time() + 60, 0
        st.markdown('<p class="gate-sub" style="color:#ef233c">ACCESS DENIED</p>', unsafe_allow_html=True)
    st.markdown('<p class="gate-foot">© SEIZE · MADRID</p>', unsafe_allow_html=True)
    st.stop()


gate()


def hero(sub=""):
    st.markdown(f'<div class="hero"><h1>SEIZE<span class="tag">SITE SELECTION ENGINE</span></h1>'
                + (f"<p>{sub}</p>" if sub else "") + "</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ data (cached)
@st.cache_resource(show_spinner="Loading Madrid data…")
def data(version=DATA_VERSION):
    return load_data()


@st.cache_data(show_spinner=False)
def districts_geojson():
    return json.loads((APP / "districts.geojson").read_text())


@st.cache_data(show_spinner=False)
def registry(version=DATA_VERSION):
    reg = pd.read_parquet(APP / "brands.parquet")
    try:
        stats = pd.read_parquet(APP / "brand_model_stats.parquet")[["brand", "auc", "reliability"]]
        reg = reg.merge(stats, on="brand", how="left")
    except Exception:
        reg["auc"], reg["reliability"] = np.nan, None
    return reg


@st.cache_data(show_spinner=False)
def listings(version=DATA_VERSION):
    return pd.read_parquet(APP / "listings.parquet")


@st.cache_resource(show_spinner=False)
def osm_points():
    return load_osm()[0]


@st.cache_resource(show_spinner=False)
def model_for(brand, version=DATA_VERSION):
    reg = registry()
    row = reg[reg["brand"] == brand]
    extra = None
    if len(row) and row.iloc[0]["sector"] == "Food & drink":
        inv, _, _ = inventory(data(), brand, osm_points())
        extra = missing_from_census(inv)
    return run(data(), brand, 5, extra_stores=extra)


@st.cache_resource(show_spinner=False)
def ranked_for(brand, n, m2_range, floors, payback, econ_json, version=DATA_VERSION):
    econ = {k: tuple(v) for k, v in json.loads(econ_json).items()} if econ_json else None
    return rank(data(), model_for(brand), listings(), econ=econ, n=n, m2_range=m2_range, floors=floors, payback_target=payback)


def brand_status(row, research):
    """Label shown in the search list, before anyone clicks."""
    d = (research or {}).get("data") or {}
    if not row["franchisable"]:
        return "✗", "Not franchisable"
    if d.get("individual_franchise_available_spain") == "no":
        return "✗", "Not franchised in Spain"
    rel = row.get("reliability")
    if isinstance(rel, str) and rel in ("unreliable", "error"):
        return "⚠", "Low-confidence model"
    if d.get("individual_franchise_available_spain") == "yes":
        return "✓", "Franchisable"
    if d.get("individual_franchise_available_spain") == "unclear":
        return "?", "Franchising unclear"
    return "·", "Not yet verified"


# ------------------------------------------------------------------ settings (per brand)
FLOOR_OPTS = ["ground", "mezzanine", "unknown", "upper", "basement"]
ECON_LABELS = {"auv_eur": "Average revenue per store (€/yr)", "cogs_pct": "Food & packaging (% sales)",
               "labour_pct": "Labour incl. social security (% sales)", "royalty_pct": "Royalty (% sales)",
               "marketing_pct": "Marketing fund (% sales)", "other_opex_pct": "Other opex (% sales)",
               "investment_eur": "Initial investment (€)", "occupancy_pct": "Occupancy paid to franchisor (% sales)"}


def settings_for(brand):
    """Effective settings: research defaults, then the user's overrides from the Settings page."""
    prof = pipeline.cached_research(brand)
    d = (prof or {}).get("data") or {}
    _, _, rent_model, m2_suggest = pipeline.econ_from_profile(prof)
    user = SS.setdefault("settings", {}).get(brand, {})
    m2 = user.get("m2_range") or m2_suggest or pipeline.DEFAULT_M2
    src = (d.get("min_floor_area_m2") or {}).get("source")
    url = (src or "").split(" ")[0] if src and src.startswith("http") else None
    m2_source = "your setting" if user.get("m2_range") else (
        (f'franchisor requirement — <a href="{url}" target="_blank">source ↗</a>' if url else "franchisor requirement (research)")
        if m2_suggest else "default — no franchisor figure found yet")
    return {"profile": prof, "m2_range": tuple(int(v) for v in m2), "m2_source": m2_source,
            "floors": tuple(user.get("floors") or ("ground", "mezzanine", "unknown")),
            "payback": float(user.get("payback") or 4.0), "overrides": user.get("econ") or {}, "rent_model": rent_model}


def safe_research(brand):
    """Run Claude research; never crash the app on API problems (e.g. no credit)."""
    try:
        prof = reasoning.research_franchise(brand)
    except Exception as e:
        msg = str(e)
        if "credit balance" in msg.lower():
            st.error("Claude research is unavailable: the Anthropic account has run out of credit. "
                     "Top up at console.anthropic.com → Plans & Billing. Ranking continues without brand economics.")
        else:
            st.error(f"Claude research failed ({type(e).__name__}). Ranking continues without brand economics.")
        return None
    if prof and prof.get("data"):
        pipeline.save_research(brand, prof)
    return prof


# ------------------------------------------------------------------ the analysis run, with a live checklist
def analyse(brand, n, allow_research=True):
    cfg = settings_for(brand)
    t0 = time.time()
    with st.status(f"Analysing {brand}…", expanded=True) as box:
        # 1 research
        prof = cfg["profile"]
        if prof is None and allow_research and reasoning.available():
            if quota.allow("claude", SS):
                st.write("⧗ **Researching the franchise** — Claude is reading the official site, directories and press (~1 min)…")
                prof = safe_research(brand)
                cfg = settings_for(brand)
                cfg["profile"] = prof
            else:
                st.write("⚠ Daily research limit reached — continuing without brand economics.")
        d = (prof or {}).get("data") or {}
        st.write(f"✓ **Franchise research** — " + (f"{len((prof or {}).get('citations') or [])} sources · franchising in Spain: "
                                                    f"{d.get('individual_franchise_available_spain', '?')}" if prof else "not available"))
        # 2 stores, competitors, location model
        R = model_for(brand)
        m = R.metrics
        st.write(f"✓ **Stores & competitors** — {m['stores']} {brand} stores (census + OpenStreetMap) · {m['competitors']:,} competitors")
        st.write(f"✓ **Location model** — accuracy {m['auc_spatial_cv']:.2f} AUC ({m['reliability']}), districts held out")
        # 3 screen + simulate + rank
        econ, missing, rent_model, _ = pipeline.econ_from_profile(prof, cfg["overrides"])
        st.write(f"⧗ **Screening {len(listings()):,} units for rent** and simulating each one's P&L…")
        K = ranked_for(brand, n, cfg["m2_range"], cfg["floors"], cfg["payback"],
                       json.dumps({k: list(v) for k, v in econ.items()}) if econ else "")
        if K.get("ok"):
            st.write(f"✓ **Screened** — {K['log']['listings']:,} units → {K['log']['pass_suitability_gate']:,} qualify "
                     f"({cfg['m2_range'][0]}–{cfg['m2_range'][1]} m², {', '.join(cfg['floors'])})")
            st.write("✓ **Ranked** — " + K["metric"] + " · cannibalisation re-scored after every pick")
            conf = confidence.score_units(R, K["sites"], prof, econ, cfg["overrides"])
            st.write("✓ **Confidence scored** — model · verified inputs · demand stability · listing quality")
        else:
            conf = []
            st.write("✗ " + K.get("reason", "No unit qualifies."))
        box.update(label=f"{brand} — analysed in {time.time() - t0:.0f}s", state="complete", expanded=False)
    O = {"R": R, "ranked": K, "econ": econ, "missing": missing, "rent_model": rent_model, "m2_range": cfg["m2_range"],
         "profile": prof, "payback": cfg["payback"], "conf": conf, "brand": brand, "m2_source": cfg["m2_source"]}
    SS.update(out=O, brand=brand, n_sites=n)
    SS.pop("memo", None)
    return O


def eur(v):
    return confidence.eur(v)


def conf_badge(c):
    g = confidence.grade(c)
    cls = {"High": "high", "Medium": "med", "Low": "low"}[g]
    return f'<span class="badge {cls}">CONFIDENCE {c}/100 · {g.upper()}</span>'


# ================================================================== ANALYSE
def page_analyse():
    hero("Pick a franchise. Seize researches its economics, screens every commercial unit for rent in Madrid, "
         "simulates each unit's P&L and ranks them by risk-adjusted return.")
    reg = registry()
    opts = reg[reg["n_stores"] >= 3].copy()
    research = {b: pipeline.cached_research(b) for b in opts["brand"].head(400)}
    stat = {r.brand: brand_status(r._asdict(), research.get(r.brand)) for r in opts.itertuples()}
    c0, c1, c2, c3 = st.columns([1.3, 3.3, 1.0, 1.0])
    sectors = ["All sectors"] + opts.groupby("sector")["n_stores"].sum().sort_values(ascending=False).index.tolist()
    sector = c0.selectbox("Sector", sectors, label_visibility="collapsed")
    if sector != "All sectors":
        opts = opts[opts["sector"] == sector]
    names = opts["brand"].tolist()
    lab = {r.brand: f"{stat[r.brand][0]}  {r.brand}  ·  {r.n_stores} stores  ·  {stat[r.brand][1]}" for r in opts.itertuples()}
    cur = SS.get("brand", "Telepizza")
    brand = c1.selectbox("Franchise", names, index=names.index(cur) if cur in names else 0, format_func=lambda b: lab.get(b, b),
                         label_visibility="collapsed")
    n_sites = c2.selectbox("Sites", [3, 5, 10], index=[3, 5, 10].index(SS.get("n_sites", 5)), label_visibility="collapsed",
                           format_func=lambda n: f"Top {n}")
    go = c3.button("ANALYSE →", type="primary", width="stretch")
    cfg = settings_for(brand)
    st.markdown(f'<p class="muted">Units {cfg["m2_range"][0]}–{cfg["m2_range"][1]} m² <i>({cfg["m2_source"]})</i> · '
                f'floors: {", ".join(cfg["floors"])} · payback target {cfg["payback"]:.0f} yrs</p>', unsafe_allow_html=True)
    st.page_link(PAGES["settings"], label="Adjust the model in Settings →")

    if go or ("out" not in SS) or (SS.get("out", {}).get("brand") != brand and go):
        try:
            analyse(brand, n_sites)
        except ValueError as e:
            st.error(str(e))
            return
    O = SS.get("out")
    if not O or O["brand"] != brand:
        st.info(f"Press **ANALYSE** to run {brand}.")
        return
    R, K, m, conf = O["R"], O["ranked"], O["R"].metrics, O["conf"]

    st.markdown('<div class="verdict"><div class="lbl">VERDICT</div>' +
                "".join(f"<p>{md_bold(l)}</p>" for l in confidence.verdict(brand, O, conf)) + "</div>", unsafe_allow_html=True)
    for w in R.warnings:
        (st.error if w.startswith("UNRELIABLE") else st.warning)(w)
    if not O["econ"]:
        miss = ", ".join(ECON_LABELS.get(x, x).split(" (")[0].lower() for x in O["missing"])
        st.info(("Claude's research found no published figure for: **" + miss + "**." if O.get("profile")
                 else "This franchise hasn't been researched yet.") +
                " Units are ranked by customers per euro of rent instead of payback. You can enter the figures in **Settings**.")
    if not K.get("ok"):
        st.error(K.get("reason"))
        return
    S = K["sites"]
    cmap = {c["rank"]: c for c in conf}

    left, right = st.columns([1.05, 1.25], gap="large")
    with left:
        st.markdown('<p class="mono muted">RANKED UNITS · hover for details</p>', unsafe_allow_html=True)
        for _, s in S.iterrows():
            c = cmap.get(int(s["rank"]), {"confidence": 0})
            if O["econ"]:
                kpi = f"{s['p_payback']:.0%} <span class='muted'>pay back ≤{O['payback']:.0f}y</span>"
                grid = [("EBITDA / yr", f"{eur(s['ebitda_p10'])} – {eur(s['ebitda_p90'])}"),
                        ("Middle case", eur(s["ebitda_p50"])), ("Payback (mid)", f"{s['payback_p50']:.1f} yrs"),
                        ("Revenue (mid)", eur(s["revenue_p50"])), ("Investment", eur(s["invest_p50"])),
                        ("P(loss)", f"{s['p_loss']:.0%}")]
            else:
                kpi = f"{s['demand_ratio']:.2f}× <span class='muted'>demand</span>"
                grid = []
            grid += [("Demand vs typical store", f"{s['demand_ratio']:.2f}×"), ("Traspaso", eur(s["traspaso_eur"]) if s["traspaso_eur"] else "none"),
                     (f"Nearest {brand}", f"{s['dist_nearest_brand_m']:,.0f} m")]
            cells = "".join(f"<div>{a}<b>{b}</b></div>" for a, b in grid)
            st.markdown(f"""<div class="card"><div class="top"><div><span class="rk">#{int(s['rank'])}</span>
<span class="addr">{s['address']}</span></div><div class="kpi">{kpi}</div></div>
<div class="sub">{s['m2']:.0f} m² · {eur(s['rent_month'])}/month · {s['eur_m2']:.1f} €/m² {conf_badge(c['confidence'])}</div>
<div class="more"><div class="grid">{cells}</div></div></div>""", unsafe_allow_html=True)
            if st.button(f"OPEN UNIT #{int(s['rank'])} →", key=f"open{int(s['rank'])}"):
                SS["unit_rank"] = int(s["rank"])
                st.switch_page(PAGES["unit"])
    with right:
        show_comp = st.toggle("Show competitors", value=False)
        legend = ['<span><i class="dot" style="background:#ef233c"></i>Recommended unit</span>',
                  f'<span><i class="dot" style="background:#f5f5f7"></i>Existing {brand}</span>']
        if show_comp:
            legend.append('<span><i class="dot" style="background:#787882"></i>Competitor</span>')
        st.markdown('<div class="legend">' + "".join(legend) + "</div>", unsafe_allow_html=True)
        st.pydeck_chart(map_deck(R, S, show_comp), height=620)

    t1, t2, t3 = st.tabs(["Filter funnel", "Location model", "Demand calibration"])
    with t1:
        f = pd.DataFrame({"step": [s.replace("_", " ") for s in K["log"]], "units": list(K["log"].values())})
        st.altair_chart(alt.Chart(f).mark_bar(color="#ef233c").encode(
            x=alt.X("units:Q"), y=alt.Y("step:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=["step", "units"]).properties(height=36 * len(f)), width="stretch")
    with t2:
        a, b = st.columns([1.2, 1])
        coef = R.coefficients.copy()
        coef["direction"] = coef["effect"].map(lambda v: "more likely" if v > 0 else "less likely")
        a.altair_chart(alt.Chart(coef).mark_bar().encode(
            x=alt.X("effect:Q", title="effect on log-odds (per std. dev.)"),
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["more likely", "less likely"], range=["#ef233c", "#787882"]),
                            legend=None)).properties(height=300), width="stretch")
        b.write(f"Spatial-CV AUC **{m['auc_spatial_cv']:.3f}** ({m['reliability']}) · folds {m['auc_folds']}")
        b.write(f"{m['stores']} stores vs {m['background_n']:,} background premises · activity `{m['category']}`")
    with t3:
        cal = demand_cal(data()) or {}
        cc = pd.DataFrame({"Demand measure": ["Residents only", "Residents + every premises (old)", "Consumer + office premises (now)"],
                           "Match with real footfall": [cal.get("baseline_residents_only"), cal.get("baseline_v0_all_premises"),
                                                        cal.get("spearman_in_sample")]})
        st.dataframe(cc, hide_index=True, width="stretch", column_config={"Match with real footfall":
                     st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
        st.caption(f"Rank correlation with the city's {cal.get('n_sensors', 31)} pedestrian counters; held-out "
                   f"{cal.get('spearman_leave_one_out', 0):.2f}.")


def md_bold(s):
    import re
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)


def map_deck(R, S, show_comp=False, focus=None, zoom=11.2):
    sites = S.copy()
    sites["label"] = sites["rank"].astype(int).astype(str)
    sites["tip"] = "<b>#" + sites["label"] + " " + sites["address"].fillna("") + "</b><br/>" + sites["m2"].round(0).astype(int).astype(str) \
        + " m² · €" + sites["rent_month"].round(0).astype(int).astype(str) + "/month"
    stores = R.stores.copy()
    stores["tip"] = "<b>" + stores["trade_name"].fillna(R.brand).astype(str) + "</b>"
    layers = [pdk.Layer("GeoJsonLayer", districts_geojson(), stroked=True, filled=False,
                        get_line_color=[140, 140, 150, 90], line_width_min_pixels=1)]
    if show_comp:
        comp = R.competitors.copy()
        comp["tip"] = "<b>" + comp["trade_name"].fillna("Competitor").astype(str) + "</b>"
        layers.append(pdk.Layer("ScatterplotLayer", comp, get_position=["lon", "lat"], get_fill_color=GREY + [170],
                                get_radius=30, radius_min_pixels=2, radius_max_pixels=6, pickable=True))
    layers += [
        pdk.Layer("ScatterplotLayer", stores, get_position=["lon", "lat"], get_fill_color=WHITE + [235],
                  get_radius=60, radius_min_pixels=4, radius_max_pixels=8, pickable=True),
        pdk.Layer("ScatterplotLayer", sites, get_position=["lon", "lat"], get_fill_color=RED + [240],
                  get_line_color=[8, 8, 10], line_width_min_pixels=2, stroked=True, get_radius=150, radius_min_pixels=11, radius_max_pixels=14, pickable=True),
        pdk.Layer("TextLayer", sites, get_position=["lon", "lat"], get_text="label", get_color=[255, 255, 255],
                  get_size=13, get_alignment_baseline="'center'", font_weight=800),
    ]
    lat, lon = (focus if focus else (40.43, -3.69))
    return pdk.Deck(layers=layers, map_provider="carto", map_style="dark",
                    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom),
                    tooltip={"html": "{tip}", "style": {"fontSize": "12px", "backgroundColor": "#121216", "color": "#f5f5f7"}})


# ================================================================== UNIT PAGE
def page_unit():
    O = SS.get("out")
    if not O or not O["ranked"].get("ok") or "unit_rank" not in SS:
        hero()
        st.info("Open a unit from the **Analyse** page.")
        st.page_link(PAGES["analyse"], label="← Back to Analyse")
        return
    R, K = O["R"], O["ranked"]
    S = K["sites"]
    s = S[S["rank"] == SS["unit_rank"]].iloc[0]
    c = next((x for x in O["conf"] if x["rank"] == int(s["rank"])), None)
    hero()
    st.page_link(PAGES["analyse"], label=f"← All {O['brand']} units")
    st.markdown(f"## #{int(s['rank'])} · {s['address']}")
    st.markdown((conf_badge(c["confidence"]) if c else "") + f'<span class="badge">{O["brand"].upper()}</span>'
                f'<span class="badge">{str(s.get("neighbourhood") or "").upper()}</span>'
                f'<a href="{s["url"]}" target="_blank"><span class="badge">IDEALISTA LISTING ↗</span></a>', unsafe_allow_html=True)

    k = st.columns(5)
    k[0].metric("Rent / month", eur(s["rent_month"]))
    k[1].metric("Size", f"{s['m2']:.0f} m²")
    rel = s["eur_m2"] / s["local_eur_m2"] if s.get("local_eur_m2") else np.nan
    k[2].metric("€/m² vs area", f"{s['eur_m2']:.1f}", f"{(rel - 1):+.0%} vs nearby" if np.isfinite(rel) else None, delta_color="inverse")
    k[3].metric("Traspaso", eur(s["traspaso_eur"]) if s["traspaso_eur"] else "none")
    k[4].metric("Demand vs typical store", f"{s['demand_ratio']:.2f}×")

    a, b = st.columns([1.1, 1], gap="large")
    with a:
        st.markdown("#### The money")
        if O["econ"]:
            rng = np.random.default_rng(int(s["rank"]) * 7)
            rev, e, pb, inv = _pnl(max(s["demand_ratio"], 0), s["rent_year"], s["traspaso_eur"], O["econ"], rng, 10_000)
            st.markdown(f"**EBITDA per year:** {eur(np.percentile(e, 10))} (pessimistic) · **{eur(np.median(e))}** (middle) · "
                        f"{eur(np.percentile(e, 90))} (optimistic)  \n**Payback:** {np.percentile(pb, 25):.1f}–{np.percentile(pb, 75):.1f} yrs "
                        f"(middle {np.median(pb):.1f}) · **P(payback ≤ {O['payback']:.0f}y): {(pb <= O['payback']).mean():.0%}** · "
                        f"P(loss): {(e <= 0).mean():.0%}  \n**Investment:** {eur(np.median(inv))} (brand + traspaso)")
            h = pd.DataFrame({"EBITDA (€/yr)": e})
            st.altair_chart(alt.Chart(h).mark_bar(color="#ef233c", opacity=.85).encode(
                x=alt.X("EBITDA (€/yr):Q", bin=alt.Bin(maxbins=45)), y=alt.Y("count()", title="simulations")).properties(height=220),
                width="stretch")
            st.caption("10,000 simulations: brand revenue and costs drawn from cited ranges, this unit's own rent and traspaso, "
                       f"±{MODEL_ERROR_SIGMA:.0%} model error.")
        else:
            st.info("No brand economics — add them in **Settings** to see this unit's P&L.")
        st.markdown("#### Why this unit")
        for line in why_lines(R, s):
            st.markdown(f"- {line}")
    with b:
        st.markdown("#### Confidence")
        if c:
            for kpart, v in c["parts"].items():
                st.markdown(f'<div class="part"><span style="flex:0 0 190px">{confidence.LABELS[kpart]}</span>'
                            f'<div class="bar"><i style="width:{v*100:.0f}%"></i></div><span class="mono">{v*100:.0f}</span>'
                            f'<span class="muted">×{confidence.WEIGHTS[kpart]:.0%}</span></div>'
                            f'<div class="muted" style="margin:-4px 0 6px 0">{c["notes"][kpart]}</div>', unsafe_allow_html=True)
        st.markdown("#### Around the unit")
        st.pydeck_chart(map_deck(R, S[S["rank"] == s["rank"]], show_comp=True, focus=(s["lat"], s["lon"]), zoom=14.3), height=330)
        st.markdown("#### Rent vs comparable units (≤ 1 km)")
        L = listings()
        near = L[np.hypot(L["x"] - s["x"], L["y"] - s["y"]) <= 1000]
        if len(near) >= 3:
            ch = alt.Chart(near).mark_bar(color="#3f3f46").encode(x=alt.X("eur_m2:Q", bin=alt.Bin(maxbins=25), title="€/m²/month"),
                                                                   y=alt.Y("count()", title="listings"))
            rule = alt.Chart(pd.DataFrame({"v": [s["eur_m2"]]})).mark_rule(color="#ef233c", size=3).encode(x="v:Q")
            st.altair_chart((ch + rule).properties(height=180), width="stretch")
            st.caption(f"{len(near)} listings within 1 km · median {near['eur_m2'].median():.1f} €/m² · this unit {s['eur_m2']:.1f} (red).")
    if s.get("description"):
        with st.expander("Listing description (Idealista)"):
            st.write(s["description"])


def why_lines(R, s):
    """Compare the unit with the brand's existing stores, feature by feature (data, not opinion)."""
    fs = R.ctx["fs"]
    store_f = fs.compute(R.ctx["stores_xy"])
    out = []
    for f, label in [("footfall", "Footfall"), ("income_1000", "Income within 1 km"), ("pop_1000", "Residents within 1 km"),
                     ("comp_500", "Competitors within 500 m"), ("transit_300", "Metro/rail stations within 300 m"),
                     ("office_500", "Offices within 500 m")]:
        if f not in s or f not in store_f:
            continue
        pct = (store_f[f] < s[f]).mean()
        val = s[f]
        fmt = f"€{val:,.0f}" if f == "income_1000" else f"{val:,.0f}"
        out.append(f"**{label}:** {fmt} — higher than {pct:.0%} of existing {R.brand} stores")
    coef = dict(zip(R.coefficients["feature"], R.coefficients["effect"]))
    top = R.coefficients.head(2)["feature"].tolist()
    out.append("The model weighs most: " + " and ".join(top).lower() + ".")
    return out


# ================================================================== SETTINGS
def page_settings():
    hero("Adjust the model. Every value starts from cited research; anything you change is marked as your assumption.")
    brand = SS.get("brand", "Telepizza")
    reg = registry()
    names = reg[reg["n_stores"] >= 3]["brand"].tolist()
    brand = st.selectbox("Brand", names, index=names.index(brand) if brand in names else 0)
    cfg = settings_for(brand)
    prof = cfg["profile"]
    d = (prof or {}).get("data") or {}
    user = SS.setdefault("settings", {}).setdefault(brand, {})

    st.markdown("### Unit")
    a, b, c = st.columns(3)
    lo = a.number_input("Minimum size (m²)", 10, 5000, int(cfg["m2_range"][0]), step=10)
    hi = b.number_input("Maximum size (m²)", 20, 10000, int(cfg["m2_range"][1]), step=10)
    floors = c.multiselect("Floors allowed", FLOOR_OPTS, list(cfg["floors"]))
    st.markdown(f'<p class="muted">Source: {cfg["m2_source"]}</p>', unsafe_allow_html=True)
    a, b = st.columns(3)[:2]
    payback = a.selectbox("Payback target", [2.0, 3.0, 4.0, 5.0, 7.0], index=[2.0, 3.0, 4.0, 5.0, 7.0].index(cfg["payback"]),
                          format_func=lambda v: f"{v:.0f} years")

    st.markdown("### Brand economics")
    src = d.get("pnl_inputs") or {}
    rows = []
    keys = ["auv_eur", "cogs_pct", "labour_pct", "royalty_pct", "marketing_pct", "other_opex_pct", "investment_eur"] + \
        (["occupancy_pct"] if cfg["rent_model"] == "franchisor_property" else [])
    for k in keys:
        v = cfg["overrides"].get(k) or src.get(k) or {}
        rows.append({"key": k, "Input": ECON_LABELS[k], "Low": v.get("low"), "Mid": v.get("mode"), "High": v.get("high"),
                     "Source": "your assumption" if k in cfg["overrides"] else (f"included in {v['included_in']}" if v.get("included_in")
                                                                               else (v.get("label") or "—")),
                     "Link": (v.get("source") or "").split(" ")[0] if k not in cfg["overrides"] else ""})
    df = pd.DataFrame(rows)
    ed = st.data_editor(df, hide_index=True, width="stretch", disabled=["key", "Input", "Source", "Link"], key=f"econ_{brand}",
                        column_config={"key": None, "Link": st.column_config.LinkColumn("Source link", display_text="open ↗")})
    if not prof:
        st.caption("Not researched yet — run the brand once on Analyse, or type the figures yourself.")

    x, y, _ = st.columns([1, 1, 3])
    if x.button("APPLY & RE-RUN", type="primary"):
        user["m2_range"] = (int(lo), int(hi)) if (int(lo), int(hi)) != tuple(cfg["m2_range"]) or "m2_range" in user else None
        user["floors"] = floors
        user["payback"] = payback
        changed = {}
        for (_, r0), (_, r1) in zip(df.iterrows(), ed.iterrows()):
            if any((pd.isna(r0[c]) != pd.isna(r1[c])) or (not pd.isna(r1[c]) and r0[c] != r1[c]) for c in ["Low", "Mid", "High"]):
                if not pd.isna(r1["Mid"]):
                    changed[r1["key"]] = {"low": r1["Low"] if not pd.isna(r1["Low"]) else r1["Mid"], "mode": r1["Mid"],
                                          "high": r1["High"] if not pd.isna(r1["High"]) else r1["Mid"], "label": "user_assumption"}
        user["econ"] = {**(user.get("econ") or {}), **changed}
        if user.get("m2_range") is None:
            user.pop("m2_range")
        SS["brand"] = brand
        SS.pop("out", None)
        st.switch_page(PAGES["analyse"])
    if y.button("RESET TO RESEARCH"):
        SS["settings"].pop(brand, None)
        SS.pop("out", None)
        st.rerun()


# ================================================================== REPORT
def page_report():
    hero("The reasoning behind the recommendation: franchise research, the data trail and the investment memo.")
    O = SS.get("out")
    if not O:
        st.info("Run an analysis on **Analyse** first.")
        return
    st.markdown(f"### {O['brand']}")
    if st.button("WRITE INVESTMENT MEMO", type="primary", disabled=not reasoning.available() or not O["ranked"].get("ok")):
        if quota.allow("claude", SS):
            with st.spinner("Claude is writing the memo from the evidence pack…"):
                try:
                    SS["memo"] = reasoning.write_report(report.evidence_from_pipeline(O))
                except Exception as e:
                    st.error("Claude is unavailable" + (": the Anthropic account has run out of credit." if "credit" in str(e).lower()
                                                        else f" ({type(e).__name__})."))
        else:
            st.error("Daily limit reached.")
    t1, t2, t3 = st.tabs(["Investment memo", "How we got here", "Franchise research"])
    with t1:
        st.markdown(SS.get("memo") or "_Press **Write investment memo**. Claude may only use numbers from the evidence pack._")
    with t2:
        st.caption("Generated by Python from this exact run — not written by an AI.")
        for i, (step, detail) in enumerate(report.trail_from_pipeline(O), 1):
            st.markdown(f'<div class="step"><b>{i}. {step}</b><br>{detail}</div>', unsafe_allow_html=True)
    with t3:
        p = O.get("profile")
        if p:
            st.markdown(p.get("summary") or "")
            if p.get("citations"):
                st.markdown("**Sources:** " + " · ".join(f"[{c.get('title') or c['url']}]({c['url']})" for c in p["citations"][:20]))
        else:
            st.caption("No research for this brand yet.")


# ================================================================== PAGE — BRAND DIRECTORY
def page_directory():
    hero("Every chain operating in Madrid, detected from the municipal premises census and OpenStreetMap.")
    reg = registry()
    research = {b: pipeline.cached_research(b) for b in reg["brand"].head(400)}
    reg["status"] = [" ".join(brand_status(r._asdict(), research.get(r.brand))) for r in reg.itertuples()]
    k = st.columns(4)
    k[0].metric("Brands detected", f"{len(reg):,}")
    k[1].metric("Franchisable sectors", f"{int(reg['franchisable'].sum()):,}")
    k[2].metric("Food & drink brands", f"{int((reg['sector'] == 'Food & drink').sum()):,}")
    k[3].metric("Seen in OpenStreetMap too", f"{int((reg['n_osm'] > 0).sum()):,}")
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
    st.dataframe(t[["status", "brand", "sector", "subcategory", "n_stores", "n_osm", "auc", "not_franchisable_reason"]],
                 hide_index=True, width="stretch", height=520, column_config={
                     "status": "Status", "auc": st.column_config.NumberColumn("Model accuracy", format="%.2f"),
                     "brand": "Brand", "sector": "Sector", "subcategory": "Main activity", "n_stores": "Stores (census)",
                     "n_osm": "In OpenStreetMap",                      "not_franchisable_reason": "Why not franchisable"})
    st.caption("How it's built: names on open premises → generic names removed (activity words, sign types like "
               "'muestra opaca') → chains = ≥3 distinct locations → spelling variants merged → sector from licences → "
               "cross-checked with OpenStreetMap. Google Maps confirmation comes next.")
    with st.expander("Brands by sector"):
        by = reg.groupby("sector").agg(brands=("brand_id", "count"), stores=("n_stores", "sum")).reset_index()
        st.altair_chart(alt.Chart(by).mark_bar(color="#ef233c").encode(
            x=alt.X("brands:Q", title="brands"), y=alt.Y("sector:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
            tooltip=["sector", "brands", "stores"]).properties(height=30 * len(by)), width="stretch")



# ================================================================== PAGE 3 — METHODOLOGY
def page_methodology():
    hero("Exactly what runs behind every number: data, scraping, models, simulations, formulas and their limits.")
    D = data()
    cal = demand_cal(D) or {}
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




PAGES = {
    "analyse": st.Page(page_analyse, title="Analyse", default=True),
    "brands": st.Page(page_directory, title="Brands", url_path="brands"),
    "settings": st.Page(page_settings, title="Settings", url_path="settings"),
    "report": st.Page(page_report, title="Report", url_path="report"),
    "methodology": st.Page(page_methodology, title="Methodology", url_path="methodology"),
    "unit": st.Page(page_unit, title="Unit", url_path="unit", visibility="hidden"),
}
st.navigation(list(PAGES.values()), position="top").run()
