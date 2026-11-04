"""Headless end-to-end run: model -> robustness MC -> Claude research -> Apify rents -> P&L MC -> Claude memo.

    python scripts/run_report.py "McDonald's" --sites 5

Reads keys from .streamlit/secrets.toml (or env). Writes runs/<brand>-<date>/ : memo.md, evidence.json, research.json, rents.json
"""
import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sec = ROOT / ".streamlit" / "secrets.toml"
if sec.exists():
    for k, v in re.findall(r'^(\w+)\s*=\s*"([^"]*)"', sec.read_text(), re.M):
        if v and not os.environ.get(k):
            os.environ[k] = v

import numpy as np  # noqa: E402
from mfi import reasoning, report, scrape  # noqa: E402
from mfi.model import pnl  # noqa: E402
from mfi.model.robustness import simulate as robustness  # noqa: E402
from mfi.model.site_model import load_data, run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("brand")
ap.add_argument("--sites", type=int, default=5)
ap.add_argument("--B", type=int, default=200)
ap.add_argument("--skip", default="", help="comma list: research,rents,memo")
ap.add_argument("--reuse", default="", help="comma list of saved steps to reuse from today's run dir: research,rents")
a = ap.parse_args()
skip = set(a.skip.split(","))
out = ROOT / "runs" / f"{re.sub(r'[^a-z0-9]+', '-', a.brand.lower()).strip('-')}-{date.today()}"
out.mkdir(parents=True, exist_ok=True)


def save(name, obj):
    (out / name).write_text(json.dumps(obj, default=str, ensure_ascii=False, indent=1))


print(f"[1/6] site model: {a.brand}")
R = run(load_data(), a.brand, a.sites)
print(f"      {R.metrics['stores']} stores · AUC {R.metrics['auc_spatial_cv']:.3f} · sites: " + "; ".join(R.sites.address))
print(f"[2/6] robustness Monte Carlo B={a.B}")
rob = robustness(R, B=a.B)
print(rob["summary"][["rank", "address", "p_selected", "demand_p50"]].round(2).to_string(index=False))

profile = None
reuse = set(a.reuse.split(","))
if "research" in reuse and (out / "research.json").exists():
    profile = json.loads((out / "research.json").read_text())
    print("[3/6] reusing saved research")
elif "research" not in skip and reasoning.available():
    print("[3/6] Claude franchise research (web search)…")
    profile = reasoning.research_franchise(a.brand)
    save("research.json", profile)
    print("      eligibility:", (profile.get("data") or {}).get("individual_franchise_available_spain"),
          "| sources:", len(profile.get("citations") or []), "| json parsed:", profile.get("data") is not None)
inputs = report.pnl_inputs_from_research(profile)

rents = {}
if "rents" in reuse and (out / "rents.json").exists():
    rents = {int(k): v for k, v in json.loads((out / "rents.json").read_text()).items()}
    print("[4/6] reusing saved rents")
elif "rents" not in skip and scrape.available():
    print("[4/6] Apify → Idealista rents near each site…")
    min_m2 = (((profile or {}).get("data") or {}).get("min_floor_area_m2") or {}).get("value")
    for r in R.sites.itertuples():
        try:
            rents[int(r.rank)] = scrape.rents_near(r.lat, r.lon, min_m2=min_m2)
            st = rents[int(r.rank)]["stats"]
            print(f"      #{r.rank} {r.address}: {st.get('n')} listings" + (f", median €{st['p50']:.1f}/m²" if "p50" in st else ""))
        except Exception as e:
            print(f"      #{r.rank} failed: {e}")
    save("rents.json", rents)
med = [v["stats"]["p50"] for v in rents.values() if "p50" in v["stats"]]
if med and report.rent_model(profile) == "market_lease":
    inputs.loc[inputs.key == "rent_eur_m2_month", ["low", "mode", "high", "source", "label"]] = [
        float(np.min(med)), float(np.median(med)), float(np.max(med)), "Idealista via Apify (site medians)", "listing_data"]

rm = report.rent_model(profile)
print(f"[5/6] P&L Monte Carlo (rent model: {rm})")
print(inputs[["parameter", "low", "mode", "high", "label"]].to_string(index=False))
po = pnl.simulate(R.sites, rob["ratio_draws"], inputs, site_rents={k: v["stats"] for k, v in rents.items()}, rent_model=rm)
if po["ok"]:
    print(po["summary"][["rank", "address", "revenue_p50", "ebitda_p10", "ebitda_p50", "ebitda_p90", "p_loss", "payback_p50_yrs"]].round(2).to_string(index=False))
else:
    print("      not run — missing:", po["missing"])

ev = report.evidence_pack(R, rob, po, rents, profile)
save("evidence.json", ev)
if "memo" not in skip and reasoning.available():
    print("[6/6] Claude investment memo…")
    memo = reasoning.write_report(ev)
    (out / "memo.md").write_text(memo or "")
    print(f"      {len(memo or '')} chars")
print("saved ->", out)
