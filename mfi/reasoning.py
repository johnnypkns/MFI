"""Claude reasoning layer. Kept separate from the maths: Claude researches inputs and explains outputs; it never
computes or alters a model number.

  research_franchise(name)   web research -> cited franchise profile + P&L input ranges (low/mode/high, source, label)
  write_report(evidence)     full investment memo from the evidence pack (model + Monte Carlo + P&L + scrapes + research).
                             Only figures present in the evidence pack may appear.

Needs ANTHROPIC_API_KEY (env var or .streamlit/secrets.toml). Without it the dashboard runs; these return None.
"""
import json
import os
import re

MODEL = os.environ.get("MFI_CLAUDE_MODEL", "claude-sonnet-5-5")


def _key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        try:
            import streamlit as st
            k = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            k = None
    return k or None


def _client():
    k = _key()
    if not k:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=k)


def available() -> bool:
    return _key() is not None


def _text(resp):
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _json_block(s):
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except json.JSONDecodeError:
        return None


def _citations(resp):
    """Inline citations first (sources actually quoted), then every page the searches returned."""
    out, seen = [], set()

    def add(url, title, kind):
        if url and url not in seen:
            seen.add(url)
            out.append({"url": url, "title": title or "", "kind": kind})
    for b in resp.content:
        for c in getattr(b, "citations", None) or []:
            add(getattr(c, "url", None), getattr(c, "title", ""), "cited")
    for b in resp.content:
        if getattr(b, "type", "") == "web_search_tool_result":
            for r in getattr(b, "content", None) or []:
                add(getattr(r, "url", None), getattr(r, "title", ""), "searched")
    return out


RESEARCH_PROMPT = """You are a franchise investment analyst. Research "{name}" for a prospective INDIVIDUAL franchisee
opening ONE unit in Madrid, Spain. Use web search. Prefer, in order: official franchise sites and recruitment pages,
company filings / annual reports / investor presentations, franchise disclosure documents, Spanish franchise
directories (Tormo, mundoFranquicia, AEF), reputable press.

Hard rules
- Never invent a number. Unknown -> null.
- Each figure: source URL + label "verified" (official/filing) or "third_party_estimate". Say if it is not Spain-specific
  (e.g. US FDD) — still useful, but flagged.
- State plainly whether an individual can franchise this brand IN SPAIN, or whether it is company-run / a master franchisee.
- Determine WHO PROVIDES THE PREMISES: does the franchisee find and lease the unit on the open market, or does the
  franchisor select/own the site and charge occupancy (rent) as % of sales? This changes the whole economics.
- P&L cost lines MUST be mutually exclusive: cogs, labour, royalty, marketing, other_opex, occupancy. If a source
  bundles one line inside another (e.g. labour inside "other operating expenses"), set the bundled line to
  {{"low":0,"mode":0,"high":0}} with "included_in": "<the line that contains it>". Never double count.
- For P&L inputs give a range (low, mode, high). Cost ratios as % of sales. If only an industry benchmark exists for the
  segment (e.g. quick-service restaurants in Spain), you may use it but label it "third_party_estimate" and say so in source.

Return a markdown summary (max 220 words), then exactly one ```json block:
{{"brand": str,
  "individual_franchise_available_spain": "yes"|"no"|"unclear",
  "eligibility_explanation": str,
  "operator_in_spain": str|null,
  "spain_store_count": {{"value": int|null, "year": int|null, "source": str|null}},
  "franchise_fee_eur": {{"value": num|null, "source": str|null, "label": str|null}},
  "min_floor_area_m2": {{"value": num|null, "source": str|null}},
  "rent_model": "market_lease"|"franchisor_property"|"unclear",
  "rent_model_explanation": str,
  "pnl_inputs": {{
     "auv_eur":        {{"low": num|null, "mode": num|null, "high": num|null, "source": str|null, "label": str|null, "note": str|null}},
     "unit_m2":        {{...same...}},
     "cogs_pct":       {{...}}, "labour_pct": {{...}}, "royalty_pct": {{...}}, "marketing_pct": {{...}},
     "other_opex_pct": {{...}}, "investment_eur": {{...}},
     "occupancy_pct": {{...only if rent_model is franchisor_property: rent/occupancy paid to franchisor as % of sales...}}
     (every item may also carry "included_in": str|null)
  }},
  "notes": [str]}}"""


def research_franchise(name: str, max_searches: int = 10):
    c = _client()
    if c is None:
        return None
    resp = c.messages.create(
        model=MODEL, max_tokens=6000,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": max_searches}],
        messages=[{"role": "user", "content": RESEARCH_PROMPT.format(name=name)}])
    txt = _text(resp)
    return {"summary": re.split(r"```json", txt)[0].strip(), "data": _json_block(txt),
            "citations": _citations(resp), "model": MODEL,
            "searches": getattr(getattr(resp.usage, "server_tool_use", None), "web_search_requests", None)}


REPORT_PROMPT = """You are the lead analyst writing the investment memo behind a franchise site-selection dashboard.
Brand: "{brand}", city: Madrid. Below is the EVIDENCE PACK — everything the platform computed or collected.

Absolute rules
- Use ONLY numbers that appear in the evidence pack. Never compute new figures, never estimate, never round in a way
  that changes meaning. If something is missing, say it is missing and what would fill it.
- Attribute every claim to its method: "the suitability model", "the robustness Monte Carlo (B draws)", "the P&L Monte
  Carlo (N draws)", "the Idealista scrape (n listings)", "Claude's franchise research (source)", "the municipal census".
- Be decisive but honest. A site can be Recommended, Conditional (state the condition) or Not recommended.
- Plain English, no hype, no filler. Use markdown.

Structure
## Verdict
3–5 sentences: can an individual open this franchise in Madrid, which site is best and why, how confident we are.
## How we got here
Numbered steps, one line each, naming the method and its key number (data used -> model -> simulation -> result).
## Site-by-site
For each site, a ### heading "#rank Address — Barrio (District) · VERDICT", then bullets:
- **Why here:** the 2–3 strongest quantitative reasons (demand vs median store, competitors, footfall proxy, income, distance to nearest own store)
- **Robustness Monte Carlo:** p_selected and demand P10–P90, and what that means
- **Financials:** P&L Monte Carlo results if present (revenue P50, EBITDA P10/P50/P90, P(loss), payback P50, P(payback ≤ target)); else what is missing
- **Market evidence:** rent scrape / Google Maps results if present; else "not yet collected"
- **Main risk:** one line
## What would change the answer
The 3 inputs the conclusion is most sensitive to (use the P&L sensitivity if present) and the data that would de-risk each.
## Limitations
Short bullets, from the evidence pack's caveats.

EVIDENCE PACK
```json
{evidence}
```"""


def write_report(evidence: dict):
    c = _client()
    if c is None:
        return None
    resp = c.messages.create(model=MODEL, max_tokens=5000, messages=[{"role": "user", "content": REPORT_PROMPT.format(
        brand=evidence.get("brand"), evidence=json.dumps(evidence, default=str, ensure_ascii=False, indent=1))}])
    return _text(resp)
