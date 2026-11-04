"""Milestone-1 data checks. Every result is stored in validation_results against a run_id.

pass = fine; warn = usable but documented caveat; fail = do not build on this data.
"""
from datetime import date

from .db import connect

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _q(conn, sql, *a):
    return conn.execute(sql, a).fetchone()


def _rate(status_pass, status_warn, x):
    return "pass" if x >= status_pass else "warn" if x >= status_warn else "fail"


@check
def boundary_counts(c):
    d, b, s = _q(c, "SELECT (SELECT count(*) FROM v_districts),(SELECT count(*) FROM v_neighbourhoods),(SELECT count(*) FROM v_census_sections)")
    ok = d == 21 and b == 131 and s > 2000
    return ("pass" if ok else "fail", f"{d} districts / {b} barrios / {s} sections", "21 / 131 / >2000", "Madrid has 21 districts and 131 barrios since 2017")


@check
def geometry_valid(c):
    bad = _q(c, """SELECT (SELECT count(*) FROM v_districts WHERE NOT ST_IsValid(geom))
                      + (SELECT count(*) FROM v_neighbourhoods WHERE NOT ST_IsValid(geom))
                      + (SELECT count(*) FROM v_census_sections WHERE NOT ST_IsValid(geom))""")[0]
    return ("pass" if bad == 0 else "fail", f"{bad} invalid", "0", None)


@check
def sections_tile_districts(c):
    a_sec, a_dis = _q(c, "SELECT (SELECT sum(ST_Area(geom)) FROM v_census_sections),(SELECT sum(ST_Area(geom)) FROM v_districts)")
    r = a_sec / a_dis
    return (_rate(0.995, 0.98, 1 - abs(1 - r)), f"section area / district area = {r:.4f}", "≈1.000",
            "Sections should exactly tile the municipality")


@check
def padron_total(c):
    pop, ref = _q(c, "SELECT sum(pop_total), max(reference_date) FROM v_section_population")
    ok = 3_000_000 <= pop <= 3_900_000
    return ("pass" if ok else "fail", f"{pop:,} residents at {ref}", "3.0M–3.9M",
            "Plausibility band around Madrid's ~3.4–3.5M register population")


@check
def padron_freshness(c):
    ref = _q(c, "SELECT max(reference_date) FROM v_section_population")[0]
    age = (date.today() - ref).days
    return ("pass" if age <= 120 else "warn", f"{age} days old ({ref})", "≤120 days", None)


@check
def padron_sections_match_boundaries(c):
    n, m = _q(c, """SELECT count(*), count(*) FILTER (WHERE c.section_code IS NOT NULL)
                    FROM v_section_population p LEFT JOIN v_census_sections c USING (section_code)""")
    r = m / n
    return (_rate(0.999, 0.98, r), f"{m}/{n} padrón sections have a polygon ({r:.2%})", "100%", None)


@check
def boundaries_have_population(c):
    n, m = _q(c, """SELECT count(*), count(p.section_code) FROM v_census_sections c
                    LEFT JOIN v_section_population p USING (section_code)""")
    r = m / n
    return (_rate(0.999, 0.98, r), f"{m}/{n} polygons have padrón data ({r:.2%})", "100%", None)


@check
def income_year(c):
    y = _q(c, "SELECT max(year) FROM v_section_income")[0]
    lag = date.today().year - y
    return ("pass" if lag <= 3 else "warn", f"latest income year {y} (lag {lag}y)", "lag ≤3y",
            "ADRH is published with a ~2-year lag; treat as structural, not current, income")


@check
def income_section_coverage(c):
    n, m = _q(c, """SELECT count(*), count(i.net_income_per_person) FROM v_census_sections c
                    LEFT JOIN v_section_income i USING (section_code)""")
    r = m / n
    return (_rate(0.97, 0.90, r), f"{m}/{n} current sections have income ({r:.2%})", "≥97%",
            "Gaps = sections redrawn after the income year, or INE suppression. Not imputed.")


@check
def income_units(c):
    lo, hi, med = _q(c, """SELECT min(net_income_per_person), max(net_income_per_person),
                                 percentile_cont(0.5) WITHIN GROUP (ORDER BY net_income_per_person)
                          FROM v_section_income""")
    ok = 4_000 <= lo and hi <= 120_000 and 10_000 <= med <= 40_000
    return ("pass" if ok else "fail", f"net €/person min {lo:,.0f} / median {med:,.0f} / max {hi:,.0f}",
            "annual EUR, median 10k–40k", "Catches thousand-separator parsing errors (19.422 must be 19,422)")


@check
def income_household_gt_person(c):
    bad = _q(c, "SELECT count(*) FROM v_section_income WHERE net_income_per_household < net_income_per_person")[0]
    return ("pass" if bad == 0 else "fail", f"{bad} sections with household < person income", "0", None)


@check
def premises_freshness(c):
    s = _q(c, "SELECT max(snapshot_date) FROM v_premises")[0]
    age = (date.today() - s).days
    return ("pass" if age <= 14 else "warn", f"snapshot {s} ({age} days)", "≤14 days", None)


@check
def premises_geocoded_street(c):
    n, g = _q(c, """SELECT count(*), count(geom) FILTER (WHERE geom_source='local') FROM v_premises
                    WHERE status='Abierto' AND access_type='Puerta Calle'""")
    r = g / n
    return (_rate(0.99, 0.95, r), f"{g:,}/{n:,} open street-level premises have their own point ({r:.2%})", "≥99%",
            "Street-level = the franchise-relevant stock")


@check
def premises_geocoded_all_open(c):
    n, g, b = _q(c, """SELECT count(*), count(geom), count(*) FILTER (WHERE geom_source <> 'local')
                       FROM v_premises WHERE status='Abierto'""")
    r = g / n
    return (_rate(0.95, 0.85, r), f"{g:,}/{n:,} open premises located ({r:.1%}); {b:,} via mall/building fallback",
            "≥95%", "Unlocated remainder is mostly upper-floor 'Interior' units with no building match")


@check
def premises_inside_madrid(c):
    g, inside = _q(c, """WITH m AS (SELECT ST_Union(geom) g FROM v_districts)
                         SELECT count(*), count(*) FILTER (WHERE ST_Intersects(m.g, p.geom))
                         FROM v_premises p, m WHERE p.geom IS NOT NULL""")
    r = inside / g
    return (_rate(0.995, 0.98, r), f"{inside:,}/{g:,} points inside municipal boundary ({r:.2%})", "≥99.5%",
            "Fails if coordinates are in the wrong CRS")


@check
def premises_section_agreement(c):
    n, agree = _q(c, """SELECT count(*), count(*) FILTER (WHERE s.section_code = p.section_code)
                        FROM v_premises p JOIN v_census_sections s ON ST_Contains(s.geom, p.geom)
                        WHERE p.geom IS NOT NULL AND p.section_code IS NOT NULL""")
    r = agree / n
    return (_rate(0.95, 0.85, r), f"declared section = spatial section for {r:.1%} of {n:,}", "≥95%",
            "Disagreement usually = points on a boundary or census coded to an older seccionado")


@check
def premises_barrio_codes(c):
    n, m = _q(c, """SELECT count(*), count(n.barrio_code) FROM v_premises p
                    LEFT JOIN v_neighbourhoods n USING (barrio_code) WHERE p.barrio_code IS NOT NULL""")
    r = m / n
    return (_rate(0.999, 0.98, r), f"{m:,}/{n:,} premises barrio codes exist in the barrio boundaries ({r:.2%})", "100%",
            "Census uses DD*100+B, boundaries DD*10+B — normalised at load")


@check
def activities_link_premises(c):
    n, m = _q(c, """SELECT count(*), count(p.id_local) FROM v_premise_activities a
                    LEFT JOIN v_premises p USING (id_local)""")
    r = m / n
    return (_rate(0.98, 0.90, r), f"{m:,}/{n:,} activities match a premise ({r:.1%})", "≥98%", None)


def run_all(run_id: int):
    rows = []
    with connect() as conn:
        for fn in CHECKS:
            try:
                status, observed, expected, detail = fn(conn)
            except Exception as e:  # a crashing check is a failed check
                conn.rollback()
                status, observed, expected, detail = "fail", f"error: {e!r}"[:500], None, None
            conn.execute("INSERT INTO validation_results VALUES (%s,%s,%s,%s,%s,%s)",
                         (run_id, fn.__name__, status, observed, expected, detail))
            rows.append((fn.__name__, status, observed))
        conn.commit()
    return rows
