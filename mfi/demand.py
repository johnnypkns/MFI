"""Layer 2 — where people actually are during the day.

Replaces the v0 demand proxy ("residents + k x every open premises"), which counted wholesale markets and
industrial estates as footfall. Now:

  Demand per census section   D_i = residents_i + a * consumer_premises_i + b * office_premises_i
  Footfall potential at point F(p) = sum_i D_i * exp(-d(p,i)/lambda) + c * rail/metro stations within 300 m

  consumer premises  shops, food & drink, personal services, clinics, education, banks' branches, leisure
  office premises    professional services, public administration, IT, consultancies  -> daytime workers
  industrial         manufacturing, construction, wholesale, logistics, auto  -> NOT demand (Mercamadrid, polígonos)

a, b, lambda and c are CALIBRATED against the Ayuntamiento's 31 permanent pedestrian counters (mean weekday
footfall 2022-2024): the combination whose potential best rank-correlates with measured footfall wins.
Leave-one-sensor-out validation is reported so the fit isn't just memorised.

    python -m mfi.demand   ->  data/app/demand.json (calibration), data/app/generators.parquet, data/app/ped_sensors.parquet
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from .model.export import APP, ROOT

TO_UTM = Transformer.from_crs(4326, 25830, always_xy=True)

CONSUMER = {"47", "56", "96", "93", "55", "85", "86", "92", "79", "68", "95", "77", "75", "64", "65", "66", "90", "91"}
OFFICE = {"58", "59", "60", "61", "62", "63", "69", "70", "71", "72", "73", "74", "78", "80", "82", "84", "88", "94"}
INDUSTRIAL = {f"{i:02d}" for i in range(1, 44)} | {"45", "46", "49", "50", "51", "52", "53", "81", "38", "39"}

A_GRID = [0, 10, 40, 80, 160, 320, 640, 1280]
B_GRID = [0, 5, 20, 80, 320]
L_GRID = [150, 250, 400, 600, 900]
C_GRID = [0, 5_000, 20_000, 80_000]


def premises_group(divisions: pd.Series) -> pd.Series:
    """Classify each premises by its first licensed division."""
    first = divisions.fillna("").str.split(",").str[0].str[:2]
    return np.select([first.isin(CONSUMER), first.isin(OFFICE), first.isin(INDUSTRIAL)],
                     ["consumer", "office", "industrial"], "other")


def load_sensors(raw_dir=None):
    raw_dir = Path(raw_dir or ROOT / "data" / "raw" / "madrid_pedestrians")
    parts = []
    for f in sorted(raw_dir.glob("300321-*.csv")):
        x = pd.read_csv(f, sep=";", encoding="utf-8-sig", low_memory=False)
        x.columns = [c.lower() for c in x.columns]
        parts.append(x[["fecha", "identificador", "peatones", "direccion", "latitude", "longitude"]])
    d = pd.concat(parts)
    d.columns = ["fecha", "id", "peatones", "direccion", "lat", "lon"]
    # coordinates are published with Spanish digit grouping: '40.422.009' = 40.422009
    for c in ["lat", "lon"]:
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(".", "", regex=False), errors="coerce")
        d[c] = d[c] / 10 ** (d[c].abs().astype("int64").astype(str).str.len() - (2 if c == "lat" else 1))
    d["peatones"] = pd.to_numeric(d["peatones"], errors="coerce")
    d["day"] = pd.to_datetime(d["fecha"].astype(str).str.split(" ").str[0], dayfirst=True, errors="coerce")
    day = d.groupby(["id", "day"]).agg(p=("peatones", "sum"), n=("peatones", "count")).reset_index()
    day = day[(day["n"] >= 22) & (day["day"].dt.dayofweek < 5)]                 # complete weekdays only
    s = day.groupby("id").agg(weekday_mean=("p", "mean"), days=("p", "size"))
    s = s.join(d.groupby("id")[["lat", "lon", "direccion"]].first())
    s = s[s["days"] >= 30].reset_index()
    s["x"], s["y"] = TO_UTM.transform(s["lon"].values, s["lat"].values)
    return s


def load_generators(path=None):
    path = path or sorted((ROOT / "data" / "raw" / "osm").glob("*/generators.json"))[-1]
    rows = []
    for e in json.loads(Path(path).read_text())["elements"]:
        t = e.get("tags", {})
        lat, lon = (e.get("lat"), e.get("lon")) if "lat" in e else ((e.get("center") or {}).get("lat"), (e.get("center") or {}).get("lon"))
        if lat is None:
            continue
        kind = ("metro" if t.get("station") == "subway" or t.get("railway") == "subway_entrance" or t.get("subway") == "yes"
                else "rail" if t.get("railway") == "station" or t.get("train") == "yes"
                else "bus" if t.get("highway") == "bus_stop"
                else "university" if t.get("amenity") in ("university", "college")
                else "school" if t.get("amenity") == "school"
                else "hospital" if t.get("amenity") in ("hospital", "clinic")
                else "mall" if t.get("shop") in ("mall", "department_store")
                else "supermarket" if t.get("shop") == "supermarket"
                else "office" if "office" in t
                else "tourism" if t.get("tourism") in ("attraction", "museum") or t.get("amenity") in ("theatre", "cinema")
                else "hotel" if t.get("tourism") == "hotel"
                else "market" if t.get("amenity") == "marketplace" else None)
        if kind:
            rows.append({"kind": kind, "name": t.get("name"), "lat": lat, "lon": lon})
    g = pd.DataFrame(rows)
    g["x"], g["y"] = TO_UTM.transform(g["lon"].values, g["lat"].values)
    return g


class DemandSurface:
    """Per-section demand components + point potential. Built once, reused by features, Huff and calibration."""

    def __init__(self, premises, sections, generators):
        s = sections.dropna(subset=["pop_total"]).reset_index(drop=True)
        self.sec = s
        self.xy = s[["x", "y"]].values
        self.tree = cKDTree(self.xy)
        op = premises[premises["status"] == "Abierto"].copy()
        op["group"] = premises_group(op["divisions"])
        cnt = op.groupby(["section_spatial", "group"]).size().unstack(fill_value=0)
        for g in ["consumer", "office", "industrial", "other"]:
            s[g] = s["section_code"].map(cnt[g] if g in cnt else {}).fillna(0).values
        self.R = s["pop_total"].values.astype(float)
        self.C = s["consumer"].values.astype(float)
        self.O = s["office"].values.astype(float)
        self.I = s["industrial"].values.astype(float)
        self.op = op
        self.grp_tree = {g: cKDTree(op.loc[op["group"] == g, ["x", "y"]].values) for g in ["consumer", "office", "industrial"]}
        self.gen = generators
        st = generators[generators["kind"].isin(["metro", "rail"])]
        self.station_tree = cKDTree(st[["x", "y"]].values)
        self.gen_tree = {k: cKDTree(v[["x", "y"]].values) for k, v in generators.groupby("kind")}

    def section_demand(self, a, b):
        return self.R + a * self.C + b * self.O

    def potential(self, pts, a, b, lam, c, cutoff=3000):
        D = self.section_demand(a, b)
        out = np.empty(len(pts))
        for k, (idx, p) in enumerate(zip(self.tree.query_ball_point(pts, cutoff), pts)):
            idx = np.asarray(idx, int)
            d = np.hypot(*(self.xy[idx] - p).T) if len(idx) else np.array([])
            out[k] = (D[idx] * np.exp(-d / lam)).sum()
        return out + c * self.stations_within(pts, 300)

    def stations_within(self, pts, r):
        return np.array([len(i) for i in self.station_tree.query_ball_point(pts, r)], float)

    def count(self, kind, pts, r):
        t = self.grp_tree.get(kind) or self.gen_tree.get(kind)
        return np.array([len(i) for i in t.query_ball_point(pts, r)], float) if t is not None else np.zeros(len(pts))

    def industrial_share(self, pts, r=200):
        i, c, o = self.count("industrial", pts, r), self.count("consumer", pts, r), self.count("office", pts, r)
        return np.where(i + c + o > 0, i / np.maximum(i + c + o, 1), 0.0)


def calibrate(surface, sensors):
    pts = sensors[["x", "y"]].values
    y = np.log(sensors["weekday_mean"].values)
    rows = []
    for lam in L_GRID:
        for a in A_GRID:
            for b in B_GRID:
                base = surface.potential(pts, a, b, lam, 0)
                st = surface.stations_within(pts, 300)
                for c in C_GRID:
                    rho = spearmanr(np.log1p(base + c * st), y).correlation
                    rows.append((lam, a, b, c, rho))
    g = pd.DataFrame(rows, columns=["lambda_m", "a_consumer", "b_office", "c_station", "spearman"])
    best = g.loc[g["spearman"].idxmax()]
    # leave-one-sensor-out: refit on n-1 sensors, predict the held-out one's rank
    loo_pred = np.empty(len(sensors))
    for i in range(len(sensors)):
        keep = np.arange(len(sensors)) != i
        sub = []
        for _, r in g.nlargest(40, "spearman").iterrows():   # search the top region of the grid
            pot = surface.potential(pts, r.a_consumer, r.b_office, r.lambda_m, r.c_station)
            sub.append((spearmanr(np.log1p(pot[keep]), y[keep]).correlation, pot[i], pot))
        _, _, pot = max(sub, key=lambda t: t[0])
        loo_pred[i] = np.log1p(pot[i])
    loo_rho = spearmanr(loo_pred, y).correlation
    baselines = {
        "residents_only": spearmanr(np.log1p(surface.potential(pts, 0, 0, best.lambda_m, 0)), y).correlation,
        "v0_all_premises": spearmanr(np.log1p(surface.potential_all_premises(pts, 20, 2000)), y).correlation
        if hasattr(surface, "potential_all_premises") else None,
    }
    return best, g, loo_rho, baselines


def main():
    from .model.site_model import load_data
    data = load_data()
    sensors = load_sensors()
    gens = load_generators()
    surf = DemandSurface(data.premises, data.sections, gens)

    # v0 proxy for comparison: residents + 20 x ALL open premises
    allp = surf.C + surf.O + surf.I + surf.sec["other"].values

    def v0(pts, k, lam):
        D = surf.R + k * allp
        return np.array([(D[np.asarray(i, int)] * np.exp(-np.hypot(*(surf.xy[np.asarray(i, int)] - p).T) / lam)).sum()
                         for i, p in zip(surf.tree.query_ball_point(pts, 3000), pts)])
    surf.potential_all_premises = v0
    best, grid, loo, base = calibrate(surf, sensors)
    cal = {"lambda_m": int(best.lambda_m), "a_consumer": float(best.a_consumer), "b_office": float(best.b_office),
           "c_station": float(best.c_station), "spearman_in_sample": float(best.spearman),
           "spearman_leave_one_out": float(loo), "n_sensors": int(len(sensors)),
           "baseline_residents_only": float(base["residents_only"]), "baseline_v0_all_premises": float(base["v0_all_premises"]),
           "sensor_period": "2022-2024 weekday means, Ayuntamiento de Madrid permanent counters (dataset 300321)"}
    (APP / "demand.json").write_text(json.dumps(cal, indent=1))
    gens.drop(columns=["x", "y"]).to_parquet(APP / "generators.parquet", index=False)
    pts = sensors[["x", "y"]].values
    sensors["potential"] = surf.potential(pts, best.a_consumer, best.b_office, best.lambda_m, best.c_station)
    sensors.drop(columns=["x", "y"]).to_parquet(APP / "ped_sensors.parquet", index=False)
    print(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()
