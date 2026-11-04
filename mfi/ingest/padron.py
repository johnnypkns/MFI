"""Padrón municipal (population register) -> demographics, long format."""
import pandas as pd

from ..fetch import fetch, set_file_meta
from .common import already_loaded, copy_df, madrid_cusec

COUNT_COLS = {
    "ESPANOLESHOMBRES": ("M", "spanish"),
    "ESPANOLESMUJERES": ("F", "spanish"),
    "EXTRANJEROSHOMBRES": ("M", "foreign"),
    "EXTRANJEROSMUJERES": ("F", "foreign"),
}


def parse(path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    raw.columns = raw.columns.str.strip()
    ref = raw["FX_DATOS_INI"].dropna().unique()
    if len(ref) != 1:
        raise ValueError(f"Padrón file has {len(ref)} reference dates, expected 1: {ref[:5]}")
    for c in COUNT_COLS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0).astype(int)  # blank cell = 0 people in register
    # A handful of register entries carry no census section (address not yet sectioned).
    # They cannot be placed on a map, so they are dropped — and reported, not hidden.
    no_sec = raw["COD_SECCION"].isna() | raw["COD_DISTRITO"].isna()
    if no_sec.any():
        print(f"    padrón: {int(raw.loc[no_sec, list(COUNT_COLS)].sum().sum())} residents in "
              f"{int(no_sec.sum())} rows have no census section — excluded")
    raw = raw[~no_sec].copy()
    raw["section_code"] = [madrid_cusec(d, s) for d, s in zip(raw["COD_DISTRITO"].astype(int),
                                                              raw["COD_SECCION"].astype(int))]
    raw["age"] = raw["COD_EDAD_INT"].str.strip().str.replace(r"\D.*$", "", regex=True).astype(int)  # '100 o +' -> 100
    long = raw.melt(id_vars=["section_code", "age"], value_vars=list(COUNT_COLS), value_name="population")
    long["sex"] = long["variable"].map(lambda v: COUNT_COLS[v][0])
    long["nationality"] = long["variable"].map(lambda v: COUNT_COLS[v][1])
    long = long.groupby(["section_code", "age", "sex", "nationality"], as_index=False)["population"].sum()
    long["reference_date"] = pd.to_datetime(ref[0]).date()
    return long


def load(run_id):
    f = fetch("madrid_padron", run_id)
    if already_loaded("demographics", f.file_id):
        return f.file_id, 0
    df = parse(f.path).assign(file_id=f.file_id)
    df = df[df["population"] > 0]  # zero rows carry no information; totals are unaffected
    copy_df("demographics", df[["file_id", "reference_date", "section_code", "age", "sex", "nationality", "population"]])
    set_file_meta(f.file_id, len(df), df["reference_date"].iloc[0])
    return f.file_id, len(df)
