"""INE ADRH (household income atlas), Madrid province table 31097 -> income_statistics.

Only the Madrid municipality (28079) is kept. INE formats numbers Spanish-style:
'19.422' = 19,422 EUR. Suppressed cells ('..', '.', blank) stay NULL — never imputed.
"""
import pandas as pd

from ..fetch import fetch, set_file_meta
from .common import already_loaded, copy_df

INDICATORS = {
    "Renta neta media por persona": "net_income_per_person",
    "Renta neta media por hogar": "net_income_per_household",
    "Media de la renta por unidad de consumo": "mean_income_per_cu",
    "Mediana de la renta por unidad de consumo": "median_income_per_cu",
    "Renta bruta media por persona": "gross_income_per_person",
    "Renta bruta media por hogar": "gross_income_per_household",
}


def es_number(s):
    s = (s or "").strip()
    if s in ("", ".", ".."):
        return None
    return float(s.replace(".", "").replace(",", "."))


def parse(path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    raw.columns = ["municipio", "distrito", "seccion", "indicador", "periodo", "total"]
    raw = raw[raw["municipio"].str.startswith("28079 ")]
    unknown = set(raw["indicador"]) - set(INDICATORS)
    if unknown:
        raise ValueError(f"Unmapped INE indicators (table changed?): {unknown}")
    lvl = raw.apply(lambda r: ("census_section", r.seccion.split()[0]) if r.seccion
                    else ("district", r.distrito.split()[0]) if r.distrito
                    else ("municipality", r.municipio.split()[0]), axis=1)
    return pd.DataFrame({
        "year": raw["periodo"].astype(int),
        "geo_level": [l for l, _ in lvl],
        "geo_code": [c for _, c in lvl],
        "indicator": raw["indicador"].map(INDICATORS),
        "value_eur": raw["total"].map(es_number),
    })


def load(run_id):
    f = fetch("ine_adrh_madrid", run_id)
    if already_loaded("income_statistics", f.file_id):
        return f.file_id, 0
    df = parse(f.path).assign(file_id=f.file_id)
    copy_df("income_statistics", df[["file_id", "year", "geo_level", "geo_code", "indicator", "value_eur"]])
    set_file_meta(f.file_id, len(df), pd.Timestamp(f"{df['year'].max()}-12-31").date())
    return f.file_id, len(df)
