"""Censo de Locales y Actividades -> premises + premise_activities."""
import pandas as pd

from ..fetch import fetch, set_file_meta
from .common import already_loaded, copy_df, madrid_cusec


def _read(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False, low_memory=False)
    df.columns = df.columns.str.strip()
    return df.apply(lambda c: c.str.strip())


def _snapshot(df):
    d = df["fx_carga"].replace("", pd.NA).dropna().unique()
    if len(d) != 1:
        raise ValueError(f"Expected one fx_carga date, found {d[:5]}")
    return pd.to_datetime(d[0], format="%d/%m/%Y").date()


def _int(s):
    return pd.to_numeric(s.replace("", pd.NA), errors="coerce").astype("Int64")


def _xy(df, xcol, ycol):
    x = pd.to_numeric(df[xcol].str.replace(",", "."), errors="coerce")
    y = pd.to_numeric(df[ycol].str.replace(",", "."), errors="coerce")
    ok = x.between(400_000, 480_000) & y.between(4_440_000, 4_520_000)  # Madrid box in UTM30N; 0.0/blank = missing
    return x.where(ok), y.where(ok)


def locate(df) -> pd.DataFrame:
    """Coordinates in order of precision, with the method recorded:
    1 'local'      the premise's own point
    2 'agrupacion' the mall/gallery it sits in (Agrupado premises)
    3 'building'   mean point of other premises at the same building address (id_ndp_edificio);
                   mostly upper-floor 'Interior' premises the census leaves at 0,0
    Anything else stays NULL. No geocoding service, no guesses."""
    x, y = _xy(df, "coordenada_x_local", "coordenada_y_local")
    src = pd.Series("local", index=df.index).where(x.notna())
    ax, ay = _xy(df, "coordenada_x_agrupacion", "coordenada_y_agrupacion")
    use = x.isna() & ax.notna()
    x, y, src = x.mask(use, ax), y.mask(use, ay), src.mask(use, "agrupacion")
    bld = df["id_ndp_edificio"].replace("", pd.NA)
    bx = x.where(src == "local").groupby(bld).transform("mean")
    by = y.where(src == "local").groupby(bld).transform("mean")
    use = x.isna() & bx.notna()
    x, y, src = x.mask(use, bx), y.mask(use, by), src.mask(use, "building")
    geom = ("SRID=25830;POINT(" + x.round(2).astype(str) + " " + y.round(2).astype(str) + ")").where(x.notna())
    return pd.DataFrame({"geom": geom, "geom_source": src})


def parse_locales(path) -> pd.DataFrame:
    df = _read(path)
    loc = locate(df)
    dist = _int(df["id_distrito_local"])
    sec = _int(df["id_seccion_censal_local"]) % 1000
    section = [madrid_cusec(d, s) if pd.notna(d) and pd.notna(s) else None for d, s in zip(dist, sec)]
    return pd.DataFrame({
        "snapshot_date": _snapshot(df),
        "id_local": _int(df["id_local"]),
        "district_code": dist,
        # census codes barrios as DD*100+B (e.g. 1704); the boundary files use DD*10+B (174). Normalise to the latter.
        "barrio_code": dist * 10 + _int(df["id_barrio_local"]) % 100,
        "section_code": section,
        "status": df["desc_situacion_local"],
        "access_type": df["desc_tipo_acceso_local"],
        "street_type": df["clase_vial_edificio"],
        "street_name": df["desc_vial_edificio"],
        "street_number": df["num_edificio"].str.lstrip("0"),
        "trade_name": df["rotulo"].replace("", None),
        "postal_code": df["cod_postal"].replace("", None),
        "building_id": _int(df["id_ndp_edificio"]),
        "geom": loc["geom"],
        "geom_source": loc["geom_source"],
    })


def parse_actividades(path) -> pd.DataFrame:
    df = _read(path)
    return pd.DataFrame({
        "snapshot_date": _snapshot(df),
        "id_local": _int(df["id_local"]),
        "section_id": df["id_seccion"], "section_desc": df["desc_seccion"],
        "division_id": df["id_division"], "division_desc": df["desc_division"],
        "epigrafe_id": df["id_epigrafe"], "epigrafe_desc": df["desc_epigrafe"],
    }).replace("", None)


def load_locales(run_id):
    f = fetch("madrid_locales", run_id)
    if already_loaded("premises", f.file_id):
        return f.file_id, 0
    df = parse_locales(f.path).assign(file_id=f.file_id)
    if df["id_local"].duplicated().any():
        raise ValueError("Duplicate id_local in Locales file")
    copy_df("premises", df)
    set_file_meta(f.file_id, len(df), df["snapshot_date"].iloc[0])
    return f.file_id, len(df)


def load_actividades(run_id):
    f = fetch("madrid_actividades", run_id)
    if already_loaded("premise_activities", f.file_id):
        return f.file_id, 0
    df = parse_actividades(f.path).assign(file_id=f.file_id)
    copy_df("premise_activities", df)
    set_file_meta(f.file_id, len(df), df["snapshot_date"].iloc[0])
    return f.file_id, len(df)
