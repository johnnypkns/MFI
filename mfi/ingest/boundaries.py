"""District, barrio and census-section polygons -> PostGIS (EPSG:25830)."""
import geopandas as gpd
from shapely.geometry import MultiPolygon

from ..fetch import fetch, set_file_meta
from .common import already_loaded, copy_df, madrid_cusec

CRS = 25830


def _multi(g):
    """Force MultiPolygon. make_valid can return GeometryCollections with stray
    line/point slivers; keep only the polygonal parts."""
    if g.geom_type == "Polygon":
        return MultiPolygon([g])
    if g.geom_type == "MultiPolygon":
        return g
    polys = [p for part in getattr(g, "geoms", []) for p in (part.geoms if part.geom_type == "MultiPolygon" else [part])
             if p.geom_type == "Polygon"]
    if not polys:
        raise ValueError(f"No polygon in geometry {g.geom_type}")
    return MultiPolygon(polys)


def _prep(gdf):
    gdf = gdf.to_crs(CRS) if gdf.crs and gdf.crs.to_epsg() != CRS else gdf.set_crs(CRS, allow_override=gdf.crs is None)
    gdf["geom"] = gdf.geometry.make_valid().apply(_multi).apply(lambda g: f"SRID={CRS};{g.wkt}")
    return gdf


def load_districts(run_id):
    f = fetch("madrid_districts", run_id)
    if already_loaded("districts", f.file_id):
        return f.file_id, 0
    g = _prep(gpd.read_file(f"zip://{f.path}"))
    df = g.assign(district_code=g["COD_DIS"].astype(int), name=g["NOMBRE"].str.strip(), file_id=f.file_id)
    copy_df("districts", df[["district_code", "name", "geom", "file_id"]])
    set_file_meta(f.file_id, len(df))
    return f.file_id, len(df)


def load_barrios(run_id):
    f = fetch("madrid_barrios", run_id)
    if already_loaded("neighbourhoods", f.file_id):
        return f.file_id, 0
    g = _prep(gpd.read_file(f"zip://{f.path}"))
    df = g.assign(barrio_code=g["COD_BAR"].astype(int), district_code=g["CODDIS"].astype(int),
                  name=g["NOMBRE"].str.strip(), file_id=f.file_id)
    copy_df("neighbourhoods", df[["barrio_code", "district_code", "name", "geom", "file_id"]])
    set_file_meta(f.file_id, len(df))
    return f.file_id, len(df)


def load_sections(run_id):
    f = fetch("madrid_sections", run_id)
    if already_loaded("census_sections", f.file_id):
        return f.file_id, 0
    g = gpd.read_file(f.path)
    if g.crs is None:  # TopoJSON carries no CRS; Madrid publishes it in lon/lat
        g = g.set_crs(4326)
    g = _prep(g)
    dist = g["COD_DIS"].astype(int)
    sec = g["COD_SECCIO"].astype(int) % 1000
    df = g.assign(section_code=[madrid_cusec(d, s) for d, s in zip(dist, sec)], district_code=dist,
                  barrio_code=g["COD_BAR"].astype(int), file_id=f.file_id)
    if df["section_code"].duplicated().any():
        raise ValueError("Duplicate section codes in boundary file")
    copy_df("census_sections", df[["section_code", "district_code", "barrio_code", "geom", "file_id"]])
    set_file_meta(f.file_id, len(df))
    return f.file_id, len(df)
