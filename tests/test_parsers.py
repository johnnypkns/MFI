"""Unit tests for the parsing rules most likely to silently corrupt data. No DB needed.
Run: python -m pytest -q
"""
import pandas as pd

from mfi.ingest.common import madrid_cusec
from mfi.ingest.income import es_number
from mfi.ingest.premises import locate


def test_cusec():
    assert madrid_cusec(2, 62) == "2807902062"
    assert madrid_cusec(17, 120) == "2807917120"


def test_es_number_thousands():
    assert es_number("19.422") == 19422
    assert es_number("1.234.567") == 1234567
    assert es_number("17,5") == 17.5


def test_es_number_suppressed_is_null():
    for s in ("", ".", "..", None):
        assert es_number(s) is None


def test_locate_fallbacks():
    df = pd.DataFrame({
        "coordenada_x_local":      ["440554.59", "0.0", "0.0", "0.0"],
        "coordenada_y_local":      ["4475338.53", "0.0", "0.0", "0.0"],
        "coordenada_x_agrupacion": ["", "440692.59", "", ""],
        "coordenada_y_agrupacion": ["", "4475369.53", "", ""],
        "id_ndp_edificio":         ["11005235", "", "11005235", "999"],
    })
    out = locate(df)
    assert list(out["geom_source"].fillna("none")) == ["local", "agrupacion", "building", "none"]
    assert out["geom"][2] == out["geom"][0]          # building fallback = sibling's point
    assert pd.isna(out["geom"][3])                   # no sibling -> stays NULL, never guessed


def test_brand_name_normalisation():
    from mfi.model.export import norm_name
    for v in ["McDonald's", "MC DONALD´S", "mcdonalds", "MC DONALD S"]:
        assert norm_name(v) == "MCDONALDS"
