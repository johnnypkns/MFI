"""Source registry. One entry per dataset. Add new sources here, never inline URLs in loaders.

Madrid's CKAN portal timestamps download URLs (they change daily), so Madrid entries
store the CKAN package id + a resource matcher and the real URL is resolved at run time.
"""

MADRID_CKAN = "https://datos.madrid.es/api/3/action/package_show"
GEOPORTAL = "https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS"

SOURCES = {
    "madrid_districts": dict(
        publisher="Ayuntamiento de Madrid", title="Distritos municipales de Madrid (shapefile)",
        dataset_url="https://datos.madrid.es/dataset/300497-0-distritos-municipales-madrid",
        url=f"{GEOPORTAL}/Distritos/Distritos.zip",
        license="CC BY 4.0", geographic_resolution="district", data_type="boundary",
        update_frequency="on change", confidence="high"),
    "madrid_barrios": dict(
        publisher="Ayuntamiento de Madrid", title="Barrios municipales de Madrid (shapefile)",
        dataset_url="https://datos.madrid.es/dataset/300496-0-barrios-madrid",
        url=f"{GEOPORTAL}/Barrios/Barrios.zip",
        license="CC BY 4.0", geographic_resolution="barrio", data_type="boundary",
        update_frequency="on change", confidence="high"),
    "madrid_sections": dict(
        publisher="Ayuntamiento de Madrid", title="Seccionado censal (TopoJSON)",
        dataset_url="https://datos.madrid.es/dataset/300724-0-seccionado-censal-mapas",
        url=f"{GEOPORTAL}/Seccionado/TopoJSON/Secciones_Censales.json",
        license="CC BY 4.0", geographic_resolution="census_section", data_type="boundary",
        update_frequency="on change", confidence="high"),
    "madrid_padron": dict(
        publisher="Ayuntamiento de Madrid", title="Padrón municipal — población por sección, edad, sexo y nacionalidad",
        dataset_url="https://datos.madrid.es/dataset/200076-0-padron",
        ckan_package="200076-0-padron", resource_match=lambda r: (r.get("format") or "").upper() == "CSV",
        license="CC BY 4.0", geographic_resolution="census_section", data_type="official_register",
        update_frequency="monthly", confidence="high"),
    "madrid_locales": dict(
        publisher="Ayuntamiento de Madrid", title="Censo de locales — Locales",
        dataset_url="https://datos.madrid.es/dataset/200085-0-censo-locales",
        ckan_package="200085-0-censo-locales",
        resource_match=lambda r: (r.get("format") or "").upper() == "CSV" and (r.get("description") or "").strip() == "Locales",
        license="CC BY 4.0", geographic_resolution="point", data_type="official_register",
        update_frequency="daily", confidence="high",
        notes="Status field matters: only 'Abierto' premises are operating."),
    "madrid_actividades": dict(
        publisher="Ayuntamiento de Madrid", title="Censo de locales — Actividades",
        dataset_url="https://datos.madrid.es/dataset/200085-0-censo-locales",
        ckan_package="200085-0-censo-locales",
        resource_match=lambda r: (r.get("format") or "").upper() == "CSV" and (r.get("description") or "").strip() == "Actividades",
        license="CC BY 4.0", geographic_resolution="point", data_type="official_register",
        update_frequency="daily", confidence="high",
        notes="Registered activity, not verified trading. Competitor checks (M3) must confirm against OSM/live sources."),
    "ine_adrh_madrid": dict(
        publisher="INE", title="Atlas de Distribución de Renta de los Hogares — Madrid provincia, tabla 31097 (renta media y mediana)",
        dataset_url="https://www.ine.es/jaxiT3/Tabla.htm?t=31097",
        url="https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/31097.csv",
        license="INE — reuse with attribution", geographic_resolution="census_section", data_type="official_statistic",
        update_frequency="annual (~2-year lag)", confidence="high",
        notes="Income year lags ~2 years. Section boundaries are those of the income year, not today's."),
}
