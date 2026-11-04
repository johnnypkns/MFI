# Data sources — Milestone 1

All seven are official, free and open-licensed. Retrieved 2026-11-04. The registry in code is `mfi/sources.py`; every download is hashed and logged in `source_files`.

| Key | Publisher | What | Resolution | Reference date | Licence |
|---|---|---|---|---|---|
| `madrid_districts` | Ayto. Madrid | District polygons (shapefile) | 21 districts | current | CC BY 4.0 |
| `madrid_barrios` | Ayto. Madrid | Barrio polygons (shapefile) | 131 barrios | current | CC BY 4.0 |
| `madrid_sections` | Ayto. Madrid | Census-section polygons (TopoJSON) | 2,462 sections | current | CC BY 4.0 |
| `madrid_padron` | Ayto. Madrid | Population register by section × age × sex × nationality | section | 1 Oct 2026 (monthly) | CC BY 4.0 |
| `ine_adrh_madrid` | INE | Household income atlas (ADRH), table 31097 | section | income year 2024 (2015–2024 held) | INE, attribution |
| `madrid_locales` | Ayto. Madrid | Censo de Locales: every commercial premise, status, address, point | point | 3 Nov 2026 (daily) | CC BY 4.0 |
| `madrid_actividades` | Ayto. Madrid | Registered activities per premise (epígrafe codes) | point | 3 Nov 2026 (daily) | CC BY 4.0 |

Dataset pages:
- https://datos.madrid.es/dataset/300497-0-distritos-municipales-madrid
- https://datos.madrid.es/dataset/300496-0-barrios-madrid
- https://datos.madrid.es/dataset/300724-0-seccionado-censal-mapas
- https://datos.madrid.es/dataset/200076-0-padron
- https://datos.madrid.es/dataset/200085-0-censo-locales
- https://www.ine.es/jaxiT3/Tabla.htm?t=31097

## Quirks found and how they're handled

| Quirk | Handling |
|---|---|
| Madrid CKAN download URLs carry a daily timestamp | Resolved at run time via `package_show` + resource matcher. Fails loudly if the match isn't exactly one file. |
| Section TopoJSON has no CRS (it's lon/lat) | Set to EPSG:4326, reprojected to 25830. |
| One section polygon becomes a GeometryCollection after `make_valid` | Polygon parts kept, slivers dropped. Area check: sections tile districts to 1.0002. |
| Padrón age `"100 o +"` | Stored as 100 (= 100+). |
| 3 padrón residents have no census section | Excluded and printed at load. Immaterial (3 of 3.5M). |
| INE numbers use `.` as thousands separator (`19.422` = €19,422) | Explicit parser + unit test + plausibility check. A wrong parse would show median ≈ €19 and fail `income_units`. |
| INE suppressed cells (`..`) | NULL. Never imputed. |
| 25% of premises have coordinates `0,0` | Mostly upper-floor "Interior" units. Fallback: mall point, then mean point of other premises at the same building address. Method stored in `geom_source`. Street-level open premises: 99.92% have their own point. |
| Census declares a section per premise | Agrees with the spatial join for 96.5%. Use the **spatial** section for analysis. |

## Known limits — read before using these numbers

1. **Income is 2024 and nominal.** Treat as structural purchasing power, not current spending.
2. **The premises census is a register, not reality.** "Abierto" means registered as open. M3 must cross-check competitors against OSM / live sources before counting them.
3. **Premises ≠ available rentals.** Nothing here says a unit is for rent. That's M5 (`properties`).
4. **Residents ≠ footfall.** Padrón counts who sleeps there. Office workers, students, tourists come later (OSM demand generators, transport).

## Scraping policy (changed 2026-11-04)

John lifted the brief's no-scraping rule for **Idealista** (rent listings) and **Google Maps** (competitors, ratings,
review counts) via Apify. Both sites' terms prohibit scraping; accepted knowingly by the founder. Low-volume runs,
raw output archived with provenance like every other source. Revisit before selling or publicly distributing the data.
