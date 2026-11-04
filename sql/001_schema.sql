-- Madrid Franchise Intelligence — Milestone 1 schema
-- Native CRS for all geometry: EPSG:25830 (ETRS89 / UTM 30N, metres). Same CRS the
-- Ayuntamiento publishes in, so distances/areas are correct without reprojection.
-- History rule: nothing is overwritten. Every fact row points at the source_file it
-- came from; "latest" is a view, not an UPDATE.

CREATE EXTENSION IF NOT EXISTS postgis;

-- ---------------------------------------------------------------- provenance
CREATE TABLE IF NOT EXISTS sources (
    source_id             serial PRIMARY KEY,
    source_key            text UNIQUE NOT NULL,      -- stable key used in code, e.g. 'madrid_padron'
    publisher             text NOT NULL,
    title                 text NOT NULL,
    dataset_url           text NOT NULL,
    license               text,
    geographic_resolution text,                      -- 'census_section' | 'barrio' | 'district' | 'point'
    data_type             text,                      -- 'official_register' | 'official_statistic' | 'boundary' | ...
    update_frequency      text,
    confidence            text NOT NULL DEFAULT 'high' CHECK (confidence IN ('high','medium','low')),
    notes                 text
);

CREATE TABLE IF NOT EXISTS research_runs (
    run_id       serial PRIMARY KEY,
    run_type     text NOT NULL,                      -- 'ingest' | 'validate' | later: 'research', 'score'
    started_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz,
    status       text NOT NULL DEFAULT 'running' CHECK (status IN ('running','ok','failed')),
    code_version text,
    params       jsonb,
    notes        text
);

CREATE TABLE IF NOT EXISTS source_files (
    file_id             serial PRIMARY KEY,
    source_id           int NOT NULL REFERENCES sources,
    run_id              int REFERENCES research_runs,
    url                 text NOT NULL,
    local_path          text NOT NULL,
    sha256              char(64) NOT NULL,
    bytes               bigint NOT NULL,
    retrieved_at        timestamptz NOT NULL DEFAULT now(),
    data_reference_date date,                        -- the date the data describes, not when we fetched it
    row_count           int,
    UNIQUE (source_id, sha256)                       -- same bytes = same file, never loaded twice
);

-- ---------------------------------------------------------------- geography
CREATE TABLE IF NOT EXISTS districts (
    district_code smallint NOT NULL,                 -- 1..21
    name          text NOT NULL,
    geom          geometry(MultiPolygon, 25830) NOT NULL,
    file_id       int NOT NULL REFERENCES source_files,
    PRIMARY KEY (district_code, file_id)
);

CREATE TABLE IF NOT EXISTS neighbourhoods (
    barrio_code   smallint NOT NULL,                 -- district*10 + barrio, e.g. 205 = Delicias
    district_code smallint NOT NULL,
    name          text NOT NULL,
    geom          geometry(MultiPolygon, 25830) NOT NULL,
    file_id       int NOT NULL REFERENCES source_files,
    PRIMARY KEY (barrio_code, file_id)
);

CREATE TABLE IF NOT EXISTS census_sections (
    section_code  char(10) NOT NULL,                 -- INE CUSEC: 28 079 DD SSS
    district_code smallint NOT NULL,
    barrio_code   smallint,
    geom          geometry(MultiPolygon, 25830) NOT NULL,
    file_id       int NOT NULL REFERENCES source_files,
    PRIMARY KEY (section_code, file_id)
);
CREATE INDEX IF NOT EXISTS census_sections_gix ON census_sections USING gist (geom);

-- ---------------------------------------------------------------- demographics
-- Padrón municipal, long format: one row per section x age x sex x nationality.
CREATE TABLE IF NOT EXISTS demographics (
    file_id        int NOT NULL REFERENCES source_files,
    reference_date date NOT NULL,
    section_code   char(10) NOT NULL,
    age            smallint NOT NULL,                -- 100 = 100+
    sex            char(1) NOT NULL CHECK (sex IN ('M','F')),
    nationality    text NOT NULL CHECK (nationality IN ('spanish','foreign')),
    population     int NOT NULL CHECK (population >= 0),
    PRIMARY KEY (file_id, section_code, age, sex, nationality)
);

-- INE Atlas de Distribución de Renta de los Hogares (ADRH). EUR, nominal.
CREATE TABLE IF NOT EXISTS income_statistics (
    file_id    int NOT NULL REFERENCES source_files,
    year       smallint NOT NULL,                    -- income year (tax year), not publication year
    geo_level  text NOT NULL CHECK (geo_level IN ('municipality','district','census_section')),
    geo_code   text NOT NULL,
    indicator  text NOT NULL,                        -- snake_case, see mfi/ingest/income.py
    value_eur  numeric(12,2),                        -- NULL = suppressed by INE ('..'), never imputed
    PRIMARY KEY (file_id, year, geo_level, geo_code, indicator)
);

-- ---------------------------------------------------------------- commercial premises
-- Censo de Locales y Actividades. A "premise" is a physical unit (local); it can
-- carry several activities. These are NOT rental listings (that is `properties`, M5).
CREATE TABLE IF NOT EXISTS premises (
    file_id        int NOT NULL REFERENCES source_files,
    snapshot_date  date NOT NULL,
    id_local       bigint NOT NULL,
    district_code  smallint,
    barrio_code    smallint,
    section_code   char(10),
    status         text,                             -- 'Abierto', 'Cerrado', 'Baja', ...
    access_type    text,                             -- 'Puerta Calle' (street level) vs 'Agrupado' (inside a mall/gallery)
    street_type    text,
    street_name    text,
    street_number  text,
    trade_name     text,                             -- rótulo
    postal_code    text,
    building_id    bigint,                           -- id_ndp_edificio: the street-number address the premise belongs to
    geom           geometry(Point, 25830),           -- NULL when no coordinate could be derived
    geom_source    text CHECK (geom_source IN ('local','agrupacion','building')),
    PRIMARY KEY (file_id, id_local)
);
CREATE INDEX IF NOT EXISTS premises_gix ON premises USING gist (geom);

CREATE TABLE IF NOT EXISTS premise_activities (
    file_id        int NOT NULL REFERENCES source_files,
    snapshot_date  date NOT NULL,
    id_local       bigint NOT NULL,
    section_id     text,                             -- CNAE-style section letter, e.g. 'I' Hostelería
    section_desc   text,
    division_id    text,                             -- e.g. '56' food & beverage service
    division_desc  text,
    epigrafe_id    text,                             -- finest activity code, e.g. '561004' bar-restaurante
    epigrafe_desc  text
);
CREATE INDEX IF NOT EXISTS premise_activities_local_ix ON premise_activities (file_id, id_local);
CREATE INDEX IF NOT EXISTS premise_activities_epi_ix   ON premise_activities (epigrafe_id);

-- ---------------------------------------------------------------- QC
CREATE TABLE IF NOT EXISTS validation_results (
    run_id     int NOT NULL REFERENCES research_runs,
    check_name text NOT NULL,
    status     text NOT NULL CHECK (status IN ('pass','warn','fail')),
    observed   text,
    expected   text,
    detail     text,
    checked_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, check_name)
);
