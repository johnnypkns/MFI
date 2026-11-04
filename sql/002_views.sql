-- "Latest" views. Pick the most recently retrieved file per source; history stays in the base tables.

CREATE OR REPLACE VIEW v_latest_file AS
SELECT DISTINCT ON (s.source_key) s.source_key, f.*
FROM source_files f JOIN sources s USING (source_id)
ORDER BY s.source_key, f.retrieved_at DESC;

CREATE OR REPLACE VIEW v_districts AS
SELECT d.* FROM districts d JOIN v_latest_file l ON l.file_id = d.file_id AND l.source_key = 'madrid_districts';

CREATE OR REPLACE VIEW v_neighbourhoods AS
SELECT n.* FROM neighbourhoods n JOIN v_latest_file l ON l.file_id = n.file_id AND l.source_key = 'madrid_barrios';

CREATE OR REPLACE VIEW v_census_sections AS
SELECT c.* FROM census_sections c JOIN v_latest_file l ON l.file_id = c.file_id AND l.source_key = 'madrid_sections';

CREATE OR REPLACE VIEW v_premises AS
SELECT p.* FROM premises p JOIN v_latest_file l ON l.file_id = p.file_id AND l.source_key = 'madrid_locales';

CREATE OR REPLACE VIEW v_premise_activities AS
SELECT a.* FROM premise_activities a JOIN v_latest_file l ON l.file_id = a.file_id AND l.source_key = 'madrid_actividades';

-- Population per section: totals + the age bands franchise models care about.
CREATE OR REPLACE VIEW v_section_population AS
SELECT d.section_code, d.reference_date,
       sum(population)                                         AS pop_total,
       sum(population) FILTER (WHERE age < 15)                 AS pop_0_14,
       sum(population) FILTER (WHERE age BETWEEN 15 AND 24)    AS pop_15_24,
       sum(population) FILTER (WHERE age BETWEEN 25 AND 44)    AS pop_25_44,
       sum(population) FILTER (WHERE age BETWEEN 45 AND 64)    AS pop_45_64,
       sum(population) FILTER (WHERE age >= 65)                AS pop_65_plus,
       sum(population) FILTER (WHERE nationality = 'foreign')  AS pop_foreign,
       round(sum(age::numeric * population) / nullif(sum(population),0), 1) AS mean_age
FROM demographics d JOIN v_latest_file l ON l.file_id = d.file_id AND l.source_key = 'madrid_padron'
GROUP BY d.section_code, d.reference_date;

CREATE OR REPLACE VIEW v_section_income AS
SELECT i.geo_code AS section_code, i.year,
       max(value_eur) FILTER (WHERE indicator = 'net_income_per_person')    AS net_income_per_person,
       max(value_eur) FILTER (WHERE indicator = 'net_income_per_household') AS net_income_per_household,
       max(value_eur) FILTER (WHERE indicator = 'median_income_per_cu')     AS median_income_per_cu
FROM income_statistics i JOIN v_latest_file l ON l.file_id = i.file_id AND l.source_key = 'ine_adrh_madrid'
WHERE i.geo_level = 'census_section'
  AND i.year = (SELECT max(year) FROM income_statistics WHERE file_id = l.file_id AND geo_level = 'census_section')
GROUP BY i.geo_code, i.year;

-- One row per section: geometry + people + money + commerce. Base layer for everything after M1.
CREATE OR REPLACE VIEW v_section_profile AS
SELECT c.section_code, c.district_code, c.barrio_code,
       round((ST_Area(c.geom) / 1e6)::numeric, 4)                          AS area_km2,
       p.pop_total, p.pop_15_24, p.pop_25_44, p.pop_65_plus, p.mean_age,
       round(p.pop_total / nullif(ST_Area(c.geom) / 1e6, 0))               AS pop_per_km2,
       inc.year AS income_year, inc.net_income_per_person, inc.net_income_per_household,
       (SELECT count(*) FROM v_premises pr
         WHERE pr.status = 'Abierto' AND pr.geom IS NOT NULL
           AND ST_Contains(c.geom, pr.geom))                               AS open_premises,
       c.geom
FROM v_census_sections c
LEFT JOIN v_section_population p USING (section_code)
LEFT JOIN v_section_income   inc USING (section_code);
