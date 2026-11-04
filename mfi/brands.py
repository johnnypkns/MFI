"""Layer 1a — brand registry: every real chain operating in Madrid, cleaned.

    python -m mfi.brands      ->  data/app/brands.parquet, data/app/brand_members.parquet

Steps
 1. Names: normalised trade names on OPEN premises in the municipal census.
 2. Drop generic names: a name made only of activity words ("BAR", "RESTAURANTE", "SIN RÓTULO", "FARMACIA")
    is a description, not a brand. The activity vocabulary is taken from the census's own activity codes,
    plus a short manual list.
 3. Chains: names on >= MIN_STORES distinct locations (premises < 30 m apart count once).
 4. Merge spelling variants: "BURGUER KING" -> "BURGER KING"; "STARBUCKS COFFEE" -> "STARBUCKS";
    "MC DONALD'S VICÁLVARO" -> "MCDONALD'S". Prefix rule + fuzzy rule (union-find).
 5. Sector = most common activity division of the chain's premises; subcategory = most common activity code.
 6. Cross-check against OpenStreetMap brand tags (food & drink): n_osm per brand.
 7. Flag sectors that cannot be bought as a franchise in Spain (pharmacies, tobacconists, banks, lottery,
    public administration) — they stay in the registry as competitors/demand signals, not as products.
"""
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .model.export import APP, norm_name

MIN_STORES = 3

SECTORS = {  # CNAE division -> sector shown in the app
    "56": "Food & drink", "47": "Retail", "96": "Beauty & personal care", "93": "Fitness & leisure",
    "85": "Education", "86": "Health clinics", "75": "Veterinary & pets", "68": "Real estate agencies",
    "79": "Travel agencies", "45": "Automotive", "77": "Rental services", "95": "Repair services",
    "53": "Parcel & postal", "55": "Hotels", "92": "Betting & gaming", "64": "Banking", "65": "Insurance",
    "66": "Financial services", "78": "Employment agencies", "74": "Professional services", "82": "Office services",
    "81": "Building services", "88": "Social services", "87": "Care homes",
}
NOT_FRANCHISABLE = {
    "64": "banks are company-owned", "65": "insurers are company-owned", "84": "public administration",
    "94": "associations", "99": "international bodies",
}
NOT_FRANCHISABLE_CODES = {  # finer exclusions by activity code prefix
    "4773": "pharmacies: owner must be a licensed pharmacist (Spanish law)",
    "4726": "tobacconists (estancos): state concession",
    "9200": "state lottery outlets (Loterías y Apuestas del Estado)",
}
EXTRA_GENERIC = """SIN ROTULO NO TIENE NOMBRE SN S N LOCAL LOCALES BAJO BAJOS PRIMERO SEGUNDO DE DEL LA LAS EL LOS Y E EN
CON SL SA SLU CB SCOOP BAZAR CHINO ALIMENTACION ALIMENTOS FRUTAS FRUTERIA VERDURAS CARNICERIA CHARCUTERIA PESCADERIA
PANADERIA PASTELERIA CONFITERIA BOLLERIA KIOSCO KIOSKO QUIOSCO LOCUTORIO LOCUTORIOS ESTANCO ADMINISTRACION LOTERIAS
LOTERIA FARMACIA PARAFARMACIA OPTICA PELUQUERIA PELUQUERIAS BARBERIA ESTETICA CENTRO CLINICA DENTAL TALLER TALLERES
MECANICO GARAJE APARCAMIENTO PARKING OFICINA OFICINAS ALMACEN TIENDA TIENDAS COMERCIAL COMERCIO MERCADO GALERIA
BAR BARES CAFE CAFETERIA CAFETERIAS RESTAURANTE RESTAURANTES TABERNA MESON CERVECERIA ASADOR PIZZERIA KEBAB HAMBURGUESERIA
COCINA COMIDAS COMIDA BEBIDAS VINOS MERCERIA DROGUERIA PERFUMERIA FERRETERIA LIBRERIA PAPELERIA ZAPATERIA MODA
LAVANDERIA TINTORERIA ACADEMIA COLEGIO ESCUELA GIMNASIO INMOBILIARIA SEGUROS BANCO VIVIENDA TRASTERO VACIO VACANTE
CERRADO OBRAS NUEVO NUEVA MADRID ESPANA HERMANOS HNOS HIJOS VIUDA
MUESTRA MUESTRAS OPACA OPACAS CORPORATIVA CORPORATIVO LUMINOSA LUMINOSO ILUMINADA ILUMINADO PUBLICITARIA PUBLICITARIO
DINTEL ROTULO ROTULOS INFORMAR LETRAS CORPOREAS BANDEROLA TOLDO PLACA SOBRE PANEL CARTEL VINILO ESCAPARATE
CASA CASITA ESQUINA RINCON TERRAZA PATIO VILLA BODEGA BODEGUITA ANTIGUA ANTIGUO COPAS GASTROBAR CATERING KEBAB DONER
ESPECIFICAR ESPECIFICADO GABINETE JURIDICO PSICOLOGICO PARQUE PLAZA DESPACHO CONSULTA CONSULTORIO
SUPERMERCADO HIPER HIPERMERCADO MINIMARKET SR SRA NAILS UNAS MANICURA PEDICURA BARBER BARBERSHOP SHOP STORE BEAUTY SALON
STUDIO STUDIOS PELUQUEROS CABALLEROS SENORAS UNISEX DETERMINADO INDICAR INDICADO DEFINIDO IDENTIFICATIVO FISIOTERAP
FISIOTERAPIA FISIOTERAPEUTA PSICOLOGIA PSICOLOGO DENTISTA PODOLOGO PODOLOGIA PODOLOGICA ODONTOLOGICA ODONTOLOGIA MEDICA
MEDICO ESTETICO MASAJE MASAJES NOTARIA GESTORIA ASESORIA REFORMAS ESTUDIO PILATES YOGA ENTRENAMIENTO FORMACION DISCOTECA
FIESTAS SALA ESCAPE PROTESIS FINCAS ADMINISTRADOR AGENCIA SCHOOL ENGLISH IDIOMAS IGLESIA PARROQUIA POLICIA MINIALMACENES
AREA GOBIERNO MEDIO AMBIENTE JUNTA MUNICIPAL DISTRITO SERVICIO SERVICIOS PIE COMPRO ORO
OPACO OPACOS ABOGADO ABOGADOS ARQUITECTO NOTARIO DESCONOCIDO LABORATORIO FLORISTERIA
NO INFORMADO INFORMADA CONSTA FIGURA NOMBRE NINGUNO NINGUNA
TAPAS TAPERIA CHURRERIA HELADERIA HORNO OBRADOR BOCATERIA SANDWICHERIA BURGER PIZZA PIZZAS SUSHI WOK BUFFET""".split()
PHRASE_MAX_VARIANT_SHARE = 0.5   # a "chain" whose locations mostly carry DIFFERENT names is a phrase, not a brand


def _plain(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()


def generic_vocabulary(codes: pd.DataFrame):
    words = set(EXTRA_GENERIC)
    for col in ["epigrafe_desc", "division_desc", "section_desc"]:
        for v in codes[col].dropna():
            words.update(w for w in re.findall(r"[A-Z]+", _plain(v)) if len(w) >= 2)
    return words


def is_generic(raw_name, vocab):
    toks = [t for t in re.findall(r"[A-Z0-9]+", _plain(raw_name)) if len(t) >= 2 or t.isdigit()]
    def gen(t):  # plural/singular tolerant: SUPERMERCADO ~ SUPERMERCADOS, PELUQUERO ~ PELUQUEROS
        return t.isdigit() or t in vocab or t + "S" in vocab or t + "ES" in vocab or (t.endswith("S") and t[:-1] in vocab) \
            or (t.endswith("ES") and t[:-2] in vocab)
    return not toks or all(gen(t) for t in toks)


def _dedupe_count(xy, r=30):
    if len(xy) <= 1:
        return len(xy)
    tree, seen, n = cKDTree(xy), set(), 0
    for i, nb in enumerate(tree.query_ball_point(xy, r)):
        if i not in seen:
            n += 1
            seen.update(nb)
    return n


class UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build(data, osm=None, codes=None):
    codes = codes if codes is not None else pd.read_csv(APP / "activity_codes.csv", dtype=str)
    vocab = generic_vocabulary(codes)
    p = data.premises
    op = p[(p["status"] == "Abierto") & p["name_norm"].notna()].copy()
    op["generic"] = op["trade_name"].map(lambda n: is_generic(n, vocab))
    named = op[~op["generic"]]

    # 3. chains by distinct locations
    counts = named.groupby("name_norm").apply(lambda g: _dedupe_count(g[["x", "y"]].values), include_groups=False)
    canon = set(counts[counts >= MIN_STORES].index)

    # 4a. token-boundary prefix attach: "KFC MADRID SUR" -> KFC, but "INECO" does not attach to "INE"
    tok = named.drop_duplicates("name_norm").set_index("name_norm")["trade_name"].map(
        lambda n: [t for t in re.findall(r"[A-Z0-9]+", _plain(n))])
    member = {}
    for n, toks in tok.items():
        best = None
        for k in range(len(toks), 0, -1):
            key = "".join(toks[:k])
            if key in canon:
                best = key
                break
        if best:
            member[n] = best
    # 4b. merge chain keys: prefix ("STARBUCKSCOFFEE" -> "STARBUCKS"), containment for long keys
    #     ("CERVECERIA100MONTADITOS" -> "100MONTADITOS"), fuzzy spelling ("BURGUERKING" ~ "BURGERKING")
    uf = UF(canon)
    ck = sorted(canon, key=len)
    for i, a_ in enumerate(ck):
        for b_ in ck[i + 1:]:
            if (len(a_) >= 5 and b_.startswith(a_)) or (len(a_) >= 8 and a_ in b_) or \
               (min(len(a_), len(b_)) >= 6 and abs(len(a_) - len(b_)) <= 3 and SequenceMatcher(None, a_, b_).ratio() >= 0.9):
                uf.union(a_, b_)
    named = named[named["name_norm"].isin(member)].copy()
    named["canon_key"] = named["name_norm"].map(member)
    named["brand_key"] = named["canon_key"].map(uf.find)
    # 4c. phrase filter: share of locations whose name IS one of the cluster's chain keys
    exact = named["name_norm"].isin(canon)
    share = exact.groupby(named["brand_key"]).mean()
    phrases = set(share[share < PHRASE_MAX_VARIANT_SHARE].index)
    named = named[~named["brand_key"].isin(phrases) | exact]   # keep exact-name members of phrase clusters only

    # 5. sector, subcategory, display name, counts
    code_desc = codes.drop_duplicates("epigrafe_id").set_index("epigrafe_id")["epigrafe_desc"].to_dict()
    rows = []
    for key, g in named.groupby("brand_key"):
        epis = g["epigrafes"].dropna().str.split(",").explode()
        epis = epis[~epis.isin(["000000", "", None])]
        top_epi = epis.value_counts().index[0] if len(epis) else None
        div = top_epi[:2] if top_epi else None
        excl = NOT_FRANCHISABLE.get(div) or next((v for k, v in NOT_FRANCHISABLE_CODES.items() if top_epi and top_epi.startswith(k)), None)
        n = _dedupe_count(g[["x", "y"]].values)
        if n < MIN_STORES:
            continue
        rows.append(dict(
            brand_key=key, brand=Counter(g.loc[g["name_norm"] == key, "trade_name"] if (g["name_norm"] == key).any()
                                         else g["trade_name"]).most_common(1)[0][0].strip().title(),
            sector=SECTORS.get(div, "Other"), division=div, top_activity=top_epi,
            subcategory=(code_desc.get(top_epi) or "").capitalize() or None,
            n_stores=n, n_premises=len(g), n_districts=int(g["district_code"].nunique()),
            name_variants=int(g["name_norm"].nunique()),
            franchisable=excl is None, not_franchisable_reason=excl,
        ))
    br = pd.DataFrame(rows)

    # 6. OSM cross-check (food & drink brands)
    if osm is not None and len(br):
        ob = osm.dropna(subset=["brand_norm"])
        osm_counts = ob.groupby("brand_norm").size()
        osm_label = ob.groupby("brand_norm")["brand"].agg(lambda s: s.value_counts().index[0])
        def osm_match(k):
            return [b for b in osm_counts.index if b == k or (len(k) >= 5 and (b.startswith(k) or k.startswith(b)) and len(b) >= 5)]
        br["n_osm"] = br["brand_key"].map(lambda k: int(osm_counts[osm_match(k)].sum())).astype(int)
        lab = br["brand_key"].map(lambda k: osm_label[osm_match(k)[0]] if osm_match(k) else None)
        br["brand"] = lab.where(lab.notna(), br["brand"])      # OSM spells brands properly ("McDonald's", "KFC")
    br = br.sort_values("n_stores", ascending=False).reset_index(drop=True)
    br.insert(0, "brand_id", range(1, len(br) + 1))
    members = named[["name_norm", "brand_key"]].drop_duplicates().merge(br[["brand_key", "brand_id"]], on="brand_key")
    return br, members


def main():
    from .model.site_model import load_data
    from .stores import load_osm
    data = load_data()
    try:
        osm, _ = load_osm()
    except Exception:
        osm = None
    br, members = build(data, osm)
    br.to_parquet(APP / "brands.parquet", index=False)
    members.to_parquet(APP / "brand_members.parquet", index=False)
    print(f"{len(br):,} brands ({br.franchisable.sum():,} franchisable) | {len(members):,} name variants")
    print(br.groupby("sector")["brand_id"].count().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
