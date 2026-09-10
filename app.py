import os
import re
import unicodedata
import io

import numpy as np
import pandas as pd
from scipy.stats import poisson
import streamlit as st

# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(
    page_title="Predicción Liga 1 Perú - V2",
    page_icon="⚽",
    layout="wide",
)

st.markdown("""
<style>
.metric-card {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    border: 1px solid #e9ecef;
}
.suggestion-box-blue {
    background-color: #e8f0fe;
    color: #1a73e8;
    padding: 12px 16px;
    border-radius: 8px;
    font-weight: 500;
    margin-top: 10px;
}
.suggestion-box-green {
    background-color: #e6f4ea;
    color: #137333;
    padding: 12px 16px;
    border-radius: 8px;
    font-weight: 500;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 1. NORMALIZACIÓN DE EQUIPOS
# =========================================================
def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFD", texto).encode(
        "ascii", "ignore"
    ).decode("utf-8")
    texto = re.sub(r"[^a-zA-Z0-9\s]", " ", texto)
    return " ".join(texto.lower().split())


# IMPORTANTE:
# FC Cajamarca y UTC Cajamarca son equipos DIFERENTES.
# También se conserva "atletico" porque distingue
# Atlético Grau y Alianza Atlético.
DICCIONARIO_EQUIPOS = {
    "juan pablo ii college": "colegio juan pablo ii",
    "juan pablo ii": "colegio juan pablo ii",
    "colegio juan pablo ii": "colegio juan pablo ii",

    "los chankas": "los chankas",
    "chankas": "los chankas",
    "chankas cyc": "los chankas",

    "sporting cristal": "sporting cristal",
    "cristal": "sporting cristal",
    "alianza lima": "alianza lima",
    "universitario": "universitario",
    "universitario de deportes": "universitario",
    "sport boys": "sport boys",

    "alianza atletico": "alianza atletico",
    "alianza atletico sullana": "alianza atletico",
    "atletico grau": "atletico grau",

    "comerciantes unidos": "comerciantes unidos",
    "comerciantes unidos de cutervo": "comerciantes unidos",

    "fbc melgar": "melgar",
    "melgar": "melgar",

    "deportivo garcilaso": "garcilaso",
    "garcilaso": "garcilaso",

    "cusco fc": "cusco",
    "cusco": "cusco",
    "cienciano": "cienciano",

    "deporte huancayo": "sport huancayo",
    "sport huancayo": "sport huancayo",

    "fc cajamarca": "fc cajamarca",
    "fc cajamarca futbol club": "fc cajamarca",
    "utc cajamarca": "utc cajamarca",
    "utc": "utc cajamarca",
    "ut c": "utc cajamarca",

    "adt": "adt",
    "adt tarma": "adt",

    "cd moquegua": "cd moquegua",
    "moquegua": "cd moquegua",
}


ALTITUDES_DEFAULT = {
    "adt": (3050, "Tarma"),
    "cienciano": (3360, "Cusco"),
    "cusco": (3360, "Cusco"),
    "garcilaso": (3360, "Cusco"),
    "sport huancayo": (3250, "Huancayo"),
    "los chankas": (2920, "Andahuaylas"),
    "utc cajamarca": (2750, "Cajamarca"),
    "fc cajamarca": (2750, "Cajamarca"),
    "comerciantes unidos": (2620, "Cutervo"),
    "melgar": (2335, "Arequipa"),
    "universitario": (150, "Lima"),
    "alianza lima": (150, "Lima"),
    "sporting cristal": (150, "Lima"),
    "sport boys": (10, "Callao"),
    "atletico grau": (50, "Piura"),
    "alianza atletico": (50, "Sullana"),
    "colegio juan pablo ii": (150, "Chongoyape"),
    "cd moquegua": (1410, "Moquegua"),
}


def estandarizar_nombre(nombre):
    txt = normalizar_texto(nombre)

    # Primero coincidencia exacta.
    if txt in DICCIONARIO_EQUIPOS:
        return DICCIONARIO_EQUIPOS[txt]

    # Solo eliminar palabras genéricas.
    txt2 = re.sub(
        r"\b(club|fc|cd|fbc|deportivo|asociacion)\b",
        " ",
        txt
    )
    txt2 = " ".join(txt2.split())

    if txt2 in DICCIONARIO_EQUIPOS:
        return DICCIONARIO_EQUIPOS[txt2]

    return txt


def resolver_columna_club(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    posibles = [
        "club", "equipo", "nombre", "team",
        "clubes", "equipos"
    ]

    for pos in posibles:
        if pos in df.columns:
            df = df.rename(columns={pos: "club"})
            break

    if "club" not in df.columns and len(df.columns) > 0:
        df["club"] = df.iloc[:, 0]

    return df


# =========================================================
# 2. CARGA DE DATOS
# =========================================================
def obtener_ruta_excel(nombre_archivo="Liga1_2026.xlsx"):
    try:
        directorio = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        directorio = os.getcwd()
    return os.path.join(directorio, nombre_archivo)


def preparar_partidos(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "local" not in df.columns or "visita" not in df.columns:
        raise ValueError(
            "Partidos_Fecha debe tener las columnas local y visita."
        )

    if "jornada" in df.columns:
        extra = df["jornada"].astype(str).str.extract(r"(\d+)")[0]
        df["jornada_num"] = extra.fillna("0")
    elif "fecha_num" in df.columns:
        df["jornada_num"] = df["fecha_num"].astype(str)
    else:
        df["jornada_num"] = "0"

    df["fecha_str"] = (
        df["fecha"].astype(str)
        if "fecha" in df.columns
        else ""
    )

    if "hora" not in df.columns:
        df["hora"] = "15:00"

    if "goles_local" not in df.columns:
        df["goles_local"] = np.nan

    if "goles_visita" not in df.columns:
        df["goles_visita"] = np.nan

    if "jugado" not in df.columns:
        df["jugado"] = False

    df["local_std"] = df["local"].apply(estandarizar_nombre)
    df["visita_std"] = df["visita"].apply(estandarizar_nombre)

    return df


def cargar_datos_excel(fuente):
    try:
        xls = pd.ExcelFile(fuente)
        hojas = xls.sheet_names

        if "Partidos_Fecha" not in hojas:
            raise ValueError("Falta la hoja Partidos_Fecha.")
        if "Tabla_Acumulada" not in hojas:
            raise ValueError("Falta la hoja Tabla_Acumulada.")

        df_partidos = preparar_partidos(
            pd.read_excel(xls, sheet_name="Partidos_Fecha")
        )

        df_geo = None
        if "Data_Geografica" in hojas:
            df_geo = pd.read_excel(
                xls, sheet_name="Data_Geografica"
            )
            df_geo = resolver_columna_club(df_geo)
            df_geo["equipo_std"] = df_geo["club"].apply(
                estandarizar_nombre
            )
        else:
            df_geo = pd.DataFrame([
                {
                    "equipo_std": k,
                    "altitud": v[0],
                    "ciudad": v[1],
                }
                for k, v in ALTITUDES_DEFAULT.items()
            ])

        df_acum = pd.read_excel(
            xls, sheet_name="Tabla_Acumulada"
        )
        df_acum = resolver_columna_club(df_acum)
        df_acum["equipo_std"] = df_acum["club"].apply(
            estandarizar_nombre
        )

        if "Tabla_Clausura" in hojas:
            df_claus = pd.read_excel(
                xls, sheet_name="Tabla_Clausura"
            )
            df_claus = resolver_columna_club(df_claus)
            df_claus["equipo_std"] = df_claus["club"].apply(
                estandarizar_nombre
            )
        else:
            df_claus = df_acum.copy()

        df_apertura = None
        df_clausura_res = None

        if "Resultados_Apertura" in hojas:
            df_apertura = preparar_partidos(
                pd.read_excel(
                    xls, sheet_name="Resultados_Apertura"
                )
            )

        if "Resultados_Clausura" in hojas:
            df_clausura_res = preparar_partidos(
                pd.read_excel(
                    xls, sheet_name="Resultados_Clausura"
                )
            )

        return (
            df_partidos,
            df_acum,
            df_claus,
            df_geo,
            df_apertura,
            df_clausura_res,
            None,
        )

    except Exception as e:
        return (
            None, None, None, None,
            None, None, str(e)
        )


def generar_bytes_excel(
    df_partidos,
    df_acum,
    df_claus,
    df_geo,
    df_apertura=None,
    df_clausura_res=None,
):
    buffer = io.BytesIO()

    with pd.ExcelWriter(
        buffer, engine="openpyxl"
    ) as writer:
        df_partidos.to_excel(
            writer,
            sheet_name="Partidos_Fecha",
            index=False,
        )
        df_acum.to_excel(
            writer,
            sheet_name="Tabla_Acumulada",
            index=False,
        )
        df_claus.to_excel(
            writer,
            sheet_name="Tabla_Clausura",
            index=False,
        )
        df_geo.to_excel(
            writer,
            sheet_name="Data_Geografica",
            index=False,
        )

        if df_apertura is not None:
            df_apertura.to_excel(
                writer,
                sheet_name="Resultados_Apertura",
                index=False,
            )

        if df_clausura_res is not None:
            df_clausura_res.to_excel(
                writer,
                sheet_name="Resultados_Clausura",
                index=False,
            )

    buffer.seek(0)
    return buffer


def guardar_cambios_excel(
    df_partidos,
    df_acum,
    df_claus,
    df_geo,
    ruta,
    df_apertura=None,
    df_clausura_res=None,
):
    try:
        with pd.ExcelWriter(
            ruta, engine="openpyxl"
        ) as writer:
            df_partidos.to_excel(
                writer,
                sheet_name="Partidos_Fecha",
                index=False,
            )
            df_acum.to_excel(
                writer,
                sheet_name="Tabla_Acumulada",
                index=False,
            )
            df_claus.to_excel(
                writer,
                sheet_name="Tabla_Clausura",
                index=False,
            )
            df_geo.to_excel(
                writer,
                sheet_name="Data_Geografica",
                index=False,
            )

            if df_apertura is not None:
                df_apertura.to_excel(
                    writer,
                    sheet_name="Resultados_Apertura",
                    index=False,
                )

            if df_clausura_res is not None:
                df_clausura_res.to_excel(
                    writer,
                    sheet_name="Resultados_Clausura",
                    index=False,
                )

        return True, "Cambios guardados con éxito."

    except Exception as e:
        return False, str(e)


# =========================================================
# 3. RIVALIDADES
# =========================================================
RIVALIDADES = {
    frozenset(
        ["atletico grau", "alianza atletico"]
    ): "Clásico Piurano",

    frozenset(
        ["cusco", "cienciano"]
    ): "Clásico Cusqueño",

    frozenset(
        ["fc cajamarca", "utc cajamarca"]
    ): "Clásico Cajamarquino",

    frozenset(
        ["fc cajamarca", "comerciantes unidos"]
    ): "Rivalidad Cajamarquina",

    frozenset(
        ["utc cajamarca", "comerciantes unidos"]
    ): "Rivalidad Cajamarquina",
}


def obtener_rivalidad(local, visita):
    clave = frozenset([local, visita])

    if clave in RIVALIDADES:
        return 1, RIVALIDADES[clave]

    return 0, "Sin rivalidad especial"


# =========================================================
# 4. HISTORIAL DE RESULTADOS
# =========================================================
def combinar_resultados(df_apertura, df_clausura):
    frames = []

    for df in [df_apertura, df_clausura]:
        if df is None or df.empty:
            continue

        d = df.copy()

        if "local_std" not in d.columns:
            d["local_std"] = d["local"].apply(
                estandarizar_nombre
            )

        if "visita_std" not in d.columns:
            d["visita_std"] = d["visita"].apply(
                estandarizar_nombre
            )

        d["goles_local_num"] = pd.to_numeric(
            d.get("goles_local", np.nan),
            errors="coerce",
        )

        d["goles_visita_num"] = pd.to_numeric(
            d.get("goles_visita", np.nan),
            errors="coerce",
        )

        d = d.dropna(
            subset=[
                "goles_local_num",
                "goles_visita_num",
            ]
        )

        frames.append(d)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames, ignore_index=True
    )


def historial_desde_partidos(df_partidos):
    d = df_partidos.copy()

    d["gl"] = pd.to_numeric(
        d["goles_local"], errors="coerce"
    )
    d["gv"] = pd.to_numeric(
        d["goles_visita"], errors="coerce"
    )

    if "jugado" in d.columns:
        jugado = d["jugado"].astype(str).str.lower().isin(
            ["true", "1", "si", "sí", "yes"]
        )
    else:
        jugado = d["gl"].notna() & d["gv"].notna()

    d = d[
        jugado
        & d["gl"].notna()
        & d["gv"].notna()
    ].copy()

    if d.empty:
        return pd.DataFrame()

    d["goles_local_num"] = d["gl"]
    d["goles_visita_num"] = d["gv"]

    return d


def ordenar_historico(df):
    if df.empty:
        return df

    d = df.copy()

    if "fecha" in d.columns:
        d["_fecha"] = pd.to_datetime(
            d["fecha"], errors="coerce"
        )
    elif "fecha_str" in d.columns:
        d["_fecha"] = pd.to_datetime(
            d["fecha_str"], errors="coerce"
        )
    else:
        d["_fecha"] = pd.NaT

    d["_jornada"] = pd.to_numeric(
        d.get("jornada_num", 0),
        errors="coerce",
    ).fillna(0)

    return d.sort_values(
        ["_fecha", "_jornada"],
        na_position="last"
    ).reset_index(drop=True)


# =========================================================
# 5. ELO
# =========================================================
ELO_INICIAL = 1500.0
ELO_K = 24.0
VENTAJA_ELO_LOCAL = 55.0


def calcular_elo_historico(df_hist):
    elo = {}
    partidos_elo = {}

    if df_hist.empty:
        return elo, partidos_elo

    hist = ordenar_historico(df_hist)

    for _, row in hist.iterrows():
        local = row.get(
            "local_std",
            estandarizar_nombre(
                row.get("local", "")
            )
        )
        visita = row.get(
            "visita_std",
            estandarizar_nombre(
                row.get("visita", "")
            )
        )

        if not local or not visita or local == visita:
            continue

        gl = pd.to_numeric(
            row.get("goles_local_num", np.nan),
            errors="coerce",
        )
        gv = pd.to_numeric(
            row.get("goles_visita_num", np.nan),
            errors="coerce",
        )

        if pd.isna(gl) or pd.isna(gv):
            continue

        elo.setdefault(local, ELO_INICIAL)
        elo.setdefault(visita, ELO_INICIAL)
        partidos_elo.setdefault(local, 0)
        partidos_elo.setdefault(visita, 0)

        if gl > gv:
            resultado = 1.0
        elif gl == gv:
            resultado = 0.5
        else:
            resultado = 0.0

        diferencia = (
            elo[local]
            + VENTAJA_ELO_LOCAL
            - elo[visita]
        )

        esperado = 1.0 / (
            1.0 + 10.0 ** (-diferencia / 400.0)
        )

        cambio = ELO_K * (
            resultado - esperado
        )

        elo[local] += cambio
        elo[visita] -= cambio

        partidos_elo[local] += 1
        partidos_elo[visita] += 1

    return elo, partidos_elo


# =========================================================
# 6. FORMA RECIENTE
# =========================================================
def obtener_forma_equipo(equipo, df_hist, n=5):
    base = {
        "partidos": 0,
        "puntos_promedio": 1.0,
        "gf_promedio": 1.25,
        "gc_promedio": 1.25,
        "diferencia_promedio": 0.0,
        "victorias": 0,
        "empates": 0,
        "derrotas": 0,
    }

    if df_hist.empty:
        return base

    h = df_hist.copy()

    if "local_std" not in h.columns:
        h["local_std"] = h["local"].apply(
            estandarizar_nombre
        )

    if "visita_std" not in h.columns:
        h["visita_std"] = h["visita"].apply(
            estandarizar_nombre
        )

    local = h[
        h["local_std"] == equipo
    ].copy()

    local["gf_eq"] = local[
        "goles_local_num"
    ]
    local["gc_eq"] = local[
        "goles_visita_num"
    ]

    local["puntos"] = np.where(
        local["gf_eq"] > local["gc_eq"],
        3,
        np.where(
            local["gf_eq"] == local["gc_eq"],
            1,
            0,
        ),
    )

    visita = h[
        h["visita_std"] == equipo
    ].copy()

    visita["gf_eq"] = visita[
        "goles_visita_num"
    ]
    visita["gc_eq"] = visita[
        "goles_local_num"
    ]

    visita["puntos"] = np.where(
        visita["gf_eq"] > visita["gc_eq"],
        3,
        np.where(
            visita["gf_eq"] == visita["gc_eq"],
            1,
            0,
        ),
    )

    partidos = pd.concat(
        [local, visita],
        ignore_index=True
    )

    if partidos.empty:
        return base

    partidos = ordenar_historico(
        partidos
    ).tail(n)

    puntos = partidos["puntos"].astype(float)
    gf = partidos["gf_eq"].astype(float)
    gc = partidos["gc_eq"].astype(float)

    return {
        "partidos": len(partidos),
        "puntos_promedio": float(
            puntos.mean()
        ),
        "gf_promedio": float(
            gf.mean()
        ),
        "gc_promedio": float(
            gc.mean()
        ),
        "diferencia_promedio": float(
            (gf - gc).mean()
        ),
        "victorias": int(
            (puntos == 3).sum()
        ),
        "empates": int(
            (puntos == 1).sum()
        ),
        "derrotas": int(
            (puntos == 0).sum()
        ),
    }


# =========================================================
# 7. FUERZA DE TABLA
# =========================================================
def valor_numerico(row, columna, defecto):
    if columna not in row.index:
        return defecto

    valor = pd.to_numeric(
        row[columna], errors="coerce"
    )

    if pd.isna(valor):
        return defecto

    return float(valor)


def obtener_fuerza_equipos(
    equipo_local,
    equipo_visita,
    df_tabla
):
    df_t = df_tabla.copy()

    if "equipo_std" not in df_t.columns:
        df_t = resolver_columna_club(df_t)
        df_t["equipo_std"] = df_t[
            "club"
        ].apply(estandarizar_nombre)

    row_loc = df_t[
        df_t["equipo_std"] == equipo_local
    ]

    row_vis = df_t[
        df_t["equipo_std"] == equipo_visita
    ]

    if not row_loc.empty:
        r = row_loc.iloc[0]
        gf_loc = valor_numerico(
            r, "gf", 1.25
        )
        gc_loc = valor_numerico(
            r, "gc", 1.25
        )
        pj_loc = max(
            1.0,
            valor_numerico(r, "pj", 1.0)
        )
    else:
        gf_loc, gc_loc, pj_loc = (
            1.25, 1.25, 1.0
        )

    if not row_vis.empty:
        r = row_vis.iloc[0]
        gf_vis = valor_numerico(
            r, "gf", 1.25
        )
        gc_vis = valor_numerico(
            r, "gc", 1.25
        )
        pj_vis = max(
            1.0,
            valor_numerico(r, "pj", 1.0)
        )
    else:
        gf_vis, gc_vis, pj_vis = (
            1.25, 1.25, 1.0
        )

    return (
        gf_loc / pj_loc,
        gc_loc / pj_loc,
        gf_vis / pj_vis,
        gc_vis / pj_vis,
    )


# =========================================================
# 8. GEOGRAFÍA
# =========================================================
def obtener_info_geo(equipo, df_geo):
    if df_geo is not None and not df_geo.empty:
        df_g = df_geo.copy()

        if "equipo_std" not in df_g.columns:
            df_g = resolver_columna_club(df_g)
            df_g["equipo_std"] = df_g[
                "club"
            ].apply(estandarizar_nombre)

        row = df_g[
            df_g["equipo_std"] == equipo
        ]

        if not row.empty:
            r = row.iloc[0]

            alt = pd.to_numeric(
                r.get("altitud", np.nan),
                errors="coerce",
            )

            ciudad = str(
                r.get("ciudad", "")
            )

            if pd.isna(alt):
                alt = ALTITUDES_DEFAULT.get(
                    equipo, (150, "Lima")
                )[0]

            if ciudad.lower() in [
                "nan", "", "none"
            ]:
                ciudad = ALTITUDES_DEFAULT.get(
                    equipo, (150, "Lima")
                )[1]

            return float(alt), ciudad

    return ALTITUDES_DEFAULT.get(
        equipo, (150, "Lima")
    )


def factor_altitud(
    alt_local,
    alt_visita
):
    # Ajuste moderado, deliberadamente menor
    # que los coeficientes de la versión anterior.
    diferencia = max(
        0.0,
        alt_local - alt_visita
    )

    if alt_local < 1800:
        f_loc, f_vis = 1.00, 1.00
    elif alt_local < 2500:
        f_loc, f_vis = 1.02, 0.96
    elif alt_local < 3000:
        f_loc, f_vis = 1.04, 0.92
    else:
        f_loc, f_vis = 1.06, 0.88

    extra = min(
        0.06,
        diferencia / 10000.0
    )

    f_vis *= 1.0 - extra
    f_loc *= 1.0 + extra * 0.35

    return f_loc, f_vis


# =========================================================
# 9. DIXON-COLES + ELO + FORMA + RIVALIDAD
# =========================================================
def tau_dixon_coles(
    x,
    y,
    lambda_local,
    mu_visita,
    rho=-0.08
):
    if x == 0 and y == 0:
        return max(
            0.0001,
            1.0 - (
                lambda_local
                * mu_visita
                * rho
            )
        )

    if x == 1 and y == 0:
        return max(
            0.0001,
            1.0 + (
                mu_visita * rho
            )
        )

    if x == 0 and y == 1:
        return max(
            0.0001,
            1.0 + (
                lambda_local * rho
            )
        )

    if x == 1 and y == 1:
        return max(
            0.0001,
            1.0 - rho
        )

    return 1.0


def calcular_dixon_coles_v2(
    equipo_local,
    equipo_visita,
    df_tabla,
    df_geo,
    df_hist,
    elo,
    partidos_elo,
    rho=-0.08,
):
    att_loc, def_loc, att_vis, def_vis = (
        obtener_fuerza_equipos(
            equipo_local,
            equipo_visita,
            df_tabla,
        )
    )

    forma_loc = obtener_forma_equipo(
        equipo_local, df_hist, 5
    )
    forma_vis = obtener_forma_equipo(
        equipo_visita, df_hist, 5
    )

    dif_forma = (
        forma_loc["puntos_promedio"]
        - forma_vis["puntos_promedio"]
    )

    ajuste_forma_loc = np.clip(
        1.0 + 0.08 * dif_forma / 3.0,
        0.92,
        1.08,
    )

    ajuste_forma_vis = np.clip(
        1.0 - 0.08 * dif_forma / 3.0,
        0.92,
        1.08,
    )

    elo_loc = float(
        elo.get(equipo_local, ELO_INICIAL)
    )
    elo_vis = float(
        elo.get(equipo_visita, ELO_INICIAL)
    )

    dif_elo = (
        elo_loc
        + VENTAJA_ELO_LOCAL
        - elo_vis
    )

    ajuste_elo = np.clip(
        1.0 + 0.00035 * dif_elo,
        0.88,
        1.12,
    )

    alt_loc, ciudad_loc = obtener_info_geo(
        equipo_local, df_geo
    )
    alt_vis, ciudad_vis = obtener_info_geo(
        equipo_visita, df_geo
    )

    home_advantage = (
        1.14
        if alt_loc < 1800
        else 1.18
    )

    factor_alt_loc, factor_alt_vis = (
        factor_altitud(
            alt_loc,
            alt_vis,
        )
    )

    rivalidad, nombre_rivalidad = (
        obtener_rivalidad(
            equipo_local,
            equipo_visita,
        )
    )

    # La rivalidad MODERA la diferencia de fuerza.
    # No impone 50/50.
    ajuste_rivalidad = 1.0

    if rivalidad:
        diferencia_ataque = np.clip(
            att_loc - att_vis,
            -1.0,
            1.0,
        )

        ajuste_rivalidad = (
            1.0
            - 0.022
            * diferencia_ataque
        )

    base_local = (
        att_loc
        * (def_vis / 1.20)
        * home_advantage
    )

    base_visita = (
        att_vis
        * (def_loc / 1.20)
    )

    lambda_local = (
        base_local
        * (0.65 + 0.35 * ajuste_elo)
        * ajuste_forma_loc
        * factor_alt_loc
        * ajuste_rivalidad
    )

    mu_visita = (
        base_visita
        * (
            0.65
            + 0.35 * (2.0 - ajuste_elo)
        )
        * ajuste_forma_vis
        * factor_alt_vis
    )

    lambda_local = float(
        np.clip(
            lambda_local,
            0.45,
            3.50,
        )
    )

    mu_visita = float(
        np.clip(
            mu_visita,
            0.30,
            3.00,
        )
    )

    max_goles = 9
    matriz = np.zeros(
        (max_goles, max_goles)
    )

    for x in range(max_goles):
        for y in range(max_goles):
            matriz[x, y] = max(
                0.0,
                poisson.pmf(
                    x, lambda_local
                )
                * poisson.pmf(
                    y, mu_visita
                )
                * tau_dixon_coles(
                    x,
                    y,
                    lambda_local,
                    mu_visita,
                    rho,
                ),
            )

    total = matriz.sum()

    if total <= 0:
        raise ValueError(
            "No se pudo construir la matriz."
        )

    matriz /= total

    p_local = float(
        np.tril(matriz, -1).sum()
    )
    p_empate = float(
        np.trace(matriz)
    )
    p_visita = float(
        np.triu(matriz, 1).sum()
    )

    p_under25 = float(
        sum(
            matriz[x, y]
            for x in range(max_goles)
            for y in range(max_goles)
            if x + y <= 2
        )
    )

    p_over25 = 1.0 - p_under25

    p_btts_si = float(
        matriz[1:, 1:].sum()
    )
    p_btts_no = 1.0 - p_btts_si

    marcador = np.unravel_index(
        np.argmax(matriz),
        matriz.shape,
    )

    return {
        "p_local": p_local,
        "p_empate": p_empate,
        "p_visita": p_visita,
        "p_over25": p_over25,
        "p_under25": p_under25,
        "p_btts_si": p_btts_si,
        "p_btts_no": p_btts_no,
        "lambda_local": lambda_local,
        "mu_visita": mu_visita,
        "altitud": alt_loc,
        "altitud_visita": alt_vis,
        "ciudad_local": ciudad_loc,
        "ciudad_visita": ciudad_vis,
        "elo_local": elo_loc,
        "elo_visita": elo_vis,
        "dif_elo": dif_elo,
        "forma_local": forma_loc,
        "forma_visita": forma_vis,
        "rivalidad": rivalidad,
        "nombre_rivalidad": nombre_rivalidad,
        "marcador_modal": (
            f"{marcador[0]}-{marcador[1]}"
        ),
        "partidos_elo_local": partidos_elo.get(
            equipo_local, 0
        ),
        "partidos_elo_visita": partidos_elo.get(
            equipo_visita, 0
        ),
    }


# =========================================================
# 10. SUGERENCIAS
# =========================================================
def generar_sugerencia(
    p_loc,
    p_emp,
    p_vis,
    nombre_loc,
    nombre_vis,
):
    probs = {
        f"Gana {nombre_loc}": p_loc,
        "Empate": p_emp,
        f"Gana {nombre_vis}": p_vis,
    }

    orden = sorted(
        probs.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    mejor, p_mejor = orden[0]
    _, p_segundo = orden[1]

    diferencia = p_mejor - p_segundo

    if p_mejor >= 0.55 and diferencia >= 0.10:
        confianza = "Alta"
    elif p_mejor >= 0.45 and diferencia >= 0.06:
        confianza = "Media-Alta"
    elif p_mejor >= 0.38:
        confianza = "Media"
    else:
        confianza = "Baja"

    if confianza == "Baja":
        if p_loc + p_emp >= 0.62:
            sugerencia = (
                f"1X — {nombre_loc} o Empate"
            )
        elif p_vis + p_emp >= 0.62:
            sugerencia = (
                f"X2 — Empate o {nombre_vis}"
            )
        else:
            sugerencia = mejor
    else:
        sugerencia = mejor

    return sugerencia, confianza


def nivel_confianza(p):
    if p >= 0.65:
        return "Alta"
    if p >= 0.55:
        return "Media-Alta"
    if p >= 0.45:
        return "Media"
    return "Baja"


# =========================================================
# 11. INTERFAZ STREAMLIT
# =========================================================
st.title(
    "⚽ Modelo de Predicción Liga 1 Perú — V2"
)
st.caption(
    "Dixon-Coles + Poisson + Elo + forma + tabla + "
    "altitud + rivalidades"
)

st.sidebar.header("📁 Archivo de Datos")

archivo_subido = st.sidebar.file_uploader(
    "Subir archivo Excel (Liga1_2026.xlsx)",
    type=["xlsx"],
)

ruta_local = obtener_ruta_excel(
    "Liga1_2026.xlsx"
)

fuente_excel = (
    archivo_subido
    if archivo_subido is not None
    else (
        ruta_local
        if os.path.exists(ruta_local)
        else None
    )
)

if st.sidebar.button("🔄 Recargar Datos"):
    st.session_state.clear()
    st.rerun()

if fuente_excel is None:
    st.error(
        "❌ No se encontró Liga1_2026.xlsx. "
        "Sube el archivo Excel en el menú lateral."
    )
    st.stop()

if (
    "df_partidos" not in st.session_state
    or archivo_subido is not None
):
    (
        df_p,
        df_a,
        df_c,
        df_g,
        df_ap,
        df_cr,
        err,
    ) = cargar_datos_excel(
        fuente_excel
    )

    if err:
        st.error(
            f"Error al procesar el Excel: {err}"
        )
        st.stop()

    st.session_state.df_partidos = df_p
    st.session_state.df_acum = df_a
    st.session_state.df_claus = df_c
    st.session_state.df_geo = df_g
    st.session_state.df_apertura = df_ap
    st.session_state.df_clausura_res = df_cr


# =========================================================
# HISTORIAL + ELO
# =========================================================
historial = combinar_resultados(
    st.session_state.df_apertura,
    st.session_state.df_clausura_res,
)

hist_partidos = historial_desde_partidos(
    st.session_state.df_partidos
)

if not hist_partidos.empty:
    historial = pd.concat(
        [historial, hist_partidos],
        ignore_index=True,
    )

if not historial.empty:
    cols = [
        c for c in [
            "fecha_str",
            "local_std",
            "visita_std",
            "goles_local_num",
            "goles_visita_num",
        ]
        if c in historial.columns
    ]

    if cols:
        historial = historial.drop_duplicates(
            subset=cols,
            keep="first",
        )

elo, partidos_elo = calcular_elo_historico(
    historial
)


# =========================================================
# SIDEBAR
# =========================================================
tabla_ref = st.sidebar.radio(
    "Tabla de Rendimiento:",
    (
        "Tabla Acumulada",
        "Tabla Clausura",
    ),
    index=0,
)

df_tabla_act = (
    st.session_state.df_acum
    if tabla_ref == "Tabla Acumulada"
    else st.session_state.df_claus
)

jornadas = sorted(
    st.session_state.df_partidos[
        "jornada_num"
    ].astype(str).unique(),
    key=lambda x: (
        int(x)
        if str(x).isdigit()
        else 9999
    ),
)

if not jornadas:
    st.error("No se encontraron jornadas.")
    st.stop()

idx_default = (
    jornadas.index("8")
    if "8" in jornadas
    else 0
)

jornada_sel = st.sidebar.selectbox(
    "Seleccionar Jornada:",
    jornadas,
    format_func=lambda x: (
        f"Jornada {x}"
    ),
    index=idx_default,
)

with st.sidebar.expander(
    "ℹ️ Componentes del modelo",
    expanded=False,
):
    st.write("**Activos:**")
    st.write("• Tabla de rendimiento")
    st.write("• Apertura + Clausura")
    st.write("• Forma últimos 5 partidos")
    st.write("• Elo histórico")
    st.write("• Localía")
    st.write("• Altitud")
    st.write("• Rivalidades")
    st.write("• Poisson + Dixon-Coles")
    st.caption(
        "Los campos xG del Excel no se utilizan: "
        "no son xG reales basados en ocasiones de gol."
    )

df_f = st.session_state.df_partidos[
    st.session_state.df_partidos[
        "jornada_num"
    ].astype(str) == str(jornada_sel)
].copy().reset_index(drop=True)

tab_pred, tab_update = st.tabs(
    [
        f"📊 Pronósticos Jornada {jornada_sel}",
        "📝 Actualizar Resultados y Marcadores",
    ]
)


# =========================================================
# PRONÓSTICOS
# =========================================================
with tab_pred:
    st.subheader(
        f"Jornada {jornada_sel} — Pronósticos ({tabla_ref})"
    )

    if df_f.empty:
        st.warning(
            "No hay partidos en esta jornada."
        )

    for idx in range(len(df_f)):
        row = df_f.iloc[idx]

        eq_loc = row["local_std"]
        eq_vis = row["visita_std"]

        nombre_loc = str(
            row["local"]
        ).strip()
        nombre_vis = str(
            row["visita"]
        ).strip()

        fecha = str(
            row.get("fecha_str", "")
        ).split(" ")[0]

        hora = str(
            row.get("hora", "15:00")
        )

        try:
            r = calcular_dixon_coles_v2(
                eq_loc,
                eq_vis,
                df_tabla_act,
                st.session_state.df_geo,
                historial,
                elo,
                partidos_elo,
            )
        except Exception as e:
            st.error(
                f"Error en {nombre_loc} vs "
                f"{nombre_vis}: {e}"
            )
            continue

        p_loc = r["p_local"]
        p_emp = r["p_empate"]
        p_vis = r["p_visita"]
        p_over = r["p_over25"]
        p_under = r["p_under25"]
        p_btts_si = r["p_btts_si"]
        p_btts_no = r["p_btts_no"]

        sug, conf = generar_sugerencia(
            p_loc,
            p_emp,
            p_vis,
            nombre_loc,
            nombre_vis,
        )

        sug_goles = (
            "Más de 2.5 Goles (+2.5)"
            if p_over > p_under
            else "Menos de 2.5 Goles (-2.5)"
        )

        conf_goles = nivel_confianza(
            max(p_over, p_under)
        )

        sug_btts = (
            "Ambos Equipos SÍ Anotan"
            if p_btts_si > p_btts_no
            else "Ambos Equipos NO Anotan"
        )

        conf_btts = nivel_confianza(
            max(p_btts_si, p_btts_no)
        )

        st.markdown(
            f"### 🏟️ {nombre_loc} vs {nombre_vis} | "
            f"{r['ciudad_local']} "
            f"({int(r['altitud'])} msnm)"
        )

        st.caption(
            f"📅 {fecha} | 🕒 {hora}"
        )

        if r["rivalidad"]:
            st.info(
                f"🔥 **{r['nombre_rivalidad']}** — "
                "se modera la influencia de la "
                "diferencia de fuerza; no se fuerza "
                "un 50/50."
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.write(
                f"**Gana {nombre_loc}**"
            )
            st.title(
                f"{p_loc * 100:.1f}%"
            )
            st.caption(
                f"Cuota justa: "
                f"{1 / max(p_loc, 0.001):.2f}"
            )

        with c2:
            st.write("**Empate**")
            st.title(
                f"{p_emp * 100:.1f}%"
            )
            st.caption(
                f"Cuota justa: "
                f"{1 / max(p_emp, 0.001):.2f}"
            )

        with c3:
            st.write(
                f"**Gana {nombre_vis}**"
            )
            st.title(
                f"{p_vis * 100:.1f}%"
            )
            st.caption(
                f"Cuota justa: "
                f"{1 / max(p_vis, 0.001):.2f}"
            )

        s1, s2 = st.columns([2, 1])

        with s1:
            st.markdown(
                f"<div class='suggestion-box-blue'>"
                f"<b>Pronóstico 1X2:</b> {sug}"
                f"</div>",
                unsafe_allow_html=True,
            )

        with s2:
            st.markdown(
                f"<div class='suggestion-box-green'>"
                f"<b>Confianza:</b> {conf}"
                f"</div>",
                unsafe_allow_html=True,
            )

        cg, cb = st.columns(2)

        with cg:
            st.markdown(
                "#### ⚽ Mercado de Goles"
            )

            st.write(
                f"**Más de 2.5:** "
                f"{p_over * 100:.1f}%"
            )
            st.progress(float(p_over))

            st.write(
                f"**Menos de 2.5:** "
                f"{p_under * 100:.1f}%"
            )
            st.progress(float(p_under))

            st.markdown(
                f"<div class='suggestion-box-blue'>"
                f"<b>Pronóstico:</b> "
                f"{sug_goles}</div>",
                unsafe_allow_html=True,
            )

            st.caption(
                f"🎯 Confianza: {conf_goles}"
            )

        with cb:
            st.markdown(
                "#### 🔥 Ambos Equipos Anotan"
            )

            st.write(
                f"**Sí:** "
                f"{p_btts_si * 100:.1f}%"
            )
            st.progress(float(p_btts_si))

            st.write(
                f"**No:** "
                f"{p_btts_no * 100:.1f}%"
            )
            st.progress(float(p_btts_no))

            st.markdown(
                f"<div class='suggestion-box-blue'>"
                f"<b>Pronóstico:</b> "
                f"{sug_btts}</div>",
                unsafe_allow_html=True,
            )

            st.caption(
                f"🎯 Confianza: {conf_btts}"
            )

        with st.expander(
            "🔎 Detalle del modelo"
        ):
            d1, d2, d3, d4 = st.columns(4)

            with d1:
                st.metric(
                    "Elo local",
                    f"{r['elo_local']:.0f}",
                )
                st.metric(
                    "Elo visita",
                    f"{r['elo_visita']:.0f}",
                )

            with d2:
                st.metric(
                    "Forma local",
                    f"{r['forma_local']['puntos_promedio']:.2f}",
                )
                st.metric(
                    "Forma visita",
                    f"{r['forma_visita']['puntos_promedio']:.2f}",
                )

            with d3:
                st.metric(
                    "λ goles local",
                    f"{r['lambda_local']:.2f}",
                )
                st.metric(
                    "μ goles visita",
                    f"{r['mu_visita']:.2f}",
                )

            with d4:
                st.metric(
                    "Marcador modal",
                    r["marcador_modal"],
                )
                st.metric(
                    "Diferencia Elo",
                    f"{r['dif_elo']:+.0f}",
                )

            st.caption(
                "λ y μ son parámetros Poisson del modelo; "
                "no representan xG de un proveedor externo."
            )

        st.divider()


# =========================================================
# ACTUALIZACIÓN DE RESULTADOS
# =========================================================
with tab_update:
    st.subheader(
        f"⚙️ Actualización — Jornada {jornada_sel}"
    )

    st.info(
        "Los resultados marcados como jugados quedan "
        "disponibles para forma y Elo en el siguiente "
        "cálculo."
    )

    with st.form(
        key=f"form_jornada_{jornada_sel}"
    ):
        actualizaciones = []

        for idx in range(len(df_f)):
            row = df_f.iloc[idx]

            c1, c2, c3, c4, c5 = st.columns(
                [3, 1, 1, 3, 2]
            )

            with c1:
                st.write(
                    f"**{row['local']}**"
                )

            with c2:
                val_loc = (
                    int(row["goles_local"])
                    if pd.notnull(
                        row["goles_local"]
                    )
                    else 0
                )

                gl = st.number_input(
                    f"GL_{idx}",
                    min_value=0,
                    max_value=15,
                    value=val_loc,
                    key=f"gl_{jornada_sel}_{idx}",
                    label_visibility="collapsed",
                )

            with c3:
                val_vis = (
                    int(row["goles_visita"])
                    if pd.notnull(
                        row["goles_visita"]
                    )
                    else 0
                )

                gv = st.number_input(
                    f"GV_{idx}",
                    min_value=0,
                    max_value=15,
                    value=val_vis,
                    key=f"gv_{jornada_sel}_{idx}",
                    label_visibility="collapsed",
                )

            with c4:
                st.write(
                    f"**{row['visita']}**"
                )

            with c5:
                jug = st.checkbox(
                    "Jugado",
                    value=bool(
                        row["jugado"]
                    ),
                    key=f"jug_{jornada_sel}_{idx}",
                )

            actualizaciones.append(
                (
                    row["local"],
                    row["visita"],
                    gl,
                    gv,
                    jug,
                )
            )

        submit = st.form_submit_button(
            "💾 Aplicar y Guardar Marcadores"
        )

    if submit:
        for (
            loc,
            vis,
            gl,
            gv,
            jug,
        ) in actualizaciones:

            mask = (
                st.session_state.df_partidos[
                    "jornada_num"
                ].astype(str).eq(
                    str(jornada_sel)
                )
                &
                st.session_state.df_partidos[
                    "local"
                ].astype(str).eq(
                    str(loc)
                )
                &
                st.session_state.df_partidos[
                    "visita"
                ].astype(str).eq(
                    str(vis)
                )
            )

            st.session_state.df_partidos.loc[
                mask, "goles_local"
            ] = gl

            st.session_state.df_partidos.loc[
                mask, "goles_visita"
            ] = gv

            st.session_state.df_partidos.loc[
                mask, "jugado"
            ] = jug

        if os.path.exists(ruta_local):
            ok, msg = guardar_cambios_excel(
                st.session_state.df_partidos,
                st.session_state.df_acum,
                st.session_state.df_claus,
                st.session_state.df_geo,
                ruta_local,
                st.session_state.df_apertura,
                st.session_state.df_clausura_res,
            )

            if ok:
                st.success(
                    "¡Marcadores guardados correctamente!"
                )
            else:
                st.warning(
                    "La sesión se actualizó, pero no "
                    f"se pudo escribir el archivo: {msg}"
                )
        else:
            st.success(
                "¡Marcadores actualizados en la sesión!"
            )

        st.rerun()

    st.write("---")
    st.markdown(
        "#### 📥 Exportar copia de seguridad"
    )

    excel_bytes = generar_bytes_excel(
        st.session_state.df_partidos,
        st.session_state.df_acum,
        st.session_state.df_claus,
        st.session_state.df_geo,
        st.session_state.df_apertura,
        st.session_state.df_clausura_res,
    )

    st.download_button(
        label="Descargar Liga1_2026_Actualizado.xlsx",
        data=excel_bytes,
        file_name="Liga1_2026_Actualizado.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )
