# -*- coding: utf-8 -*-

"""

Modelo de Predicción Liga 1 Perú - Quipus Data

Versión revisada con interacción hora-clima-altitud:

- Lee las ocho hojas del archivo Liga1_2026.xlsx.
- Usa Poisson con corrección Dixon-Coles.
- Filtra los partidos históricos anteriores a cada encuentro usando la fecha real.
- Recalcula K/L/M/N cronológicamente y no depende de referencias Prom_* desplazadas del Excel.
- Incorpora forma, localía, altitud, descanso y contexto competitivo; xG es configurable
  y se mantiene en 0% por defecto hasta demostrar mejora fuera de muestra.
- Integra estadísticas de tiros, tiros a puerta, córners, faltas y tarjetas por CSV,
  FootyStats API o API-Football, en un módulo separado del 1X2.
- Aplica realmente los factores contextuales después de la lambda base, evitando ajustes calculados pero no utilizados.
- Muestra resaltado en las letras, no en el fondo de las filas.
- Añade una lectura orientativa de presión competitiva: título/puestos internacionales,

  zona media y lucha por evitar el descenso.

- La presión competitiva se muestra como contexto y solo aplica ajustes muy pequeños

  y configurables a los goles esperados; no es una variable causal validada.

Instalación:

    pip install streamlit pandas numpy scipy openpyxl

Ejecución:

    streamlit run app_quipus_data_4_0.py

"""



import base64

import os

import unicodedata
import time
import requests
import re

from io import BytesIO



import numpy as np

import pandas as pd

import streamlit as st

from scipy.stats import poisson



st.set_page_config(

    page_title="Quipus Data 4.0 - Liga 1 Perú",

    page_icon="⚽",

    layout="wide",

)



# ---------------------------------------------------------------------
# Estilo: fondos neutros; colores solo en texto/señales/estados
# ---------------------------------------------------------------------

st.markdown("""
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    div[data-testid="stDataFrame"] {width: 100%;}
    div[data-testid="stDataFrame"] th {font-weight: 700 !important;}
    .small-note {font-size: .88rem; color: #6b7280;}
</style>
""", unsafe_allow_html=True)



LOGO_PATH = "logo_quipus.png"

if os.path.exists(LOGO_PATH):

    with open(LOGO_PATH, "rb") as f:

        logo_b64 = base64.b64encode(f.read()).decode("utf-8")

    st.markdown(

        f'<div style="margin-bottom:10px"><img src="data:image/png;base64,{logo_b64}" '

        'style="max-width:280px;width:100%;height:auto"></div>',

        unsafe_allow_html=True,

    )

else:

    st.markdown("### 📊 Quipus Data")



st.title("⚽ Quipus Data 4.0 | Liga 1 Perú")

st.caption(

    "Modelo estadístico orientativo. Las probabilidades son estimaciones, no garantías. "

    "La tensión competitiva se trata como contexto, no como certeza de rendimiento."

)



# ---------------------------------------------------------------------

# 1. Utilidades y normalización

# ---------------------------------------------------------------------

ALIAS_EQUIPOS = {

    "alianza atletico": "alianza atletico",

    "alianza lima": "alianza lima",

    "atletico grau": "atletico grau",

    "academia cantolao": "academia cantolao",

    "cantolao": "academia cantolao",

    "cajamarca": "fc cajamarca",

    "fc cajamarca": "fc cajamarca",

    "comerciantes unidos": "comerciantes unidos",

    "cusco": "cusco fc",

    "cusco fc": "cusco fc",

    "deportivo garcilaso": "deportivo garcilaso",

    "garcilaso": "deportivo garcilaso",

    "juan pablo ii": "juan pablo ii college",

    "juan pablo ii college": "juan pablo ii college",

    "melgar": "fbc melgar",

    "fbc melgar": "fbc melgar",

    "sport boys": "sport boys",

    "sport boys association": "sport boys",

    "sport huancayo": "sport huancayo",

    "sporting cristal": "sporting cristal",

    "universitario": "universitario",

    "utc": "utc cajamarca",

    "utc cajamarca": "utc cajamarca",

    "chankas": "chankas cyc",

    "chankas cyc": "chankas cyc",

    "los chankas": "chankas cyc",

    "cienciano": "cienciano",

    "adt": "adt",

    "cd moquegua": "cd moquegua",

    "moquegua": "cd moquegua",

}



HOJAS_ESPERADAS = [

    "Partidos_Fecha", "Resultados_Apertura", "Resultados_Clausura",

    "Tabla_Acumulada", "Tabla_Clausura", "Historial_H2H",

    "Data_Geografica", "Formulas",

]



def limpio(valor):

    if pd.isna(valor):

        return ""

    return str(valor).strip()



def normalizar_nombre(nombre):

    if not isinstance(nombre, str):

        return ""

    n = unicodedata.normalize("NFD", nombre.strip().lower())

    n = "".join(c for c in n if unicodedata.category(c) != "Mn")

    n = " ".join(n.split())

    return ALIAS_EQUIPOS.get(n, n)



def normalizar_columnas(df):

    df = df.copy()

    df.columns = [

        unicodedata.normalize("NFD", str(c).strip().lower())

        .encode("ascii", "ignore").decode("ascii")

        .replace(" ", "_")

        for c in df.columns

    ]

    renombres = {

        "equipo_local": "local", "equipo_visitante": "visita",

        "visitante": "visita", "goles_local": "gl",

        "goles_visitante": "gv", "goles_visita": "gv",

        "club": "equipo", "situacion_actual": "situacion_actual",

        "pts": "pts", "rango": "rango", "pos": "pos",

        "gl": "gl", "gv": "gv",

    }

    df = df.rename(columns=renombres)

    for c in ("local", "visita", "equipo", "club"):

        if c in df.columns:

            df[c + "_std"] = df[c].map(normalizar_nombre)

    if "fecha" in df.columns:

        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)

    numericas = {

        "gl", "gv", "pj", "pg", "pe", "pp", "gf", "gc", "dg", "pts",

        "altitud_local", "altitud_visita", "delta_altitud_msnm",

        "dias_descanso_local", "dias_descanso_visita", "altitud_msnm",

        "pos", "rango", "xg_local", "xg_visita", "prom_gf_local",

        "prom_gc_local", "prom_gf_visita", "prom_gc_visita",

    }

    for c in df.columns:

        if c in numericas or c.startswith("xg_") or c.startswith("prom_"):

            df[c] = pd.to_numeric(

                df[c].astype(str).str.replace(",", ".", regex=False),

                errors="coerce",

            )

    return df



def a_float(v, default=np.nan):

    try:

        if pd.isna(v):

            return default

        return float(str(v).replace(",", "."))

    except (ValueError, TypeError):

        return default



def parsear_hora(v):

    if pd.isna(v):

        return "--:--"

    if hasattr(v, "strftime"):

        try:

            return v.strftime("%H:%M")

        except Exception:

            pass

    return str(v)[:5]



def primera_columna(df, candidatas):

    for c in candidatas:

        if c in df.columns:

            return c

    return None



# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# 1.1. Capa de datos complementarios (CSV + API-Football)
# ---------------------------------------------------------------------
COLUMNAS_STATS = [
    "fecha", "local", "visita",
    "tiros_local", "tiros_visita",
    "tiros_puerta_local", "tiros_puerta_visita",
    "corners_local", "corners_visita",
    "faltas_local", "faltas_visita",
    "amarillas_local", "amarillas_visita",
    "rojas_local", "rojas_visita",
]

ALIASES_STATS = {
    "date": "fecha", "match_date": "fecha", "fixture_date": "fecha",
    "home": "local", "home_team": "local", "equipo_local": "local",
    "away": "visita", "away_team": "visita", "equipo_visitante": "visita",
    "shots_home": "tiros_local", "home_shots": "tiros_local",
    "total_shots_home": "tiros_local", "tiros_local": "tiros_local",
    "shots_away": "tiros_visita", "away_shots": "tiros_visita",
    "total_shots_away": "tiros_visita", "tiros_visita": "tiros_visita",
    "shots_on_target_home": "tiros_puerta_local",
    "home_shots_on_target": "tiros_puerta_local",
    "tiros_a_puerta_local": "tiros_puerta_local",
    "shots_on_target_away": "tiros_puerta_visita",
    "away_shots_on_target": "tiros_puerta_visita",
    "tiros_a_puerta_visita": "tiros_puerta_visita",
    "corners_home": "corners_local", "home_corners": "corners_local",
    "corner_kicks_home": "corners_local", "corners_local": "corners_local",
    "corners_away": "corners_visita", "away_corners": "corners_visita",
    "corner_kicks_away": "corners_visita", "corners_visita": "corners_visita",
    "fouls_home": "faltas_local", "home_fouls": "faltas_local",
    "faltas_local": "faltas_local",
    "fouls_away": "faltas_visita", "away_fouls": "faltas_visita",
    "faltas_visita": "faltas_visita",
    "yellow_cards_home": "amarillas_local", "home_yellow_cards": "amarillas_local",
    "tarjetas_amarillas_local": "amarillas_local", "amarillas_local": "amarillas_local",
    "yellow_cards_away": "amarillas_visita", "away_yellow_cards": "amarillas_visita",
    "tarjetas_amarillas_visita": "amarillas_visita", "amarillas_visita": "amarillas_visita",
    "red_cards_home": "rojas_local", "home_red_cards": "rojas_local",
    "tarjetas_rojas_local": "rojas_local", "rojas_local": "rojas_local",
    "red_cards_away": "rojas_visita", "away_red_cards": "rojas_visita",
    "tarjetas_rojas_visita": "rojas_visita", "rojas_visita": "rojas_visita",
}

def normalizar_estadisticas_partidos(df):
    """Normaliza CSV/API a una estructura única. No inventa datos ausentes."""
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNAS_STATS)
    out = df.copy()
    cols = []
    for c in out.columns:
        c0 = unicodedata.normalize("NFD", str(c).strip().lower())
        c0 = "".join(ch for ch in c0 if unicodedata.category(ch) != "Mn")
        c0 = re.sub(r"[^a-z0-9]+", "_", c0).strip("_")
        cols.append(ALIASES_STATS.get(c0, c0))
    out.columns = cols
    # También admite los nombres canónicos ya normalizados.
    out = out.loc[:, ~out.columns.duplicated()].copy()
    for c in COLUMNAS_STATS:
        if c not in out.columns:
            out[c] = np.nan
    out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce", dayfirst=True)
    for c in COLUMNAS_STATS:
        if c not in ("fecha", "local", "visita"):
            out[c] = pd.to_numeric(
                out[c].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            )
    out["local"] = out["local"].fillna("").astype(str).str.strip()
    out["visita"] = out["visita"].fillna("").astype(str).str.strip()
    out["local_std"] = out["local"].map(normalizar_nombre)
    out["visita_std"] = out["visita"].map(normalizar_nombre)
    out = out[(out["local_std"] != "") & (out["visita_std"] != "")]
    return out.sort_values("fecha", na_position="last").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def descargar_stats_footystats(api_key, season_id):
    """
    Descarga partidos y nombres de equipos desde FootyStats.
    Usa league-matches + league-teams porque el endpoint de partidos puede
    devolver IDs de equipo en lugar de nombres legibles.
    """
    base = "https://api.football-data-api.com"
    params = {"key": api_key, "season_id": int(season_id), "max_per_page": 1000}
    rm = requests.get(f"{base}/league-matches", params=params, timeout=30)
    rm.raise_for_status()
    pm = rm.json()
    if pm.get("success") is False:
        raise ValueError(f"FootyStats: {pm.get('message', pm)}")
    match_rows = pm.get("data", [])
    if isinstance(match_rows, dict):
        match_rows = match_rows.get("matches", match_rows.get("data", []))
    if not isinstance(match_rows, list):
        raise ValueError("La respuesta de FootyStats no contiene una lista de partidos.")

    rt = requests.get(
        f"{base}/league-teams",
        params={"key": api_key, "season_id": int(season_id), "include": "stats"},
        timeout=30,
    )
    rt.raise_for_status()
    pt = rt.json()
    team_rows = pt.get("data", [])
    if isinstance(team_rows, dict):
        team_rows = team_rows.get("teams", team_rows.get("data", []))
    team_map = {}
    if isinstance(team_rows, list):
        for team in team_rows:
            if isinstance(team, dict) and team.get("id") is not None:
                team_map[str(team.get("id"))] = team.get("name", team.get("cleanName", ""))

    filas = []
    for match in match_rows:
        if not isinstance(match, dict) or match.get("status") != "complete":
            continue
        home = match.get("home_name") or match.get("homeTeam") or team_map.get(str(match.get("homeID")), "")
        away = match.get("away_name") or match.get("awayTeam") or team_map.get(str(match.get("awayID")), "")
        if not home or not away:
            continue
        ts = match.get("date_unix")
        try:
            fecha = pd.to_datetime(int(ts), unit="s", utc=True).tz_convert(None) if ts else pd.NaT
        except (TypeError, ValueError, OverflowError):
            fecha = pd.NaT
        filas.append({
            "fecha": fecha, "local": home, "visita": away, "fuente": "FootyStats API",
            "tiros_local": match.get("team_a_shots"),
            "tiros_visita": match.get("team_b_shots"),
            "tiros_puerta_local": match.get("team_a_shotsOnTarget"),
            "tiros_puerta_visita": match.get("team_b_shotsOnTarget"),
            "corners_local": match.get("team_a_corners"),
            "corners_visita": match.get("team_b_corners"),
            "faltas_local": match.get("team_a_fouls"),
            "faltas_visita": match.get("team_b_fouls"),
            "amarillas_local": match.get("team_a_yellow_cards"),
            "amarillas_visita": match.get("team_b_yellow_cards"),
            "rojas_local": match.get("team_a_red_cards"),
            "rojas_visita": match.get("team_b_red_cards"),
        })
    return normalizar_estadisticas_partidos(pd.DataFrame(filas))


def obtener_estadistica_api(stats, nombre):
    """Extrae un valor de la respuesta API-Football, tolerando etiquetas alternativas."""
    if not isinstance(stats, list):
        return np.nan
    equivalencias = {
        "tiros": ["Total Shots"],
        "tiros_puerta": ["Shots on Goal", "Shots on Target"],
        "corners": ["Corner Kicks"],
        "faltas": ["Fouls"],
        "amarillas": ["Yellow Cards"],
        "rojas": ["Red Cards"],
    }
    buscados = equivalencias.get(nombre, [])
    for item in stats:
        if str(item.get("type", "")).strip() in buscados:
            valor = item.get("value")
            if valor is None or valor == "":
                return np.nan
            try:
                return float(str(valor).replace("%", "").replace(",", "."))
            except (TypeError, ValueError):
                return np.nan
    return np.nan

@st.cache_data(ttl=3600, show_spinner=False)
def descargar_stats_api_football(api_key, league_id, season, limite=30):
    """
    Recupera estadísticas de partidos finalizados desde API-Football.
    Requiere una clave válida y que el plan permita la competición solicitada.
    El límite evita realizar llamadas masivas en cada ejecución.
    """
    base = "https://v3.football.api-sports.io"
    headers = {"x-apisports-key": api_key}
    r = requests.get(
        f"{base}/fixtures",
        headers=headers,
        params={"league": int(league_id), "season": int(season)},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("errors"):
        raise ValueError(f"API-Football: {payload['errors']}")
    fixtures = payload.get("response", [])
    finalizados = [
        x for x in fixtures
        if x.get("fixture", {}).get("status", {}).get("short") in {"FT", "AET", "PEN"}
    ]
    finalizados = sorted(
        finalizados, key=lambda x: x.get("fixture", {}).get("date", ""), reverse=True
    )[:max(1, min(int(limite), 50))]
    filas = []
    for n, fx in enumerate(finalizados, start=1):
        fid = fx.get("fixture", {}).get("id")
        teams = fx.get("teams", {})
        home = teams.get("home", {}).get("name", "")
        away = teams.get("away", {}).get("name", "")
        fecha = fx.get("fixture", {}).get("date")
        if not fid or not home or not away:
            continue
        rs = requests.get(
            f"{base}/fixtures/statistics", headers=headers,
            params={"fixture": fid}, timeout=30
        )
        rs.raise_for_status()
        body = rs.json()
        response = body.get("response", [])
        hstats, astats = [], []
        for block in response:
            team_name = block.get("team", {}).get("name", "")
            if team_name == home:
                hstats = block.get("statistics", [])
            elif team_name == away:
                astats = block.get("statistics", [])
        row = {"fecha": fecha, "local": home, "visita": away}
        for metric in ("tiros", "tiros_puerta", "corners", "faltas", "amarillas", "rojas"):
            row[f"{metric}_local"] = obtener_estadistica_api(hstats, metric)
            row[f"{metric}_visita"] = obtener_estadistica_api(astats, metric)
        filas.append(row)
        # Pequeña pausa para no disparar peticiones consecutivas.
        if n < len(finalizados):
            time.sleep(0.15)
    return normalizar_estadisticas_partidos(pd.DataFrame(filas))

def predecir_estadisticas_secundarias(local, visita, fecha_partido=None):
    """
    Estimación independiente de tiros, tiros a puerta, córners, faltas y tarjetas.
    Usa promedios de los últimos cinco registros disponibles por equipo.
    No modifica lambdas ni probabilidades 1X2.
    """
    df = globals().get("estadisticas_partidos", pd.DataFrame())
    vacio = {
        "Tiros esperados total": np.nan, "Tiros a puerta esperados total": np.nan,
        "Córners esperados total": np.nan, "Faltas esperadas total": np.nan,
        "Tarjetas amarillas esperadas total": np.nan,
        "Partidos estadísticos disponibles": 0,
    }
    if (not APLICAR_STATS_SECUNDARIAS or df is None or df.empty
            or fecha_partido is None or pd.isna(fecha_partido)):
        return vacio
    h = df.copy()
    fecha_obj = pd.Timestamp(fecha_partido).normalize()
    h = h[h["fecha"].notna() & (h["fecha"].dt.normalize() < fecha_obj)]
    eq_l, eq_v = normalizar_nombre(local), normalizar_nombre(visita)
    if h.empty:
        return vacio

    metricas = {
        "tiros": ("tiros_local", "tiros_visita"),
        "tiros_puerta": ("tiros_puerta_local", "tiros_puerta_visita"),
        "corners": ("corners_local", "corners_visita"),
        "faltas": ("faltas_local", "faltas_visita"),
        "amarillas": ("amarillas_local", "amarillas_visita"),
    }
    salidas = {}
    n_total = 0
    for nombre, (col_l, col_v) in metricas.items():
        valores_equipo = []
        for eq in (eq_l, eq_v):
            loc = h[h["local_std"] == eq][col_l]
            vis = h[h["visita_std"] == eq][col_v]
            serie = pd.concat([loc, vis]).dropna().tail(5)
            valores_equipo.append(float(serie.mean()) if len(serie) else np.nan)
        # No se presenta un total si falta la muestra de uno de los dos clubes.
        if all(np.isfinite(v) for v in valores_equipo):
            salidas[nombre] = float(sum(valores_equipo))
            n_total = max(n_total, min(
                len(pd.concat([h[h["local_std"] == eq_l][col_l],
                               h[h["visita_std"] == eq_l][col_v]]).dropna().tail(5)),
                len(pd.concat([h[h["local_std"] == eq_v][col_l],
                               h[h["visita_std"] == eq_v][col_v]]).dropna().tail(5))
            ))
        else:
            salidas[nombre] = np.nan
    return {
        "Tiros esperados total": round(salidas["tiros"], 2) if np.isfinite(salidas["tiros"]) else np.nan,
        "Tiros a puerta esperados total": round(salidas["tiros_puerta"], 2) if np.isfinite(salidas["tiros_puerta"]) else np.nan,
        "Córners esperados total": round(salidas["corners"], 2) if np.isfinite(salidas["corners"]) else np.nan,
        "Faltas esperadas total": round(salidas["faltas"], 2) if np.isfinite(salidas["faltas"]) else np.nan,
        "Tarjetas amarillas esperadas total": round(salidas["amarillas"], 2) if np.isfinite(salidas["amarillas"]) else np.nan,
        "Partidos estadísticos disponibles": int(n_total),
    }



# 2. Carga del Excel y lectura robusta de la hoja Formulas

# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)

def cargar_excel_completo(bytes_excel):

    xls = pd.ExcelFile(BytesIO(bytes_excel))

    disponibles = xls.sheet_names

    originales = {

        h: pd.read_excel(xls, sheet_name=h) if h in disponibles else pd.DataFrame()

        for h in HOJAS_ESPERADAS

    }

    datos = {

        k: normalizar_columnas(v)

        for k, v in originales.items() if k != "Formulas"

    }

    datos["formulas_original"] = originales["Formulas"]

    datos["hojas_disponibles"] = disponibles



    # Promedios globales: buscar etiquetas; si no se encuentran, calcularlos

    # desde resultados históricos en vez de depender de números de fila fijos.

    prior_local, prior_visita = np.nan, np.nan

    formulas = originales["Formulas"]

    if not formulas.empty and formulas.shape[1] >= 4:

        etiquetas = formulas.iloc[:, 1].fillna("").astype(str).str.lower()

        valores = formulas.iloc[:, 3]

        for etiqueta, valor_bruto in zip(etiquetas, valores):

            valor = a_float(valor_bruto)

            if not np.isfinite(valor):

                continue

            if "mgoles local" in etiqueta or "mgoles_local" in etiqueta:

                prior_local = valor

            elif "mgoles visita" in etiqueta or "mgoles_visita" in etiqueta:

                prior_visita = valor



    def promedio_goles(hojas, col):

        vals = []

        for h in hojas:

            d = datos[h]

            if col in d.columns:

                vals.extend(pd.to_numeric(d[col], errors="coerce").dropna().tolist())

        return float(np.mean(vals)) if vals else np.nan



    if not np.isfinite(prior_local) or prior_local <= 0:

        prior_local = promedio_goles(

            ["Resultados_Apertura", "Resultados_Clausura"], "gl"

        )

    if not np.isfinite(prior_visita) or prior_visita <= 0:

        prior_visita = promedio_goles(

            ["Resultados_Apertura", "Resultados_Clausura"], "gv"

        )

    datos["prior_local"] = float(np.clip(prior_local if np.isfinite(prior_local) else 1.45, .3, 3.5))

    datos["prior_visita"] = float(np.clip(prior_visita if np.isfinite(prior_visita) else 1.20, .3, 3.5))

    return datos



st.sidebar.header("📂 Archivo de datos")

archivo_subido = st.sidebar.file_uploader("Cargar Excel actualizado", type=["xlsx"])

archivo_predeterminado = next(

    (

        p for p in [

            "Liga1_2026.xlsx",

            "Liga1_2026(1).xlsx",

            "Liga1_2026(2).xlsx",
            "Liga1_2026(3).xlsx",

        ]

        if os.path.exists(p)

    ),

    None,

)

if archivo_subido is not None:

    bytes_excel, nombre_archivo = archivo_subido.getvalue(), archivo_subido.name

elif archivo_predeterminado:

    with open(archivo_predeterminado, "rb") as f:

        bytes_excel = f.read()

    nombre_archivo = archivo_predeterminado

else:

    st.error("No se encontró el Excel. Cárguelo desde el panel lateral.")

    st.stop()



try:

    datos = cargar_excel_completo(bytes_excel)

except Exception as exc:

    st.error(f"No se pudo leer el Excel: {exc}")

    st.stop()



st.sidebar.caption(f"Archivo activo: {nombre_archivo}")


# Carga mixta de estadísticas complementarias.
st.sidebar.subheader("📊 Datos complementarios")
archivo_stats = st.sidebar.file_uploader(
    "Importar estadísticas de partidos (CSV)", type=["csv"], key="csv_stats_4"
)
if archivo_stats is not None:
    try:
        stats_csv = pd.read_csv(archivo_stats)
        st.session_state["stats_quipus_4"] = normalizar_estadisticas_partidos(stats_csv)
        st.sidebar.success(f"CSV cargado: {len(st.session_state['stats_quipus_4'])} partidos.")
    except Exception as exc:
        st.sidebar.error(f"No se pudo leer el CSV de estadísticas: {exc}")


with st.sidebar.expander("Conectar FootyStats API", expanded=False):
    footy_key = st.text_input("Clave API FootyStats", value="", type="password", key="footy_key_4")
    footy_season = st.text_input("ID de temporada (season_id)", value="", key="footy_season_4")
    st.caption("FootyStats requiere que la temporada y la Liga 1 estén habilitadas en tu cuenta.")
    if st.button("Importar desde FootyStats", key="btn_footy_stats_4"):
        if not footy_key or not footy_season.strip().isdigit():
            st.error("Introduce la clave API y el ID numérico de temporada.")
        else:
            try:
                with st.spinner("Descargando partidos y estadísticas de FootyStats..."):
                    footy_stats = descargar_stats_footystats(footy_key, int(footy_season))
                if footy_stats.empty:
                    st.warning("FootyStats no devolvió partidos finalizados con estadísticas.")
                else:
                    st.session_state["stats_quipus_4"] = footy_stats
                    st.success(f"FootyStats importó {len(footy_stats)} partidos.")
            except Exception as exc:
                st.error(f"No se pudo consultar FootyStats: {exc}")

with st.sidebar.expander("Conectar API-Football", expanded=False):
    api_key_default = ""
    try:
        api_key_default = st.secrets.get("API_FOOTBALL_KEY", "")
    except Exception:
        pass
    api_key = st.text_input("API key", value=api_key_default, type="password", key="api_football_key_4")
    api_league = st.text_input("ID de Liga 1 en API-Football", value="", key="api_league_4")
    api_season = st.number_input("Temporada", min_value=2010, max_value=2035, value=2026, key="api_season_4")
    api_limit = st.slider("Últimos partidos a consultar", 5, 50, 30, key="api_limit_4")
    if st.button("Consultar estadísticas de la API", key="btn_api_stats_4"):
        if not api_key or not api_league.strip().isdigit():
            st.error("Introduce una API key y el ID numérico de la competición.")
        else:
            try:
                with st.spinner("Consultando partidos y estadísticas..."):
                    api_stats = descargar_stats_api_football(api_key, int(api_league), int(api_season), int(api_limit))
                if api_stats.empty:
                    st.warning("La API no devolvió estadísticas utilizables para esos partidos.")
                else:
                    st.session_state["stats_quipus_4"] = api_stats
                    st.success(f"Se importaron estadísticas de {len(api_stats)} partidos.")
            except Exception as exc:
                st.error(f"No se pudo consultar API-Football: {exc}")

if "stats_quipus_4" not in st.session_state:
    st.session_state["stats_quipus_4"] = pd.DataFrame(columns=COLUMNAS_STATS)
estadisticas_partidos = normalizar_estadisticas_partidos(st.session_state["stats_quipus_4"])


faltantes = [h for h in HOJAS_ESPERADAS if h not in datos["hojas_disponibles"]]

if faltantes:

    st.warning("No se encontraron estas hojas: " + ", ".join(faltantes))
if datos.get("alerta_xg_apertura_visita"):
    st.warning(datos["alerta_xg_apertura_visita"])
if not estadisticas_partidos.empty:
    st.sidebar.caption(f"Estadísticas complementarias activas: {len(estadisticas_partidos)} partidos.")
else:
    st.sidebar.info(
        "Sin estadísticas complementarias cargadas. Los mercados de tiros, córners, "
        "faltas y tarjetas aparecerán como no disponibles."
    )




if not estadisticas_partidos.empty:
    with st.sidebar.expander("Vista previa de datos complementarios"):
        st.dataframe(estadisticas_partidos.tail(12), use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar estadísticas normalizadas",
            data=estadisticas_partidos.to_csv(index=False).encode("utf-8-sig"),
            file_name="quipus_estadisticas_normalizadas.csv",
            mime="text/csv",
            key="download_stats_4",
        )

partidos = datos["Partidos_Fecha"].copy()

if partidos.empty or not {"local", "visita"}.issubset(partidos.columns):

    st.error("La hoja Partidos_Fecha debe contener las columnas Local y Visita.")

    st.stop()



# ---------------------------------------------------------------------

# 3. Histórico común

# ---------------------------------------------------------------------

def construir_historico(datos):

    partes = []

    for hoja, peso in [("Resultados_Apertura", .40), ("Resultados_Clausura", .60)]:

        df = datos[hoja].copy()

        if df.empty or not {"local", "visita", "gl", "gv"}.issubset(df.columns):

            continue

        df["fuente"] = hoja

        df["peso_base"] = peso

        if "jornada" not in df.columns:

            df["jornada"] = np.arange(1, len(df) + 1)

        df["jornada_num"] = pd.to_numeric(

            df["jornada"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"

        )

        df["gl"] = pd.to_numeric(df["gl"], errors="coerce")

        df["gv"] = pd.to_numeric(df["gv"], errors="coerce")

        df = df.dropna(subset=["gl", "gv", "local_std", "visita_std"])

        # xG se conserva únicamente cuando existe como dato real.

        # No se sustituye por goles reales.

        for xg in ("xg_local", "xg_visita"):

            if xg not in df.columns:

                df[xg] = np.nan

            df[xg] = pd.to_numeric(df[xg], errors="coerce")

        if hoja == "Resultados_Apertura" and "xg_visita" in df.columns:
            xgv = df["xg_visita"].dropna()
            if len(xgv) >= 20 and xgv.nunique() <= 5:
                df["xg_visita"] = np.nan
                datos["alerta_xg_apertura_visita"] = (
                    "xG visitante del Apertura excluido: solo presenta "
                    f"{xgv.nunique()} valores distintos en {len(xgv)} registros."
                )

        partes.append(df)

    return pd.concat(partes, ignore_index=True, sort=False) if partes else pd.DataFrame()



historico = construir_historico(datos)



def preparar_tabla(df):

    df = df.copy()

    for c in ("pj", "pg", "pe", "pp", "gf", "gc", "dg", "pts", "rango", "pos"):

        if c in df.columns:

            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df



tabla_acumulada = preparar_tabla(datos["Tabla_Acumulada"])

tabla_clausura = preparar_tabla(datos["Tabla_Clausura"])



def fila_equipo_tabla(tabla, equipo):

    if tabla.empty or "equipo_std" not in tabla.columns:

        return None

    z = tabla[tabla["equipo_std"] == normalizar_nombre(equipo)]

    return None if z.empty else z.iloc[0]



# ---------------------------------------------------------------------

# 4. Contexto competitivo: puestos, urgencia y tensión de final de torneo

# ---------------------------------------------------------------------

def objetivo_equipo(fila, total_equipos=18):

    if fila is None:

        return "Sin datos de tabla"

    situacion = str(fila.get("situacion_actual", "")).lower()

    rango = a_float(fila.get("rango", fila.get("pos")))

    if any(x in situacion for x in ["descenso directo", "zona de descenso"]):

        return "Descenso directo"

    if any(x in situacion for x in ["riesgo directo", "peligro de descenso", "descenso"]):

        return "Lucha por la permanencia"

    if "libertadores" in situacion:

        return "Puestos de Libertadores / título"

    if "sudamericana" in situacion:

        return "Puestos de Sudamericana"

    if np.isfinite(rango):

        if rango <= 4:

            return "Pelea alta: título / Libertadores"

        if rango <= 8:

            return "Pelea por torneo internacional"

        if rango >= total_equipos - 1:

            return "Descenso directo"

        if rango >= total_equipos - 4:

            return "Lucha por la permanencia"

        if rango >= 9:

            return "Zona media"

    return "Objetivo por confirmar"



def urgencia_equipo(fila, total_equipos=18):

    """

    Índice descriptivo de 0 a 1. No representa una probabilidad.

    Sube en los extremos de la tabla y cerca de las zonas objetivo.

    """

    if fila is None:

        return 0.25

    situacion = str(fila.get("situacion_actual", "")).lower()

    rango = a_float(fila.get("rango", fila.get("pos")))

    pts = a_float(fila.get("pts"))

    pj = a_float(fila.get("pj"))

    if not np.isfinite(rango):

        return 0.25

    if any(x in situacion for x in ["descenso directo", "zona de descenso"]):

        base = 0.95

    elif any(x in situacion for x in ["riesgo directo", "peligro de descenso", "descenso"]):

        base = 0.85

    elif "libertadores" in situacion:

        base = 0.78 if rango <= 4 else 0.62

    elif "sudamericana" in situacion:

        base = 0.68

    elif rango <= 4:

        base = 0.78

    elif rango <= 8:

        base = 0.62

    elif rango >= total_equipos - 4:

        base = 0.78

    else:

        base = 0.30



    # Con pocos partidos por jugar, la presión de los objetivos suele aumentar.

    pj_max = 34.0

    restantes = max(0.0, pj_max - pj) if np.isfinite(pj) else 5.0

    factor_cierre = float(np.clip((8.0 - restantes) / 8.0, 0.0, 1.0))

    return float(np.clip(base * (0.70 + 0.30 * factor_cierre), 0.0, 1.0))



def presion_partido(fila_l, fila_v):

    ul, uv = urgencia_equipo(fila_l), urgencia_equipo(fila_v)

    ol = objetivo_equipo(fila_l)

    ov = objetivo_equipo(fila_v)

    media = (ul + uv) / 2

    if media >= .78:

        nivel = "Muy alta"

    elif media >= .60:

        nivel = "Alta"

    elif media >= .40:

        nivel = "Media"

    else:

        nivel = "Baja"

    return {

        "Objetivo local": ol,

        "Objetivo visita": ov,

        "Urgencia local (0-1)": round(ul, 2),

        "Urgencia visita (0-1)": round(uv, 2),

        "Presión competitiva": nivel,

        "Índice presión (0-1)": round(media, 2),

    }



# ---------------------------------------------------------------------

# 5. Rendimiento de equipo con resultados, xG, tablas y forma

# ---------------------------------------------------------------------

def filtrar_historico_previo(fecha_partido=None, jornada_partido=None):

    """Usa fecha real como corte principal. Nunca usa partidos del mismo día."""

    h = historico.copy()

    if h.empty:

        return h



    if fecha_partido is not None and pd.notna(fecha_partido) and "fecha" in h.columns:

        fecha_obj = pd.Timestamp(fecha_partido).normalize()

        fechas = pd.to_datetime(h["fecha"], errors="coerce")

        # Solo registros con fecha conocida y estrictamente anteriores.

        h = h[fechas.notna() & (fechas.dt.normalize() < fecha_obj)].copy()

    elif jornada_partido is not None and pd.notna(jornada_partido):

        try:

            jn = int(jornada_partido)

            h = h[(h["fuente"] != "Resultados_Clausura") | (h["jornada_num"] < jn)].copy()

        except (ValueError, TypeError):

            pass

    return h





def promedios_excel_equipo(equipo, fecha_partido=None, jornada_partido=None):

    """Reproduce las cuatro primeras fórmulas de Resultados_Clausura."""

    eq = normalizar_nombre(equipo)

    h = filtrar_historico_previo(fecha_partido, jornada_partido)



    if h.empty:

        return {"K": np.nan, "L": np.nan, "M": np.nan, "N": np.nan,

                "n_local": 0, "n_visita": 0}



    local = h[h["local_std"] == eq].copy()

    visita = h[h["visita_std"] == eq].copy()



    return {

        "K": float(local["gl"].mean()) if not local.empty else np.nan,

        "L": float(local["gv"].mean()) if not local.empty else np.nan,

        "M": float(visita["gv"].mean()) if not visita.empty else np.nan,

        "N": float(visita["gl"].mean()) if not visita.empty else np.nan,

        "n_local": len(local),

        "n_visita": len(visita),

    }





def factores_altitud_excel(altitud):

    """Replica exactamente los factores de las fórmulas Excel."""

    alt = a_float(altitud)

    if not np.isfinite(alt):

        return 1.00, 1.00

    f_local = 1.20 if alt >= 2000 else (1.10 if alt >= 1000 else 1.00)

    f_visita = 0.75 if alt >= 2000 else (0.85 if alt >= 1000 else 1.00)

    return f_local, f_visita





def lambdas_clausura_excel(local, visita, fecha_partido=None, jornada_partido=None,
                           altitud_excel_j=np.nan):
    """
    Calcula goles esperados usando solo datos previos al partido:
    - base de goles anotados/recibidos por condición local/visita;
    - ajuste experimental con xG, cuando hay suficientes observaciones previas;
    - factores de altitud de la fórmula histórica del Excel.

    El ajuste xG es configurable y no se interpreta como mejora garantizada.
    """
    ml = promedios_excel_equipo(local, fecha_partido, jornada_partido)
    mv = promedios_excel_equipo(visita, fecha_partido, jornada_partido)
    h = filtrar_historico_previo(fecha_partido, jornada_partido)

    # Promedios de referencia calculados solo con partidos disponibles antes
    # de la jornada evaluada, para evitar fuga temporal en backtesting.
    if not h.empty:
        d2 = pd.to_numeric(h["gl"], errors="coerce").mean()
        d6 = pd.to_numeric(h["gv"], errors="coerce").mean()
    else:
        d2, d6 = datos["prior_local"], datos["prior_visita"]

    f_local, f_visita = factores_altitud_excel(altitud_excel_j)

    lam_l_goles = (
        np.nan if not all(np.isfinite([ml["K"], mv["N"], d2])) or d2 <= 0
        else (ml["K"] * mv["N"] / d2)
    )
    lam_v_goles = (
        np.nan if not all(np.isfinite([mv["M"], ml["L"], d6])) or d6 <= 0
        else (mv["M"] * ml["L"] / d6)
    )

    # xG: promedios de creación y concesión separados por condición de juego.
    # Se utilizan únicamente los xG de partidos anteriores al partido objetivo.
    xg_l, xg_v = np.nan, np.nan
    peso_xg = float(np.clip(PESO_XG_MODELO, 0.0, 0.50))
    if not h.empty and {"xg_local", "xg_visita", "local_std", "visita_std"}.issubset(h.columns):
        eq_l, eq_v = normalizar_nombre(local), normalizar_nombre(visita)
        hx = h.copy()
        # Las columnas xG del usuario ya llevan ajuste de altitud. Para poder
        # volver a modelar la sede del partido objetivo sin duplicar el ajuste,
        # quitamos de cada observación histórica el factor de altitud que se le
        # aplicó en su propio estadio. Después el factor del estadio objetivo se
        # aplica una sola vez al lambda final.
        if "altitud_local" in hx.columns:
            def neutralizar_xg_fila(r, col):
                val = a_float(r.get(col))
                alt_hist = a_float(r.get("altitud_local"))
                fl_hist, fv_hist = factores_altitud_excel(alt_hist)
                factor_hist = fl_hist if col == "xg_local" else fv_hist
                return val / factor_hist if np.isfinite(val) and factor_hist > 0 else np.nan
            hx["xg_local_neutro"] = hx.apply(lambda r: neutralizar_xg_fila(r, "xg_local"), axis=1)
            hx["xg_visita_neutro"] = hx.apply(lambda r: neutralizar_xg_fila(r, "xg_visita"), axis=1)
        else:
            hx["xg_local_neutro"] = pd.to_numeric(hx["xg_local"], errors="coerce")
            hx["xg_visita_neutro"] = pd.to_numeric(hx["xg_visita"], errors="coerce")

        hl = hx[hx["local_std"] == eq_l]
        hv = hx[hx["visita_std"] == eq_v]
        # Para lambda local: xG producido en casa por local y xG concedido
        # fuera por visitante. Para lambda visitante, el sentido inverso.
        xg_for_l = pd.to_numeric(hl["xg_local_neutro"], errors="coerce").dropna()
        xg_against_v = pd.to_numeric(hv["xg_local_neutro"], errors="coerce").dropna()
        xg_for_v = pd.to_numeric(hv["xg_visita_neutro"], errors="coerce").dropna()
        xg_against_l = pd.to_numeric(hl["xg_visita_neutro"], errors="coerce").dropna()
        prior_xg_l = pd.to_numeric(hx["xg_local_neutro"], errors="coerce").mean()
        prior_xg_v = pd.to_numeric(hx["xg_visita_neutro"], errors="coerce").mean()

        if (min(len(xg_for_l), len(xg_against_v), len(xg_for_v), len(xg_against_l)) >= 3
                and np.isfinite(prior_xg_l) and prior_xg_l > 0
                and np.isfinite(prior_xg_v) and prior_xg_v > 0):
            xg_l = float(xg_for_l.mean() * xg_against_v.mean() / prior_xg_l)
            xg_v = float(xg_for_v.mean() * xg_against_l.mean() / prior_xg_v)

    # Mezcla conservadora goles/xG; si no hay suficientes datos xG, se mantiene
    # la estimación basada en goles. Altitud se aplica una sola vez al final.
    lam_l = lam_l_goles
    lam_v = lam_v_goles
    if np.isfinite(lam_l_goles) and np.isfinite(xg_l):
        lam_l = (1 - peso_xg) * lam_l_goles + peso_xg * xg_l
    if np.isfinite(lam_v_goles) and np.isfinite(xg_v):
        lam_v = (1 - peso_xg) * lam_v_goles + peso_xg * xg_v

    lam_l = lam_l * f_local if np.isfinite(lam_l) else np.nan
    lam_v = lam_v * f_visita if np.isfinite(lam_v) else np.nan

    return {
        "lambda_local": lam_l,
        "lambda_visita": lam_v,
        "local": ml,
        "visita": mv,
        "D2": d2,
        "D6": d6,
        "lambda_local_goles": lam_l_goles * f_local if np.isfinite(lam_l_goles) else np.nan,
        "lambda_visita_goles": lam_v_goles * f_visita if np.isfinite(lam_v_goles) else np.nan,
        "lambda_local_xg": xg_l * f_local if np.isfinite(xg_l) else np.nan,
        "lambda_visita_xg": xg_v * f_visita if np.isfinite(xg_v) else np.nan,
        "peso_xg": peso_xg if np.isfinite(xg_l) and np.isfinite(xg_v) else 0.0,
        "altitud_excel_j": a_float(altitud_excel_j),
        "factor_altitud_local": f_local,
        "factor_altitud_visita": f_visita,
    }




def metricas_equipo(datos, equipo, fecha_partido=None, jornada_partido=None):

    """Métricas descriptivas previas al partido, sin fuga temporal."""

    eq = normalizar_nombre(equipo)

    h = filtrar_historico_previo(fecha_partido, jornada_partido)

    if h.empty:

        pl, pv = datos["prior_local"], datos["prior_visita"]

        return {"gf": pl, "ga": pv, "gf_local": pl, "ga_local": pv,

                "gf_visita": pv, "ga_visita": pl, "n": 0, "forma": .5}



    loc = h[h["local_std"] == eq].copy()

    vis = h[h["visita_std"] == eq].copy()

    recientes = pd.concat([

        loc.assign(gf=loc["gl"], ga=loc["gv"]),

        vis.assign(gf=vis["gv"], ga=vis["gl"])

    ], ignore_index=True)

    if "fecha" in recientes.columns:

        recientes = recientes.sort_values("fecha")

    recientes = recientes.tail(5)

    forma = .5

    if not recientes.empty:

        puntos = np.where(recientes["gf"] > recientes["ga"], 3,

                          np.where(recientes["gf"] == recientes["ga"], 1, 0))

        forma = float(np.mean(puntos) / 3.0)



    gf_local = float(loc["gl"].mean()) if not loc.empty else datos["prior_local"]

    ga_local = float(loc["gv"].mean()) if not loc.empty else datos["prior_visita"]

    gf_visita = float(vis["gv"].mean()) if not vis.empty else datos["prior_visita"]

    ga_visita = float(vis["gl"].mean()) if not vis.empty else datos["prior_local"]

    return {

        "gf": max(.2, np.nanmean([gf_local, gf_visita])),

        "ga": max(.2, np.nanmean([ga_local, ga_visita])),

        "gf_local": max(.2, gf_local), "ga_local": max(.2, ga_local),

        "gf_visita": max(.2, gf_visita), "ga_visita": max(.2, ga_visita),

        "n": len(loc) + len(vis), "forma": forma,

    }



# ---------------------------------------------------------------------

# 6. Factores contextuales

# ---------------------------------------------------------------------

PESO_XG_MODELO = 0.00  # xG sigue desactivado por defecto hasta validar mejora fuera de muestra.
PESO_XG_MODELO = st.sidebar.slider(
    "Peso de xG en el modelo 1X2", min_value=0.0, max_value=0.50,
    value=float(PESO_XG_MODELO), step=0.05,
    help="0% usa solo goles; cualquier peso mayor mezcla xG. Valida el ajuste con backtesting."
)

APLICAR_MICROAJUSTE_PRESION = False
APLICAR_H2H_MODELO = True
APLICAR_STATS_SECUNDARIAS = True



def obtener_contexto(datos, partido):

    loc, vis = limpio(partido.get("local")), limpio(partido.get("visita"))

    eq_loc, eq_vis = normalizar_nombre(loc), normalizar_nombre(vis)

    f_loc, f_vis, rho = 1.0, 1.0, -.08

    geo = datos["Data_Geografica"]



    def info_geo(eq):

        if geo.empty or "equipo_std" not in geo.columns:

            return None

        z = geo[geo["equipo_std"] == eq]

        return None if z.empty else z.iloc[0]



    il, iv = info_geo(eq_loc), info_geo(eq_vis)

    # Para el partido actual, la sede concreta indicada en Partidos_Fecha tiene
    # prioridad. Luego se busca el estadio exacto en Data_Geografica y, por
    # último, se recurre a la ficha geográfica del club local.
    estadio_partido = limpio(partido.get("estadio"))
    ciudad_partido = limpio(partido.get("ciudad"))
    sede_geo = None
    if not geo.empty and "estadio" in geo.columns and estadio_partido:
        coincidencia = geo[geo["estadio"].astype(str).str.strip().str.casefold() == estadio_partido.casefold()]
        if not coincidencia.empty:
            sede_geo = coincidencia.iloc[0]
    if sede_geo is None and not geo.empty and "ciudad" in geo.columns and ciudad_partido:
        coincidencia = geo[geo["ciudad"].astype(str).str.strip().str.casefold() == ciudad_partido.casefold()]
        if not coincidencia.empty:
            sede_geo = coincidencia.iloc[0]
    if sede_geo is None:
        sede_geo = il

    alt_l_fixture = a_float(partido.get("altitud_local"))
    alt_l_geo = a_float(sede_geo.get("altitud_msnm")) if sede_geo is not None else np.nan
    alt_l = alt_l_fixture if np.isfinite(alt_l_fixture) else alt_l_geo
    alt_v = a_float(partido.get("altitud_visita"))
    if not np.isfinite(alt_v):
        alt_v = a_float(iv.get("altitud_msnm")) if iv is not None else np.nan

    delta = alt_l - alt_v if np.isfinite(alt_l) and np.isfinite(alt_v) else np.nan



    # La altitud NO se vuelve a aplicar aquí: la metodología Excel

    # la incorpora directamente en lambda mediante J (altitud del estadio).



    # El clima del partido se toma del estadio/localidad donde se juega,
    # no se compara el clima de origen de ambos clubes. Priorizamos la sede
    # informada en Partidos_Fecha y usamos Data_Geografica como respaldo.
    clima_estadio = str(sede_geo.get("tipo_clima", "")).lower() if sede_geo is not None else ""
    cancha_l = str(sede_geo.get("tipo_cancha", "")).lower() if sede_geo is not None else ""

    if "sintetic" in cancha_l or "sintética" in cancha_l:
        f_loc *= 1.01
        f_vis *= .99

    descanso_l = a_float(partido.get("dias_descanso_local"), 7)
    descanso_v = a_float(partido.get("dias_descanso_visita"), 7)
    if descanso_l < 4:
        f_loc *= .95
    if descanso_v < 4:
        f_vis *= .95

    # Interacción conservadora entre hora de inicio y condiciones del estadio.
    # No es una medición meteorológica en vivo: es una regla contextual basada
    # en el tipo de clima y altitud registrados en el Excel. Se mantiene pequeña
    # porque los factores principales de altitud ya se aplican en lambda.
    hora = parsear_hora(partido.get("hora"))
    hora_num = None
    try:
        hora_num = int(hora.split(":")[0])
    except (ValueError, TypeError, IndexError):
        pass

    calor_extremo = ("calor extremo" in clima_estadio) or ("caluroso" in clima_estadio)
    calor = calor_extremo or ("calor" in clima_estadio)
    if hora_num is not None:
        # Sullana y sedes marcadas con calor: el tramo 12:00-15:59 suele
        # representar mayor carga térmica que un inicio a las 15:00 exactas
        # no garantiza temperatura concreta; por eso el ajuste es leve.
        if calor and 12 <= hora_num <= 15:
            f_loc *= 1.015
            f_vis *= .975
        elif calor and (hora_num == 11 or hora_num == 16):
            f_loc *= 1.005
            f_vis *= .990

        # En altura, jugar al mediodía puede aumentar la exigencia física.
        # Como el ajuste de altitud base ya existe, aquí solo se modela una
        # interacción hora-altitud pequeña, evitando duplicar un gran factor.
        if np.isfinite(alt_l):
            if alt_l >= 2500 and 11 <= hora_num <= 14:
                f_loc *= 1.005
                f_vis *= .985
            elif alt_l >= 1500 and 11 <= hora_num <= 14:
                f_loc *= 1.002
                f_vis *= .992



    # Historial H2H: ajuste limitado para no sobreponderar enfrentamientos antiguos.

    h2h = datos["Historial_H2H"]

    if APLICAR_H2H_MODELO and not h2h.empty and {"local_std", "visita_std", "gl", "gv"}.issubset(h2h.columns):

        h = h2h[

            ((h2h["local_std"] == eq_loc) & (h2h["visita_std"] == eq_vis)) |

            ((h2h["local_std"] == eq_vis) & (h2h["visita_std"] == eq_loc))

        ].copy()



        # Si Historial_H2H tiene Fecha, usar solo enfrentamientos previos.

        # Esto evita fuga temporal al evaluar partidos históricos.

        if "fecha" in h.columns and pd.notna(partido.get("fecha")):

            fechas_h2h = pd.to_datetime(h["fecha"], errors="coerce")

            fecha_obj = pd.Timestamp(partido.get("fecha")).normalize()

            h = h[fechas_h2h.notna() & (fechas_h2h.dt.normalize() < fecha_obj)].copy()



        h = h.tail(8).copy()

        if not h.empty:

            bal = []

            for _, r in h.iterrows():

                if r["local_std"] == eq_loc:

                    bal.append(1 if r["gl"] > r["gv"] else (-1 if r["gl"] < r["gv"] else 0))

                else:

                    bal.append(1 if r["gv"] > r["gl"] else (-1 if r["gv"] < r["gl"] else 0))

            balance = float(np.mean(bal))

            f_loc *= float(np.clip(1 + .025 * balance, .975, 1.025))

            f_vis *= float(np.clip(1 - .020 * balance, .980, 1.020))

            grandes = {"alianza lima", "universitario", "sporting cristal"}

            if eq_loc in grandes and eq_vis in grandes:

                rho = -.12



    fila_l = fila_equipo_tabla(tabla_acumulada, loc)

    fila_v = fila_equipo_tabla(tabla_acumulada, vis)

    presion = presion_partido(fila_l, fila_v)



    # Presión: ajuste pequeño y simétrico al nivel de goles esperados.

    # No se fuerza a que un equipo "deba ganar"; la incertidumbre se comunica.

    if APLICAR_MICROAJUSTE_PRESION and presion["Índice presión (0-1)"] >= .80:

        f_loc *= 1.015

        f_vis *= 1.015



    return (

        float(np.clip(f_loc, .70, 1.35)),

        float(np.clip(f_vis, .65, 1.30)),

        rho,

        {

            "Altitud local": alt_l, "Altitud visita": alt_v, "Delta usado": delta,
            "Estadio": estadio_partido, "Ciudad": ciudad_partido,
            "Clima estadio": clima_estadio, "Hora inicio": hora,
            "Factor contextual local": float(f_loc),
            "Factor contextual visita": float(f_vis),
            "Descanso local": descanso_l, "Descanso visita": descanso_v,
            **presion,

        },

    )



def matriz_dixon_coles(lambda_loc, lambda_vis, rho=-.08, max_goles=9):

    matriz = np.zeros((max_goles, max_goles), dtype=float)

    for i in range(max_goles):

        for j in range(max_goles):

            p = poisson.pmf(i, lambda_loc) * poisson.pmf(j, lambda_vis)

            tau = 1.0

            if i == 0 and j == 0: tau = 1 - lambda_loc * lambda_vis * rho

            elif i == 1 and j == 0: tau = 1 + lambda_vis * rho

            elif i == 0 and j == 1: tau = 1 + lambda_loc * rho

            elif i == 1 and j == 1: tau = 1 - rho

            matriz[i, j] = max(0.0, p * tau)

    total = matriz.sum()

    return matriz / total if total > 0 else matriz



# ---------------------------------------------------------------------

# 7. Pronóstico de partido

# ---------------------------------------------------------------------

def pronosticar_partido(datos, partido):

    loc, vis = limpio(partido.get("local")), limpio(partido.get("visita"))

    fecha = partido.get("fecha")

    jornada_num = partido.get("jornada_num")



    f_loc, f_vis, rho, contexto = obtener_contexto(datos, partido)



    # Metodología principal: promedios de goles y factores de altitud.
    # Corrección metodológica: la altitud que afecta el partido es la del
    # estadio del equipo local (Altitud_Local), no la ciudad de origen del
    # equipo visitante. La fórmula del Excel usa Altitud_Visita por error.

    altitud_estadio = a_float(partido.get("altitud_local"), np.nan)

    excel = lambdas_clausura_excel(
        loc, vis, fecha, jornada_num, altitud_estadio
    )



    lambda_loc = excel["lambda_local"]

    lambda_vis = excel["lambda_visita"]



    # Respaldo si algún equipo todavía no tiene suficientes datos.

    if not np.isfinite(lambda_loc) or not np.isfinite(lambda_vis):

        ml = metricas_equipo(datos, loc, fecha, jornada_num)

        mv = metricas_equipo(datos, vis, fecha, jornada_num)

        lambda_loc = ((ml["gf_local"] + mv["ga_visita"]) / 2) if not np.isfinite(lambda_loc) else lambda_loc

        lambda_vis = ((mv["gf_visita"] + ml["ga_local"]) / 2) if not np.isfinite(lambda_vis) else lambda_vis



    # Ajustes contextuales pequeños posteriores a la fórmula Excel.

    # La altitud NO se reaplica: ya está incorporada en lambda mediante J.

    # Aquí se aplican los factores calculados por obtener_contexto: clima,

    # cancha, descanso, horario, H2H y, opcionalmente, presión.

    lambda_loc *= f_loc

    lambda_vis *= f_vis



    lambda_loc = float(np.clip(lambda_loc, .20, 4.0))

    lambda_vis = float(np.clip(lambda_vis, .20, 4.0))



    matriz = matriz_dixon_coles(lambda_loc, lambda_vis, rho)

    p_local = float(np.tril(matriz, -1).sum() * 100)

    p_empate = float(np.diag(matriz).sum() * 100)

    p_visita = float(np.triu(matriz, 1).sum() * 100)

    p_over = float(sum(matriz[i, j] for i in range(len(matriz)) for j in range(len(matriz)) if i + j >= 3) * 100)

    p_under = 100 - p_over

    p_btts = float(matriz[1:, 1:].sum() * 100)



    i_modal, j_modal = np.unravel_index(np.argmax(matriz), matriz.shape)

    probs = {"Local": p_local, "Empate": p_empate, "Visita": p_visita}

    ganador = max(probs, key=probs.get)

    orden = sorted(probs.values(), reverse=True)

    margen = orden[0] - orden[1]



    # -----------------------------------------------------------------

    # Marcador compatible

    # -----------------------------------------------------------------

    # No basta con condicionar al 1X2. El marcador mostrado debe ser

    # compatible también con los mercados que el sistema está recomendando.

    #

    # Ejemplo:

    #   Local + Over 2.5 + BTTS -> 3-0 NO es compatible.

    #   Local + Over 2.5 + BTTS -> 2-1 sí es compatible.

    mascara = np.zeros_like(matriz)



    for i in range(matriz.shape[0]):

        for j in range(matriz.shape[1]):

            if ganador == "Local":

                ok = i > j

            elif ganador == "Visita":

                ok = j > i

            else:

                ok = i == j



            # Solo condicionamos el mercado si forma parte de la

            # recomendación automática.

            if p_over >= 55:

                ok = ok and (i + j >= 3)

            elif p_under >= 58:

                ok = ok and (i + j <= 2)



            if p_btts >= 55:

                ok = ok and (i >= 1 and j >= 1)



            if ok:

                mascara[i, j] = matriz[i, j]



    if mascara.sum() > 0:

        i_pron, j_pron = np.unravel_index(np.argmax(mascara), mascara.shape)

    else:

        # Respaldo matemático: condicionado únicamente al 1X2.

        if ganador == "Local":

            mascara = np.tril(matriz, -1)

        elif ganador == "Visita":

            mascara = np.triu(matriz, 1)

        else:

            mascara = np.diag(np.diag(matriz))

        i_pron, j_pron = np.unravel_index(np.argmax(mascara), mascara.shape)



    criterios_marcador = ["1X2"]

    if p_over >= 55:

        criterios_marcador.append("Over 2.5")

    elif p_under >= 58:

        criterios_marcador.append("Under 2.5")

    if p_btts >= 55:

        criterios_marcador.append("BTTS")



    if max(probs.values()) >= 60 and margen >= 12:

        estado, senal = "FIJA", "🟢"

    elif margen <= 7 or max(probs.values()) < 40:

        estado, senal = "CERRADO", "🔴"

    else:

        estado, senal = "INTERMEDIO", "🟡"



    recomendacion = (

        f"{ganador} (confianza alta)" if estado == "FIJA" else

        f"Partido cerrado — {ganador} es una ligera preferencia" if estado == "CERRADO" else

        f"{ganador} (confianza media)"

    )

    if p_over >= 55:

        recomendacion += " | Más de 2.5 goles"

    elif p_under >= 58:

        recomendacion += " | Menos de 2.5 goles"

    if p_btts >= 55:

        recomendacion += " | Ambos marcan"



    dia = limpio(partido.get("dia")) or (pd.Timestamp(fecha).day_name() if pd.notna(fecha) else "Por definir")

    dia = {"Monday":"Lunes","Tuesday":"Martes","Wednesday":"Miércoles","Thursday":"Jueves","Friday":"Viernes","Saturday":"Sábado","Sunday":"Domingo"}.get(dia, dia)



    resultado = {

        "Señal": senal, "Estado": estado, "Día": dia or "Por definir",

        "Hora": parsear_hora(partido.get("hora")), "Local": loc, "Visita": vis,

        "λ Local": round(lambda_loc, 2), "λ Visita": round(lambda_vis, 2),

        "% Local": round(p_local, 2), "% Empate": round(p_empate, 2), "% Visita": round(p_visita, 2),

        "Más de 2.5 goles": f"{p_over:.2f}%", "Menos de 2.5 goles": f"{p_under:.2f}%", "Ambos marcan: Sí": f"{p_btts:.2f}%",

        "Marcador compatible": f"{i_pron} - {j_pron}",

        "Marcador modal": f"{i_modal} - {j_modal}",

        "Pronóstico 1X2": ganador, "Recomendacion": recomendacion,

        "Prob. principal": round(max(probs.values()), 2), "Margen 1X2": round(margen, 2),

        "Prom_GF_Local": round(excel["local"]["K"], 3) if np.isfinite(excel["local"]["K"]) else np.nan,

        "Prom_GC_Local": round(excel["local"]["L"], 3) if np.isfinite(excel["local"]["L"]) else np.nan,

        "Prom_GF_Visita": round(excel["visita"]["M"], 3) if np.isfinite(excel["visita"]["M"]) else np.nan,

        "Prom_GC_Visita": round(excel["visita"]["N"], 3) if np.isfinite(excel["visita"]["N"]) else np.nan,

                "D2": round(excel["D2"], 4) if np.isfinite(excel["D2"]) else np.nan,
        "D6": round(excel["D6"], 4) if np.isfinite(excel["D6"]) else np.nan,
        "λ Local base goles": round(excel["lambda_local_goles"], 2) if np.isfinite(excel["lambda_local_goles"]) else np.nan,
        "λ Visita base goles": round(excel["lambda_visita_goles"], 2) if np.isfinite(excel["lambda_visita_goles"]) else np.nan,
        "λ Local xG": round(excel["lambda_local_xg"], 2) if np.isfinite(excel["lambda_local_xg"]) else np.nan,
        "λ Visita xG": round(excel["lambda_visita_xg"], 2) if np.isfinite(excel["lambda_visita_xg"]) else np.nan,
        "Peso xG aplicado": round(excel["peso_xg"], 2),

        "Altitud del estadio usada por fórmula": round(altitud_estadio, 1) if np.isfinite(altitud_estadio) else np.nan,
        "Factores contexto local": round(f_loc, 3),
        "Factores contexto visita": round(f_vis, 3),
        "Criterios marcador": ", ".join(criterios_marcador),
        "Presión competitiva": contexto.get("presion_texto", "No disponible") if isinstance(contexto, dict) else "No disponible",
    }
    resultado.update(predecir_estadisticas_secundarias(loc, vis, fecha))
    return resultado


# ---------------------------------------------------------------------
# 8. Interfaz y presentación de pronósticos
# ---------------------------------------------------------------------
st.subheader("Pronósticos de partidos")
st.caption(
    "Las probabilidades son estimaciones estadísticas. Antes de usar el modelo para decisiones, "
    "valídelo con resultados históricos y sin información futura."
)

partidos_ui = partidos.copy()
if "jornada" in partidos_ui.columns:
    partidos_ui["jornada_num"] = pd.to_numeric(
        partidos_ui["jornada"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
else:
    partidos_ui["jornada_num"] = np.nan

if "fecha" in partidos_ui.columns:
    partidos_ui["fecha"] = pd.to_datetime(partidos_ui["fecha"], errors="coerce", dayfirst=True)
else:
    partidos_ui["fecha"] = pd.NaT

# Compatibilidad con distintas denominaciones de columnas de jornada.
if "jornada_num" not in partidos_ui.columns:
    partidos_ui["jornada_num"] = np.nan

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    jornadas = sorted(partidos_ui["jornada_num"].dropna().astype(int).unique().tolist())
    opciones_jornada = ["Todas"] + [str(j) for j in jornadas]
    seleccion_jornada = st.selectbox("Jornada", opciones_jornada, index=0)
with col2:
    estados_filtro = st.selectbox("Tipo de señal", ["Todas", "FIJA", "INTERMEDIO", "CERRADO"])
with col3:
    equipos = sorted(set(partidos_ui["local"].dropna().astype(str)) | set(partidos_ui["visita"].dropna().astype(str)))
    filtro_equipo = st.selectbox("Equipo", ["Todos"] + equipos, index=0)

filtrados = partidos_ui.copy()
if seleccion_jornada != "Todas":
    filtrados = filtrados[filtrados["jornada_num"] == int(seleccion_jornada)]
if filtro_equipo != "Todos":
    filtrados = filtrados[
        filtrados["local"].astype(str).eq(filtro_equipo)
        | filtrados["visita"].astype(str).eq(filtro_equipo)
    ]

if filtrados.empty:
    st.info("No hay partidos que coincidan con los filtros seleccionados.")
else:
    resultados = []
    errores = []
    barra = st.progress(0, text="Calculando pronósticos...")
    total = len(filtrados)
    for idx, (_, partido) in enumerate(filtrados.iterrows(), start=1):
        try:
            resultados.append(pronosticar_partido(datos, partido))
        except Exception as exc:
            errores.append(f"{partido.get('local', '?')} vs {partido.get('visita', '?')}: {exc}")
        barra.progress(idx / total, text=f"Calculando partido {idx} de {total}...")
    barra.empty()

    if errores:
        with st.expander(f"Advertencias de cálculo ({len(errores)})", expanded=False):
            for error in errores:
                st.warning(error)

    if resultados:
        df_resultados = pd.DataFrame(resultados)
        if estados_filtro != "Todas":
            df_resultados = df_resultados[df_resultados["Estado"] == estados_filtro]

        if not df_resultados.empty:
            # Orden de columnas principal para lectura del pronóstico.
            columnas_principales = [
                "Señal", "Estado", "Día", "Hora", "Local", "Visita",
                "% Local", "% Empate", "% Visita", "Pronóstico 1X2",
                "Marcador compatible", "Más de 2.5 goles",
                "Menos de 2.5 goles", "Ambos marcan: Sí",
                "Tiros esperados total", "Tiros a puerta esperados total",
                "Córners esperados total", "Faltas esperadas total",
                "Tarjetas amarillas esperadas total", "Recomendacion",
            ]
            # El marcador modal se conserva internamente para métricas de backtesting,
            # pero se elimina de la vista principal y del CSV de pronósticos.
            df_vista = df_resultados.drop(columns=["Marcador modal"], errors="ignore")
            columnas = [c for c in columnas_principales if c in df_vista.columns]
            columnas += [c for c in df_vista.columns if c not in columnas]

            def color_texto(valor):
                texto = str(valor).upper()
                if texto == "FIJA" or "🟢" in texto:
                    return "color: #15803d; font-weight: 700;"
                if texto == "CERRADO" or "🔴" in texto:
                    return "color: #dc2626; font-weight: 700;"
                if texto == "INTERMEDIO" or "🟡" in texto:
                    return "color: #b45309; font-weight: 700;"
                return ""

            def numero_porcentaje(valor):
                try:
                    return float(str(valor).replace("%", "").strip())
                except (TypeError, ValueError):
                    return np.nan

            def resaltar_probabilidad(valor):
                # Resalta en verde cualquier porcentaje igual o superior al 60%.
                numero = numero_porcentaje(valor)
                if np.isfinite(numero) and numero >= 60:
                    return "color: #15803d; font-weight: 700;"
                return ""

            # Tabla resumen: una fila por partido, con porcentajes de hasta dos decimales.
            columnas_resumen = [
                "Señal", "Local", "Visita", "Pronóstico 1X2",
                "% Local", "% Empate", "% Visita",
                "Más de 2.5 goles", "Menos de 2.5 goles", "Ambos marcan: Sí",
                "Tiros esperados total", "Tiros a puerta esperados total",
                "Córners esperados total",
            ]
            columnas_resumen = [c for c in columnas_resumen if c in df_resultados.columns]
            df_resumen = df_resultados[columnas_resumen].copy()

            for col in ["% Local", "% Empate", "% Visita"]:
                if col in df_resumen.columns:
                    df_resumen[col] = df_resumen[col].map(
                        lambda v: f"{float(v):.2f}%" if pd.notna(v) else ""
                    )

            def aplicar_estilos(tabla, columnas_tabla):
                estilo = tabla.style
                for col in [c for c in ["Señal", "Estado", "Pronóstico 1X2"] if c in columnas_tabla]:
                    estilo = estilo.map(color_texto, subset=[col])
                # Detecta columnas de porcentajes y también valores que ya incluyen el signo %.
                columnas_porcentaje = [
                    c for c in columnas_tabla
                    if "%" in c or "Prob." in c or "probabilidad" in c.lower()
                ]
                for col in columnas_porcentaje:
                    estilo = estilo.map(resaltar_probabilidad, subset=[col])
                for col in columnas_tabla:
                    if tabla[col].astype(str).str.contains("%", regex=False).any():
                        estilo = estilo.map(resaltar_probabilidad, subset=[col])
                return estilo

            st.subheader("Tabla resumen de resultados")
            st.dataframe(
                aplicar_estilos(df_resumen, list(df_resumen.columns)),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Detalle de pronósticos")
            styler = aplicar_estilos(df_vista[columnas], columnas)
            formato = {
                c: "{:.2f}%" for c in ["% Local", "% Empate", "% Visita"]
                if c in columnas
            }
            st.dataframe(
                styler.format(formato, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )

            csv = df_vista.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Descargar pronósticos en CSV",
                data=csv,
                file_name="pronosticos_liga1.csv",
                mime="text/csv",
            )

            with st.expander("Ver datos técnicos del modelo", expanded=False):
                tecnicas = [
                    "λ Local", "λ Visita", "Prob. principal", "Margen 1X2",
                    "Prom_GF_Local", "Prom_GC_Local", "Prom_GF_Visita", "Prom_GC_Visita",
                    "D2", "D6", "Altitud del estadio usada por fórmula",
                    "Factores contexto local", "Factores contexto visita", "Criterios marcador",
                    "Presión competitiva", "Tiros esperados total",
                    "Tiros a puerta esperados total", "Córners esperados total",
                    "Faltas esperadas total", "Tarjetas amarillas esperadas total",
                    "Partidos estadísticos disponibles",
                ]
                cols_tec = [c for c in tecnicas if c in df_resultados.columns]
                st.dataframe(df_resultados[cols_tec], use_container_width=True, hide_index=True)
        else:
            st.info("No hay resultados para el tipo de señal seleccionado.")


# ---------------------------------------------------------------------
# 9. Backtesting cronológico (solo Clausura, con Apertura como histórico previo)
# ---------------------------------------------------------------------
def ejecutar_backtesting_clausura(datos):
    """
    Evalúa cada partido del Clausura con los resultados del Apertura y solo
    las jornadas anteriores del Clausura. Los partidos de la misma jornada
    se excluyen todos del entrenamiento porque el Excel no trae fecha real
    para cada resultado histórico.
    """
    df = datos.get("Resultados_Clausura", pd.DataFrame()).copy()
    columnas_req = {"jornada", "local", "visita", "gl", "gv"}
    if df.empty or not columnas_req.issubset(df.columns):
        return pd.DataFrame(), "La hoja Resultados_Clausura no contiene las columnas necesarias."

    df["jornada_num"] = pd.to_numeric(
        df["jornada"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    df["gl"] = pd.to_numeric(df["gl"], errors="coerce")
    df["gv"] = pd.to_numeric(df["gv"], errors="coerce")
    df = df.dropna(subset=["jornada_num", "local", "visita", "gl", "gv"])
    df = df.sort_values(["jornada_num", "local", "visita"]).reset_index(drop=True)

    resultados_bt = []
    for _, partido in df.iterrows():
        # No se evalúa si faltan datos identificables.
        try:
            pred = pronosticar_partido(datos, partido)
            gl_real, gv_real = int(partido["gl"]), int(partido["gv"])
            real = "Local" if gl_real > gv_real else ("Visita" if gl_real < gv_real else "Empate")
            probs = {
                "Local": float(pred["% Local"]) / 100.0,
                "Empate": float(pred["% Empate"]) / 100.0,
                "Visita": float(pred["% Visita"]) / 100.0,
            }
            p_real = max(1e-6, min(1.0, probs[real]))
            brier = sum((probs[k] - (1.0 if k == real else 0.0)) ** 2 for k in probs)
            resultados_bt.append({
                "Jornada": int(partido["jornada_num"]),
                "Local": partido["local"],
                "Visita": partido["visita"],
                "Goles reales": f"{gl_real}-{gv_real}",
                "Resultado real": real,
                "Pronóstico 1X2": pred["Pronóstico 1X2"],
                "Acierto 1X2": int(pred["Pronóstico 1X2"] == real),
                "Marcador modal": pred["Marcador modal"],
                "Acierto marcador exacto": int(pred["Marcador modal"] == f"{gl_real} - {gv_real}"),
                "Prob. Local": probs["Local"],
                "Prob. Empate": probs["Empate"],
                "Prob. Visita": probs["Visita"],
                "Brier": brier,
                "Log loss": -np.log(p_real),
                "λ Local": pred["λ Local"],
                "λ Visita": pred["λ Visita"],
            })
        except Exception as exc:
            resultados_bt.append({
                "Jornada": int(partido["jornada_num"]),
                "Local": partido.get("local", ""),
                "Visita": partido.get("visita", ""),
                "Error": str(exc),
            })
    return pd.DataFrame(resultados_bt), None


with st.expander("📈 Backtesting: evaluación con partidos ya jugados", expanded=False):
    st.write(
        "El backtesting simula cada partido del Clausura usando el Apertura como "
        "histórico previo y solo jornadas anteriores del Clausura. Todos los partidos "
        "de una misma jornada quedan fuera del entrenamiento. Como la hoja histórica "
        "no contiene fechas reales por partido, el corte temporal se realiza por jornada. "
        "El modelo también considera el estadio, la altitud y la hora de inicio; los "
        "ajustes hora-clima son conservadores y no sustituyen datos meteorológicos reales."
    )
    st.warning(
        "La evaluación no es una garantía de rendimiento futuro. Las tablas de posiciones "
        "actuales solo se muestran como contexto; no se utilizan para ajustar la probabilidad "
        "durante el backtesting. El ajuste xG es experimental y debe compararse con el modelo "
        "sin xG antes de decidir si conservarlo."
    )
    if st.button("Ejecutar backtesting del Clausura", key="btn_backtesting"):
        with st.spinner("Comparando el modelo con y sin xG, respetando el orden de las jornadas..."):
            peso_original = PESO_XG_MODELO
            h2h_original = APLICAR_H2H_MODELO
            stats_original = APLICAR_STATS_SECUNDARIAS
            # H2H y estadísticas secundarias se desactivan durante backtesting:
            # los resultados históricos no tienen fecha individual verificable.
            APLICAR_H2H_MODELO = False
            APLICAR_STATS_SECUNDARIAS = False
            try:
                PESO_XG_MODELO = 0.0
                df_bt, mensaje_bt = ejecutar_backtesting_clausura(datos)
                PESO_XG_MODELO = 0.20
                df_con_xg, mensaje_con_xg = ejecutar_backtesting_clausura(datos)
            finally:
                PESO_XG_MODELO = peso_original
                APLICAR_H2H_MODELO = h2h_original
                APLICAR_STATS_SECUNDARIAS = stats_original
            mensaje_sin_xg = None
        if mensaje_bt or mensaje_con_xg:
            st.error(mensaje_bt or mensaje_con_xg)
        elif df_bt.empty:
            st.info("No hay partidos históricos suficientes para evaluar.")
        else:
            if "Error" in df_bt.columns:
                errores_bt = df_bt[df_bt["Error"].notna()] if df_bt["Error"].notna().any() else pd.DataFrame()
                df_ok = df_bt[df_bt["Error"].isna()].copy() if df_bt["Error"].notna().any() else df_bt.copy()
            else:
                errores_bt = pd.DataFrame()
                df_ok = df_bt.copy()

            if not errores_bt.empty:
                st.warning(f"{len(errores_bt)} partidos no pudieron evaluarse.")
                st.dataframe(errores_bt[["Jornada", "Local", "Visita", "Error"]], use_container_width=True, hide_index=True)

            if not df_ok.empty and "Acierto 1X2" in df_ok.columns:
                n = len(df_ok)
                accuracy = df_ok["Acierto 1X2"].mean() * 100
                exact = df_ok["Acierto marcador exacto"].mean() * 100
                brier_mean = df_ok["Brier"].mean()
                logloss_mean = df_ok["Log loss"].mean()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Partidos evaluados", f"{n}")
                c2.metric("Acierto 1X2", f"{accuracy:.1f}%")
                c3.metric("Marcador exacto", f"{exact:.1f}%")
                c4.metric("Brier medio", f"{brier_mean:.3f}")
                st.caption(f"Log loss medio: {logloss_mean:.3f}. Para Brier y log loss, valores menores son mejores.")
                referencia_local = (df_ok["Resultado real"] == "Local").mean() * 100
                st.caption(
                    f"Referencia simple: acertar siempre al equipo local habría logrado "
                    f"{referencia_local:.1f}% en estos mismos partidos."
                )
                if not df_con_xg.empty and "Acierto 1X2" in df_con_xg.columns:
                    df_con_xg_ok = df_con_xg[df_con_xg["Acierto 1X2"].notna()].copy()
                    if not df_con_xg_ok.empty:
                        comparacion = pd.DataFrame([
                            {
                                "Modelo": "Sin xG (peso 0%)",
                                "Partidos": len(df_ok),
                                "Acierto 1X2 (%)": round(df_ok["Acierto 1X2"].mean() * 100, 2),
                                "Brier medio": round(df_ok["Brier"].mean(), 4),
                                "Log loss medio": round(df_ok["Log loss"].mean(), 4),
                            },
                            {
                                "Modelo": "Con xG (peso 20%)",
                                "Partidos": len(df_con_xg_ok),
                                "Acierto 1X2 (%)": round(df_con_xg_ok["Acierto 1X2"].mean() * 100, 2),
                                "Brier medio": round(df_con_xg_ok["Brier"].mean(), 4),
                                "Log loss medio": round(df_con_xg_ok["Log loss"].mean(), 4),
                            },
                        ])
                        st.subheader("Comparación del modelo con y sin xG")
                        st.dataframe(comparacion, use_container_width=True, hide_index=True)
                        st.caption("No se debe conservar el ajuste xG solo porque suba el porcentaje de acierto: compare también Brier y log loss. Menor Brier/log loss es mejor.")
                st.subheader("Resultados por jornada")
                por_jornada = df_ok.groupby("Jornada", as_index=False).agg(
                    Partidos=("Acierto 1X2", "count"),
                    Acierto_1X2=("Acierto 1X2", "mean"),
                    Brier_medio=("Brier", "mean"),
                    Log_loss_medio=("Log loss", "mean"),
                )
                por_jornada["Acierto 1X2"] = (por_jornada["Acierto_1X2"] * 100).map(lambda v: f"{v:.2f}%")
                por_jornada["Brier medio"] = por_jornada["Brier_medio"].round(3)
                por_jornada["Log loss medio"] = por_jornada["Log_loss_medio"].round(3)
                st.dataframe(
                    por_jornada[["Jornada", "Partidos", "Acierto 1X2", "Brier medio", "Log loss medio"]],
                    use_container_width=True, hide_index=True
                )
                st.subheader("Detalle de partidos")
                df_bt_vista = df_ok.drop(columns=["Marcador modal"], errors="ignore")
                st.dataframe(df_bt_vista, use_container_width=True, hide_index=True)
                st.download_button(
                    "Descargar resultados del backtesting (CSV)",
                    data=df_bt_vista.to_csv(index=False).encode("utf-8-sig"),
                    file_name="backtesting_liga1.csv",
                    mime="text/csv",
                    key="download_backtesting",
                )
            else:
                st.info("No se pudieron evaluar partidos válidos.")

st.divider()
st.caption(
    "Nota metodológica: el modelo no garantiza aciertos. Las etiquetas FIJA, INTERMEDIO y CERRADO "
    "son reglas internas basadas en probabilidades estimadas, no certezas."
)
