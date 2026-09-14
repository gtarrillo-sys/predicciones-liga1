# app_liga1_v4.py
# Modelo de Predicción Liga 1 Perú - V4
# Uso: streamlit run app_liga1_v4.py
#
# Cambios principales:
# - Usa Apertura + Clausura en orden cronológico.
# - No usa xG_Local/xG_Visita como variables predictoras.
# - Elo cronológico.
# - Forma últimos 5.
# - Ataque/defensa dinámicos con suavizado.
# - Altitud con efecto moderado.
# - Rivalidades con ajuste moderado.
# - H2H se muestra como diagnóstico; por defecto NO altera la probabilidad,
#   porque el backtesting realizado sobre el histórico disponible no justificó
#   un peso fijo para H2H.
# - Incluye automáticamente los 8 resultados de Fecha 9 proporcionados por el usuario
#   si todavía no aparecen en las hojas de resultados.
# - Permite seleccionar Jornada 10, Jornada 11 u otra jornada disponible.
# - Permite actualizar resultados desde la interfaz y descargar un Excel actualizado.

import io
import os
import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson
import google.generativeai as genai


# =========================================================
# 1. CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Liga 1 Perú - Predictor V4",
    page_icon="⚽",
    layout="wide",
)

ARCHIVO_DEFECTO = "Liga1_2026.xlsx"

def obtener_analisis_ia(local, visita, altitud, desc_l, desc_v, p_l, p_e, p_v):
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        return "Configura la clave GOOGLE_API_KEY en los Secrets de Streamlit para activar este análisis."
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Eres un analista deportivo experto en la Liga 1 peruana.
    Analiza brevemente el partido: {local} vs {visita}.
    - Datos estadísticos: Gana Local {p_l}%, Empate {p_e}%, Gana Visita {p_v}%.
    - Altitud del estadio: {altitud} msnm.
    - Días de descanso: Local ({desc_l} días) vs Visita ({desc_v} días).
    
    Redacta una conclusión táctica muy corta de 3 líneas (en un solo párrafo) que sirva como veredicto del impacto físico de la altura o descanso en el resultado.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Error al generar el análisis táctico de IA."

# Parámetros calibrables
ELO_INICIAL = 1500.0
ELO_K = 24.0
ELO_LOCAL = 55.0

VENTANA_FORMA = 5
VENTANA_RATES = 8

# Suavizado de goles por partido
PRIOR_GF = 1.20
PRIOR_GA = 1.20
PRIOR_PESO = 5.0

# Peso de componentes
PESO_ELO = 0.28
PESO_FORMA = 0.12
PESO_TASA = 0.60

# Altitud: deliberadamente moderada.
# La ventaja principal de la localía se mantiene separada.
ALTITUD_REFERENCIA = 1500.0
ALTITUD_COEF_LOCAL = 0.000035
ALTITUD_COEF_VISITA = 0.000045
ALTITUD_MAX_FACTOR = 1.16
ALTITUD_MIN_VISITA = 0.86

RHO_DIXON_COLES = -0.08

# H2H: por el backtesting realizado, queda en 0 por defecto.
# Se puede activar experimentalmente desde la barra lateral.
H2H_PESO_DEFECTO = 0.00
H2H_MAX = 10


# =========================================================
# 2. NORMALIZACIÓN DE EQUIPOS
# =========================================================

def quitar_acentos(texto):
    texto = "" if texto is None else str(texto)
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def normalizar_nombre(texto):
    """
    Normalización conservadora.
    NO elimina 'atletico', porque Atlético Grau y Alianza Atlético
    son clubes distintos.
    """
    s = quitar_acentos(texto).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    equivalencias = {
        "fc cajamarca": "fc cajamarca",
        "fc cajamarca peru": "fc cajamarca",
        "utc": "utc cajamarca",
        "utc cajamarca": "utc cajamarca",
        "ut cajamarca": "utc cajamarca",

        "atletico grau": "atletico grau",
        "atletico grau piura": "atletico grau",
        "alianza atletico": "alianza atletico",

        "juan pablo ii": "juan pablo ii college",
        "juan pablo ii college": "juan pablo ii college",

        "comerciantes unidos": "comerciantes unidos",
        "chankas cyc": "chankas cyc",
        "los chankas": "chankas cyc",
        "deportivo garcilaso": "deportivo garcilaso",
        "garcilaso": "deportivo garcilaso",

        "sporting cristal": "sporting cristal",
        "cristal": "sporting cristal",
        "sport boys": "sport boys",
        "universitario": "universitario",
        "universitario de deportes": "universitario",
        "alianza lima": "alianza lima",
        "melgar": "melgar",
        "fbc melgar": "melgar",
        "cusco fc": "cusco fc",
        "cienciano": "cienciano",
        "adt": "adt",
        "cd moquegua": "cd moquegua",
        "deportivo moquegua": "cd moquegua",
        "sport huancayo": "sport huancayo",
    }

    return equivalencias.get(s, s)


# Rivalidades: se compara como pareja sin importar localía.
RIVALIDADES = {
    frozenset(["atletico grau", "alianza atletico"]): "Clásico Piurano",
    frozenset(["cusco fc", "cienciano"]): "Clásico Cusqueño",
    frozenset(["fc cajamarca", "utc cajamarca"]): "Clásico Cajamarquino",
    frozenset(["fc cajamarca", "comerciantes unidos"]): "Rivalidad Cajamarquina",
    frozenset(["utc cajamarca", "comerciantes unidos"]): "Rivalidad Cajamarquina",
}


def nombre_rivalidad(local, visita):
    return RIVALIDADES.get(
        frozenset([normalizar_nombre(local), normalizar_nombre(visita)]),
        "",
    )


# =========================================================
# 3. ALTITUDES DE RESPALDO
# =========================================================

ALTITUDES_DEFAULT = {
    "alianza lima": (150, "Lima"),
    "universitario": (150, "Lima"),
    "sporting cristal": (150, "Lima"),
    "sport boys": (10, "Callao"),
    "atletico grau": (50, "Sullana"),
    "alianza atletico": (50, "Sullana"),
    "utc cajamarca": (2750, "Cajamarca"),
    "fc cajamarca": (2750, "Cajamarca"),
    "comerciantes unidos": (2620, "Cutervo"),
    "chankas cyc": (2920, "Andahuaylas"),
    "cusco fc": (3399, "Cusco"),
    "cienciano": (3399, "Cusco"),
    "deportivo garcilaso": (3399, "Cusco"),
    "melgar": (2335, "Arequipa"),
    "sport huancayo": (3259, "Huancayo"),
    "adt": (3050, "Tarma"),
    "cd moquegua": (1410, "Moquegua"),
    "juan pablo ii college": (150, "Chongoyape"),
}


# =========================================================
# 4. RESULTADOS DE FECHA 9 PROPORCIONADOS POR EL USUARIO
# =========================================================

# Esta lista ya se encuentra integrada en las hojas del archivo Excel.
RESULTADOS_FECHA_9 = []


# =========================================================
# 5. UTILIDADES DE COLUMNAS
# =========================================================

def buscar_columna(df, candidatos):
    mapa = {str(c).strip().lower(): c for c in df.columns}
    for c in candidatos:
        if c.lower() in mapa:
            return mapa[c.lower()]
    return None


def estandarizar_columnas(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


# =========================================================
# 6. CARGA DEL EXCEL
# =========================================================

def cargar_excel(fuente):
    xls = pd.ExcelFile(fuente)
    hojas = xls.sheet_names

    requeridas = [
        "Partidos_Fecha",
        "Resultados_Apertura",
        "Resultados_Clausura",
        "Tabla_Acumulada",
    ]
    faltantes = [h for h in requeridas if h not in hojas]
    if faltantes:
        raise ValueError(f"Faltan hojas obligatorias: {faltantes}")

    partidos = estandarizar_columnas(
        pd.read_excel(xls, sheet_name="Partidos_Fecha")
    )
    apertura = estandarizar_columnas(
        pd.read_excel(xls, sheet_name="Resultados_Apertura")
    )
    clausura = estandarizar_columnas(
        pd.read_excel(xls, sheet_name="Resultados_Clausura")
    )
    acumulada = estandarizar_columnas(
        pd.read_excel(xls, sheet_name="Tabla_Acumulada")
    )

    clausura_tabla = (
        estandarizar_columnas(
            pd.read_excel(xls, sheet_name="Tabla_Clausura")
        )
        if "Tabla_Clausura" in hojas
        else acumulada.copy()
    )

    if "Data_Geografica" in hojas:
        geo = estandarizar_columnas(
            pd.read_excel(xls, sheet_name="Data_Geografica")
        )
    else:
        geo = pd.DataFrame()

    if "Historial_H2H" in hojas:
        h2h = estandarizar_columnas(
            pd.read_excel(xls, sheet_name="Historial_H2H")
        )
    else:
        h2h = pd.DataFrame(
            columns=["Fecha", "Local", "Visitante", "GL", "GV", "Temporada"]
        )

    return {
        "partidos": partidos,
        "apertura": apertura,
        "clausura": clausura,
        "acumulada": acumulada,
        "tabla_clausura": clausura_tabla,
        "geo": geo,
        "h2h": h2h,
        "hojas": hojas,
    }


# =========================================================
# 7. PREPARAR RESULTADOS
# =========================================================

def preparar_resultados(df, fase):
    df = df.copy()

    col_jornada = buscar_columna(df, ["Jornada", "Fecha_Num"])
    col_local = buscar_columna(df, ["Local"])
    col_visita = buscar_columna(df, ["Visita", "Visitante"])
    col_gl = buscar_columna(df, ["Goles_Local", "GL"])
    col_gv = buscar_columna(df, ["Goles_Visita", "GV"])
    col_fecha = buscar_columna(df, ["Fecha"])

    if not all([col_local, col_visita, col_gl, col_gv]):
        return pd.DataFrame(
            columns=[
                "fase", "jornada", "fecha", "local", "visita",
                "gl", "gv", "local_std", "visita_std"
            ]
        )

    out = pd.DataFrame()
    out["fase"] = fase
    out["jornada"] = (
        df[col_jornada].astype(str) if col_jornada else ""
    )

    if col_fecha:
        out["fecha"] = pd.to_datetime(df[col_fecha], errors="coerce")
    else:
        out["fecha"] = pd.NaT

    out["local"] = df[col_local].astype(str).str.strip()
    out["visita"] = df[col_visita].astype(str).str.strip()
    out["gl"] = pd.to_numeric(df[col_gl], errors="coerce")
    out["gv"] = pd.to_numeric(df[col_gv], errors="coerce")

    out["local_std"] = out["local"].apply(normalizar_nombre)
    out["visita_std"] = out["visita"].apply(normalizar_nombre)

    # Si no hay fecha en el Excel, se mantiene el orden de lectura.
    out["_orden"] = np.arange(len(out))
    out = out.dropna(subset=["gl", "gv"]).copy()

    return out


def crear_historico(datos):
    ap = preparar_resultados(datos["apertura"], "Apertura")
    cl = preparar_resultados(datos["clausura"], "Clausura")

    hist = pd.concat([ap, cl], ignore_index=True)

    # En el Excel, Apertura aparece antes que Clausura.
    # Para datos con fecha real, ordenar cronológicamente.
    if hist["fecha"].notna().any():
        hist["_fecha_sort"] = hist["fecha"].fillna(pd.Timestamp("1900-01-01"))
        hist = hist.sort_values(
            ["_fecha_sort", "_orden"], kind="stable"
        ).drop(columns=["_fecha_sort"])

    hist = hist.reset_index(drop=True)
    return hist


def agregar_fecha_9_si_falta(hist):
    """
    Agrega los resultados de Fecha 9 dados por el usuario únicamente
    si el mismo partido y marcador no están ya en el histórico.
    """
    hist = hist.copy()

    claves = set()
    for _, r in hist.iterrows():
        claves.add(
            (
                normalizar_nombre(r["local"]),
                normalizar_nombre(r["visita"]),
                int(r["gl"]),
                int(r["gv"]),
            )
        )

    nuevas = []
    for jornada, fecha, local, visita, gl, gv in RESULTADOS_FECHA_9:
        clave = (
            normalizar_nombre(local),
            normalizar_nombre(visita),
            gl,
            gv,
        )
        if clave not in claves:
            nuevas.append({
                "fase": "Clausura",
                "jornada": jornada,
                "fecha": pd.to_datetime(fecha),
                "local": local,
                "visita": visita,
                "gl": gl,
                "gv": gv,
                "local_std": normalizar_nombre(local),
                "visita_std": normalizar_nombre(visita),
                "_orden": 100000 + len(nuevas),
            })

    if nuevas:
        hist = pd.concat([hist, pd.DataFrame(nuevas)], ignore_index=True)
        hist["_fecha_sort"] = hist["fecha"].fillna(pd.Timestamp("1900-01-01"))
        hist = hist.sort_values(
            ["_fecha_sort", "_orden"], kind="stable"
        ).drop(columns=["_fecha_sort"]).reset_index(drop=True)

    return hist


# =========================================================
# 8. DATOS GEOGRÁFICOS
# =========================================================

def construir_geo(df_geo):
    geo = {}

    if not df_geo.empty:
        col_club = buscar_columna(df_geo, ["Club", "Equipo", "Nombre"])
        col_ciudad = buscar_columna(df_geo, ["Ciudad"])
        col_alt = buscar_columna(df_geo, ["Altitud_msnm", "Altitud", "Altitud_ms"])
        if col_club and col_alt:
            for _, r in df_geo.iterrows():
                equipo = normalizar_nombre(r[col_club])
                try:
                    alt = float(r[col_alt])
                except Exception:
                    alt = ALTITUDES_DEFAULT.get(equipo, (0, ""))[0]
                ciudad = (
                    str(r[col_ciudad])
                    if col_ciudad and pd.notna(r[col_ciudad])
                    else ALTITUDES_DEFAULT.get(equipo, (alt, ""))[1]
                )
                geo[equipo] = (alt, ciudad)

    for equipo, valor in ALTITUDES_DEFAULT.items():
        geo.setdefault(equipo, valor)

    return geo


# =========================================================
# 9. ELO CRONOLÓGICO
# =========================================================

def calcular_elo_prepartido(hist):
    """
    Devuelve un DataFrame con el Elo disponible ANTES de cada partido.
    """
    elo = {}
    registros = []

    for idx, r in hist.iterrows():
        loc = r["local_std"]
        vis = r["visita_std"]

        elo.setdefault(loc, ELO_INICIAL)
        elo.setdefault(vis, ELO_INICIAL)

        elo_loc = elo[loc]
        elo_vis = elo[vis]

        esperado_loc = 1.0 / (
            1.0 + 10.0 ** (-(elo_loc + ELO_LOCAL - elo_vis) / 400.0)
        )

        if r["gl"] > r["gv"]:
            resultado = 1.0
        elif r["gl"] == r["gv"]:
            resultado = 0.5
        else:
            resultado = 0.0

        registros.append({
            "idx": idx,
            "elo_local_pre": elo_loc,
            "elo_visita_pre": elo_vis,
            "elo_diff_pre": elo_loc + ELO_LOCAL - elo_vis,
            "elo_expected_local_pre": esperado_loc,
        })

        elo[loc] += ELO_K * (resultado - esperado_loc)
        elo[vis] += ELO_K * ((1.0 - resultado) - (1.0 - esperado_loc))

    return pd.DataFrame(registros).set_index("idx"), elo


# =========================================================
# 10. FORMA RECIENTE
# =========================================================

def calcular_forma(hist, equipo, antes_de=None, ventana=5):
    h = hist.copy()

    if antes_de is not None:
        if pd.notna(antes_de):
            h = h[h["fecha"].isna() | (h["fecha"] < antes_de)]

    h = h[
        (h["local_std"] == equipo) |
        (h["visita_std"] == equipo)
    ].copy()

    if h.empty:
        return {
            "puntos": 0.5,
            "gf": 1.20,
            "ga": 1.20,
            "n": 0,
        }

    if h["fecha"].notna().any():
        h["_fecha"] = h["fecha"].fillna(pd.Timestamp("1900-01-01"))
        h = h.sort_values("_fecha")
    else:
        h = h.sort_index()

    h = h.tail(ventana)

    puntos = 0
    gf = 0
    ga = 0

    for _, r in h.iterrows():
        if r["local_std"] == equipo:
            gf += r["gl"]
            ga += r["gv"]
            if r["gl"] > r["gv"]:
                puntos += 3
            elif r["gl"] == r["gv"]:
                puntos += 1
        else:
            gf += r["gv"]
            ga += r["gl"]
            if r["gv"] > r["gl"]:
                puntos += 3
            elif r["gv"] == r["gl"]:
                puntos += 1

    n = len(h)
    return {
        "puntos": puntos / max(3 * n, 1),
        "gf": gf / max(n, 1),
        "ga": ga / max(n, 1),
        "n": n,
    }


# =========================================================
# 11. TASAS DE ATAQUE / DEFENSA PREPARTIDO
# =========================================================

def calcular_tasas_equipo(hist, equipo, antes_de=None, ventana=8):
    h = hist.copy()

    if antes_de is not None and pd.notna(antes_de):
        h = h[h["fecha"].isna() | (h["fecha"] < antes_de)]

    h = h[
        (h["local_std"] == equipo) |
        (h["visita_std"] == equipo)
    ].copy()

    if h.empty:
        return PRIOR_GF, PRIOR_GA, 0

    if h["fecha"].notna().any():
        h["_fecha"] = h["fecha"].fillna(pd.Timestamp("1900-01-01"))
        h = h.sort_values("_fecha")

    h = h.tail(ventana)

    gf = []
    ga = []
    for _, r in h.iterrows():
        if r["local_std"] == equipo:
            gf.append(float(r["gl"]))
            ga.append(float(r["gv"]))
        else:
            gf.append(float(r["gv"]))
            ga.append(float(r["gl"]))

    n = len(gf)
    media_gf = (sum(gf) + PRIOR_GF * PRIOR_PESO) / (n + PRIOR_PESO)
    media_ga = (sum(ga) + PRIOR_GA * PRIOR_PESO) / (n + PRIOR_PESO)

    return media_gf, media_ga, n


# =========================================================
# 12. INFORMACIÓN DE TABLA
# =========================================================

def tabla_dict(df_tabla):
    if df_tabla is None or df_tabla.empty:
        return {}

    col_club = buscar_columna(df_tabla, ["Club", "Equipo"])
    col_pts = buscar_columna(df_tabla, ["Pts", "Puntos"])
    col_gf = buscar_columna(df_tabla, ["GF"])
    col_gc = buscar_columna(df_tabla, ["GC"])
    col_pj = buscar_columna(df_tabla, ["PJ"])

    if not col_club:
        return {}

    salida = {}
    for _, r in df_tabla.iterrows():
        equipo = normalizar_nombre(r[col_club])
        salida[equipo] = {
            "pts": float(r[col_pts]) if col_pts and pd.notna(r[col_pts]) else 0,
            "gf": float(r[col_gf]) if col_gf and pd.notna(r[col_gf]) else 0,
            "gc": float(r[col_gc]) if col_gc and pd.notna(r[col_gc]) else 0,
            "pj": float(r[col_pj]) if col_pj and pd.notna(r[col_pj]) else 0,
        }
    return salida


# =========================================================
# 13. H2H
# =========================================================

def preparar_h2h(df_h2h):
    if df_h2h is None or df_h2h.empty:
        return pd.DataFrame(
            columns=["fecha", "local", "visita", "gl", "gv",
                     "local_std", "visita_std"]
        )

    d = df_h2h.copy()

    c_fecha = buscar_columna(d, ["Fecha"])
    c_local = buscar_columna(d, ["Local"])
    c_vis = buscar_columna(d, ["Visitante", "Visita"])
    c_gl = buscar_columna(d, ["GL", "Goles_Local"])
    c_gv = buscar_columna(d, ["GV", "Goles_Visita"])

    if not all([c_fecha, c_local, c_vis, c_gl, c_gv]):
        return pd.DataFrame(
            columns=["fecha", "local", "visita", "gl", "gv",
                     "local_std", "visita_std"]
        )

    out = pd.DataFrame({
        "fecha": pd.to_datetime(d[c_fecha], errors="coerce"),
        "local": d[c_local].astype(str).str.strip(),
        "visita": d[c_vis].astype(str).str.strip(),
        "gl": pd.to_numeric(d[c_gl], errors="coerce"),
        "gv": pd.to_numeric(d[c_gv], errors="coerce"),
    }).dropna(subset=["fecha", "gl", "gv"])

    out["local_std"] = out["local"].apply(normalizar_nombre)
    out["visita_std"] = out["visita"].apply(normalizar_nombre)
    return out.sort_values("fecha").reset_index(drop=True)


def obtener_h2h(df_h2h, local, visita, fecha_partido=None, n=10):
    if df_h2h is None or df_h2h.empty:
        return pd.DataFrame()

    loc = normalizar_nombre(local)
    vis = normalizar_nombre(visita)
    d = df_h2h.copy()

    if fecha_partido is not None and pd.notna(fecha_partido):
        d = d[d["fecha"] < fecha_partido].copy()

    d = d[
        (
            (d["local_std"] == loc) &
            (d["visita_std"] == vis)
        ) |
        (
            (d["local_std"] == vis) &
            (d["visita_std"] == loc)
        )
    ].copy()

    return d.sort_values("fecha", ascending=False).head(n).reset_index(drop=True)


def h2h_indice(df_h2h, local, visita, fecha_partido=None):
    """
    Índice del rendimiento del local en H2H, entre 0 y 1.
    Se usa solamente si el usuario activa H2H experimental.
    """
    h = obtener_h2h(
        df_h2h, local, visita, fecha_partido=fecha_partido, n=H2H_MAX
    )

    if h.empty:
        return None, h

    hoy = (
        pd.Timestamp(fecha_partido)
        if fecha_partido is not None and pd.notna(fecha_partido)
        else h["fecha"].max() + pd.Timedelta(days=1)
    )

    pesos = []
    resultados = []

    for _, r in h.iterrows():
        dias = max((hoy - r["fecha"]).days, 0)
        anos = dias / 365.25

        if anos <= 2:
            w = 1.00
        elif anos <= 4:
            w = 0.75
        elif anos <= 6:
            w = 0.50
        elif anos <= 8:
            w = 0.30
        else:
            w = 0.15

        # Resultado desde la perspectiva del equipo local del partido futuro.
        if r["local_std"] == normalizar_nombre(local):
            if r["gl"] > r["gv"]:
                res = 1.0
            elif r["gl"] == r["gv"]:
                res = 0.5
            else:
                res = 0.0
        else:
            if r["gv"] > r["gl"]:
                res = 1.0
            elif r["gv"] == r["gl"]:
                res = 0.5
            else:
                res = 0.0

        pesos.append(w)
        resultados.append(res)

    indice = float(np.average(resultados, weights=pesos))
    return indice, h


# =========================================================
# 14. DIXON-COLES
# =========================================================

def tau_dixon_coles(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return max(0.01, 1 - lam * mu * rho)
    if x == 0 and y == 1:
        return max(0.01, 1 + lam * rho)
    if x == 1 and y == 0:
        return max(0.01, 1 + mu * rho)
    if x == 1 and y == 1:
        return max(0.01, 1 - rho)
    return 1.0


def matriz_dixon_coles(lam, mu, rho=RHO_DIXON_COLES, max_goles=8):
    m = np.zeros((max_goles + 1, max_goles + 1))

    for x in range(max_goles + 1):
        for y in range(max_goles + 1):
            p = poisson.pmf(x, lam) * poisson.pmf(y, mu)
            p *= tau_dixon_coles(x, y, lam, mu, rho)
            m[x, y] = max(0.0, p)

    total = m.sum()
    if total > 0:
        m /= total
    return m


# =========================================================
# 15. CÁLCULO DE FUERZA Y PROBABILIDADES
# =========================================================

def obtener_elo_actual(hist):
    _, elo_final = calcular_elo_prepartido(hist)
    return elo_final


def fuerza_partido(
    local,
    visita,
    fecha_partido,
    hist,
    elo_actual,
    geo,
    tabla_actual=None,
):
    loc = normalizar_nombre(local)
    vis = normalizar_nombre(visita)

    elo_loc = elo_actual.get(loc, ELO_INICIAL)
    elo_vis = elo_actual.get(vis, ELO_INICIAL)

    # Diferencia Elo estandarizada.
    elo_score = np.clip((elo_loc + ELO_LOCAL - elo_vis) / 400.0, -2, 2)

    form_loc = calcular_forma(hist, loc, fecha_partido, VENTANA_FORMA)
    form_vis = calcular_forma(hist, vis, fecha_partido, VENTANA_FORMA)

    gf_l, ga_l, n_l = calcular_tasas_equipo(
        hist, loc, fecha_partido, VENTANA_RATES
    )
    gf_v, ga_v, n_v = calcular_tasas_equipo(
        hist, vis, fecha_partido, VENTANA_RATES
    )

    # Índice de forma: rango aproximado -1 a +1.
    forma_score = np.clip(form_loc["puntos"] - form_vis["puntos"], -1, 1)

    # Diferencia ofensiva/defensiva.
    tasa_score = np.clip(
        ((gf_l - ga_v) - (gf_v - ga_l)) / 3.0,
        -1.5,
        1.5,
    )

    # Tabla actual: se utiliza como apoyo para pronósticos futuros.
    tabla_score = 0.0
    if tabla_actual:
        a = tabla_actual.get(loc, {})
        b = tabla_actual.get(vis, {})
        pj_a = max(a.get("pj", 0), 1)
        pj_b = max(b.get("pj", 0), 1)

        pts_a = a.get("pts", 0) / pj_a
        pts_b = b.get("pts", 0) / pj_b

        tabla_score = np.clip((pts_a - pts_b) / 3.0, -1.0, 1.0)

    # Mezcla:
    # tasa + Elo + forma + una pequeña señal de tabla.
    fuerza = (
        PESO_ELO * elo_score
        + PESO_FORMA * forma_score
        + PESO_TASA * tasa_score
        + 0.08 * tabla_score
    )

    # Tasas base de goles.
    ataque_local = np.clip(gf_l, 0.45, 2.70)
    defensa_local = np.clip(ga_l, 0.45, 2.70)

    ataque_visita = np.clip(gf_v, 0.45, 2.70)
    defensa_visita = np.clip(ga_v, 0.45, 2.70)

    # Fuerza global moderada.
    ataque_local *= np.exp(0.16 * fuerza)
    defensa_visita *= np.exp(-0.10 * fuerza)

    ataque_visita *= np.exp(-0.13 * fuerza)
    defensa_local *= np.exp(0.08 * fuerza)

    # Localía base.
    lam = np.sqrt(ataque_local * defensa_visita) * 1.10
    mu = np.sqrt(ataque_visita * defensa_local) * 0.93

    # Altitud.
    alt_l, ciudad = geo.get(loc, ALTITUDES_DEFAULT.get(loc, (0, "")))
    alt_v, _ = geo.get(vis, ALTITUDES_DEFAULT.get(vis, (0, "")))

    delta = alt_l - alt_v

    factor_alt_local = 1.0 + ALTITUD_COEF_LOCAL * max(delta, 0)
    factor_alt_visita = 1.0 - ALTITUD_COEF_VISITA * max(delta, 0)

    factor_alt_local = np.clip(
        factor_alt_local, 1.0, ALTITUD_MAX_FACTOR
    )
    factor_alt_visita = np.clip(
        factor_alt_visita, ALTITUD_MIN_VISITA, 1.0
    )

    lam *= factor_alt_local
    mu *= factor_alt_visita

    # Rivalidad: reduce ligeramente la distancia de fuerzas.
    rivalidad = nombre_rivalidad(local, visita)
    if rivalidad:
        # Acerca las tasas al promedio para evitar sobrevalorar diferencias.
        promedio = (lam + mu) / 2
        lam = 0.92 * lam + 0.08 * promedio
        mu = 0.92 * mu + 0.08 * promedio

    lam = float(np.clip(lam, 0.35, 3.40))
    mu = float(np.clip(mu, 0.25, 2.80))

    return {
        "lambda": lam,
        "mu": mu,
        "elo_local": elo_loc,
        "elo_visita": elo_vis,
        "forma_local": form_loc,
        "forma_visita": form_vis,
        "gf_local": gf_l,
        "ga_local": ga_l,
        "gf_visita": gf_v,
        "ga_visita": ga_v,
        "alt_local": alt_l,
        "alt_visita": alt_v,
        "ciudad": ciudad,
        "rivalidad": rivalidad,
        "fuerza": fuerza,
    }


def calcular_probabilidades(
    local,
    visita,
    fecha_partido,
    hist,
    elo_actual,
    geo,
    tabla_actual,
    df_h2h,
    peso_h2h=0.0,
):
    f = fuerza_partido(
        local, visita, fecha_partido, hist, elo_actual, geo, tabla_actual
    )

    m = matriz_dixon_coles(f["lambda"], f["mu"])

    # Convención correcta:
    # filas = goles local; columnas = goles visitante.
    p_local = float(np.tril(m, -1).sum())
    p_empate = float(np.trace(m))
    p_visita = float(np.triu(m, 1).sum())

    # H2H experimental: solamente si se activa manualmente.
    h2h_idx, h2h_df = h2h_indice(
        df_h2h, local, visita, fecha_partido
    )

    if peso_h2h > 0 and h2h_idx is not None and len(h2h_df) >= 3:
        # Mezcla conservadora y explícita.
        # El resto conserva la distribución del modelo.
        h2h_local = 0.34 + 0.32 * h2h_idx
        h2h_emp = 0.30
        h2h_vis = 1.0 - h2h_local - h2h_emp
        h2h_vis = max(h2h_vis, 0.05)

        total = h2h_local + h2h_emp + h2h_vis
        h2h_local /= total
        h2h_emp /= total
        h2h_vis /= total

        p_local = (1 - peso_h2h) * p_local + peso_h2h * h2h_local
        p_empate = (1 - peso_h2h) * p_empate + peso_h2h * h2h_emp
        p_visita = (1 - peso_h2h) * p_visita + peso_h2h * h2h_vis

    total = p_local + p_empate + p_visita
    p_local /= total
    p_empate /= total
    p_visita /= total

    p_under = float(
        sum(
            m[i, j]
            for i in range(m.shape[0])
            for j in range(m.shape[1])
            if i + j <= 2
        )
    )
    p_over = 1.0 - p_under

    p_btts_si = float(m[1:, 1:].sum())
    p_btts_no = 1.0 - p_btts_si

    # Marcador modal.
    ix = np.unravel_index(np.argmax(m), m.shape)
    marcador = f"{ix[0]} - {ix[1]}"

    return {
        **f,
        "p_local": float(p_local),
        "p_empate": float(p_empate),
        "p_visita": float(p_visita),
        "p_over25": float(p_over),
        "p_under25": float(p_under),
        "p_btts_si": float(p_btts_si),
        "p_btts_no": float(p_btts_no),
        "marcador": marcador,
        "h2h_indice": h2h_idx,
        "h2h": h2h_df,
    }


# =========================================================
# 16. RECOMENDACIONES
# =========================================================

def recomendacion_1x2(p1, px, p2, local, visita):
    probs = {
        local: p1,
        "Empate": px,
        visita: p2,
    }

    orden = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    primero, segundo = orden[0], orden[1]

    if primero[1] >= 0.50 and primero[1] - segundo[1] >= 0.10:
        return f"Gana {primero[0]}", "Alta"

    if primero[1] >= 0.40 and primero[1] - segundo[1] >= 0.07:
        return f"Gana {primero[0]}", "Media-Alta"

    if p1 + px >= 0.63 and p1 >= p2:
        return f"1X: {local} o Empate", "Media-Alta"

    if p2 + px >= 0.63 and p2 >= p1:
        return f"X2: Empate o {visita}", "Media-Alta"

    return "Doble opción / mercado prudente", "Media"


# =========================================================
# 17. EXPORTACIÓN
# =========================================================

def generar_excel(datos, partidos):
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        partidos.to_excel(writer, sheet_name="Partidos_Fecha", index=False)

        for clave, hoja in [
            ("apertura", "Resultados_Apertura"),
            ("clausura", "Resultados_Clausura"),
            ("acumulada", "Tabla_Acumulada"),
            ("tabla_clausura", "Tabla_Clausura"),
            ("geo", "Data_Geografica"),
            ("h2h", "Historial_H2H"),
        ]:
            datos[clave].to_excel(writer, sheet_name=hoja, index=False)

    buffer.seek(0)
    return buffer


# =========================================================
# 18. APLICAR RESULTADOS INGRESADOS EN PARTIDOS_FECHA
# =========================================================

def preparar_partidos(df):
    d = df.copy()

    c_jornada = buscar_columna(d, ["Jornada", "Fecha_Num"])
    c_fecha = buscar_columna(d, ["Fecha"])
    c_hora = buscar_columna(d, ["Hora"])
    c_local = buscar_columna(d, ["Local"])
    c_vis = buscar_columna(d, ["Visita", "Visitante"])
    c_gl = buscar_columna(d, ["Goles_Local", "GL"])
    c_gv = buscar_columna(d, ["Goles_Visita", "GV"])

    if not c_local or not c_vis:
        raise ValueError("Partidos_Fecha debe tener Local y Visita.")

    out = d.copy()

    out["_jornada"] = (
        out[c_jornada].astype(str) if c_jornada else ""
    )
    out["_fecha"] = (
        pd.to_datetime(out[c_fecha], errors="coerce")
        if c_fecha else pd.NaT
    )
    out["_hora"] = (
        out[c_hora].astype(str) if c_hora else "15:00"
    )
    out["_local"] = out[c_local].astype(str).str.strip()
    out["_visita"] = out[c_vis].astype(str).str.strip()

    if c_gl:
        out["_gl"] = pd.to_numeric(out[c_gl], errors="coerce")
    else:
        out["_gl"] = np.nan

    if c_gv:
        out["_gv"] = pd.to_numeric(out[c_gv], errors="coerce")
    else:
        out["_gv"] = np.nan

    return out


def actualizar_partidos_desde_ui(partidos, resultados_ui):
    d = partidos.copy()

    for idx, valores in resultados_ui.items():
        gl, gv = valores
        d.loc[idx, "Goles_Local"] = gl
        d.loc[idx, "Goles_Visita"] = gv

    return d


# =========================================================
# 19. INTERFAZ
# =========================================================

st.title("⚽ Modelo de Predicción Liga 1 Perú — V4")
st.caption(
    "Dixon-Coles + Elo + forma reciente + ataque/defensa + altitud + rivalidades"
)

st.sidebar.header("📁 Datos")

archivo_subido = st.sidebar.file_uploader(
    "Sube tu Excel actualizado",
    type=["xlsx"],
)

ruta_local = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ARCHIVO_DEFECTO,
)

fuente = archivo_subido if archivo_subido is not None else (
    ruta_local if os.path.exists(ruta_local) else None
)

if fuente is None:
    st.error(
        "No se encontró el Excel. Sube Liga1_2026.xlsx desde el menú lateral."
    )
    st.stop()

try:
    datos = cargar_excel(fuente)
    partidos_raw = preparar_partidos(datos["partidos"])
    historico = crear_historico(datos)

    # Incorpora Fecha 9 solamente si todavía no está en Resultados_Clausura.
    historico = agregar_fecha_9_si_falta(historico)

    geo = construir_geo(datos["geo"])
    tabla_acum = tabla_dict(datos["acumulada"])
    tabla_claus = tabla_dict(datos["tabla_clausura"])
    h2h = preparar_h2h(datos["h2h"])

except Exception as e:
    st.error(f"Error leyendo el Excel: {e}")
    st.stop()


# =========================================================
# 20. BARRA LATERAL
# =========================================================

st.sidebar.success(
    f"Histórico utilizado: {len(historico)} partidos"
)

jornadas = (
    partidos_raw["_jornada"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

def numero_jornada(x):
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else 999

jornadas = sorted(jornadas, key=numero_jornada)

      # Filtrado directo sin errores de espacios
jornadas_validas = list(jornadas)
if not jornadas_validas: jornadas_validas = ["Jornada 10", "Jornada 11"]
    preferidas = [j for j in jornadas_validas if numero_jornada(j) in]
    jornada_default = preferidas[0] if preferidas else (jornadas_validas[0] if jornadas_validas else "")
    jornada = st.sidebar.selectbox("Seleccionar jornada", jornadas_validas, index=jornadas_validas.index(jornada_default) if (jornada_default in jornadas_validas) else 0)

tabla_opcion = st.sidebar.radio(
    "Tabla de referencia",
    ["Acumulada", "Clausura"],
    index=0,
)

tabla_actual = tabla_acum if tabla_opcion == "Acumulada" else tabla_claus

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Ajustes experimentales")

usar_h2h = st.sidebar.checkbox(
    "Activar H2H experimental",
    value=False,
    help=(
        "Por el backtesting realizado, el H2H no se activa por defecto. "
        "Puedes probarlo, pero no sustituye la calibración."
    ),
)

peso_h2h = 0.0
if usar_h2h:
    peso_h2h = st.sidebar.slider(
        "Peso H2H",
        min_value=0.00,
        max_value=0.15,
        value=H2H_PESO_DEFECTO,
        step=0.01,
    )

# =========================================================
# 21. ELO ACTUAL
# =========================================================

elo_actual = obtener_elo_actual(historico)


# =========================================================
# 22. FILTRO DE JORNADA
# =========================================================

df_jornada = partidos_raw[
    partidos_raw["_jornada"].astype(str) == str(jornada)
].copy()

if df_jornada.empty:
    st.warning(f"No hay partidos para {jornada}.")
    st.stop()

st.subheader(
    f"📊 Pronósticos — {jornada} — Tabla {tabla_opcion}"
)

st.info(
    "Las probabilidades se calculan con la información histórica disponible "
    "antes del partido. Los resultados de Fecha 9 proporcionados fueron "
    "incorporados si aún no estaban en las hojas de resultados."
)

# =========================================================
# 23. PRONÓSTICOS
# =========================================================

pronosticos = []

for idx, row in df_jornada.iterrows():
    local = row["_local"]
    visita = row["_visita"]
    fecha = row["_fecha"]

    if pd.isna(fecha):
        fecha = None

    pred = calcular_probabilidades(
        local=local,
        visita=visita,
        fecha_partido=fecha,
        hist=historico,
        elo_actual=elo_actual,
        geo=geo,
        tabla_actual=tabla_actual,
        df_h2h=h2h,
        peso_h2h=peso_h2h,
    )

    rec, conf = recomendacion_1x2(
        pred["p_local"],
        pred["p_empate"],
        pred["p_visita"],
        local,
        visita,
    )

    pronosticos.append({
        "Jornada": jornada,
        "Fecha": (
            fecha.strftime("%Y-%m-%d")
            if fecha is not None else ""
        ),
        "Local": local,
        "Visita": visita,
        "P_Local": pred["p_local"],
        "P_Empate": pred["p_empate"],
        "P_Visita": pred["p_visita"],
        "Over_2.5": pred["p_over25"],
        "Under_2.5": pred["p_under25"],
        "BTTS_Si": pred["p_btts_si"],
        "BTTS_No": pred["p_btts_no"],
        "Marcador_Modal": pred["marcador"],
        "Recomendacion": rec,
        "Confianza": conf,
        "Lambda_Local": pred["lambda"],
        "Lambda_Visita": pred["mu"],
        "Elo_Local": pred["elo_local"],
        "Elo_Visita": pred["elo_visita"],
        "Altitud_Local": pred["alt_local"],
        "Altitud_Visita": pred["alt_visita"],
        "Rivalidad": pred["rivalidad"],
        "H2H_Indice": pred["h2h_indice"],
    })

    with st.container():
        st.markdown(
            f"### 🏟️ {local} vs {visita}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                f"Gana {local}",
                f"{pred['p_local']*100:.1f}%",
                f"Cuota justa {1/max(pred['p_local'], 0.001):.2f}",
            )

        with c2:
            st.metric(
                "Empate",
                f"{pred['p_empate']*100:.1f}%",
                f"Cuota justa {1/max(pred['p_empate'], 0.001):.2f}",
            )

        with c3:
            st.metric(
                f"Gana {visita}",
                f"{pred['p_visita']*100:.1f}%",
                f"Cuota justa {1/max(pred['p_visita'], 0.001):.2f}",
            )

        c4, c5, c6, c7 = st.columns(4)

        with c4:
            st.write(
                f"**Recomendación:** {rec} ({conf})"
            )

        with c5:
            mercado = (
                "Más de 2.5"
                if pred["p_over25"] >= pred["p_under25"]
                else "Menos de 2.5"
            )
            st.write(
                f"**Goles:** {mercado} — "
                f"{max(pred['p_over25'], pred['p_under25'])*100:.1f}%"
            )

        with c6:
            btts = (
                "Sí"
                if pred["p_btts_si"] >= pred["p_btts_no"]
                else "No"
            )
            st.write(
                f"**BTTS:** {btts} — "
                f"{max(pred['p_btts_si'], pred['p_btts_no'])*100:.1f}%"
            )

        with c7:
            st.write(
                f"**Marcador modal:** {pred['marcador']}"
            )

        ciudad = pred["ciudad"]
        st.caption(
            f"📍 {ciudad} | "
            f"Altitud local: {pred['alt_local']:.0f} msnm | "
            f"Altitud visita: {pred['alt_visita']:.0f} msnm"
        )

        if pred["rivalidad"]:
            st.caption(
                f"🔥 {pred['rivalidad']}"
            )

        if pred["h2h"] is not None and not pred["h2h"].empty:
            with st.expander(
                f"Historial H2H — {len(pred['h2h'])} enfrentamientos previos"
            ):
                h = pred["h2h"][
                    ["fecha", "local", "gl", "gv", "visita"]
                ].copy()
                h["fecha"] = h["fecha"].dt.strftime("%Y-%m-%d")
                st.dataframe(
                    h,
                    use_container_width=True,
                    hide_index=True,
                )
                if pred["h2h_indice"] is not None:
                    st.caption(
                        f"Índice H2H desde la perspectiva del local: "
                        f"{pred['h2h_indice']:.3f}. "
                        f"No altera el modelo salvo que actives H2H experimental."
                    )

        st.divider()


# =========================================================
# 24. TABLA RESUMEN
# =========================================================

df_pred = pd.DataFrame(pronosticos)

st.subheader("📋 Resumen de pronósticos")

df_resumen = df_pred[
    [
        "Fecha", "Local", "Visita",
        "P_Local", "P_Empate", "P_Visita",
        "Over_2.5", "BTTS_Si",
        "Marcador_Modal", "Recomendacion", "Confianza"
    ]
].copy()

for c in [
    "P_Local", "P_Empate", "P_Visita",
    "Over_2.5", "BTTS_Si"
]:
    df_resumen[c] = (df_resumen[c] * 100).round(1)

df_resumen = df_resumen.rename(columns={
    "P_Local": "% Local",
    "P_Empate": "% Empate",
    "P_Visita": "% Visita",
    "Over_2.5": "% +2.5",
    "BTTS_Si": "% BTTS Sí",
})

st.dataframe(
    df_resumen,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# 25. ACTUALIZAR RESULTADOS
# =========================================================

st.subheader("📝 Actualizar resultados")

st.write(
    "Cuando termine la jornada, puedes ingresar los marcadores aquí. "
    "Luego descarga el Excel actualizado."
)

resultados_ui = {}

for idx, row in df_jornada.iterrows():
    a, b, c, d = st.columns([3, 1, 1, 2])

    with a:
        st.write(
            f"**{row['_local']} – {row['_visita']}**"
        )

    with b:
        gl = st.number_input(
            "GL",
            min_value=0,
            max_value=15,
            value=0,
            step=1,
            key=f"gl_{idx}",
        )

    with c:
        gv = st.number_input(
            "GV",
            min_value=0,
            max_value=15,
            value=0,
            step=1,
            key=f"gv_{idx}",
        )

    with d:
        if st.checkbox(
            "Partido jugado",
            value=False,
            key=f"jugado_{idx}",
        ):
            resultados_ui[idx] = (gl, gv)


if st.button("💾 Preparar Excel actualizado"):
    partidos_actualizados = datos["partidos"].copy()

    # Asegura columnas de goles.
    if "Goles_Local" not in partidos_actualizados.columns:
        partidos_actualizados["Goles_Local"] = np.nan
    if "Goles_Visita" not in partidos_actualizados.columns:
        partidos_actualizados["Goles_Visita"] = np.nan

    # Los índices de df_jornada son índices de partidos_raw.
    for idx, (gl, gv) in resultados_ui.items():
        # Recuperamos el índice original de la fila.
        fila = df_jornada.loc[idx]
        local = fila["_local"]
        visita = fila["_visita"]
        fecha = fila["_fecha"]

        c_local = partidos_actualizados["Local"].astype(str).str.strip()
        c_vis = partidos_actualizados["Visita"].astype(str).str.strip()

        mask = (
            (c_local.apply(normalizar_nombre) == normalizar_nombre(local))
            & (c_vis.apply(normalizar_nombre) == normalizar_nombre(visita))
        )

        if "Fecha" in partidos_actualizados.columns and pd.notna(fecha):
            fechas_excel = pd.to_datetime(
                partidos_actualizados["Fecha"],
                errors="coerce"
            )
            mask &= (
                fechas_excel.dt.strftime("%Y-%m-%d")
                == pd.Timestamp(fecha).strftime("%Y-%m-%d")
            )

        if mask.any():
            partidos_actualizados.loc[
                mask, "Goles_Local"
            ] = gl
            partidos_actualizados.loc[
                mask, "Goles_Visita"
            ] = gv

    datos_export = datos.copy()
    datos_export["partidos"] = partidos_actualizados

    archivo_bytes = generar_excel(
        datos_export,
        partidos_actualizados,
    )

    nombre_salida = (
        f"Liga1_2026_actualizado_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    )

    st.download_button(
        "⬇️ Descargar Excel actualizado",
        data=archivo_bytes,
        file_name=nombre_salida,
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )


# =========================================================
# 26. DIAGNÓSTICO
# =========================================================

with st.expander("🔎 Diagnóstico del modelo"):
    st.write(
        f"**Partidos históricos utilizados:** {len(historico)}"
    )
    st.write(
        f"**Registros H2H disponibles:** {len(h2h)}"
    )
    st.write(
        f"**Peso H2H actual:** {peso_h2h:.2%}"
    )

    if jornada:
        st.write(
            f"**Jornada seleccionada:** {jornada}"
        )

    st.markdown("### Elo actual")
    elo_tabla = pd.DataFrame(
        [
            {
                "Equipo": k,
                "Elo": round(v, 1),
            }
            for k, v in sorted(
                elo_actual.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]
    )
    st.dataframe(
        elo_tabla,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Parámetros principales")
    st.json({
        "ELO inicial": ELO_INICIAL,
        "ELO K": ELO_K,
        "ventana forma": VENTANA_FORMA,
        "ventana tasas": VENTANA_RATES,
        "peso Elo": PESO_ELO,
        "peso forma": PESO_FORMA,
        "peso tasas": PESO_TASA,
        "rho Dixon-Coles": RHO_DIXON_COLES,
        "peso H2H": peso_h2h,
        "usar H2H": usar_h2h,
    })

    st.warning(
        "El H2H se mantiene desactivado por defecto porque el backtesting "
        "realizado sobre el histórico disponible no mostró mejora con un "
        "peso fijo. Puede probarse experimentalmente desde la barra lateral."
    )
# =========================================================
# SECCIÓN PREMIUM: DIAGNÓSTICO AVANZADO CON IA GEMINI
# =========================================================

st.markdown("---")
st.markdown("### 🏆 Panel de Análisis Avanzado e Inteligencia Artificial")

# 1. Buscamos si hay un partido seleccionado en la interfaz
# Si tu selector de la barra lateral guarda el partido en una variable, la usamos.
# Si no, tomamos de forma automática el primer partido de la jornada elegida para el diagnóstico.
if 'partidos_filtrados' in locals() and not partidos_filtrados.empty:
    # Usamos un selector dinámico al final para que el usuario elija qué partido auditar con IA
    partido_ia = st.selectbox(
        "🔍 Selecciona un partido de la jornada para generar el informe táctico de la IA:",
        partidos_filtrados.apply(lambda r: f"{r['Local']} vs {r['Visita']}", axis=1)
    )
    
    # Extraemos la fila exacta del partido elegido
    fila_partido = partidos_filtrados[partidos_filtrados.apply(lambda r: f"{r['Local']} vs {r['Visita']}", axis=1) == partido_ia].iloc[0]
    
    # Intentamos recuperar las variables o asignamos valores por defecto si no existen en la fila
    loc = fila_partido['Local']
    vis = fila_partido['Visita']
    alt_local = fila_partido.get('Altitud_Local', ALTITUDES_DEFAULT.get(normalizar_nombre(loc), (150,))[0])
    delta_altitud = fila_partido.get('Delta_Altitud_msnm', 0)
    d_local = fila_partido.get('Dias_Descanso_Local', 7)
    d_visita = fila_partido.get('Dias_Descanso_Visita', 7)
    
    # Recuperamos las probabilidades que calculó tu distribución de Poisson previamente en el script
    # Si las variables globales no están accesibles directamente, se asume una estimación referencial
    p_L = locals().get('prob_L', 0.45) * 100
    p_E = locals().get('prob_E', 0.30) * 100
    p_V = locals().get('prob_V', 0.25) * 100
    
    # --- RENDERIZADO DE LA TARJETA PREMIUM ---
    with st.container(border=True):
        st.markdown(f"#### 🏟️ Análisis de Campo: {loc} vs {vis}")
        st.caption(
            f"🏔️ Altitud de la Sede: {alt_local} msnm | "
            f"📉 Impacto Geográfico (Delta): {delta_altitud}m | "
            f"🏃 Descanso: {d_local}d (Local) vs {d_visita}d (Visita)"
        )
        
        # Estructura de pestañas para modernizar la visualización
        tab_metricas, tab_computo_ia = st.tabs(["📊 Probabilidades Críticas", "🤖 Reporte Técnico Gemini Flash"])
        
        with tab_metricas:
            st.write("")
            col_l, col_e, col_v = st.columns(3)
            with col_l:
                st.metric(label=f"⚽ Probabilidad Gana {loc}", value=f"{p_L:.1f}%")
            with col_e:
                st.metric(label="🤝 Probabilidad Empate", value=f"{p_E:.1f}%")
            with col_v:
                st.metric(label=f"🏃 Probabilidad Gana {vis}", value=f"{p_V:.1f}%")
            
            st.markdown("---")
            # Recomendación destacada con caja verde de éxito
            rec_texto = fila_partido.get('Recomendacion', 'Revisar mercados secundarios (Goles/Tarjetas)')
            st.success(f"🎯 **Sugerencia Analítica:** {rec_texto}")
            
        with tab_computo_ia:
            st.write("")
            st.markdown("##### 🕵️‍♂️ Veredicto de Rendimiento y Desgaste Físico")
            
            # Botón disparador para controlar el consumo de tokens y evitar recargas infinitas
            if st.button("🤖 Generar Diagnóstico Contextual de IA"):
                with st.spinner("Gemini Flash está procesando las variables geográficas y de descanso..."):
                    try:
                        reporte_final = obtener_analisis_ia(
                            loc, vis, alt_local, d_local, d_visita,
                            round(p_L, 1), round(p_E, 1), round(p_V, 1)
                        )
                        st.info(reporte_final)
                    except Exception as error_ia:
                        st.error("No se pudo conectar con el motor de IA. Verifica tus Secrets en Streamlit.")
            else:
                st.caption("Presiona el botón superior para activar los agentes y generar el texto analítico.")
else:
    st.info("Selecciona una jornada disponible en el panel para habilitar los diagnósticos avanzados.")
