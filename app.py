# -*- coding: utf-8 -*-
"""
Modelo de Predicción Liga 1 Perú - Quipus Data
Requisitos:
    pip install streamlit pandas numpy scipy openpyxl
Ejecución:
    streamlit run app_liga1.py
Coloque Liga1_2026.xlsx en la misma carpeta o cargue el Excel desde la aplicación.
"""

import base64
import os
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

st.set_page_config(
    page_title="Modelo de Predicción Liga 1 Perú - Quipus Data",
    page_icon="⚽",
    layout="wide",
)

st.markdown(
    """
    <style>
    table, th, td {
        border: 1px solid #9ca592 !important;
        border-collapse: collapse !important;
        text-align: center !important;
        white-space: normal !important;
        word-break: break-word !important;
    }
    th { background-color: #7b8870 !important; color: white !important; font-weight: bold; }
    div[data-testid="stDataFrame"] { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

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

st.title("⚽ Modelo de Predicción Liga 1 Perú")
st.caption(
    "Modelo estadístico orientativo. Las probabilidades son estimaciones, no garantías de resultado."
)

# ---------------------------------------------------------------------------
# 1. Normalización y utilidades
# ---------------------------------------------------------------------------

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

def texto_limpio(valor):
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
        "equipo_local": "local",
        "equipo_visitante": "visita",
        "visitante": "visita",
        "goles_local": "gl",
        "goles_visitante": "gv",
        "goles_visita": "gv",
        "club": "equipo",
        "situacion_actual": "situacion_actual",
        "pts": "pts",
    }
    df = df.rename(columns=renombres)

    for col in ("local", "visita", "equipo"):
        if col in df.columns:
            df[col + "_std"] = df[col].map(normalizar_nombre)

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)

    for col in df.columns:
        if col.startswith("xg_") or col.startswith("prom_"):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            )
    for col in ("gl", "gv", "pj", "pg", "pe", "pp", "gf", "gc", "dg", "pts",
                "altitud_local", "altitud_visita", "delta_altitud_msnm",
                "dias_descanso_local", "dias_descanso_visita", "altitud_msnm",
                "pos", "rango"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def columna(df, nombre, default=np.nan):
    if nombre in df.columns:
        return df[nombre]
    return pd.Series(default, index=df.index)

def a_float(valor, default=np.nan):
    try:
        if pd.isna(valor):
            return default
        return float(str(valor).replace(",", "."))
    except (ValueError, TypeError):
        return default

def parsear_hora(valor):
    if pd.isna(valor):
        return "--:--"
    if hasattr(valor, "strftime"):
        try:
            return valor.strftime("%H:%M")
        except Exception:
            pass
    s = str(valor)
    return s[:5] if len(s) >= 5 else s

# ---------------------------------------------------------------------------
# 2. Carga del libro Excel completo
# ---------------------------------------------------------------------------

HOJAS_ESPERADAS = [
    "Partidos_Fecha", "Resultados_Apertura", "Resultados_Clausura",
    "Tabla_Acumulada", "Tabla_Clausura", "Historial_H2H",
    "Data_Geografica", "Formulas"
]

@st.cache_data(show_spinner=False)
def cargar_excel_completo(bytes_excel):
    from io import BytesIO
    xls = pd.ExcelFile(BytesIO(bytes_excel))
    disponibles = xls.sheet_names
    originales = {}
    for hoja in HOJAS_ESPERADAS:
        if hoja in disponibles:
            originales[hoja] = pd.read_excel(xls, sheet_name=hoja)
        else:
            originales[hoja] = pd.DataFrame()

    datos = {k: normalizar_columnas(v) for k, v in originales.items() if k != "Formulas"}
    datos["formulas_original"] = originales["Formulas"]

    for key in ("Resultados_Apertura", "Resultados_Clausura"):
        df = datos[key].copy()
        for c in ("xg_local", "xg_visita", "prom_gf_local", "prom_gc_local",
                  "prom_gf_visita", "prom_gc_visita"):
            if c in df.columns:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce"
                )
        datos[key] = df

    formulas = originales["Formulas"]
    prior_local, prior_visita = 1.45, 1.20
    if not formulas.empty and formulas.shape[1] >= 4:
        etiqueta_col = formulas.iloc[:, 1].astype(str).str.lower()
        valor_col = formulas.iloc[:, 3]
        for i, etiqueta in etiqueta_col.items():
            valor = a_float(valor_col.loc[i])
            if not np.isfinite(valor):
                continue
            if "mgoles local" in etiqueta or "goles local" in etiqueta:
                prior_local = valor
            elif "mgoles visita" in etiqueta or "goles visita" in etiqueta:
                prior_visita = valor

        if prior_local == 1.45 and len(formulas) > 0:
            v = a_float(formulas.iloc[0, 3])
            if np.isfinite(v):
                prior_local = v
        if prior_visita == 1.20 and len(formulas) > 4:
            v = a_float(formulas.iloc[4, 3])
            if np.isfinite(v):
                prior_visita = v

    datos["prior_local"] = max(0.3, prior_local)
    datos["prior_visita"] = max(0.3, prior_visita)
    datos["hojas_disponibles"] = disponibles
    return datos

st.sidebar.header("📂 Archivo de datos")
archivo_subido = st.sidebar.file_uploader("Cargar Excel actualizado", type=["xlsx"])
archivo_predeterminado = next(
    (p for p in ["Liga1_2026.xlsx", "Liga1_2026(2).xlsx"] if os.path.exists(p)),
    None,
)

if archivo_subido is not None:
    bytes_excel = archivo_subido.getvalue()
    nombre_archivo = archivo_subido.name
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
faltantes = [h for h in HOJAS_ESPERADAS if h not in datos["hojas_disponibles"]]
if faltantes:
    st.warning("Faltan hojas opcionales o esperadas: " + ", ".join(faltantes))

partidos = datos["Partidos_Fecha"].copy()
if partidos.empty or not {"local", "visita"}.issubset(partidos.columns):
    st.error("La hoja Partidos_Fecha debe contener las columnas Local y Visita.")
    st.stop()

# ---------------------------------------------------------------------------
# 3. Construcción de base histórica común y rendimiento
# ---------------------------------------------------------------------------

def construir_historico(datos):
    partes = []
    for hoja, peso in [("Resultados_Apertura", 0.40), ("Resultados_Clausura", 0.60)]:
        df = datos[hoja].copy()
        requeridas = {"local", "visita", "gl", "gv"}
        if df.empty or not requeridas.issubset(df.columns):
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
        df["xg_local"] = pd.to_numeric(df.get("xg_local", df["gl"]), errors="coerce").fillna(df["gl"])
        df["xg_visita"] = pd.to_numeric(df.get("xg_visita", df["gv"]), errors="coerce").fillna(df["gv"])
        partes.append(df)
    if not partes:
        return pd.DataFrame()
    return pd.concat(partes, ignore_index=True, sort=False)

historico = construir_historico(datos)

def preparar_tabla(df):
    df = df.copy()
    if df.empty or "equipo" not in df.columns:
        return df
    for c in ("pj", "pg", "pe", "pp", "gf", "gc", "dg", "pts"):
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

def metricas_equipo(datos, equipo, fecha_partido=None, jornada_partido=None):
    eq = normalizar_nombre(equipo)
    h = historico.copy()
    if h.empty:
        return {
            "gf": datos["prior_local"], "ga": datos["prior_visita"],
            "gf_local": datos["prior_local"], "ga_local": datos["prior_visita"],
            "gf_visita": datos["prior_visita"], "ga_visita": datos["prior_local"],
            "n": 0, "forma": 0.5
        }

    if fecha_partido is not None and pd.notna(fecha_partido) and "fecha" in h.columns:
        fechas = pd.to_datetime(h["fecha"], errors="coerce")
        if fechas.notna().any():
            h = h[fechas.isna() | (fechas < pd.Timestamp(fecha_partido))]

    if jornada_partido is not None and pd.notna(jornada_partido):
        try:
            jn = int(jornada_partido)
            if "jornada_num" in h.columns:
                h = h[(h["fuente"] != "Resultados_Clausura") | h["jornada_num"].lt(jn)]
        except (TypeError, ValueError):
            pass

    loc = h[h["local_std"] == eq].copy()
    vis = h[h["visita_std"] == eq].copy()
    if loc.empty and vis.empty:
        gf = datos["prior_local"]
        ga = datos["prior_visita"]
        return {"gf": gf, "ga": ga, "gf_local": gf, "ga_local": ga,
                "gf_visita": ga, "ga_visita": gf, "n": 0, "forma": 0.5}

    gf_l = (0.60 * loc["gl"] + 0.40 * loc["xg_local"]).tolist()
    ga_l = (0.60 * loc["gv"] + 0.40 * loc["xg_visita"]).tolist()
    gf_v = (0.60 * vis["gv"] + 0.40 * vis["xg_visita"]).tolist()
    ga_v = (0.60 * vis["gl"] + 0.40 * vis["xg_local"]).tolist()

    def promedio_ponderado(valores, filas):
        if not valores:
            return np.nan
        pesos = []
        for _, r in filas.iterrows():
            base = 0.40 if r.get("fuente") == "Resultados_Apertura" else 0.60
            pesos.append(base)
        pesos = np.asarray(pesos, dtype=float)
        if len(pesos) >= 1:
            pesos[-min(5, len(pesos)):] *= 1.25
        return float(np.average(np.asarray(valores, dtype=float), weights=pesos))

    gf_local = promedio_ponderado(gf_l, loc)
    ga_local = promedio_ponderado(ga_l, loc)
    gf_visita = promedio_ponderado(gf_v, vis)
    ga_visita = promedio_ponderado(ga_v, vis)

    fila_ac = fila_equipo_tabla(tabla_acumulada, equipo)
    fila_cl = fila_equipo_tabla(tabla_clausura, equipo)
    ajustes_gf, ajustes_ga = [], []
    for fila, peso in [(fila_ac, 0.25), (fila_cl, 0.35)]:
        if fila is not None:
            pj = a_float(fila.get("pj"))
            if np.isfinite(pj) and pj > 0:
                f_gf = a_float(fila.get("gf")) / pj
                f_ga = a_float(fila.get("gc")) / pj
                if np.isfinite(f_gf) and np.isfinite(f_ga):
                    ajustes_gf.append((f_gf, peso))
                    ajustes_ga.append((f_ga, peso))

    gf_hist = np.nanmean([np.nanmean(gf_l) if gf_l else np.nan,
                          np.nanmean(gf_v) if gf_v else np.nan])
    ga_hist = np.nanmean([np.nanmean(ga_l) if ga_l else np.nan,
                          np.nanmean(ga_v) if ga_v else np.nan])
    if not np.isfinite(gf_hist):
        gf_hist = datos["prior_local"]
    if not np.isfinite(ga_hist):
        ga_hist = datos["prior_visita"]

    if ajustes_gf:
        tabla_gf = np.average([x[0] for x in ajustes_gf], weights=[x[1] for x in ajustes_gf])
        gf_hist = 0.75 * gf_hist + 0.25 * tabla_gf
    if ajustes_ga:
        tabla_ga = np.average([x[0] for x in ajustes_ga], weights=[x[1] for x in ajustes_ga])
        ga_hist = 0.75 * ga_hist + 0.25 * tabla_ga

    recientes = pd.concat([loc.assign(gf=loc["gl"], ga=loc["gv"]),
                           vis.assign(gf=vis["gv"], ga=vis["gl"])], ignore_index=True)
    recientes = recientes.tail(5)
    forma = 0.5
    if not recientes.empty:
        puntos = np.where(recientes["gf"] > recientes["ga"], 3,
                          np.where(recientes["gf"] == recientes["ga"], 1, 0))
        forma = float(np.mean(puntos) / 3.0)

    n = len(loc) + len(vis)
    return {
        "gf": max(0.2, float(gf_hist)),
        "ga": max(0.2, float(ga_hist)),
        "gf_local": max(0.2, float(gf_local) if np.isfinite(gf_local) else gf_hist),
        "ga_local": max(0.2, float(ga_local) if np.isfinite(ga_local) else ga_hist),
        "gf_visita": max(0.2, float(gf_visita) if np.isfinite(gf_visita) else gf_hist),
        "ga_visita": max(0.2, float(ga_visita) if np.isfinite(ga_visita) else ga_hist),
        "n": n, "forma": forma
    }

# ---------------------------------------------------------------------------
# 4. Factores contextuales
# ---------------------------------------------------------------------------

def obtener_contexto(datos, partido):
    loc, vis = texto_limpio(partido.get("local")), texto_limpio(partido.get("visita"))
    eq_loc, eq_vis = normalizar_nombre(loc), normalizar_nombre(vis)
    f_loc, f_vis, rho = 1.0, 1.0, -0.08

    geo = datos["Data_Geografica"]
    info_l = geo[geo.get("equipo_std", pd.Series(dtype=str)) == eq_loc] if not geo.empty and "equipo_std" in geo.columns else pd.DataFrame()
    info_v = geo[geo.get("equipo_std", pd.Series(dtype=str)) == eq_vis] if not geo.empty and "equipo_std" in geo.columns else pd.DataFrame()
    alt_l = a_float(info_l.iloc[0].get("altitud_msnm")) if not info_l.empty else a_float(partido.get("altitud_local"), 0)
    alt_v = a_float(info_v.iloc[0].get("altitud_msnm")) if not info_v.empty else a_float(partido.get("altitud_visita"), 0)
    delta_archivo = a_float(partido.get("delta_altitud_msnm"))
    delta = delta_archivo if np.isfinite(delta_archivo) and delta_archivo != 0 else alt_l - alt_v

    if delta > 2000:
        f_loc *= 1.16
        f_vis *= 0.78
    elif delta > 1200:
        f_loc *= 1.09
        f_vis *= 0.87
    elif delta < -2000:
        f_loc *= 0.92
        f_vis *= 1.06
    elif delta < -1200:
        f_loc *= 0.95
        f_vis *= 1.03

    clima_l = str(info_l.iloc[0].get("tipo_clima", "")).lower() if not info_l.empty else ""
    clima_v = str(info_v.iloc[0].get("tipo_clima", "")).lower() if not info_v.empty else ""
    cancha_l = str(info_l.iloc[0].get("tipo_cancha", "")).lower() if not info_l.empty else ""
    if "calor" in clima_l and "calor" not in clima_v:
        f_loc *= 1.03
        f_vis *= 0.96
    if "sintetic" in cancha_l or "sintetic" in cancha_l:
        f_loc *= 1.02
        f_vis *= 0.98

    descanso_l = a_float(partido.get("dias_descanso_local"), 7)
    descanso_v = a_float(partido.get("dias_descanso_visita"), 7)
    if descanso_l < 4:
        f_loc *= 0.94
    if descanso_v < 4:
        f_vis *= 0.94

    hora = parsear_hora(partido.get("hora"))
    try:
        hora_num = int(hora.split(":")[0])
        if hora_num in (11, 12, 13) and delta > 1000:
            f_loc *= 1.025
            f_vis *= 0.98
    except Exception:
        pass

    ac = fila_equipo_tabla(tabla_acumulada, loc)
    av = fila_equipo_tabla(tabla_acumulada, vis)
    if ac is not None:
        sit = str(ac.get("situacion_actual", "")).lower()
        if any(k in sit for k in ("libertadores", "sudamericana", "titulo", "titulo")):
            f_loc *= 1.035
        elif any(k in sit for k in ("descenso", "peligro", "riesgo")):
            f_loc *= 1.04
    if av is not None:
        sit = str(av.get("situacion_actual", "")).lower()
        if any(k in sit for k in ("libertadores", "sudamericana", "titulo", "titulo")):
            f_vis *= 1.025
        elif any(k in sit for k in ("descenso", "peligro", "riesgo")):
            f_vis *= 1.025

    h2h = datos["Historial_H2H"]
    if not h2h.empty and {"local_std", "visita_std"}.issubset(h2h.columns):
        h = h2h[
            ((h2h["local_std"] == eq_loc) & (h2h["visita_std"] == eq_vis))
            | ((h2h["local_std"] == eq_vis) & (h2h["visita_std"] == eq_loc))
        ].copy()
        if not h.empty and {"gl", "gv"}.issubset(h.columns):
            h["gl"] = pd.to_numeric(h["gl"], errors="coerce")
            h["gv"] = pd.to_numeric(h["gv"], errors="coerce")
            h = h.dropna(subset=["gl", "gv"]).tail(8)
            if not h.empty:
                balance_local = []
                for _, r in h.iterrows():
                    if r["local_std"] == eq_loc:
                        balance_local.append(1 if r["gl"] > r["gv"] else (-1 if r["gl"] < r["gv"] else 0))
                    else:
                        balance_local.append(1 if r["gv"] > r["gl"] else (-1 if r["gv"] < r["gl"] else 0))
                balance = float(np.mean(balance_local))
                f_loc *= float(np.clip(1 + 0.035 * balance, 0.965, 1.035))
                f_vis *= float(np.clip(1 - 0.025 * balance, 0.975, 1.025))
                grandes = {"alianza lima", "universitario", "sporting cristal"}
                if eq_loc in grandes and eq_vis in grandes:
                    rho = -0.12

    return float(np.clip(f_loc, 0.70, 1.35)), float(np.clip(f_vis, 0.65, 1.30)), rho, {
        "Altitud local": alt_l, "Altitud visita": alt_v, "Delta usado": delta,
        "Descanso local": descanso_l, "Descanso visita": descanso_v
    }

# ---------------------------------------------------------------------------
# 5. Poisson / Dixon-Coles y pronósticos
# ---------------------------------------------------------------------------

def matriz_dixon_coles(lambda_loc, lambda_vis, rho=-0.08, max_goles=8):
    matriz = np.zeros((max_goles, max_goles), dtype=float)
    for i in range(max_goles):
        for j in range(max_goles):
            p = poisson.pmf(i, lambda_loc) * poisson.pmf(j, lambda_vis)
            tau = 1.0
            if i == 0 and j == 0:
                tau = 1 - lambda_loc * lambda_vis * rho
            elif i == 1 and j == 0:
                tau = 1 + lambda_loc * rho
            elif i == 0 and j == 1:
                tau = 1 + lambda_vis * rho
            elif i == 1 and j == 1:
                tau = 1 - rho
            matriz[i, j] = max(0.0, p * tau)
    total = matriz.sum()
    return matriz / total if total > 0 else matriz

def pronosticar_partido(datos, partido):
    loc, vis = texto_limpio(partido.get("local")), texto_limpio(partido.get("visita"))
    fecha = partido.get("fecha")
    jornada_num = partido.get("jornada_num")
    mloc = metricas_equipo(datos, loc, fecha, jornada_num)
    mvis = metricas_equipo(datos, vis, fecha, jornada_num)
    f_loc, f_vis, rho, contexto = obtener_contexto(datos, partido)

    lambda_loc = ((mloc["gf_local"] + mvis["ga_visita"]) / 2.0) * f_loc
    lambda_vis = ((mvis["gf_visita"] + mloc["ga_local"]) / 2.0) * f_vis

    prior_l, prior_v = datos["prior_local"], datos["prior_visita"]
    lambda_loc = 0.88 * lambda_loc + 0.12 * prior_l
    lambda_vis = 0.88 * lambda_vis + 0.12 * prior_v
    lambda_loc = float(np.clip(lambda_loc, 0.20, 4.0))
    lambda_vis = float(np.clip(lambda_vis, 0.20, 4.0))

    matriz = matriz_dixon_coles(lambda_loc, lambda_vis, rho=rho)
    p_local = float(np.tril(matriz, -1).sum() * 100)
    p_empate = float(np.diag(matriz).sum() * 100)
    p_visita = float(np.triu(matriz, 1).sum() * 100)
    p_over25 = float(sum(matriz[i, j] for i in range(matriz.shape[0])
                         for j in range(matriz.shape[1]) if i + j >= 3) * 100)
    p_under25 = 100 - p_over25
    p_btts = float(matriz[1:, 1:].sum() * 100)

    i_modal, j_modal = np.unravel_index(
        np.argmax(matriz), matriz.shape
    )

    probabilidades_1x2 = {"Local": p_local, "Empate": p_empate, "Visita": p_visita}

    # Marcador más probable condicionado al pronóstico 1X2 (CORREGIDO)
    if p_local >= p_empate and p_local >= p_visita:
        mascara = np.tril(matriz, -1)
    elif p_visita >= p_empate and p_visita >= p_local:
        mascara = np.triu(matriz, 1)
    else:
        mascara = np.diag(np.diag(matriz))

    i_pron, j_pron = np.unravel_index(
        np.argmax(mascara), mascara.shape
    )
    ganador = max(probabilidades_1x2, key=probabilidades_1x2.get)
    prob_ganador = probabilidades_1x2[ganador]
    orden_probs = sorted(probabilidades_1x2.values(), reverse=True)
    margen = orden_probs[0] - orden_probs[1]

    if prob_ganador >= 60 and margen >= 12:
        estado_pronostico = "FIJA"
        senal = "🟢"
    elif margen <= 7 or prob_ganador < 40:
        estado_pronostico = "CERRADO"
        senal = "🔴"
    else:
        estado_pronostico = "INTERMEDIO"
        senal = "🟡"

    if estado_pronostico == "FIJA":
        recomendacion = f"{ganador} (confianza alta)"
    elif estado_pronostico == "CERRADO":
        recomendacion = f"Partido cerrado — {ganador} es solo una ligera preferencia"
    else:
        recomendacion = f"{ganador} (confianza media)"
    if p_over25 >= 55:
        recomendacion += " | Más de 2.5 goles"
    elif p_under25 >= 58:
        recomendacion += " | Menos de 2.5 goles"
    if p_btts >= 55:
        recomendacion += " | Ambos marcan"

    return {
        "Día": texto_limpio(partido.get("dia")) or (
            pd.Timestamp(fecha).day_name() if pd.notna(fecha) else "Por definir"
        ),
        "Hora": parsear_hora(partido.get("hora")),
        "Local": loc,
        "Visita": vis,
        "λ Local": round(lambda_loc, 2),
        "λ Visita": round(lambda_vis, 2),
        "% Local": round(p_local, 1),
        "% Empate": round(p_empate, 1),
        "% Visita": round(p_visita, 1),
        "Más de 2.5 goles": round(p_over25, 1),
        "Menos de 2.5 goles": round(p_under25, 1),
        "Ambos marcan: Sí": round(p_btts, 1),
        "Marcador_Modal": f"{i_modal} - {j_modal}",
        "Marcador_Pronosticado": f"{i_pron} - {j_pron}",
        "Pronóstico 1X2": ganador,
        "Estado": estado_pronostico,
        "Señal": senal,
        "Recomendacion": recomendacion,
        "Prob. principal": round(max(p_local, p_empate, p_visita), 1),
        "Margen 1X2": round(margen, 1),
        "Contexto": contexto,
    }

# ---------------------------------------------------------------------------
# 6. Interfaz de usuario
# ---------------------------------------------------------------------------

st.sidebar.header("⚽ Selección de Jornada")
if "jornada" in partidos.columns:
    jornadas = partidos["jornada"].dropna().astype(str).unique().tolist()
    jornadas = sorted(jornadas, key=lambda x: int("".join(c for c in x if c.isdigit()) or 0))
else:
    jornadas = []
if jornadas:
    jornada_sel = st.sidebar.selectbox("Seleccionar Jornada", jornadas)
    partidos_jornada = partidos[partidos["jornada"].astype(str) == jornada_sel].copy()
else:
    jornada_sel = "Todas"
    partidos_jornada = partidos.copy()

st.sidebar.header("📊 Opciones")
tipo_vista = st.sidebar.radio("Vista de pronósticos", ["Vista Resumida", "Vista Completa"])
max_goles_mostrar = st.sidebar.checkbox("Mostrar métricas de contexto", value=False)

if "fecha" in partidos_jornada.columns:
    partidos_jornada["fecha"] = pd.to_datetime(partidos_jornada["fecha"], errors="coerce")
partidos_jornada["jornada_num"] = pd.to_numeric(
    partidos_jornada.get("jornada", pd.Series(index=partidos_jornada.index, dtype=str))
    .astype(str).str.extract(r"(\d+)")[0], errors="coerce"
)

resultados = []
errores = []
for _, partido in partidos_jornada.iterrows():
    try:
        resultados.append(pronosticar_partido(datos, partido))
    except Exception as exc:
        errores.append(f"{partido.get('local', '?')} vs {partido.get('visita', '?')}: {exc}")

df_pronosticos = pd.DataFrame(resultados)
if errores:
    with st.expander("Advertencias de cálculo"):
        for e in errores:
            st.warning(e)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Pronósticos", "🏆 Tabla acumulada", "🥇 Tabla Clausura",
    "📚 Historial H2H", "🧾 Control de datos"
])

with tab1:
    st.subheader(f"Pronósticos — {jornada_sel}")
    if df_pronosticos.empty:
        st.info("No hay partidos disponibles para esta selección.")
    else:
        def colorear_fila(fila):
            estado = fila.get("Estado", "")
            if estado == "FIJA":
                color = "background-color: #d9ead3; color: #14532d;"
            elif estado == "CERRADO":
                color = "background-color: #f4cccc; color: #7f1d1d;"
            else:
                color = "background-color: #fff2cc; color: #783f04;"
            return [color] * len(fila)

        if tipo_vista == "Vista Resumida":
            cols = ["Señal", "Estado", "Día", "Hora", "Local", "Visita",
                    "Marcador_Modal", "Pronóstico 1X2", "Recomendacion"]
            vista = df_pronosticos[cols]
            st.dataframe(
                vista.style.apply(colorear_fila, axis=1),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "🟢 FIJA: probabilidad principal ≥ 60% y ventaja ≥ 12 puntos. "
                "🔴 CERRADO: diferencia entre las dos opciones más probables ≤ 7 puntos "
                "o ninguna opción supera el 40%. 🟡 INTERMEDIO: los demás casos. "
                "El marcador modal es el marcador exacto más probable; el pronóstico 1X2 "
                "es el resultado con mayor probabilidad acumulada."
            )
        else:
            vista_completa = df_pronosticos.drop(columns=["Contexto"], errors="ignore")
            st.dataframe(
                vista_completa.style.apply(colorear_fila, axis=1),
                use_container_width=True,
                hide_index=True,
            )
        csv = df_pronosticos.drop(columns=["Contexto"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Descargar pronósticos CSV", csv,
                           f"pronosticos_{str(jornada_sel).replace(' ', '_')}.csv",
                           "text/csv")
        if max_goles_mostrar:
            for _, r in df_pronosticos.iterrows():
                with st.expander(f"{r['Local']} vs {r['Visita']} — contexto"):
                    st.write(r.get("Contexto", {}))

with tab2:
    st.subheader("🏆 Tabla acumulada")
    if not tabla_acumulada.empty:
        st.dataframe(tabla_acumulada.drop(columns=["equipo_std"], errors="ignore"),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No se encontró información en Tabla_Acumulada.")

with tab3:
    st.subheader("🥇 Tabla de posiciones del Clausura")
    if not tabla_clausura.empty:
        st.dataframe(tabla_clausura.drop(columns=["equipo_std"], errors="ignore"),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No se encontró información en Tabla_Clausura.")

with tab4:
    st.subheader("📚 Historial de enfrentamientos directos")
    h2h = datos["Historial_H2H"]
    if not h2h.empty:
        st.dataframe(h2h.drop(columns=["local_std", "visita_std"], errors="ignore"),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No se encontró información en Historial_H2H.")

with tab5:
    st.subheader("🧾 Control de hojas y registros")
    resumen = []
    for hoja in HOJAS_ESPERADAS:
        df = datos.get(hoja, pd.DataFrame())
        resumen.append({"Hoja": hoja, "Filas": len(df), "Columnas": len(df.columns),
                        "Estado": "Cargada" if hoja in datos["hojas_disponibles"] else "No encontrada"})
    st.dataframe(pd.DataFrame(resumen), use_container_width=True, hide_index=True)
    st.write("Columnas detectadas por hoja:")
    for hoja in HOJAS_ESPERADAS:
        df = datos.get(hoja, pd.DataFrame())
        with st.expander(hoja):
            st.write(list(df.columns))
