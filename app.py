# -*- coding: utf-8 -*-
"""
Modelo de Predicción Liga 1 Perú - Quipus Data
Versión revisada:
- Lee las ocho hojas del archivo Liga1_2026.xlsx.
- Usa Poisson con corrección Dixon-Coles.
- Filtra los partidos históricos anteriores a cada encuentro usando la fecha real.
- Recalcula K/L/M/N cronológicamente y no depende de referencias Prom_* desplazadas del Excel.
- Incorpora forma, localía, xG, tablas, altitud, descanso y contexto competitivo.
- Aplica realmente los factores contextuales después de la lambda base, evitando ajustes calculados pero no utilizados.
- Muestra resaltado en las letras, no en el fondo de las filas.
- Añade una lectura orientativa de presión competitiva: título/puestos internacionales,
  zona media y lucha por evitar el descenso.
- La presión competitiva se muestra como contexto y solo aplica ajustes muy pequeños
  y configurables a los goles esperados; no es una variable causal validada.
Instalación:
    pip install streamlit pandas numpy scipy openpyxl
Ejecución:
    streamlit run app_liga1.py
"""

import base64
import os
import unicodedata
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

st.set_page_config(
    page_title="Modelo de Predicción Liga 1 Perú - Quipus Data",
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

st.title("⚽ Modelo de Predicción Liga 1 Perú")
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
faltantes = [h for h in HOJAS_ESPERADAS if h not in datos["hojas_disponibles"]]
if faltantes:
    st.warning("No se encontraron estas hojas: " + ", ".join(faltantes))

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
