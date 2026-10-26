# -*- coding: utf-8 -*-
"""
Modelo de Predicción Liga 1 Perú - Quipus Data

Versión revisada:

Lee las ocho hojas del archivo Liga1_2026.xlsx.
Usa Poisson con corrección Dixon-Coles.
Filtra los partidos históricos anteriores a cada encuentro usando la fecha real.
Recalcula K/L/M/N cronológicamente y no depende de referencias Prom_* desplazadas del Excel.
Incorpora forma, localía, xG, tablas, altitud, descanso y contexto competitivo.
Aplica realmente los factores contextuales después de la lambda base, evitando ajustes calculados pero no utilizados.
Muestra resaltado en las letras, no en el fondo de las filas.
Añade una lectura orientativa de presión competitiva: título/puestos internacionales,
  zona media y lucha por evitar el descenso.

La presión competitiva se muestra como contexto y solo aplica ajustes muy pequeños
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

    Reproduce la metodología de Resultados_Clausura:

      lambda_local     = ((K * N) / D2) * factor

      lambda_visitante = ((M * L) / D6) * factor



    IMPORTANTE:

    En el Excel actualizado, la columna J es Altitud_Visita.

    Las fórmulas originales usan J para aplicar los factores de altitud,

    por lo que se replica exactamente esa referencia.

    """

    ml = promedios_excel_equipo(local, fecha_partido, jornada_partido)

    mv = promedios_excel_equipo(visita, fecha_partido, jornada_partido)

    d2, d6 = datos["prior_local"], datos["prior_visita"]

    f_local, f_visita = factores_altitud_excel(altitud_excel_j)



    lam_l = (

        np.nan

        if not all(np.isfinite([ml["K"], mv["N"], d2])) or d2 <= 0

        else (ml["K"] * mv["N"] / d2) * f_local

    )

    lam_v = (

        np.nan

        if not all(np.isfinite([mv["M"], ml["L"], d6])) or d6 <= 0

        else (mv["M"] * ml["L"] / d6) * f_visita

    )



    return {

        "lambda_local": lam_l,

        "lambda_visita": lam_v,

        "local": ml,

        "visita": mv,

        "D2": d2,

        "D6": d6,

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

APLICAR_MICROAJUSTE_PRESION = False



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

    alt_l = a_float(il.get("altitud_msnm")) if il is not None else a_float(partido.get("altitud_local"), 0)

    alt_v = a_float(iv.get("altitud_msnm")) if iv is not None else a_float(partido.get("altitud_visita"), 0)

    delta = alt_l - alt_v if np.isfinite(alt_l) and np.isfinite(alt_v) else np.nan



    # La altitud NO se vuelve a aplicar aquí: la metodología Excel

    # la incorpora directamente en lambda mediante J (altitud del estadio).



    clima_l = str(il.get("tipo_clima", "")).lower() if il is not None else ""

    clima_v = str(iv.get("tipo_clima", "")).lower() if iv is not None else ""

    cancha_l = str(il.get("tipo_cancha", "")).lower() if il is not None else ""

    if "calor" in clima_l and "calor" not in clima_v:

        f_loc *= 1.02; f_vis *= .98

    if "sintetic" in cancha_l or "sintética" in cancha_l:

        f_loc *= 1.02; f_vis *= .98



    descanso_l = a_float(partido.get("dias_descanso_local"), 7)

    descanso_v = a_float(partido.get("dias_descanso_visita"), 7)

    if descanso_l < 4: f_loc *= .95

    if descanso_v < 4: f_vis *= .95



    hora = parsear_hora(partido.get("hora"))

    try:

        if int(hora.split(":")[0]) in (11, 12, 13) and np.isfinite(delta) and delta > 1000:

            f_loc *= 1.02; f_vis *= .99

    except Exception:

        pass



    # Historial H2H: ajuste limitado para no sobreponderar enfrentamientos antiguos.

    h2h = datos["Historial_H2H"]

    if not h2h.empty and {"local_std", "visita_std", "gl", "gv"}.issubset(h2h.columns):

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

            elif i == 1 and j == 0: tau = 1 + lambda_loc * rho

            elif i == 0 and j == 1: tau = 1 + lambda_vis * rho

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



    # Metodología principal: fórmulas de Resultados_Clausura.

    # La altitud ya está incorporada en esta etapa.

    # La fórmula Excel usa la columna J = Altitud_Visita.

    # No usar Altitud_Local aquí: hacerlo produciría un modelo distinto al Excel.

    altitud_j = a_float(partido.get("altitud_visita"), np.nan)

    excel = lambdas_clausura_excel(

        loc, vis, fecha, jornada_num, altitud_j

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



    return {

        "Señal": senal, "Estado": estado, "Día": dia or "Por definir",

        "Hora": parsear_hora(partido.get("hora")), "Local": loc, "Visita": vis,

        "λ Local": round(lambda_loc, 2), "λ Visita": round(lambda_vis, 2),

        "% Local": round(p_local, 1), "% Empate": round(p_empate, 1), "% Visita": round(p_visita, 1),

        "Más de 2.5 goles": f"{p_over:.1f}%", "Menos de 2.5 goles": f"{p_under:.1f}%", "Ambos marcan: Sí": f"{p_btts:.1f}%",

        "Marcador compatible": f"{i_pron} - {j_pron}",

        "Marcador modal": f"{i_modal} - {j_modal}",

        "Pronóstico 1X2": ganador, "Recomendacion": recomendacion,

        "Prob. principal": round(max(probs.values()), 1), "Margen 1X2": round(margen, 1),

        "Prom_GF_Local": round(excel["local"]["K"], 3) if np.isfinite(excel["local"]["K"]) else np.nan,

        "Prom_GC_Local": round(excel["local"]["L"], 3) if np.isfinite(excel["local"]["L"]) else np.nan,

        "Prom_GF_Visita": round(excel["visita"]["M"], 3) if np.isfinite(excel["visita"]["M"]) else np.nan,

        "Prom_GC_Visita": round(excel["visita"]["N"], 3) if np.isfinite(excel["visita"]["N"]) else np.nan,

        "D2": round(excel["D2"], 4), "D6": round(excel["D6"], 4),

        "Altitud usada por fórmula (J)": 
