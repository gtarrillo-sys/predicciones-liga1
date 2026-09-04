import math
import os
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y CONSTANTES
# =========================================================
st.set_page_config(
    page_title="Predicciones Liga 1 - Modelo Híbrido Integral",
    page_icon="⚽",
    layout="wide",
)

PLAZAS_CALOR = [
    "Piura",
    "Sullana",
    "Tarapoto",
    "Chiclayo",
    "Iquitos",
    "Chongoyape",
]
PLAZAS_SINTETICAS = [
    "Juliaca",
    "Andahuaylas",
    "Trujillo",
    "Nueva Cajamarca",
    "Cajamarca",
    "Cutervo",
]


# =========================================================
# 2. CARGA DE DATOS ROBUSTA (SOPORTE LOCAL Y STREAMLIT CLOUD)
# =========================================================
@st.cache_data(ttl=60)
def cargar_datos():
    archivos_locales = os.listdir(".")
    archivo_excel = next(
        (
            f
            for f in archivos_locales
            if "liga1" in f.lower() and f.endswith(".xlsx")
        ),
        None,
    )

    if not archivo_excel:
        return (
            None,
            f"No se encontró el archivo Excel en la raíz. Archivos detectados: {archivos_locales}",
        )

    try:
        xls = pd.ExcelFile(archivo_excel, engine="openpyxl")
        hojas = xls.sheet_names

        # Selección inteligente de hojas por palabras clave
        hoja_partidos = next(
            (
                h
                for h in hojas
                if "partido" in h.lower() or "fecha" in h.lower()
            ),
            hojas[0],
        )
        hoja_acumulada = next(
            (
                h
                for h in hojas
                if "acumulad" in h.lower() or "tabla_acumulada" in h.lower()
            ),
            hojas[1] if len(hojas) > 1 else hojas[0],
        )
        hoja_clausura = next(
            (
                h
                for h in hojas
                if "clausura" in h.lower() or "tabla_clausura" in h.lower()
            ),
            hojas[2] if len(hojas) > 2 else hojas[0],
        )

        df_partidos = pd.read_excel(xls, hoja_partidos)
        df_acumulada = pd.read_excel(xls, hoja_acumulada)
        df_clausura = pd.read_excel(xls, hoja_clausura)

        # Limpiar espacios en los nombres de columnas
        for df in [df_partidos, df_acumulada, df_clausura]:
            if df is not None and not df.empty:
                df.columns = df.columns.astype(str).str.strip()

        return (df_partidos, df_acumulada, df_clausura), None

    except Exception as e:
        return None, f"Error al procesar el archivo Excel: {str(e)}"


resultado_carga, error_msg = cargar_datos()

if resultado_carga is not None:
    df_partidos, df_acumulada, df_clausura = resultado_carga
else:
    st.warning(
        f"⚠️ **Nota de carga:** {error_msg}. Se ha activado la interfaz con datos por defecto."
    )
    df_partidos = pd.DataFrame(
        {
            "Jornada": ["Jornada 8"],
            "Local": ["Alianza Atlético"],
            "Visita": ["UTC Cajamarca"],
            "Hora": ["15:15"],
            "Ciudad": ["Sullana"],
            "Estadio": ["Estadio Campeones del 36"],
        }
    )
    df_acumulada = pd.DataFrame()
    df_clausura = pd.DataFrame()


# =========================================================
# 3. FUNCIONES DE EXTRACCIÓN Y LECTURA
# =========================================================
def obtener_posicion_excel(df_tabla, equipo_nombre):
    """Busca al equipo en el DataFrame y devuelve el valor de la columna 'Rango'"""
    if df_tabla is None or df_tabla.empty:
        return 10

    col_club = next(
        (c for c in df_tabla.columns if "club" in str(c).lower()),
        df_tabla.columns[1] if len(df_tabla.columns) > 1 else df_tabla.columns[0],
    )

    coincidencia = df_tabla[
        df_tabla[col_club].astype(str).str.strip().str.lower()
        == str(equipo_nombre).strip().lower()
    ]

    if not coincidencia.empty:
        if "Rango" in df_tabla.columns:
            try:
                return int(coincidencia.iloc[0]["Rango"])
            except Exception:
                pass
        return coincidencia.index[0] + 1

    return 10


def obtener_puntos_racha_excel(df_clausura, equipo_nombre):
    """Suma los puntos de las 5 columnas a partir de 'Últimos 5 partidos'"""
    if df_clausura is None or df_clausura.empty:
        return 7

    col_club = next(
        (c for c in df_clausura.columns if "club" in str(c).lower()),
        df_clausura.columns[1] if len(df_clausura.columns) > 1 else df_clausura.columns[0],
    )

    coincidencia = df_clausura[
        df_clausura[col_club].astype(str).str.strip().str.lower()
        == str(equipo_nombre).strip().lower()
    ]

    if coincidencia.empty:
        return 7

    fila = coincidencia.iloc[0]
    cols_list = list(df_clausura.columns)

    idx_racha = next(
        (
            i
            for i, c in enumerate(cols_list)
            if "últim" in str(c).lower() or "ultim" in str(c).lower()
        ),
        None,
    )

    if idx_racha is not None:
        puntos = 0
        cols_5 = cols_list[idx_racha : idx_racha + 5]
        for col in cols_5:
            val = str(fila[col]).strip().lower()
            if "gan" in val:
                puntos += 3
            elif "emp" in val:
                puntos += 1
        return puntos

    return 7


# =========================================================
# 4. MOTORES MATEMÁTICOS DE PREDICCIÓN
# =========================================================
def calcular_poisson(lambda_l, lambda_v, max_g=5):
    mat = np.zeros((max_g + 1, max_g + 1))
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p_i = ((lambda_l**i) * math.exp(-lambda_l)) / math.factorial(i)
            p_j = ((lambda_v**j) * math.exp(-lambda_v)) / math.factorial(j)
            mat[i][j] = p_i * p_j

    p_loc = np.sum(np.tril(mat, -1))
    p_emp = np.sum(np.diag(mat))
    p_vis = np.sum(np.triu(mat, 1))
    p_bts = sum(
        mat[i][j] for i in range(1, max_g + 1) for j in range(1, max_g + 1)
    )

    return p_loc, p_emp, p_vis, p_bts


def funcion_sigmoide(z):
    return 1.0 / (1.0 + math.exp(-z))


def modelo_regresion_logistica(features):
    (
        lambda_l,
        lambda_v,
        clima_3pm,
        goleada,
        desgaste,
        sintetica,
        urg_l,
        urg_v,
        racha_l,
        racha_v,
        pos_ac_l,
        pos_ac_v,
        pos_cl_l,
        pos_cl_v,
    ) = features

    factor_racha = (racha_l - racha_v) / 15.0

    pos_efectiva_l = (0.60 * pos_ac_l) + (0.40 * pos_cl_l)
    pos_efectiva_v = (0.60 * pos_ac_v) + (0.40 * pos_cl_v)

    f_pos_l = (18 - pos_efectiva_l + 1) / 18.0
    f_pos_v = (18 - pos_efectiva_v + 1) / 18.0
    dif_posicion = f_pos_l - f_pos_v

    z_1x2 = (
        0.15
        + (0.30 * (lambda_l - lambda_v))
        - (0.25 * clima_3pm)
        - (0.25 * goleada)
        + (0.40 * desgaste)
        + (0.35 * sintetica)
        + (0.15 * (urg_l - urg_v))
        + (0.50 * factor_racha)
        + (0.45 * dif_posicion)
    )

    z_bts = (
        -0.10
        + (0.25 * (lambda_l + lambda_v))
        - (0.50 * clima_3pm)
        - (0.35 * goleada)
        - (0.20 * desgaste)
    )

    return funcion_sigmoide(z_1x2), funcion_sigmoide(z_bts)


# =========================================================
# 5. INTERFAZ STREAMLIT
# =========================================================
st.title("⚽ Predicciones Liga 1 - Análisis Multicapa Integral")

with st.sidebar:
    st.header("🗓️ Selección de Partido")
    jornadas = df_partidos["Jornada"].dropna().unique()
    jornada_sel = st.selectbox("Seleccionar Jornada", jornadas)

    df_jornada = df_partidos[df_partidos["Jornada"] == jornada_sel].copy()
    df_jornada["Duelo"] = (
        df_jornada["Local"].astype(str)
        + " vs "
        + df_jornada["Visita"].astype(str)
    )
    partido_sel = st.selectbox(
        "Seleccionar Duelo", df_jornada["Duelo"].unique()
    )

    fila = df_jornada[df_jornada["Duelo"] == partido_sel].iloc[0]
    local = str(fila.get("Local", "Local")).strip()
    visita = str(fila.get("Visita", "Visita")).strip()
    hora = str(fila.get("Hora", "15:15"))
    plaza = str(fila.get("Ciudad", "Sullana"))
    estadio = str(fila.get("Estadio", "Estadio Campeones del 36"))

    # Carga automática de posiciones y rachas
    pos_ac_l_auto = obtener_posicion_excel(df_acumulada, local)
    pos_ac_v_auto = obtener_posicion_excel(df_acumulada, visita)

    pos_cl_l_auto = obtener_posicion_excel(df_clausura, local)
    pos_cl_v_auto = obtener_posicion_excel(df_clausura, visita)

    pts_racha_l_auto = obtener_puntos_racha_excel(df_clausura, local)
    pts_racha_v_auto = obtener_puntos_racha_excel(df_clausura, visita)

    st.markdown("---")
    st.header("🏆 Posiciones en Tablas")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.caption("Tabla Acumulada")
        pos_ac_l = st.number_input(f"Acum. {local}", 1, 18, int(pos_ac_l_auto))
        pos_ac_v = st.number_input(f"Acum. {visita}", 1, 18, int(pos_ac_v_auto))
    with col_t2:
        st.caption("Tabla Clausura")
        pos_cl_l = st.number_input(f"Claus. {local}", 1, 18, int(pos_cl_l_auto))
        pos_cl_v = st.number_input(f"Claus. {visita}", 1, 18, int(pos_cl_v_auto))

    st.markdown("---")
    st.header("🔥 Racha de los Últimos 5 Partidos")
    pts_racha_l = st.slider(
        f"Pts Últimos 5 - {local}", 0, 15, int(pts_racha_l_auto)
    )
    pts_racha_v = st.slider(
        f"Pts Últimos 5 - {visita}", 0, 15, int(pts_racha_v_auto)
    )

    st.markdown("---")
    st.header("⚙️ Promedios de Goles (λ)")
    lambda_l = st.number_input(
        f"Prom. Goles Local ({local})", 0.5, 4.0, 2.33, 0.05
    )
    lambda_v = st.number_input(
        f"Prom. Goles Visita ({visita})", 0.1, 4.0, 0.67, 0.05
    )

    st.markdown("---")
    st.header("⚙️ Entorno")
    es_sintetica_auto = any(
        p.lower() in plaza.lower() for p in PLAZAS_SINTETICAS
    )
    tipo_cancha = st.selectbox(
        "Gramado",
        ["Natural", "Sintético"],
        index=1 if es_sintetica_auto else 0,
    )

    urgencia_l = st.slider("Urgencia Local", 1, 5, 4)
    urgencia_v = st.slider("Urgencia Visita", 1, 5, 3)

    vino_de_golear = st.checkbox(
        "Local viene de golear", value=True if "Alianza" in local else False
    )
    desgaste_logistico = st.checkbox(
        "Alerta de viaje en Visita", value=True if "UTC" in visita else False
    )

# Cálculos Ambientales
try:
    h_dec = int(str(hora).split(":")[0]) + int(str(hora).split(":")[1]) / 60.0
except Exception:
    h_dec = 15.25

es_calor_extremo = (
    1
    if (
        any(p.lower() in plaza.lower() for p in PLAZAS_CALOR)
        and 13.0 <= h_dec <= 15.5
    )
    else 0
)
es_sintetica = 1 if tipo_cancha == "Sintético" else 0
es_goleada = 1 if vino_de_golear else 0
es_desgaste = 1 if desgaste_logistico else 0

# Evaluación de Modelos
p_loc_poi, p_emp_poi, p_vis_poi, p_bts_poi = calcular_poisson(
    lambda_l, lambda_v
)
features = [
    lambda_l,
    lambda_v,
    es_calor_extremo,
    es_goleada,
    es_desgaste,
    es_sintetica,
    urgencia_l,
    urgencia_v,
    pts_racha_l,
    pts_racha_v,
    pos_ac_l,
    pos_ac_v,
    pos_cl_l,
    pos_cl_v,
]
p_1x2_log, p_bts_log = modelo_regresion_logistica(features)

prob_1x_final = ((p_loc_poi + p_emp_poi) + p_1x2_log) / 2.0
prob_bts_final = (p_bts_poi + p_bts_log) / 2.0

# Despliegue de Resultados
st.subheader(f"📊 INFORME DETALLADO: {local} vs {visita}")
st.markdown(
    f"📍 **Plaza:** {plaza} | ⏰ **Hora:** {hora} hrs | 🏟️ **Estadio:** {estadio} ({tipo_cancha})"
)

if es_calor_extremo:
    st.error(
        "⚠️ **FILTRO CLIMÁTICO ACTIVO:** Calor Extremo detectado en la plaza. El ritmo tiende a desacelerar en el 2do tiempo."
    )

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("### 🏆 Capa 1: Posición y Momentum")
    st.write(
        f"* **Posición Acumulada (60%):** {local} ({pos_ac_l}°) vs {visita} ({pos_ac_v}°)"
    )
    st.write(
        f"* **Posición Clausura (40%):** {local} ({pos_cl_l}°) vs {visita} ({pos_cl_v}°)"
    )
    st.write(
        f"* **Racha Reciente:** {local} ({pts_racha_l} pts) vs {visita} ({pts_racha_v} pts)"
    )
    st.write(
        f"* **Nivel de Urgencia:** Local {urgencia_l}/5 | Visita {urgencia_v}/5"
    )

    st.markdown("### 📈 Capa 2: Cancha y Logística")
    st.write(
        f"* **Racha Local:** {'Viene de golear / Racha positiva' if es_goleada else 'Forma estándar'}"
    )
    st.write(
        f"* **Estado Visita:** {'Desgaste de viaje' if es_desgaste else 'Traslado normal'}"
    )

with col_c2:
    st.markdown("### 🧮 Capa 3: Métricas Híbridas Integradas")
    st.write(f"* **Poisson (1X):** {((p_loc_poi + p_emp_poi)*100):.1f}%")
    st.write(f"* **Regresión Logística (1X Ponderado):** {(p_1x2_log*100):.1f}%")
    st.write(f"* **Poisson (Ambos Anotan):** {(p_bts_poi*100):.1f}%")
    st.write(f"* **Regresión Logística (Ambos Anotan):** {(p_bts_log*100):.1f}%")

st.markdown("---")
st.markdown("### 🧠 Capa 4: Sugerencia Final del Sistema")

if prob_1x_final >= 0.60:
    if prob_1x_final >= 0.72:
        rec_resultado = f"**Gana {local} Seco** (Dominio de posición ponderada, racha reciente y localía)."
    else:
        rec_resultado = f"**Doble Oportunidad: {local} o Empate (1X)** (Excelente respaldo estadístico)."

    if es_calor_extremo:
        rec_goles = "**Menos de 2.5 Goles / Ambos Anotan: NO** (Ajuste por desaceleración climática)."
    else:
        rec_goles = (
            "**Más de 1.5 Goles**"
            if prob_bts_final > 0.50
            else "**Ambos Anotan: NO**"
        )
else:
    rec_resultado = (
        f"**Doble Oportunidad: {visita} o Empate (X2)** / Pronóstico reservado."
    )
    rec_goles = "**Menos de 2.5 Goles**"

st.success(f"🎯 **Sugerencia de Resultado:** {rec_resultado}")
st.info(f"⚽ **Sugerencia de Goles:** {rec_goles}")
