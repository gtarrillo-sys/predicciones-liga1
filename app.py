import math
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# 1. CONFIGURACIÓN Y LECTURA
# =========================================================
st.set_page_config(
    page_title="Predicciones Liga 1 - Calibración Precisa",
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


@st.cache_data
def cargar_datos():
    try:
        xls = pd.ExcelFile("liga1_data.xlsx")
        df_partidos = pd.read_excel(
            xls,
            "Partidos_Fecha"
            if "Partidos_Fecha" in xls.sheet_names
            else xls.sheet_names[0],
        )
        df_geo = (
            pd.read_excel(xls, "Data_Geografica")
            if "Data_Geografica" in xls.sheet_names
            else pd.DataFrame()
        )
        df_tabla = (
            pd.read_excel(xls, "Tabla_Posiciones_Clausura")
            if "Tabla_Posiciones_Clausura" in xls.sheet_names
            else pd.DataFrame()
        )

        df_res = (
            pd.read_excel(xls, "Resultados_Clausura")
            if "Resultados_Clausura" in xls.sheet_names
            else pd.DataFrame()
        )

        df_partidos.columns = df_partidos.columns.astype(str).str.strip()
        if not df_geo.empty:
            df_geo.columns = df_geo.columns.astype(str).str.strip()
        if not df_tabla.empty:
            df_tabla.columns = df_tabla.columns.astype(str).str.strip()
        if not df_res.empty:
            df_res.columns = df_res.columns.astype(str).str.strip()

        return df_partidos, df_geo, df_tabla, df_res
    except Exception:
        return None, None, None, None


df_partidos, df_geo, df_tabla, df_res = cargar_datos()


# =========================================================
# 2. MOTORES MATEMÁTICOS
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
    ) = features

    z_1x2 = (
        0.20
        + (0.35 * (lambda_l - lambda_v))
        - (0.25 * clima_3pm)
        - (0.30 * goleada)
        + (0.45 * desgaste)
        + (0.40 * sintetica)
        + (0.15 * (urg_l - urg_v))
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
# 3. INTERFAZ STREAMLIT
# =========================================================
st.title("⚽ Predicciones Liga 1 - Análisis Integral")

if df_partidos is not None:
    with st.sidebar:
        st.header("🗓️ Selección de Partido")
        jornadas = df_partidos["Jornada"].dropna().unique()
        jornada_sel = st.selectbox("Seleccionar Jornada", jornadas)

        df_jornada = df_partidos[
            df_partidos["Jornada"] == jornada_sel
        ].copy()
        df_jornada["Duelo"] = (
            df_jornada["Local"].astype(str)
            + " vs "
            + df_jornada["Visita"].astype(str)
        )
        partido_sel = st.selectbox(
            "Seleccionar Duelo", df_jornada["Duelo"].unique()
        )

        fila = df_jornada[df_jornada["Duelo"] == partido_sel].iloc[0]
        local = str(fila.get("Local", "Local"))
        visita = str(fila.get("Visita", "Visita"))
        hora = str(fila.get("Hora", "15:15"))
        plaza = str(fila.get("Ciudad", "Sullana"))
        estadio = str(
            fila.get("Estadio", "Estadio Campeones del 36")
        )

        # Ajuste por defecto específico para Alianza Atlético (2.33 si es local)
        def_lambda_l = (
            2.33
            if "Alianza" in local
            else 1.80
        )
        def_lambda_v = 0.67 if "UTC" in visita else 0.90

        st.markdown("---")
        st.header("⚙️ Variables de Promedio de Gol (λ)")
        lambda_l = st.number_input(
            f"Prom. Goles Local ({local})",
            0.5,
            4.0,
            float(def_lambda_l),
            0.05,
        )
        lambda_v = st.number_input(
            f"Prom. Goles Visita ({visita})",
            0.1,
            4.0,
            float(def_lambda_v),
            0.05,
        )

        st.markdown("---")
        st.header("⚙️ Entorno y Forma")
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
            "Alerta de viaje en Visita",
            value=True if "UTC" in visita else False,
        )

    # Conversión de Hora
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
    ]
    p_1x2_log, p_bts_log = modelo_regresion_logistica(features)

    prob_1x_final = ((p_loc_poi + p_emp_poi) + p_1x2_log) / 2.0
    prob_bts_final = (p_bts_poi + p_bts_log) / 2.0

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
        st.markdown("### 🛡️ Capa 1: Presión por Objetivos (Tabla)")
        st.write(f"* **Nivel de Urgencia Local:** {urgencia_l}/5")
        st.write(f"* **Nivel de Urgencia Visita:** {urgencia_v}/5")

        st.markdown("### 📈 Capa 2: Forma, Cancha y Logística")
        st.write(
            f"* **Racha Local:** {'Viene de golear / Racha positiva' if es_goleada else 'Forma estándar'}"
        )
        st.write(
            f"* **Estado Visita:** {'Desgaste de viaje' if es_desgaste else 'Traslado normal'}"
        )

    with col_c2:
        st.markdown("### 🧮 Capa 3: Métricas Híbridas")
        st.write(f"* **Poisson (1X):** {((p_loc_poi + p_emp_poi)*100):.1f}%")
        st.write(f"* **Regresión Logística (1X):** {(p_1x2_log*100):.1f}%")
        st.write(f"* **Poisson (Ambos Anotan):** {(p_bts_poi*100):.1f}%")
        st.write(
            f"* **Regresión Logística (Ambos Anotan):** {(p_bts_log*100):.1f}%"
        )

    st.markdown("---")
    st.markdown("### 🧠 Capa 4: Sugerencia Final del Sistema")

    if prob_1x_final >= 0.60:
        if prob_1x_final >= 0.72:
            rec_resultado = f"**Gana {local} Seco** (Dominio claro de métricas)."
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

else:
    st.error("No se pudo leer el archivo `liga1_data.xlsx`.")
