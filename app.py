import math
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# 1. CONFIGURACIÓN Y LECTURA MULTI-HOJA
# =========================================================
st.set_page_config(
    page_title="Predicciones Liga 1 - Sistema Integral",
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
def cargar_todas_las_hojas():
    """Lee las 4 pestañas de liga1_data.xlsx"""
    try:
        xls = pd.ExcelFile("liga1_data.xlsx")

        df_partidos = (
            pd.read_excel(xls, "Partidos_Fecha")
            if "Partidos_Fecha" in xls.sheet_names
            else pd.read_excel(xls, xls.sheet_names[0])
        )

        df_res = (
            pd.read_excel(xls, "Resultados_Clausura")
            if "Resultados_Clausura" in xls.sheet_names
            else pd.DataFrame()
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

        return df_partidos, df_res, df_geo, df_tabla
    except Exception:
        return None, None, None, None


df_partidos, df_res, df_geo, df_tabla = cargar_todas_las_hojas()


def calcular_lambdas_dinamicos(local, visita, df_resultados):
    """Calcula el rendimiento real en base a los partidos jugados en Resultados_Clausura."""
    if df_resultados.empty or len(df_resultados.columns) < 5:
        return 1.8, 0.8

    # Detección posicional de columnas:
    # Col 0: Jornada | Col 1: Local | Col 2: Goles_Local | Col 3: Goles_Visita | Col 4: Visita
    col_loc = df_resultados.columns[1]
    col_gl = df_resultados.columns[2]
    col_gv = df_resultados.columns[3]
    col_vis = df_resultados.columns[4]

    # Partidos de local del equipo Local
    p_loc = df_resultados[
        df_resultados[col_loc].astype(str).str.contains(local, case=False)
    ]
    g_favor_loc = (
        pd.to_numeric(p_loc[col_gl], errors="coerce").mean()
        if not p_loc.empty
        else 1.8
    )

    # Partidos de visita del equipo Visita
    p_vis = df_resultados[
        df_resultados[col_vis].astype(str).str.contains(visita, case=False)
    ]
    g_rec_vis = (
        pd.to_numeric(p_vis[col_gl], errors="coerce").mean()
        if not p_vis.empty
        else 1.4
    )
    g_favor_vis = (
        pd.to_numeric(p_vis[col_gv], errors="coerce").mean()
        if not p_vis.empty
        else 0.8
    )

    lambda_loc = round(
        (
            g_favor_loc
            if not np.isnan(g_favor_loc)
            else 1.8 + (g_rec_vis if not np.isnan(g_rec_vis) else 1.4)
        )
        / 2.0,
        2,
    )
    lambda_vis = round(
        g_favor_vis if not np.isnan(g_favor_vis) else 0.8, 2
    )

    return max(0.5, lambda_loc), max(0.3, lambda_vis)


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
    tiene_res = df_res is not None and not df_res.empty

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

        # Cálculo dinámico de Lambdas usando la hoja Resultados_Clausura
        lambda_l_auto, lambda_v_auto = calcular_lambdas_dinamicos(
            local, visita, df_res
        )

        st.markdown("---")
        st.header("⚙️ Variables Contextuales")
        es_sintetica_auto = any(
            p.lower() in plaza.lower() for p in PLAZAS_SINTETICAS
        )
        tipo_cancha = st.selectbox(
            "Gramado",
            ["Natural", "Sintético"],
            index=1 if es_sintetica_auto else 0,
        )

        lambda_l = st.number_input(
            f"Prom. Goles Local ({local})",
            0.5,
            4.0,
            float(lambda_l_auto),
            0.1,
        )
        lambda_v = st.number_input(
            f"Prom. Goles Visita ({visita})",
            0.3,
            4.0,
            float(lambda_v_auto),
            0.1,
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
        f"📍 **Plaza:** {plaza} | ⏰ **Hora:** {hora} hrs | 📊 **Historial Clausura Integrado:** {'Sí' if tiene_res else 'No'}"
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
