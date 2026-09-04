import datetime
import math
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# 1. CONFIGURACIÓN Y LECTURA AUTOMÁTICA DEL FIXTURE
# =========================================================
st.set_page_config(
    page_title="Predicciones Liga 1 - Fixture Integrado",
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
def cargar_datos_excel():
    """Lee el archivo liga1_data.xlsx y limpia los encabezados."""
    try:
        df = pd.read_excel("liga1_data.xlsx", sheet_name="Fixture")
    except Exception:
        df = pd.read_excel("liga1_data.xlsx")

    df.columns = df.columns.astype(str).str.strip()
    return df


try:
    df_fixture = cargar_datos_excel()
    excel_conectado = True
except Exception:
    excel_conectado = False


# =========================================================
# 2. MOTORES MATEMÁTICOS (POISSON + SIGMOIDE)
# =========================================================
def calcular_poisson(lambda_local, lambda_visita, max_goles=5):
    matriz_prob = np.zeros((max_goles + 1, max_goles + 1))
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            prob_i = (
                (lambda_local**i) * math.exp(-lambda_local)
            ) / math.factorial(i)
            prob_j = (
                (lambda_visita**j) * math.exp(-lambda_visita)
            ) / math.factorial(j)
            matriz_prob[i][j] = prob_i * prob_j

    prob_local = np.sum(np.tril(matriz_prob, -1))
    prob_empate = np.sum(np.diag(matriz_prob))
    prob_visita = np.sum(np.triu(matriz_prob, 1))

    prob_bts = 0
    for i in range(1, max_goles + 1):
        for j in range(1, max_goles + 1):
            prob_bts += matriz_prob[i][j]

    return prob_local, prob_empate, prob_visita, prob_bts


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
# 3. INTERFAZ Y PROCESAMIENTO
# =========================================================
st.title("⚽ Predicciones Liga 1 - Sistema con Fixture Real")

if excel_conectado:
    st.success("🟢 **Base de datos conectada correctamente desde Excel.**")

    # Mapeo de columnas del Excel
    col_jornada = (
        "Jornada" if "Jornada" in df_fixture.columns else df_fixture.columns[0]
    )
    col_local = (
        "Local" if "Local" in df_fixture.columns else df_fixture.columns[3]
    )
    col_visita = (
        "Visita" if "Visita" in df_fixture.columns else df_fixture.columns[4]
    )
    col_ciudad = (
        "Ciudad" if "Ciudad" in df_fixture.columns else df_fixture.columns[6]
    )

    with st.sidebar:
        st.header("🗓️ Navegación de Partidos")

        # Filtro de Jornada
        jornadas_disponibles = df_fixture[col_jornada].unique()
        jornada_sel = st.selectbox("Seleccionar Jornada", jornadas_disponibles)

        # Filtrar partidos por la jornada seleccionada
        df_jornada = df_fixture[df_fixture[col_jornada] == jornada_sel].copy()
        df_jornada["Duelo"] = (
            df_jornada[col_local].astype(str)
            + " vs "
            + df_jornada[col_visita].astype(str)
        )

        partido_sel = st.selectbox(
            "Seleccionar Partido", df_jornada["Duelo"].unique()
        )

        # Obtener datos de la fila
        fila = df_jornada[df_jornada["Duelo"] == partido_sel].iloc[0]

        st.markdown("---")
        st.header("⚙️ Parámetros del Partido")

        local = str(fila.get("Local", "Local"))
        visita = str(fila.get("Visita", "Visita"))
        hora = str(fila.get("Hora", "15:00"))
        plaza = str(fila.get("Ciudad", "Lima"))
        estadio = str(fila.get("Estadio", "Estadio Principal"))

        st.info(
            f"📍 **Plaza:** {plaza}\n\n🏟️ **Estadio:** {estadio}\n\n⏰ **Hora:** {hora}"
        )

        # Determinación automática de superficie
        es_sintetica_auto = any(
            p.lower() in plaza.lower() for p in PLAZAS_SINTETICAS
        )
        tipo_cancha = st.selectbox(
            "Gramado",
            ["Natural", "Sintético"],
            index=1 if es_sintetica_auto else 0,
        )

        lambda_l = st.number_input(
            "Prom. Goles Local (λ)", 0.5, 4.0, float(fila.get("Lambda_Local", 1.5)), 0.1
        )
        lambda_v = st.number_input(
            "Prom. Goles Visita (λ)", 0.5, 4.0, float(fila.get("Lambda_Visita", 1.0)), 0.1
        )

        urgencia_l = st.slider(
            "Urgencia Local", 1, 5, int(fila.get("Urgencia_Local", 3))
        )
        urgencia_v = st.slider(
            "Urgencia Visita", 1, 5, int(fila.get("Urgencia_Visita", 3))
        )

        vino_de_golear = st.checkbox(
            "Local viene de golear", value=bool(fila.get("Goleada_Previa", 0))
        )
        desgaste_logistico = st.checkbox(
            "Alerta de viaje en Visita", value=bool(fila.get("Desgaste_Viaje", 0))
        )

    # Convertir hora a formato decimal para filtro térmico
    try:
        h_str = str(hora).split(":")[0]
        m_str = str(hora).split(":")[1]
        hora_dec = int(h_str) + int(m_str) / 60.0
    except Exception:
        hora_dec = 15.0

    # Evaluación de Filtros
    es_calor_extremo = (
        1
        if (
            any(p.lower() in plaza.lower() for p in PLAZAS_CALOR)
            and 13.0 <= hora_dec <= 15.5
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

    # =========================================================
    # 4. INFORME DETALLADO
    # =========================================================
    st.markdown("---")
    st.subheader(f"📊 INFORME DETALLADO: {local} vs {visita}")
    st.markdown(
        f"🗓️ **{jornada_sel}** | ⏰ **Hora:** {hora} hrs ({plaza}) | 🏟️ **Estadio:** {estadio} ({tipo_cancha})"
    )

    if es_calor_extremo:
        st.error(
            "⚠️ **FILTRO CLIMÁTICO ACTIVO:** Calor Extremo detectado en la plaza. El ritmo tiende a desacelerar en el 2do tiempo."
        )
    if es_goleada:
        st.warning(
            "⚠️ **ALERTA DE REGRESIÓN A LA MEDIA:** El local viene de golear. Se ajusta a la baja la expectativa de goleada."
        )

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("### 🛡️ Capa 1: Presión por Objetivos (Tabla)")
        st.write(f"* **Nivel de Urgencia Local:** {urgencia_l}/5")
        st.write(f"* **Nivel de Urgencia Visita:** {urgencia_v}/5")

        st.markdown("### 📈 Capa 2: Forma, Cancha y Logística")
        st.write(
            f"* **Racha Local:** {'Viene de golear' if es_goleada else 'Forma estándar'}"
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

    if es_calor_extremo and es_goleada:
        rec_resultado = f"**Doble Oportunidad: {local} o Empate (1X)** o Victoria ajustada por 1 gol."
        rec_goles = "**Menos de 2.5 Goles / Ambos Anotan: NO** (Ajuste por desaceleración climática)."
    elif prob_1x_final > 0.65:
        rec_resultado = (
            f"**Gana {local} Seco** o **1X** (Alta coincidencia de modelos)."
        )
        rec_goles = (
            "**Más de 1.5 Goles**"
            if prob_bts_final > 0.5
            else "**Ambos Anotan: NO**"
        )
    else:
        rec_resultado = (
            "**Partido Reservado / Doble Oportunidad Visita o Empate**"
        )
        rec_goles = "**Menos de 2.5 Goles**"

    st.success(f"🎯 **Sugerencia de Resultado:** {rec_resultado}")
    st.info(f"⚽ **Sugerencia de Goles:** {rec_goles}")

else:
    st.error(
        "🔴 **No se encontró el archivo `liga1_data.xlsx`.** Revisa que esté subido a GitHub."
    )
