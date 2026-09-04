import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predicciones Liga 1 - Modelo Dinámico",
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


@st.cache_data
def cargar_datos_completos():
    try:
        xls = pd.ExcelFile("liga1_data.xlsx")

        df_partidos = pd.read_excel(
            xls,
            "Partidos_Fecha"
            if "Partidos_Fecha" in xls.sheet_names
            else xls.sheet_names[0],
        )

        df_jugados = (
            pd.read_excel(xls, "Resultados_Clausura")
            if "Resultados_Clausura" in xls.sheet_names
            else pd.DataFrame()
        )

        df_partidos.columns = df_partidos.columns.astype(str).str.strip()
        if not df_jugados.empty:
            df_jugados.columns = df_jugados.columns.astype(str).str.strip()

        return df_partidos, df_jugados
    except Exception:
        return None, None


df_partidos, df_jugados = cargar_datos_completos()


def calcular_lambdas_dinamicos(equipo_local, equipo_visita, df_historial):
    """Calcula el rendimiento real en base a los partidos jugados en el Clausura."""
    if df_historial.empty:
        return 1.6, 0.9  # Valores estándar si no hay historial

    # Partidos de local del equipo Local
    partidos_loc = df_historial[
        df_historial["Local"].astype(str).str.contains(equipo_local, case=False)
    ]
    goles_f_loc = (
        partidos_loc["Goles_Local"].mean() if not partidos_loc.empty else 1.5
    )

    # Partidos de visita del equipo Visita
    partidos_vis = df_historial[
        df_historial["Visita"]
        .astype(str)
        .str.contains(equipo_visita, case=False)
    ]
    goles_rec_vis = (
        partidos_vis["Goles_Local"].mean() if not partidos_vis.empty else 1.2
    )
    goles_f_vis = (
        partidos_vis["Goles_Visita"].mean() if not partidos_vis.empty else 0.8
    )

    # Estimación ponderada de Lambda
    lambda_local = max(0.5, round((goles_f_loc + goles_rec_vis) / 2.0, 2))
    lambda_visita = max(0.4, round(goles_f_vis, 2))

    return lambda_local, lambda_visita


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


# INTERFAZ Y PROCESAMIENTO
st.title("⚽ Predicciones Liga 1 - Análisis con Historial de Partidos")

if df_partidos is not None:
    with st.sidebar:
        st.header("🗓️ Partido a Analizar")
        jornadas = df_partidos["Jornada"].dropna().unique()
        jornada_sel = st.selectbox("Jornada", jornadas)

        df_jornada = df_partidos[df_partidos["Jornada"] == jornada_sel].copy()
        df_jornada["Duelo"] = (
            df_jornada["Local"].astype(str)
            + " vs "
            + df_jornada["Visita"].astype(str)
        )
        partido_sel = st.selectbox("Duelo", df_jornada["Duelo"].unique())

        fila = df_jornada[df_jornada["Duelo"] == partido_sel].iloc[0]
        local = str(fila.get("Local", "Local"))
        visita = str(fila.get("Visita", "Visita"))
        hora = str(fila.get("Hora", "15:15"))
        plaza = str(fila.get("Ciudad", "Sullana"))

        # Cálculo dinámico de Lambdas
        lambda_l_auto, lambda_v_auto = calcular_lambdas_dinamicos(
            local, visita, df_jugados
        )

        st.markdown("---")
        st.header("⚙️ Promedios Detectados")
        lambda_l = st.number_input(
            f"λ Gol Local ({local})", 0.5, 4.0, float(lambda_l_auto), 0.1
        )
        lambda_v = st.number_input(
            f"λ Gol Visita ({visita})", 0.3, 4.0, float(lambda_v_auto), 0.1
        )

    p_loc, p_emp, p_vis, p_bts = calcular_poisson(lambda_l, lambda_v)
    prob_1x = p_loc + p_emp

    st.subheader(f"📊 Análisis Evaluado: {local} vs {visita}")
    st.write(
        f"📍 **Plaza:** {plaza} | ⏰ **Hora:** {hora} | 📊 **Historial Clausura Integrado:** {'Sí' if not df_jugados.empty else 'No'}"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilidad Gana Local / Empate (1X)", f"{prob_1x*100:.1f}%")
        st.metric("Probabilidad Victoria Local Seca", f"{p_loc*100:.1f}%")
    with col2:
        st.metric("Probabilidad Ambos Anotan (BTS)", f"{p_bts*100:.1f}%")
        st.metric("Probabilidad Empate", f"{p_emp*100:.1f}%")

    st.markdown("---")
    st.markdown("### 🎯 Sugerencia de Apuesta")

    if prob_1x >= 0.65:
        st.success(
            f"✅ **Doble Oportunidad {local} o Empate (1X)** (Elevado respaldo por rendimiento en Clausura)"
        )
    else:
        st.warning(
            "⚠️ **Partido de Pronóstico Reservado / Probar hándicap o goles**"
        )
else:
    st.error("No se pudo cargar el archivo Excel.")
