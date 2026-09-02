import datetime
import math
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

# =========================================================
# 1. CONFIGURACIÓN DE PÁGINA Y CONSTANTES
# =========================================================
st.set_page_config(
    page_title="Predicciones Liga 1 - Engine V2", page_icon="⚽", layout="wide"
)

PLAZAS_CALOR = ["Piura", "Sullana", "Tarapoto", "Chiclayo", "Iquitos"]
PLAZAS_SINTETICAS = [
    "Juliaca",
    "Andahuaylas",
    "Trujillo (Mansiche)",
    "Nueva Cajamarca",
]

# =========================================================
# 2. MOTORES MATEMÁTICOS (POISSON Y REGRESIÓN LOGÍSTICA)
# =========================================================


def calcular_poisson(lambda_local, lambda_visita, max_goles=5):
    """Calcula la matriz de probabilidades de marcadores usando Poisson."""
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

    prob_over_25 = 0
    prob_bts = 0
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            if i + j > 2.5:
                prob_over_25 += matriz_prob[i][j]
            if i > 0 and j > 0:
                prob_bts += matriz_prob[i][j]

    return prob_local, prob_empate, prob_visita, prob_over_25, prob_bts


def modelo_regresion_logistica(features_partido):
    """Simula/Ejecuta la predicción ajustada por el clasificador logístico en base a variables sintéticas y cualitativas."""
    # Matriz sintética de entrenamiento para calibración del modelo en Liga 1
    # Features: [prom_l, prom_v, clima_3pm, vino_golear, viaje_malo, sintetica, urg_l, urg_v]
    X_dummy = np.array(
        [
            [1.8, 0.8, 1, 1, 1, 0, 5, 2],  # Caso Grau vs Melgar (Calor, goleada previa, viaje)
            [2.1, 1.2, 0, 0, 0, 0, 4, 4],  # Clásico en Lima (Noche, natural)
            [1.2, 0.9, 1, 0, 0, 1, 2, 2],  # Partido de mitad de tabla en sintético
            [0.9, 1.5, 0, 0, 1, 0, 1, 5],  # Visita necesitada contra local cómodo
            [2.5, 0.5, 0, 1, 0, 1, 5, 1],  # Local fuerte en sintético
        ]
    )

    # Outcomes: [1X2 (1: Local, 0: No Local), Ambos Anotan (1: Sí, 0: No)]
    y_1x2_dummy = np.array([1, 1, 0, 0, 1])
    y_bts_dummy = np.array([0, 1, 0, 0, 0])

    clf_1x2 = LogisticRegression().fit(X_dummy, y_1x2_dummy)
    clf_bts = LogisticRegression().fit(X_dummy, y_bts_dummy)

    prob_1x2 = clf_1x2.predict_proba([features_partido])[0][1]
    prob_bts = clf_bts.predict_proba([features_partido])[0][1]

    return prob_1x2, prob_bts


# =========================================================
# 3. INTERFAZ DE USUARIO (SIDEBAR - PARÁMETROS DE LA FECHA)
# =========================================================
st.title("⚽ Predicciones Liga 1 - Engine Híbrido (Poisson + Logística)")
st.markdown(
    "Sistema de análisis predictivo ajustado por clima, logística, cancha y tabla."
)

with st.sidebar:
    st.header("⚙️ Configuración del Partido")
    local = st.text_input("Equipo Local", "Atlético Grau")
    visita = st.text_input("Equipo Visitante", "FBC Melgar")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fecha = st.date_input("Fecha", datetime.date(2026, 9, 1))
    with col_f2:
        hora = st.text_input("Hora (HH:MM)", "15:00")

    plaza = st.selectbox(
        "Ciudad / Plaza",
        [
            "Piura",
            "Sullana",
            "Lima",
            "Arequipa",
            "Cajamarca",
            "Chiclayo",
            "Tarapoto",
            "Juliaca",
            "Andahuaylas",
        ],
    )
    tipo_cancha = st.selectbox(
        "Tipo de Gramado",
        ["Natural", "Sintético"],
        index=1 if plaza in PLAZAS_SINTETICAS else 0,
    )

    st.subheader("📊 Datos Recientes y Contexto")
    lambda_l = st.number_input(
        "Prom. Goles Esperados Local (λ)", 0.5, 4.0, 1.6, 0.1
    )
    lambda_v = st.number_input(
        "Prom. Goles Esperados Visita (λ)", 0.5, 4.0, 1.1, 0.1
    )

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        urgencia_l = st.slider("Urgencia Local (1-5)", 1, 5, 5)
    with col_u2:
        urgencia_v = st.slider("Urgencia Visita (1-5)", 1, 5, 3)

    st.subheader("🚨 Filtros de Impacto")
    vino_de_golear = st.checkbox("Local viene de golear (3+ goles)", value=True)
    desgaste_logistico = st.checkbox(
        "Problemas de viaje/vuelos en la Visita", value=True
    )

# =========================================================
# 4. PROCESAMIENTO Y LÓGICA DE LAS 4 CAPAS
# =========================================================
if st.button("🚀 Calcular Predicción Híbrida"):

    # Evaluaciones condicionales
    try:
        hora_dec = int(hora.split(":")[0]) + int(hora.split(":")[1]) / 60.0
    except Exception:
        hora_dec = 15.0

    es_calor_extremo = 1 if (plaza in PLAZAS_CALOR and 13.0 <= hora_dec <= 15.5) else 0
    es_sintetica = 1 if tipo_cancha == "Sintético" else 0
    es_goleada = 1 if vino_de_golear else 0
    es_desgaste = 1 if desgaste_logistico else 0

    # 1. Cálculo por Poisson
    p_loc_poi, p_emp_poi, p_vis_poi, p_over_poi, p_bts_poi = calcular_poisson(
        lambda_l, lambda_v
    )

    # 2. Cálculo por Regresión Logística
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

    # 3. Ensamble Híbrido (Ponderación 50/50)
    prob_1x_final = ((p_loc_poi + p_emp_poi) + p_1x2_log) / 2.0
    prob_bts_final = (p_bts_poi + p_bts_log) / 2.0

    # Output Visual en Streamlit
    st.markdown("---")
    st.subheader(f"📊 INFORME DETALLADO: {local} vs {visita}")
    st.markdown(
        f"📅 **Fecha:** {fecha.strftime('%d/%m/%Y')} | ⏰ **Hora:** {hora} hrs ({plaza}) | 🏟️ **Césped:** {tipo_cancha}"
    )

    # Alertas visuales
    if es_calor_extremo:
        st.error(
            "⚠️ **FILTRO CLIMÁTICO ACTIVO:** Calor Extremo (3:00 p.m.). El ritmo del partido tiende a aletargarse en la segunda mitad."
        )
    if es_goleada:
        st.warning(
            "⚠️ **ALERTA DE REGRESIÓN A LA MEDIA:** El local viene de golear. Bajan las probabilidades de un marcador abultado por planteamiento precavido del rival."
        )
    if es_sintetica:
        st.info(
            "🏟️ **FACTOR CANCHA:** Césped sintético. Pica más rápido el balón y genera bote irregular para la visita."
        )

    # Renderizado de Capas
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("### 🛡️ Capa 1: Presión por Objetivos (Tabla)")
        st.write(
            f"* **Nivel de Urgencia Local:** {urgencia_l}/5 {'🔥 (Pelea Arriba)' if urgencia_l >=4 else ''}"
        )
        st.write(
            f"* **Nivel de Urgencia Visita:** {urgencia_v}/5 {'🔥 (Pelea Arriba)' if urgencia_v >=4 else ''}"
        )

        st.markdown("### 📈 Capa 2: Forma, Cancha y Logística")
        st.write(
            f"* **Racha Local:** {'Viene de golear (Posible desaceleración)' if es_goleada else 'Forma estándar'}"
        )
        st.write(
            f"* **Estado de la Visita:** {'Desgaste crítico por vuelos/traslado' if es_desgaste else 'Traslado normal'}"
        )

    with col_c2:
        st.markdown("### 🧮 Capa 3: Métricas de Modelos Híbridos")
        st.write(f"* **Poisson (1X):** {((p_loc_poi + p_emp_poi)*100):.1f}%")
        st.write(f"* **Regresión Logística (1X):** {(p_1x2_log*100):.1f}%")
        st.write(f"* **Poisson (Ambos Anotan):** {(p_bts_poi*100):.1f}%")
        st.write(
            f"* **Regresión Logística (Ambos Anotan):** {(p_bts_log*100):.1f}%"
        )

    # CAPA 4: SUGERENCIA FINAL
    st.markdown("---")
    st.markdown("### 🧠 Capa 4: Sugerencia Final del Sistema")

    # Regla de decisión para sugerencia
    if es_calor_extremo and es_goleada:
        rec_resultado = f"**Doble Oportunidad: {local} o Empate (1X)** o Victoria ajustada del Local por 1 gol."
        rec_goles = "**Menos de 2.5 Goles / Ambos Anotan: NO** (Ajuste por desaceleración climática y repliegue rival)."
    elif prob_1x_final > 0.65:
        rec_resultado = f"**Gana {local} Seco** o **1X** (Alta coincidencia de modelos)."
        rec_goles = (
            "**Más de 1.5 Goles**"
            if prob_bts_final > 0.5
            else "**Ambos Anotan: NO**"
        )
    else:
        rec_resultado = "**Partido Reservado / Doble Oportunidad Visita o Empate**"
        rec_goles = "**Menos de 2.5 Goles**"

    st.success(f"🎯 **Sugerencia de Resultado:** {rec_resultado}")
    st.info(f"⚽ **Sugerencia de Goles:** {rec_goles}")
