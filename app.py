import datetime
import math
import numpy as np
import streamlit as st

# =========================================================
# 1. CONFIGURACIÓN GENERAL Y MAPEOS DE PLAZAS
# =========================================================
st.set_page_config(
    page_title="Predicciones Liga 1 - Fecha 8", page_icon="⚽", layout="wide"
)

PLAZAS_CALOR = ["Piura", "Sullana", "Tarapoto", "Chiclayo", "Iquitos"]
PLAZAS_SINTETICAS = [
    "Juliaca",
    "Andahuaylas",
    "Trujillo (Mansiche)",
    "Nueva Cajamarca",
    "Cajamarca",
]

# =========================================================
# 2. PROGRAMACIÓN OFICIAL Y BASE DE DATOS DE LA FECHA 8
# =========================================================
PARTIDOS_FECHA_8 = {
    "FC Cajamarca vs Cienciano": {
        "local": "FC Cajamarca",
        "visita": "Cienciano",
        "fecha": datetime.date(2026, 9, 4),
        "hora": "13:00",
        "plaza": "Cajamarca",
        "cancha": "Sintético",
        "lambda_l": 1.4,
        "lambda_v": 1.0,
        "urgencia_l": 4,
        "urgencia_v": 3,
        "goleada": False,
        "desgaste": False,
    },
    "Alianza Atlético vs UTC": {
        "local": "Alianza Atlético",
        "visita": "UTC",
        "fecha": datetime.date(2026, 9, 4),
        "hora": "15:15",
        "plaza": "Sullana",
        "cancha": "Natural",
        "lambda_l": 1.5,
        "lambda_v": 0.9,
        "urgencia_l": 3,
        "urgencia_v": 3,
        "goleada": False,
        "desgaste": True,
    },
    "Deporte Huancayo vs Sport Boys": {
        "local": "Deporte Huancayo",
        "visita": "Sport Boys",
        "fecha": datetime.date(2026, 9, 5),
        "hora": "15:00",
        "plaza": "Huancayo",
        "cancha": "Natural",
        "lambda_l": 1.7,
        "lambda_v": 0.8,
        "urgencia_l": 4,
        "urgencia_v": 2,
        "goleada": False,
        "desgaste": False,
    },
    "Cusco FC vs CD Moquegua": {
        "local": "Cusco FC",
        "visita": "CD Moquegua",
        "fecha": datetime.date(2026, 9, 5),
        "hora": "18:00",
        "plaza": "Cusco",
        "cancha": "Natural",
        "lambda_l": 2.0,
        "lambda_v": 0.7,
        "urgencia_l": 5,
        "urgencia_v": 2,
        "goleada": False,
        "desgaste": False,
    },
    "Universitario vs Comerciantes Unidos": {
        "local": "Universitario",
        "visita": "Comerciantes Unidos",
        "fecha": datetime.date(2026, 9, 5),
        "hora": "20:30",
        "plaza": "Lima",
        "cancha": "Natural",
        "lambda_l": 2.3,
        "lambda_v": 0.6,
        "urgencia_l": 5,
        "urgencia_v": 2,
        "goleada": True,
        "desgaste": False,
    },
    "Sporting Cristal vs Chankas CYC": {
        "local": "Sporting Cristal",
        "visita": "Chankas CYC",
        "fecha": datetime.date(2026, 9, 6),
        "hora": "11:00",
        "plaza": "Lima",
        "cancha": "Natural",
        "lambda_l": 2.1,
        "lambda_v": 0.9,
        "urgencia_l": 5,
        "urgencia_v": 3,
        "goleada": False,
        "desgaste": False,
    },
    "Deportivo Garcilaso vs Atlético Grau": {
        "local": "Deportivo Garcilaso",
        "visita": "Atlético Grau",
        "fecha": datetime.date(2026, 9, 6),
        "hora": "13:30",
        "plaza": "Cusco",
        "cancha": "Natural",
        "lambda_l": 1.5,
        "lambda_v": 0.9,
        "urgencia_l": 3,
        "urgencia_v": 4,
        "goleada": False,
        "desgaste": False,
    },
    "Colegio Juan Pablo II vs Alianza Lima": {
        "local": "Colegio Juan Pablo II",
        "visita": "Alianza Lima",
        "fecha": datetime.date(2026, 9, 6),
        "hora": "15:00",
        "plaza": "Chiclayo",
        "cancha": "Natural",
        "lambda_l": 0.9,
        "lambda_v": 1.8,
        "urgencia_l": 2,
        "urgencia_v": 5,
        "goleada": False,
        "desgaste": False,
    },
    "Melgar vs ADT": {
        "local": "Melgar",
        "visita": "ADT",
        "fecha": datetime.date(2026, 9, 6),
        "hora": "18:30",
        "plaza": "Arequipa",
        "cancha": "Natural",
        "lambda_l": 1.9,
        "lambda_v": 1.0,
        "urgencia_l": 4,
        "urgencia_v": 3,
        "goleada": False,
        "desgaste": False,
    },
}

# =========================================================
# 3. MOTORES MATEMÁTICOS (POISSON + LOGÍSTICA SIGMOIDE NATIVA)
# =========================================================


def calcular_poisson(lambda_local, lambda_visita, max_goles=5):
    """Calcula matriz de probabilidades exactas de gol usando la distribución de Poisson."""
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
    """Calcula la probabilidad logística mediante la curva sigmoide $1 / (1 + e^{-z})$."""
    return 1.0 / (1.0 + math.exp(-z))


def modelo_regresion_logistica(features):
    """
    Motor nativo de Regresión Logística calibrado para la Liga 1.
    Features: [lambda_l, lambda_v, clima_3pm, goleada, desgaste, sintetica, urg_l, urg_v]
    """
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

    # Ajuste de coeficientes ponderados para Victoria/Doble Oportunidad Local (z_1x2)
    z_1x2 = (
        0.20
        + (0.35 * (lambda_l - lambda_v))
        - (0.25 * clima_3pm)
        - (0.30 * goleada)
        + (0.45 * desgaste)
        + (0.40 * sintetica)
        + (0.15 * (urg_l - urg_v))
    )

    # Ajuste de coeficientes ponderados para Ambos Anotan / BTS (z_bts)
    z_bts = (
        -0.10
        + (0.25 * (lambda_l + lambda_v))
        - (0.50 * clima_3pm)
        - (0.35 * goleada)
        - (0.20 * desgaste)
    )

    return funcion_sigmoide(z_1x2), funcion_sigmoide(z_bts)


# =========================================================
# 4. INTERFAZ DE USUARIO Y CONTROL
# =========================================================
st.title("⚽ Predicciones Liga 1 - Programación Fecha 8")
st.markdown(
    "Selecciona cualquier partido de la jornada oficial para cargar sus parámetros de clima, canchas, logística y urgencia."
)

with st.sidebar:
    st.header("🗓️ Selección de Partido")
    partido_sel = st.selectbox("Elegir Duelo", list(PARTIDOS_FECHA_8.keys()))

    # Cargar diccionario según selección
    datos = PARTIDOS_FECHA_8[partido_sel]

    st.markdown("---")
    st.header("⚙️ Parámetros del Partido")
    local = st.text_input("Local", datos["local"])
    visita = st.text_input("Visita", datos["visita"])
    fecha = st.date_input("Fecha", datos["fecha"])
    hora = st.text_input("Hora (HH:MM)", datos["hora"])
    plaza = st.text_input("Ciudad / Plaza", datos["plaza"])
    tipo_cancha = st.selectbox(
        "Gramado",
        ["Natural", "Sintético"],
        index=0 if datos["cancha"] == "Natural" else 1,
    )

    lambda_l = st.number_input(
        "Prom. Goles Local (λ)", 0.5, 4.0, float(datos["lambda_l"]), 0.1
    )
    lambda_v = st.number_input(
        "Prom. Goles Visita (λ)", 0.5, 4.0, float(datos["lambda_v"]), 0.1
    )

    urgencia_l = st.slider("Urgencia Local", 1, 5, int(datos["urgencia_l"]))
    urgencia_v = st.slider("Urgencia Visita", 1, 5, int(datos["urgencia_v"]))

    vino_de_golear = st.checkbox(
        "Local viene de golear", value=bool(datos["goleada"])
    )
    desgaste_logistico = st.checkbox(
        "Alerta de viaje en la Visita", value=bool(datos["desgaste"])
    )

# =========================================================
# 5. EXECUCIÓN DEL MODELO Y GENERACIÓN DEL INFORME
# =========================================================
try:
    hora_dec = int(hora.split(":")[0]) + int(hora.split(":")[1]) / 60.0
except Exception:
    hora_dec = 15.0

es_calor_extremo = 1 if (plaza in PLAZAS_CALOR and 13.0 <= hora_dec <= 15.5) else 0
es_sintetica = 1 if tipo_cancha == "Sintético" else 0
es_goleada = 1 if vino_de_golear else 0
es_desgaste = 1 if desgaste_logistico else 0

# Procesar Poisson
p_loc_poi, p_emp_poi, p_vis_poi, p_bts_poi = calcular_poisson(lambda_l, lambda_v)

# Procesar Regresión Logística
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

# Ensamble Híbrido (Ponderado)
prob_1x_final = ((p_loc_poi + p_emp_poi) + p_1x2_log) / 2.0
prob_bts_final = (p_bts_poi + p_bts_log) / 2.0

# REPORTE VISUAL
st.markdown("---")
st.subheader(f"📊 INFORME DETALLADO: {local} vs {visita}")
st.markdown(
    f"📅 **Fecha:** {fecha.strftime('%d/%m/%Y')} | ⏰ **Hora:** {hora} hrs ({plaza}) | 🏟️ **Césped:** {tipo_cancha}"
)

# Render de alertas
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

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("### 🛡️ Capa 1: Presión por Objetivos (Tabla)")
    st.write(
        f"* **Nivel de Urgencia Local:** {urgencia_l}/5 {'🔥 (Pelea Título)' if urgencia_l >=4 else ''}"
    )
    st.write(
        f"* **Nivel de Urgencia Visita:** {urgencia_v}/5 {'🔥 (Pelea Título)' if urgencia_v >=4 else ''}"
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
    st.write(f"* **Regresión Logística (Ambos Anotan):** {(p_bts_log*100):.1f}%")

st.markdown("---")
st.markdown("### 🧠 Capa 4: Sugerencia Final del Sistema")

# Decisiones del sistema
if es_calor_extremo and es_goleada:
    rec_resultado = f"**Doble Oportunidad: {local} o Empate (1X)** o Victoria ajustada del Local por 1 gol."
    rec_goles = "**Menos de 2.5 Goles / Ambos Anotan: NO** (Ajuste por desaceleración climática y repliegue rival)."
elif prob_1x_final > 0.65:
    rec_resultado = f"**Gana {local} Seco** o **1X** (Alta coincidencia de modelos)."
    rec_goles = (
        "**Más de 1.5 Goles**" if prob_bts_final > 0.5 else "**Ambos Anotan: NO**"
    )
else:
    rec_resultado = "**Partido Reservado / Doble Oportunidad Visita o Empate**"
    rec_goles = "**Menos de 2.5 Goles**"

st.success(f"🎯 **Sugerencia de Resultado:** {rec_resultado}")
st.info(f"⚽ **Sugerencia de Goles:** {rec_goles}")
