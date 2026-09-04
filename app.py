import datetime
import math
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# 1. CONFIGURACIÓN Y LECTURA TOLERANTE DEL EXCEL
# =========================================================
st.set_page_config(
    page_title="Predicciones Liga 1 - Motor Vinculado a Excel",
    page_icon="⚽",
    layout="wide",
)

PLAZAS_CALOR = ["Piura", "Sullana", "Tarapoto", "Chiclayo", "Iquitos"]
PLAZAS_SINTETICAS = [
    "Juliaca",
    "Andahuaylas",
    "Trujillo (Mansiche)",
    "Nueva Cajamarca",
    "Cajamarca",
]


@st.cache_data
def cargar_datos_excel():
    """Lee el archivo liga1_data.xlsx desde el repositorio."""
    try:
        df = pd.read_excel("liga1_data.xlsx", sheet_name="Fecha8")
    except Exception:
        df = pd.read_excel("liga1_data.xlsx")

    # Limpiar espacios en blanco de los nombres de columnas
    df.columns = df.columns.astype(str).str.strip()
    return df


try:
    df_fecha8 = cargar_datos_excel()
    excel_conectado = True
except Exception as e:
    excel_conectado = False


def obtener_columna(df, posibles_nombres, valor_defecto=None):
    """Busca una columna entre varias opciones posibles para evitar KeyError."""
    for nombre in posibles_nombres:
        for col in df.columns:
            if col.lower() == nombre.lower():
                return df[col]
    if valor_defecto is not None:
        return pd.Series([valor_defecto] * len(df))
    return None


# =========================================================
# 2. MOTORES MATEMÁTICOS (POISSON + SIGMOIDE NATIVA)
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
# 3. INTERFAZ EN STREAMLIT
# =========================================================
st.title("⚽ Predicciones Liga 1 - Motor Vinculado a Excel")

if excel_conectado:
    st.success("🟢 **Base de datos `liga1_data.xlsx` conectada con éxito.**")

    # Mapeo flexible de columnas
    col_local = obtener_columna(
        df_fecha8, ["Local", "Equipo Local", "Equipo_Local", "Home"]
    )
    col_visita = obtener_columna(
        df_fecha8,
        ["Visita", "Visitante", "Equipo Visita", "Equipo_Visitante", "Away"],
    )

    if col_local is not None and col_visita is not None:
        df_fecha8["Partido"] = (
            col_local.astype(str) + " vs " + col_visita.astype(str)
        )
    else:
        df_fecha8["Partido"] = [
            f"Partido {i+1}" for i in range(len(df_fecha8))
        ]

    with st.sidebar:
        st.header("🗓️ Selección desde Excel")
        partido_sel = st.selectbox(
            "Elegir Duelo", df_fecha8["Partido"].unique()
        )

        # Fila del partido seleccionado
        idx = df_fecha8[df_fecha8["Partido"] == partido_sel].index[0]
        row = df_fecha8.iloc[idx]

        st.markdown("---")
        st.header("⚙️ Parámetros Extraídos")

        val_local = str(
            row.get(
                "Local",
                row.get(
                    "Equipo Local", row.get("Equipo_Local", "Equipo Local")
                ),
            )
        )
        val_visita = str(
            row.get(
                "Visita",
                row.get(
                    "Visitante",
                    row.get(
                        "Equipo Visita",
                        row.get("Equipo_Visitante", "Equipo Visitante"),
                    ),
                ),
            )
        )

        local = st.text_input("Local", val_local)
        visita = st.text_input("Visita", val_visita)
        hora = st.text_input(
            "Hora (HH:MM)",
            str(row.get("Hora", row.get("HORA", row.get("Time", "15:00")))),
        )
        plaza = st.text_input(
            "Ciudad / Plaza",
            str(
                row.get(
                    "Plaza", row.get("Ciudad", row.get("PLAZA", "Lima"))
                )
            ),
        )

        val_cancha = str(
            row.get("Cancha", row.get("Gramado", row.get("CANCHA", "Natural")))
        )
        tipo_cancha = st.selectbox(
            "Gramado",
            ["Natural", "Sintético"],
            index=1 if "sint" in val_cancha.lower() else 0,
        )

        lambda_l = st.number_input(
            "Prom. Goles Local (λ)",
            0.5,
            4.0,
            float(
                row.get(
                    "Lambda_Local",
                    row.get(
                        "Prom_Goles_Local", row.get("Lambda Local", 1.5)
                    ),
                )
            ),
            0.1,
        )
        lambda_v = st.number_input(
            "Prom. Goles Visita (λ)",
            0.5,
            4.0,
            float(
                row.get(
                    "Lambda_Visita",
                    row.get(
                        "Prom_Goles_Visita", row.get("Lambda Visita", 1.0)
                    ),
                )
            ),
            0.1,
        )

        urgencia_l = st.slider(
            "Urgencia Local",
            1,
            5,
            int(
                row.get(
                    "Urgencia_Local", row.get("Urgencia Local", 3)
                )
            ),
        )
        urgencia_v = st.slider(
            "Urgencia Visita",
            1,
            5,
            int(
                row.get(
                    "Urgencia_Visita", row.get("Urgencia Visita", 3)
                )
            ),
        )

        vino_de_golear = st.checkbox(
            "Local viene de golear",
            value=bool(
                row.get(
                    "Goleada_Previa", row.get("Goleada Previa", 0)
                )
            ),
        )
        desgaste_logistico = st.checkbox(
            "Alerta de viaje en la Visita",
            value=bool(
                row.get(
                    "Desgaste_Viaje", row.get("Desgaste Viaje", 0)
                )
            ),
        )

    # =========================================================
    # 4. PROCESAMIENTO Y REPORTE
    # =========================================================
    try:
        hora_dec = int(str(hora).split(":")[0]) + int(str(hora).split(":")[1]) / 60.0
    except Exception:
        hora_dec = 15.0

    es_calor_extremo = (
        1 if (plaza in PLAZAS_CALOR and 13.0 <= hora_dec <= 15.5) else 0
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

    st.markdown("---")
    st.subheader(f"📊 INFORME DETALLADO: {local} vs {visita}")
    st.markdown(
        f"⏰ **Hora:** {hora} hrs ({plaza}) | 🏟️ **Césped:** {tipo_cancha}"
    )

    if es_calor_extremo:
        st.error(
            "⚠️ **FILTRO CLIMÁTICO ACTIVO:** Calor Extremo (3:00 p.m.). El ritmo tiende a aletargarse en el 2do tiempo."
        )
    if es_goleada:
        st.warning(
            "⚠️ **ALERTA DE REGRESIÓN A LA MEDIA:** El local viene de golear. Bajan las expectativas de un marcador abultado."
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
        "🔴 **No se encontró el archivo `liga1_data.xlsx` en el repositorio.** Sube el archivo a GitHub para activar la lectura automática."
    )
