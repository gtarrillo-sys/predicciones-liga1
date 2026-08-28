import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# Configuración de cabeceras para la API (Mantenemos la misma estructura por si deseas vincularla)
API_KEY = "cee4cebcd9msh82842660a6a542cp1710c9jsnfc3990e9eac0"
API_HOST = "free-api-live-football-data.p.rapidapi.com"

headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

LIGA1_LEAGUE_ID = 281  # ID referencial para Perú
TEMPORADA_ACTUAL = 2026

# =========================================================
# BASE DE DATOS CALIBRADA: CONTEXTO GEOGRÁFICO LIGA 1
# =========================================================
def obtener_base_respaldo_peru():
    # GF_L/GC_L y GF_V/GC_V calibrados según la tendencia histórica del fútbol peruano
    # Geografía: Costa, Altura, Selva
    # Acceso: Normal, Dificil (Tramos largos de bus / Sin aeropuerto directo)
    return {
        'Universitario': {'PJ_L': 8, 'GF_L': 2.40, 'GC_L': 0.50, 'PJ_V': 8, 'GF_V': 1.30, 'GC_V': 0.90, 'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal'},
        'Alianza Lima': {'PJ_L': 8, 'GF_L': 2.20, 'GC_L': 0.60, 'PJ_V': 8, 'GF_V': 1.40, 'GC_V': 0.85, 'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal'},
        'Sporting Cristal': {'PJ_L': 8, 'GF_L': 2.80, 'GC_L': 0.80, 'PJ_V': 8, 'GF_V': 1.50, 'GC_V': 1.10, 'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal'},
        'Melgar': {'PJ_L': 8, 'GF_L': 2.10, 'GC_L': 0.70, 'PJ_V': 8, 'GF_V': 1.10, 'GC_V': 1.20, 'Geografia': 'Altura', 'Pasto': 'Natural', 'Acceso': 'Normal'},
        'Cienciano': {'PJ_L': 8, 'GF_L': 1.80, 'GC_L': 0.85, 'PJ_V': 8, 'GF_V': 0.90, 'GC_V': 1.40, 'Geografia': 'Altura', 'Pasto': 'Natural', 'Acceso': 'Normal'},
        'ADT Tarma': {'PJ_L': 8, 'GF_L': 1.95, 'GC_L': 0.65, 'PJ_V': 8, 'GF_V': 0.80, 'GC_V': 1.60, 'Geografia': 'Altura', 'Pasto': 'Union', 'Acceso': 'Dificil'},
        'Comerciantes Unidos': {'PJ_L': 8, 'GF_L': 1.60, 'GC_L': 1.10, 'PJ_V': 8, 'GF_V': 0.85, 'GC_V': 1.90, 'Geografia': 'Altura', 'Pasto': 'Sintetico', 'Acceso': 'Dificil'},
        'Los Chankas': {'PJ_L': 8, 'GF_L': 2.00, 'GC_L': 0.90, 'PJ_V': 8, 'GF_V': 0.70, 'GC_V': 1.80, 'Geografia': 'Altura', 'Pasto': 'Natural', 'Acceso': 'Dificil'},
        'Sport Huancayo': {'PJ_L': 8, 'GF_L': 1.70, 'GC_L': 0.80, 'PJ_V': 8, 'GF_V': 0.75, 'GC_V': 1.70, 'Geografia': 'Altura', 'Pasto': 'Natural', 'Acceso': 'Normal'},
        'Cusco FC': {'PJ_L': 8, 'GF_L': 1.90, 'GC_L': 0.75, 'PJ_V': 8, 'GF_V': 0.80, 'GC_V': 1.50, 'Geografia': 'Altura', 'Pasto': 'Natural', 'Acceso': 'Normal'},
        'Atlético Grau': {'PJ_L': 8, 'GF_L': 1.65, 'GC_L': 0.80, 'PJ_V': 8, 'GF_V': 1.00, 'GC_V': 1.30, 'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal'},
        'Alianza Atlético': {'PJ_L': 8, 'GF_L': 1.40, 'GC_L': 0.90, 'PJ_V': 8, 'GF_V': 0.75, 'GC_V': 1.50, 'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal'},
        'UTC Cajamarca': {'PJ_L': 8, 'GF_L': 1.60, 'GC_L': 1.00, 'PJ_V': 8, 'GF_V': 0.70, 'GC_V': 1.85, 'Geografia': 'Altura', 'Pasto': 'Sintetico', 'Acceso': 'Normal'},
        'Carlos A. Mannucci': {'PJ_L': 8, 'GF_L': 1.30, 'GC_L': 1.40, 'PJ_V': 8, 'GF_V': 0.80, 'GC_V': 2.10, 'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal'},
        'César Vallejo': {'PJ_L': 8, 'GF_L': 1.50, 'GC_L': 1.10, 'PJ_V': 8, 'GF_V': 0.85, 'GC_V': 1.75, 'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal'},
        'Sport Boys': {'PJ_L': 8, 'GF_L': 1.45, 'GC_L': 1.20, 'PJ_V': 8, 'GF_V': 0.70, 'GC_V': 2.00, 'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal'},
        'Unión Comercio': {'PJ_L': 8, 'GF_L': 1.35, 'GC_L': 1.80, 'PJ_V': 8, 'GF_V': 0.80, 'GC_V': 2.50, 'Geografia': 'Selva', 'Pasto': 'Natural', 'Acceso': 'Dificil'},
        'Deportivo Garcilaso': {'PJ_L': 8, 'GF_L': 1.75, 'GC_L': 1.00, 'PJ_V': 8, 'GF_V': 0.85, 'GC_V': 1.65, 'Geografia': 'Altura', 'Pasto': 'Natural', 'Acceso': 'Normal'}
    }

# =========================================================
# OBTENER FIXTURE DE LA JORNADA PERUANA
# =========================================================
def obtener_partidos_peru():
    # Cartelera de partidos ajustable por fecha para análisis directo
    return [
        {"local": "Comerciantes Unidos", "visita": "Universitario", "fecha": "Sábado", "hora": "15:30"},
        {"local": "Alianza Lima", "visita": "ADT Tarma", "fecha": "Sábado", "hora": "20:00"},
        {"local": "Melgar", "visita": "Sporting Cristal", "fecha": "Domingo", "hora": "13:00"},
        {"local": "Los Chankas", "visita": "Sport Boys", "fecha": "Domingo", "hora": "15:30"},
        {"local": "UTC Cajamarca", "visita": "César Vallejo", "fecha": "Lunes", "hora": "15:00"},
        {"local": "Cusco FC", "visita": "Alianza Atlético", "fecha": "Lunes", "hora": "18:00"}
    ]

# =========================================================
# INTERFAZ GRÁFICA DE STREAMLIT
# =========================================================
st.set_page_config(page_title="Liga 1 Context Predictor", page_icon="🇵🇪", layout="centered")
st.title("🇵🇪 LIGA 1 — ANALIZADOR GEOGRÁFICO AISLADO")
st.caption("Predicciones optimizadas para el fútbol peruano: Altura, efecto viaje en bus y césped.")

db_equipos = obtener_base_respaldo_peru()
partidos_jornada = obtener_partidos_peru()

tab1, tab2 = st.tabs(["📅 Predicciones de la Fecha", "🏔️ Matriz de Climas"])

with tab1:
    for partido in partidos_jornada:
        local, visita = partido['local'], partido['visita']
        if local not in db_equipos or visita not in db_equipos: continue
            
        factor_ataque_visita = 1.0
        factor_defensa_local = 1.0
        alertas = []
        
        # 1. EL FACTOR REY: SHOCK DE ALTURA / OXÍGENO
        geo_l = db_equipos[local]['Geografia']
        geo_v = db_equipos[visita]['Geografia']
        
        if geo_l == 'Altura' and geo_v == 'Costa':
            # Equipo de la Costa sufre un bajón físico severo en la altura (Cusco, Tarma, Cutervo, etc.)
            factor_ataque_visita *= 0.78  # Pierde un 22% de capacidad ofensiva por falta de aire
            alertas.append(f"🏔️🥵 **Shock Hipóxico (Altura):** {visita} (Costa) sube a la altura de {local}. Ventaja física local determinante.")
        elif geo_l == 'Costa' and geo_v == 'Altura':
            # Los equipos de altura perdiendo el plus geográfico al bajar al llano
            factor_ataque_visita *= 0.92  # Pierden ligera adaptabilidad en velocidad de balón
            alertas.append(f"🌊 **Bajada al Llano:** {visita} pierde su ventaja de altura al jugar en la costa frente a {local}.")
        elif geo_l == 'Selva' and geo_v == 'Costa':
            # El factor del calor sofocante e intenso de la selva (Tarapoto)
            factor_ataque_visita *= 0.85
            alertas.append(f"🌴☀️ **Calor de Selva:** {visita} expuesto a la alta temperatura y humedad de {local}.")

        # 2. EL EFECTO VIAJE EN CARRETERA (Caso Cutervo, Andahuaylas, Tarma)
        if db_equipos[local]['Acceso'] == 'Dificil' and db_equipos[visita]['Acceso'] == 'Normal':
            factor_ataque_visita *= 0.90  # 10% de penalización por acumulación de horas en bus
            alertas.append(f"🚌 **Desgaste por Carretera (Efecto Bus):** Ruta compleja de acceso para {visita} para llegar a {local}.")

        # 3. VARIABLE DE SUPERFICIE (Césped Sintético)
        if db_equipos[local]['Pasto'] == 'Sintetico':
            factor_defensa_local *= 0.90  # El local domina el rebote rápido del balón plástico
            alertas.append("👟 **Césped Artificial:** Ventaja táctica local por control del bote rápido.")

        # =========================================================
        # CÁLCULO DIRECTO AISLADO (SIN PROMEDIOS DE LIGA)
        # =========================================================
        lambda_local = (db_equipos[local]['GF_L'] + db_equipos[visita]['GC_V']) / 2.0
        lambda_local = lambda_local / factor_defensa_local  # Potencia al local
        
        lambda_visita = (db_equipos[visita]['GF_V'] + db_equipos[local]['GC_L']) / 2.0
        lambda_visita = lambda_visita * factor_ataque_visita  # Castiga a la visita por geografía

        # Distribución de Poisson para el cálculo de probabilidades
        p_local = [poisson.pmf(i, lambda_local) for i in range(6)]
        p_visita = [poisson.pmf(i, lambda_visita) for i in range(6)]
        
        prob_l, prob_e, prob_v, prob_under = 0.0, 0.0, 0.0, 0.0
        for i in range(6):
            for j in range(6):
                p_comb = p_local[i] * p_visita[j]
                if i > j: prob_l += p_comb
                elif i == j: prob_e += p_comb
                else: prob_v += p_comb
                if (i + j) < 3: prob_under += p_comb
        
        total_p = prob_l + prob_e + prob_v
        pct_l = round((prob_l / total_p) * 100, 1)
        pct_e = round((prob_e / total_p) * 100, 1)
        pct_v = round((prob_v / total_p) * 100, 1)
        pct_over = round((1.0 - prob_under) * 100, 1)
        pct_under_f = round(prob_under * 100, 1)
        
        # Despliegue visual de la tarjeta del partido
        with st.container(border=True):
            st.caption(f"📅 {partido['fecha']} — ⏰ {partido['hora']}")
            titulo_fija = " 🔥 FIJA DE VALOR" if pct_l >= 70.0 or pct_v >= 70.0 else ""
            st.markdown(f"### 🏟️ {local} vs {visita}{titulo_fija}")
            
            for a in alertas: 
                st.markdown(a)
            
            st.markdown("**📊 Probabilidades de Resultado:**")
            col1, col2, col3 = st.columns(3)
            col1.success(f"🟢 Local: {pct_l}%")
            col2.warning(f"🟡 Empate: {pct_e}%")
            col3.info(f"🔵 Visita: {pct_v}%")
            
            st.markdown("**⚽ Goles (Línea de 2.5):**")
            col_g1, col_g2 = st.columns(2)
            col_g1.markdown(f"📈 **Más de 2.5 (Over):** {pct_over}%" + (" 🔥" if pct_over >= 70 else ""))
            col_g2.markdown(f"📉 **Menos de 2.5 (Under):** {pct_under_f}%")

with tab2:
    st.info("### 🗺️ Configuración del Mapa Geográfico de la Liga 1")
    st.write("El sistema evalúa el desgaste físico según la localía de los clubes peruanos:")
    
    df_mostrar = pd.DataFrame.from_dict(db_equipos, orient='index')[['Geografia', 'Pasto', 'Acceso']]
    st.dataframe(df_mostrar, height=400)
