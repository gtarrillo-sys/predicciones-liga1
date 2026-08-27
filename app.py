import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# ==========================================
# 1. BASE DE DATOS DE EQUIPOS
# ==========================================
@st.cache_data
def cargar_y_calcular_estadisticas():
    db_dinamica = {
        'Alianza Lima':       {'PJ_L': 5, 'GF_L': 2.10, 'GC_L': 0.60, 'PJ_V': 5, 'GF_V': 1.50, 'GC_V': 0.90},
        'Universitario':      {'PJ_L': 5, 'GF_L': 2.40, 'GC_L': 0.40, 'PJ_V': 5, 'GF_V': 1.30, 'GC_V': 0.70},
        'Sporting Cristal':   {'PJ_L': 5, 'GF_L': 3.00, 'GC_L': 0.80, 'PJ_V': 5, 'GF_V': 1.80, 'GC_V': 1.10},
        'Melgar':             {'PJ_L': 5, 'GF_L': 2.20, 'GC_L': 0.70, 'PJ_V': 5, 'GF_V': 1.20, 'GC_V': 1.30},
        'Cienciano':          {'PJ_L': 5, 'GF_L': 1.40, 'GC_L': 1.10, 'PJ_V': 5, 'GF_V': 1.10, 'GC_V': 1.40},
        'Cusco FC':           {'PJ_L': 5, 'GF_L': 1.70, 'GC_L': 0.90, 'PJ_V': 5, 'GF_V': 0.90, 'GC_V': 1.60},
        'ADT Tarma':          {'PJ_L': 5, 'GF_L': 2.00, 'GC_L': 0.70, 'PJ_V': 5, 'GF_V': 0.80, 'GC_V': 1.60},
        'Sport Huancayo':     {'PJ_L': 5, 'GF_L': 1.50, 'GC_L': 1.20, 'PJ_V': 5, 'GF_V': 0.60, 'GC_V': 2.20},
        'Atlético Grau':      {'PJ_L': 5, 'GF_L': 1.40, 'GC_L': 0.80, 'PJ_V': 5, 'GF_V': 1.50, 'GC_V': 1.00},
        'Los Chankas':        {'PJ_L': 5, 'GF_L': 1.80, 'GC_L': 1.00, 'PJ_V': 5, 'GF_V': 0.70, 'GC_V': 2.00},
        'Comerciantes Unidos':{'PJ_L': 5, 'GF_L': 1.30, 'GC_L': 1.40, 'PJ_V': 5, 'GF_V': 1.00, 'GC_V': 2.30},
        'Sport Boys':         {'PJ_L': 5, 'GF_L': 1.10, 'GC_L': 1.30, 'PJ_V': 5, 'GF_V': 0.60, 'GC_V': 2.00},
        'UTC Cajamarca':      {'PJ_L': 5, 'GF_L': 1.40, 'GC_L': 1.20, 'PJ_V': 5, 'GF_V': 0.70, 'GC_V': 1.90},
        'Carlos A. Mannucci': {'PJ_L': 5, 'GF_L': 1.00, 'GC_L': 1.80, 'PJ_V': 5, 'GF_V': 0.80, 'GC_V': 2.40},
        'César Vallejo':      {'PJ_L': 5, 'GF_L': 1.20, 'GC_L': 1.30, 'PJ_V': 5, 'GF_V': 0.70, 'GC_V': 1.80},
        'Alianza Atlético':   {'PJ_L': 5, 'GF_L': 1.10, 'GC_L': 1.00, 'PJ_V': 5, 'GF_V': 0.50, 'GC_V': 1.70},
        'Unión Comercio':     {'PJ_L': 5, 'GF_L': 2.10, 'GC_L': 2.10, 'PJ_V': 5, 'GF_V': 0.60, 'GC_V': 2.50},
        'Deportivo Garcilaso':{'PJ_L': 5, 'GF_L': 1.30, 'GC_L': 1.20, 'PJ_V': 5, 'GF_V': 0.80, 'GC_V': 1.90}
    }
    prom_gf_l = sum(e['GF_L'] for e in db_dinamica.values()) / len(db_dinamica)
    prom_gc_l = sum(e['GC_L'] for e in db_dinamica.values()) / len(db_dinamica)
    return db_dinamica, prom_gf_l, prom_gc_l

db_equipos, prom_gf_l, prom_gc_l = cargar_y_calcular_estadisticas()

# ==========================================
# 2. CALENDARIO DE PARTIDOS (CLAUSURA)
# ==========================================
# Aquí puedes ir expandiendo o cambiando los partidos de cada fecha fácilmente
calendario_fechas = {
    'Fecha 7': [
        ('Sporting Cristal', 'Alianza Lima'),
        ('Universitario', 'Melgar'),
        ('Cienciano', 'Atlético Grau'),
        ('Sport Boys', 'Cusco FC'),
        ('ADT Tarma', 'Los Chankas')
    ],
    'Fecha 8': [
        ('Alianza Lima', 'Cienciano'),
        ('Sporting Cristal', 'Melgar'),
        ('Universitario', 'Sport Boys'),
        ('Atlético Grau', 'Los Chankas'),
        ('Cusco FC', 'ADT Tarma')
    ],
    'Fecha 9': [
        ('Los Chankas', 'Alianza Lima'),
        ('Melgar', 'Universitario'),
        ('Cienciano', 'Sporting Cristal'),
        ('Sport Boys', 'Atlético Grau'),
        ('ADT Tarma', 'Sport Huancayo')
    ]
}

# ==========================================
# 3. INTERFAZ VISUAL DE STREAMLIT
# ==========================================
st.set_page_config(page_title="Predicciones Liga 1", page_icon="🏆", layout="centered")

st.title("🏆 Sistema de Predicciones - Liga 1 Clausura")
st.write("Selecciona una jornada y haz clic en el partido para ver la predicción.")

# Paso 1: Seleccionar Fecha
jornada_seleccionada = st.selectbox("📅 Elige la Fecha del Torneo:", list(calendario_fechas.keys()))

# Paso 2: Mostrar partidos de esa fecha
partidos_disponibles = calendario_fechas[jornada_seleccionada]
opciones_partidos = [f"{p[0]} vs {p[1]}" for p in partidos_disponibles]

partido_elegido = st.selectbox("⚽ Selecciona el Partido:", opciones_partidos)

# Encontrar qué equipos juegan según la opción seleccionada
index_partido = opciones_partidos.index(partido_elegido)
local, visita = partidos_disponibles[index_partido]

# ==========================================
# 4. CALCULO MATEMÁTICO DE POISSON
# ==========================================
st.markdown("---")
st.subheader(f"📊 Análisis: {local} (Local) vs {visita} (Visitante)")

# Fórmulas de Fuerza
fuerza_of_l = db_equipos[local]['GF_L'] / prom_gf_l
fuerza_def_v = db_equipos[visita]['GC_V'] / prom_gf_l
lambda_local = fuerza_of_l * fuerza_def_v * prom_gf_l

fuerza_of_v = db_equipos[visita]['GF_V'] / prom_gc_l
fuerza_def_l = db_equipos[local]['GC_L'] / prom_gc_l
lambda_visita = fuerza_of_v * fuerza_def_l * prom_gc_l

# Mostrar métricas estéticas
col1, col2 = st.columns(2)
with col1:
    st.metric(f"Goles Esperados {local}", round(lambda_local, 2))
with col2:
    st.metric(f"Goles Esperados {visita}", round(lambda_visita, 2))

# Cálculo rápido de probabilidades simples
prob_local_anote = (1 - poisson.pmf(0, lambda_local)) * 100
prob_visita_anote = (1 - poisson.pmf(0, lambda_visita)) * 100

st.info(f"💡 Probabilidad de que {local} anote al menos un gol: **{round(prob_local_anote, 1)}%**\n\n"
        f"💡 Probabilidad de que {visita} anote al menos un gol: **{round(prob_visita_anote, 1)}%**")
