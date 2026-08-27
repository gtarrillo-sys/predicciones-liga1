import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# ==========================================
# 1. BASE DE DATOS COMPLETA (18 EQUIPOS)
# ==========================================
@st.cache_data
def cargar_y_calcular_estadisticas():
    # Diccionario oficial con todos los equipos de la Liga 1
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
        'Unión Comercio':     {'PJ_L': 5, 'GF_L': 1.20, 'GC_L': 2.10, 'PJ_V': 5, 'GF_V': 0.60, 'GC_V': 2.50},
        'Deportivo Garcilaso':{'PJ_L': 5, 'GF_L': 1.30, 'GC_L': 1.20, 'PJ_V': 5, 'GF_V': 0.80, 'GC_V': 1.90}
    }
    
    promedio_gf_local = sum(e['GF_L'] for e in db_dinamica.values()) / len(db_dinamica)
    promedio_gc_local = sum(e['GC_L'] for e in db_dinamica.values()) / len(db_dinamica)
    
    return db_dinamica, promedio_gf_local, promedio_gc_local

# ==========================================
# 2. PROCESAMIENTO MATEMÁTICO (POISSON)
# ==========================================
db_equipos, prom_gf_l, prom_gc_l = cargar_y_calcular_estadisticas()

st.set_page_config(page_title="Predicciones Liga 1", page_icon="🏆", layout="centered")

st.title("🏆 Sistema de Predicciones - Liga 1")
st.write("Cálculos matemáticos basados en el modelo de Poisson.")

# Selectores de equipos ordenados alfabéticamente
lista_equipos = sorted(list(db_equipos.keys()))
local = st.selectbox("Selecciona Equipo Local:", lista_equipos)
visita = st.selectbox("Selecciona Equipo Visitante:", lista_equipos)

if local != visita:
    # Fórmulas de Fuerza
    fuerza_of_l = db_equipos[local]['GF_L'] / prom_gf_l
    fuerza_def_v = db_equipos[visita]['GC_V'] / prom_gf_l
    lambda_local = fuerza_of_l * fuerza_def_v * prom_gf_l
    
    fuerza_of_v = db_equipos[visita]['GF_V'] / prom_gc_l
    fuerza_def_l = db_equipos[local]['GC_L'] / prom_gc_l
    lambda_visita = fuerza_of_v * fuerza_def_l * prom_gc_l
    
    # Diseño de métricas en pantalla
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Goles Esperados {local}", round(lambda_local, 2))
    with col2:
        st.metric(f"Goles Esperados {visita}", round(lambda_visita, 2))
else:
    st.warning("Por favor, selecciona dos equipos diferentes.")
