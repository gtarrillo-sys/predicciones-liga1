import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# ==========================================
# 1. BASE DE DATOS OPTIMIZADA
# ==========================================
@st.cache_data
def cargar_y_calcular_estadisticas():
    # Estructura limpia y directa para evitar bloqueos de red en el servidor
    db_dinamica = {
        'Sporting Cristal': {'PJ_L': 5, 'GF_L': 3.00, 'GC_L': 0.80, 'PJ_V': 5, 'GF_V': 1.80, 'GC_V': 1.10},
        'Sport Huancayo':   {'PJ_L': 5, 'GF_L': 1.50, 'GC_L': 1.20, 'PJ_V': 5, 'GF_V': 0.60, 'GC_V': 2.20},
        'Atlético Grau':    {'PJ_L': 5, 'GF_L': 1.40, 'GC_L': 0.80, 'PJ_V': 5, 'GF_V': 1.50, 'GC_V': 1.00},
        'Sport Boys':       {'PJ_L': 5, 'GF_L': 1.10, 'GC_L': 1.30, 'PJ_V': 5, 'GF_V': 0.60, 'GC_V': 2.00},
        'Universitario':    {'PJ_L': 5, 'GF_L': 2.40, 'GC_L': 0.40, 'PJ_V': 5, 'GF_V': 1.30, 'GC_V': 0.70},
        'Alianza Lima':     {'PJ_L': 5, 'GF_L': 2.10, 'GC_L': 0.60, 'PJ_V': 5, 'GF_V': 1.50, 'GC_V': 0.90},
        'ADT Tarma':        {'PJ_L': 5, 'GF_L': 2.00, 'GC_L': 0.70, 'PJ_V': 5, 'GF_V': 0.80, 'GC_V': 1.60},
        'CD Moquegua':      {'PJ_L': 5, 'GF_L': 1.00, 'GC_L': 1.20, 'PJ_V': 5, 'GF_V': 0.50, 'GC_V': 2.10}
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

# Selectores de equipos
local = st.selectbox("Selecciona Equipo Local:", list(db_equipos.keys()))
visita = st.selectbox("Selecciona Equipo Visitante:", list(db_equipos.keys()))

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
