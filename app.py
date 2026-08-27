import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# ==========================================
# 1. BASE DE DATOS DE LOS 18 EQUIPOS OFICIALES
# ==========================================
@st.cache_data
def cargar_y_calcular_estadisticas():
    db_dinamica = {
        'Alianza Lima':        {'PJ_L': 5, 'GF_L': 2.10, 'GC_L': 0.60, 'PJ_V': 5, 'GF_V': 1.50, 'GC_V': 0.90, 'Region': 'Costa'},
        'Universitario':       {'PJ_L': 5, 'GF_L': 2.40, 'GC_L': 0.40, 'PJ_V': 5, 'GF_V': 1.30, 'GC_V': 0.70, 'Region': 'Costa'},
        'Sporting Cristal':    {'PJ_L': 5, 'GF_L': 3.00, 'GC_L': 0.80, 'PJ_V': 5, 'GF_V': 1.80, 'GC_V': 1.10, 'Region': 'Costa'},
        'Melgar':              {'PJ_L': 5, 'GF_L': 2.20, 'GC_L': 0.70, 'PJ_V': 5, 'GF_V': 1.20, 'GC_V': 1.30, 'Region': 'Altura'},
        'Cienciano':           {'PJ_L': 5, 'GF_L': 1.40, 'GC_L': 1.10, 'PJ_V': 5, 'GF_V': 1.10, 'GC_V': 1.40, 'Region': 'Altura'},
        'Cusco FC':            {'PJ_L': 5, 'GF_L': 1.70, 'GC_L': 0.90, 'PJ_V': 5, 'GF_V': 0.90, 'GC_V': 1.60, 'Region': 'Altura'},
        'ADT Tarma':           {'PJ_L': 5, 'GF_L': 2.00, 'GC_L': 0.70, 'PJ_V': 5, 'GF_V': 0.80, 'GC_V': 1.60, 'Region': 'Altura'},
        'Sport Huancayo':      {'PJ_L': 5, 'GF_L': 1.50, 'GC_L': 1.20, 'PJ_V': 5, 'GF_V': 0.60, 'GC_V': 2.20, 'Region': 'Altura'},
        'Atlético Grau':       {'PJ_L': 5, 'GF_L': 1.40, 'GC_L': 0.80, 'PJ_V': 5, 'GF_V': 1.50, 'GC_V': 1.00, 'Region': 'Norte'},
        'Los Chankas':         {'PJ_L': 5, 'GF_L': 1.80, 'GC_L': 1.00, 'PJ_V': 5, 'GF_V': 0.70, 'GC_V': 2.00, 'Region': 'Altura'},
        'Comerciantes Unidos': {'PJ_L': 5, 'GF_L': 1.30, 'GC_L': 1.40, 'PJ_V': 5, 'GF_V': 1.00, 'GC_V': 2.30, 'Region': 'Altura'},
        'Sport Boys':          {'PJ_L': 5, 'GF_L': 1.10, 'GC_L': 1.30, 'PJ_V': 5, 'GF_V': 0.60, 'GC_V': 2.00, 'Region': 'Costa'},
        'UTC':                 {'PJ_L': 5, 'GF_L': 1.40, 'GC_L': 1.20, 'PJ_V': 5, 'GF_V': 0.70, 'GC_V': 1.90, 'Region': 'Altura'},
        'FC Cajamarca':        {'PJ_L': 5, 'GF_L': 1.10, 'GC_L': 1.50, 'PJ_V': 5, 'GF_V': 0.60, 'GC_V': 2.10, 'Region': 'Altura'},
        'CD Moquegua':         {'PJ_L': 5, 'GF_L': 1.20, 'GC_L': 1.30, 'PJ_V': 5, 'GF_V': 0.70, 'GC_V': 1.80, 'Region': 'Costa'},
        'Alianza Atlético':    {'PJ_L': 5, 'GF_L': 1.10, 'GC_L': 1.00, 'PJ_V': 5, 'GF_V': 0.50, 'GC_V': 1.70, 'Region': 'Norte'},
        'Juan Pablo II':       {'PJ_L': 5, 'GF_L': 0.90, 'GC_L': 1.80, 'PJ_V': 5, 'GF_V': 0.50, 'GC_V': 2.40, 'Region': 'Norte'},
        'Deportivo Garcilaso': {'PJ_L': 5, 'GF_L': 1.30, 'GC_L': 1.20, 'PJ_V': 5, 'GF_V': 0.80, 'GC_V': 1.90, 'Region': 'Altura'}
    }
    prom_gf_l = sum(e['GF_L'] for e in db_dinamica.values()) / len(db_dinamica)
    prom_gc_l = sum(e['GC_L'] for e in db_dinamica.values()) / len(db_dinamica)
    return db_dinamica, prom_gf_l, prom_gc_l

db_equipos, prom_gf_l, prom_gc_l = cargar_y_calcular_estadisticas()

# ==========================================
# 2. CALENDARIO REAL DE PARTIDOS
# ==========================================
calendario_fechas = {
    'Jornada 7': [
        ('Comerciantes Unidos', 'FC Cajamarca'),
        ('Los Chankas', 'Juan Pablo II'),
        ('UTC', 'Universitario'),
        ('Alianza Lima', 'Deportivo Garcilaso'),
        ('CD Moquegua', 'Alianza Atlético'),
        ('ADT Tarma', 'Sport Huancayo'),
        ('Sport Boys', 'Sporting Cristal'),
        ('Cienciano', 'Cusco FC'),
        ('Atlético Grau', 'Melgar')
    ]
}

# ==========================================
# 3. INTERFAZ VISUAL NATIVA
# ==========================================
st.set_page_config(page_title="LIGA 1 - PREDICTOR ESTADÍSTICO", page_icon="🏆", layout="centered")

st.title("🏆 LIGA 1 - PREDICTOR ESTADÍSTICO")
st.write("Algoritmo predictivo con factores climáticos: Altura y Calor del Norte.")

jornada_seleccionada = st.selectbox("📅 Selecciona la Fecha del Torneo:", list(calendario_fechas.keys()))
st.markdown("---")

for local, visita in calendario_fechas[jornada_seleccionada]:
    
    reg_local = db_equipos[local]['Region']
    reg_visita = db_equipos[visita]['Region']
    
    # --- MODIFICADORES GEOGRÁFICOS ---
    factor_ataque_visita = 1.0
    factor_defensa_local = 1.0
    alerta_clima = ""

    if reg_local == 'Altura' and reg_visita != 'Altura':
        factor_ataque_visita = 0.75
        factor_defensa_local = 0.85
        alerta_clima = "🏔️ **Alerta:** Factor Altura activo"
    elif reg_local == 'Norte' and reg_visita != 'Norte':
        factor_ataque_visita = 0.85
        alerta_clima = "☀️ **Alerta:** Factor Calor del Norte activo"

    # --- CÁLCULOS MATEMÁTICOS DE POISSON ---
    fuerza_of_l = db_equipos[local]['GF_L'] / prom_gf_l
    fuerza_def_v = db_equipos[visita]['GC_V'] / prom_gf_l
    lambda_local = fuerza_of_l * fuerza_def_v * prom_gf_l * (1 / factor_defensa_local)
    
    fuerza_of_v = db_equipos[visita]['GF_V'] / prom_gc_l
    fuerza_def_l = db_equipos[local]['GC_L'] / prom_gc_l
    lambda_visita = fuerza_of_v * fuerza_def_l * prom_gc_l * factor_ataque_visita
    
    max_goles = 6
    p_local = [poisson.pmf(i, lambda_local) for i in range(max_goles)]
    p_visita = [poisson.pmf(i, lambda_visita) for i in range(max_goles)]
    
    prob_l, prob_e, prob_v = 0.0, 0.0, 0.0
    max_prob_marcador = -1
    marcador_exacto = (0, 0)
    
    for i in range(max_goles):
        for j in range(max_goles):
            p_combinada = p_local[i] * p_visita[j]
            if i > j: prob_l += p_combinada
            elif i == j: prob_e += p_combinada
            else: prob_v += p_combinada
            
            if p_combinada > max_prob_marcador:
                max_prob_marcador = p_combinada
                marcador_exacto = (i, j)
                
    total = prob_l + prob_e + prob_v
    pct_l = round((prob_l / total) * 100, 1)
    pct_e = round((prob_e / total) * 100, 1)
    pct_v = round((prob_v / total) * 100, 1)
    
    es_fija = pct_l >= 80.0 or pct_v >= 80.0
    
    # --- RENDERIZADO VISUAL ---
    with st.container(border=True):
        if es_fija:
            st.markdown(f"### 🏟️ {local} vs {visita} :orange[**🔥 FIJA**]")
        else:
            st.markdown(f"### 🏟️ {local} vs {visita}")
            
        if alerta_clima:
            st.caption(alerta_clima)
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"🟢 Local: {pct_l}%")
        with col2:
            st.warning(f"🟡 Empate: {pct_e}%")
        with col3:
            st.info(f"🔵 Visita: {pct_v}%")
            
        st.markdown(f"**Resultado calculated:** `{local} {marcador_exacto[0]} - {marcador_exacto[1]} {visita}`")
        st.write("")
