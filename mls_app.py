import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# ==========================================
# 1. BASE DE DATOS MLS (MÉTRICAS Y GEOGRAFÍA)
# ==========================================
@st.cache_data
def cargar_estadisticas_mls():
    db_mls = {
        'Atlanta United':    {'PJ_L': 6, 'GF_L': 1.70, 'GC_L': 1.20, 'PJ_V': 6, 'GF_V': 1.10, 'GC_V': 1.60, 'Conf': 'Este',  'Pasto': 'Sintetico'},
        'Austin FC':         {'PJ_L': 6, 'GF_L': 1.50, 'GC_L': 1.10, 'PJ_V': 6, 'GF_V': 1.00, 'GC_V': 1.50, 'Conf': 'Oeste', 'Pasto': 'Natural'},
        'Charlotte FC':      {'PJ_L': 6, 'GF_L': 1.30, 'GC_L': 0.85, 'PJ_V': 6, 'GF_V': 1.00, 'GC_V': 1.40, 'Conf': 'Este',  'Pasto': 'Sintetico'},
        'Chicago Fire':      {'PJ_L': 6, 'GF_L': 1.40, 'GC_L': 1.50, 'PJ_V': 6, 'GF_V': 1.10, 'GC_V': 1.80, 'Conf': 'Este',  'Pasto': 'Natural'},
        'Colorado Rapids':   {'PJ_L': 6, 'GF_L': 2.00, 'GC_L': 1.20, 'PJ_V': 6, 'GF_V': 1.20, 'GC_V': 1.80, 'Conf': 'Oeste', 'Pasto': 'Natural'},
        'Columbus Crew':     {'PJ_L': 6, 'GF_L': 2.10, 'GC_L': 0.90, 'PJ_V': 6, 'GF_V': 1.50, 'GC_V': 1.10, 'Conf': 'Este',  'Pasto': 'Natural'},
        'D.C. United':       {'PJ_L': 6, 'GF_L': 1.50, 'GC_L': 1.60, 'PJ_V': 6, 'GF_V': 1.20, 'GC_V': 1.90, 'Conf': 'Este',  'Pasto': 'Natural'},
        'FC Cincinnati':     {'PJ_L': 6, 'GF_L': 1.90, 'GC_L': 1.10, 'PJ_V': 6, 'GF_V': 1.60, 'GC_V': 1.30, 'Conf': 'Este',  'Pasto': 'Natural'},
        'FC Dallas':         {'PJ_L': 6, 'GF_L': 1.60, 'GC_L': 1.20, 'PJ_V': 6, 'GF_V': 0.95, 'GC_V': 1.70, 'Conf': 'Oeste', 'Pasto': 'Natural'},
        'Houston Dynamo':    {'PJ_L': 6, 'GF_L': 1.40, 'GC_L': 0.95, 'PJ_V': 6, 'GF_V': 1.25, 'GC_V': 1.35, 'Conf': 'Oeste', 'Pasto': 'Natural'},
        'Inter Miami':       {'PJ_L': 6, 'GF_L': 2.40, 'GC_L': 1.20, 'PJ_V': 6, 'GF_V': 1.80, 'GC_V': 1.50, 'Conf': 'Este',  'Pasto': 'Natural'},
        'LA Galaxy':         {'PJ_L': 6, 'GF_L': 2.50, 'GC_L': 1.30, 'PJ_V': 6, 'GF_V': 1.60, 'GC_V': 1.60, 'Conf': 'Oeste', 'Pasto': 'Natural'},
        'LAFC':              {'PJ_L': 6, 'GF_L': 2.30, 'GC_L': 1.00, 'PJ_V': 6, 'GF_V': 1.40, 'GC_V': 1.40, 'Conf': 'Oeste', 'Pasto': 'Natural'},
        'Minnesota United':  {'PJ_L': 6, 'GF_L': 1.80, 'GC_L': 1.40, 'PJ_V': 6, 'GF_V': 1.50, 'GC_V': 1.60, 'Conf': 'Oeste', 'Pasto': 'Natural'},
        'CF Montréal':       {'PJ_L': 6, 'GF_L': 1.60, 'GC_L': 1.50, 'PJ_V': 6, 'GF_V': 1.10, 'GC_V': 2.00, 'Conf': 'Este',  'Pasto': 'Natural'},
        'Nashville SC':      {'PJ_L': 6, 'GF_L': 1.45, 'GC_L': 1.15, 'PJ_V': 6, 'GF_V': 1.00, 'GC_V': 1.50, 'Conf': 'Este',  'Pasto': 'Natural'},
        'New England':       {'PJ_L': 6, 'GF_L': 1.40, 'GC_L': 1.50, 'PJ_V': 6, 'GF_V': 1.15, 'GC_V': 1.70, 'Conf': 'Este',  'Pasto': 'Natural'},
        'New York City FC':  {'PJ_L': 6, 'GF_L': 1.85, 'GC_L': 1.10, 'PJ_V': 6, 'GF_V': 1.00, 'GC_V': 1.50, 'Conf': 'Este',  'Pasto': 'Natural'},
        'New York Red Bulls':{'PJ_L': 6, 'GF_L': 1.70, 'GC_L': 1.00, 'PJ_V': 6, 'GF_V': 1.15, 'GC_V': 1.35, 'Conf': 'Este',  'Pasto': 'Natural'},
        'Orlando City':      {'PJ_L': 6, 'GF_L': 1.60, 'GC_L': 1.30, 'PJ_V': 6, 'GF_V': 1.30, 'GC_V': 1.60, 'Conf': 'Este',  'Pasto': 'Natural'},
        'Philadelphia Union':{'PJ_L': 6, 'GF_L': 1.90, 'GC_L': 1.50, 'PJ_V': 6, 'GF_V': 1.40, 'GC_V': 1.60, 'Conf': 'Este',  'Pasto': 'Natural'},
        'Portland Timbers':  {'PJ_L': 6, 'GF_L': 2.20, 'GC_L': 1.50, 'PJ_V': 6, 'GF_V': 1.30, 'GC_V': 1.90, 'Conf': 'Oeste', 'Pasto': 'Sintetico'},
        'Real Salt Lake':    {'PJ_L': 6, 'GF_L': 2.20, 'GC_L': 1.10, 'PJ_V': 6, 'GF_V': 1.30, 'GC_V': 1.50, 'Conf': 'Oeste', 'Pasto': 'Natural'},
        'San Diego FC':      {'PJ_L': 6, 'GF_L': 1.60, 'GC_L': 1.20, 'PJ_V': 6, 'GF_V': 1.10, 'GC_V': 1.50, 'Conf': 'Oeste', 'Pasto': 'Natural'},
        'San Jose Earthquakes':{'PJ_L': 6, 'GF_L': 1.40, 'GC_L': 1.90, 'PJ_V': 6, 'GF_V': 0.90, 'GC_V': 2.50, 'Conf': 'Oeste', 'Pasto': 'Natural'},
        'Seattle Sounders':  {'PJ_L': 6, 'GF_L': 1.80, 'GC_L': 0.85, 'PJ_V': 6, 'GF_V': 1.20, 'GC_V': 1.20, 'Conf': 'Oeste', 'Pasto': 'Sintetico'},
        'Sporting KC':       {'PJ_L': 6, 'GF_L': 1.60, 'GC_L': 1.60, 'PJ_V': 6, 'GF_V': 1.00, 'GC_V': 2.10, 'Conf': 'Oeste', 'Pasto': 'Natural'},
        'St. Louis City':    {'PJ_L': 6, 'GF_L': 1.70, 'GC_L': 1.40, 'PJ_V': 6, 'GF_V': 1.10, 'GC_V': 1.80, 'Conf': 'Oeste', 'Pasto': 'Natural'},
        'Toronto FC':        {'PJ_L': 6, 'GF_L': 1.35, 'GC_L': 1.40, 'PJ_V': 6, 'GF_V': 1.10, 'GC_V': 1.60, 'Conf': 'Este',  'Pasto': 'Natural'},
        'Vancouver Whitecaps':{'PJ_L': 6, 'GF_L': 1.60, 'GC_L': 1.25, 'PJ_V': 6, 'GF_V': 1.45, 'GC_V': 1.40, 'Conf': 'Oeste', 'Pasto': 'Natural'}
    }
    prom_gf_l = sum(e['GF_L'] for e in db_mls.values()) / len(db_mls)
    prom_gc_l = sum(e['GC_L'] for e in db_mls.values()) / len(db_mls)
    return db_mls, prom_gf_l, prom_gc_l

db_equipos, prom_gf_l, prom_gc_l = cargar_estadisticas_mls()

# ==========================================
# 2. CALENDARIO REAL DE LA JORNADA MLS
# ==========================================
calendario_mls = {
    'Jornada Completa': [
        ('Seattle Sounders', 'Chicago Fire'),
        ('D.C. United', 'LAFC'),
        ('Inter Miami', 'CF Montréal'),
        ('Atlanta United', 'Charlotte FC'),
        ('Toronto FC', 'New York City FC'),
        ('New York Red Bulls', 'Philadelphia Union'),
        ('Nashville SC', 'FC Cincinnati'),
        ('Minnesota United', 'Orlando City'),
        ('Houston Dynamo', 'San Jose Earthquakes'),
        ('Sporting KC', 'Vancouver Whitecaps'),
        ('Colorado Rapids', 'Real Salt Lake'),
        ('San Diego FC', 'LA Galaxy'),
        ('Portland Timbers', 'Austin FC'),
        ('Columbus Crew', 'New New England')
    ]
}

# ==========================================
# 3. INTERFAZ VISUAL NATIVA
# ==========================================
st.set_page_config(page_title="MLS - PROGRAMACIÓN COMPLETA", page_icon="⚽", layout="centered")

st.title("⚽ MLS - PREDICTOR DE LA JORNADA")
st.write("Análisis completo 1X2 y mercado Over/Under para todos los partidos programados.")

semana_seleccionada = st.selectbox("📅 Selecciona la Jornada:", list(calendario_mls.keys()))
st.markdown("---")

for local, visita in calendario_mls[semana_seleccionada]:
    
    # Control por si algún equipo del fixture no está mapeado en las métricas base
    if local not in db_equipos or visita not in db_equipos:
        continue
        
    conf_local, pasto_local = db_equipos[local]['Conf'], db_equipos[local]['Pasto']
    conf_visita, pasto_visita = db_equipos[visita]['Conf'], db_equipos[visita]['Pasto']
    
    # --- MODIFICADORES DE VALOR ---
    factor_ataque_visita = 1.0
    factor_defensa_local = 1.0
    alertas = []

    if conf_local != conf_visita:
        factor_ataque_visita *= 0.85
        alertas.append("✈️ **Viaje Largo:** Cruce Interconferencia (Este vs Oeste)")
        
    if pasto_local == 'Sintetico' and pasto_visita == 'Natural':
        factor_defensa_local *= 0.90
        alertas.append("👟 **Césped Artificial:** Ventaja adaptativa local")

    # --- CÁLCULOS MATEMÁTICOS (POISSON) ---
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
    prob_under = 0.0
    
    for i in range(max_goles):
        for j in range(max_goles):
            p_combinada = p_local[i] * p_visita[j]
            
            if i > j: prob_l += p_combinada
            elif i == j: prob_e += p_combinada
            else: prob_v += p_combinada
            
            if (i + j) < 3:
                prob_under += p_combinada
                
    prob_over = 1.0 - prob_under
    
    total_1x2 = prob_l + prob_e + prob_v
    pct_l = round((prob_l / total_1x2) * 100, 1)
    pct_e = round((prob_e / total_1x2) * 100, 1)
    pct_v = round((prob_v / total_1x2) * 100, 1)
    
    pct_under = round(prob_under * 100, 1)
    pct_over = round(prob_over * 100, 1)
    
    es_fija = pct_l >= 70.0 or pct_v >= 70.0

    # --- RENDERIZADO VISUAL ---
    with st.container(border=True):
        if es_fija:
            st.markdown(f"### 🏟️ {local} vs {visita} :orange[**🔥 FIJA**]")
        else:
            st.markdown(f"### 🏟️ {local} vs {visita}")
            
        if alertas:
            for alerta in alertas:
                st.caption(alerta)
        
        st.markdown("**📊 Resultado del Partido:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"🟢 Local: {pct_l}%")
        with col2:
            st.warning(f"🟡 Empate: {pct_e}%")
        with col3:
            st.info(f"🔵 Visita: {pct_v}%")
            
        st.markdown("**⚽ Total de Goles:**")
        col_over, col_under = st.columns(2)
        with col_over:
            if pct_over >= 60.0:
                st.markdown(f"📈 **Más de 2.5 (Over):** :green[{pct_over}%] 🔥")
            else:
                st.markdown(f"📈 **Más de 2.5 (Over):** {pct_over}%")
        with col_under:
            if pct_under >= 60.0:
                st.markdown(f"📉 **Menos de 2.5 (Under):** :green[{pct_under}%] 🔥")
            else:
                st.markdown(f"📉 **Menos de 2.5 (Under):** {pct_under}%")
                
        st.write("")
