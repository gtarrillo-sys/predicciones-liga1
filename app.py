import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# ==========================================
# 1. BASE DE DATOS DE RENDIMIENTO (18 EQUIPOS)
# ==========================================
@st.cache_data
def cargar_y_calcular_estadisticas():
    db_dinamica = {
        'Alianza Lima':        {'PJ_L': 5, 'GF_L': 2.10, 'GC_L': 0.60, 'PJ_V': 5, 'GF_V': 1.50, 'GC_V': 0.90},
        'Universitario':       {'PJ_L': 5, 'GF_L': 2.40, 'GC_L': 0.40, 'PJ_V': 5, 'GF_V': 1.30, 'GC_V': 0.70},
        'Sporting Cristal':    {'PJ_L': 5, 'GF_L': 3.00, 'GC_L': 0.80, 'PJ_V': 5, 'GF_V': 1.80, 'GC_V': 1.10},
        'Melgar':              {'PJ_L': 5, 'GF_L': 2.20, 'GC_L': 0.70, 'PJ_V': 5, 'GF_V': 1.20, 'GC_V': 1.30},
        'Cienciano':           {'PJ_L': 5, 'GF_L': 1.40, 'GC_L': 1.10, 'PJ_V': 5, 'GF_V': 1.10, 'GC_V': 1.40},
        'Cusco FC':            {'PJ_L': 5, 'GF_L': 1.70, 'GC_L': 0.90, 'PJ_V': 5, 'GF_V': 0.90, 'GC_V': 1.60},
        'ADT Tarma':           {'PJ_L': 5, 'GF_L': 2.00, 'GC_L': 0.70, 'PJ_V': 5, 'GF_V': 0.80, 'GC_V': 1.60},
        'Sport Huancayo':      {'PJ_L': 5, 'GF_L': 1.50, 'GC_L': 1.20, 'PJ_V': 5, 'GF_V': 0.60, 'GC_V': 2.20},
        'Atlético Grau':       {'PJ_L': 5, 'GF_L': 1.40, 'GC_L': 0.80, 'PJ_V': 5, 'GF_V': 1.50, 'GC_V': 1.00},
        'Los Chankas':         {'PJ_L': 5, 'GF_L': 1.80, 'GC_L': 1.00, 'PJ_V': 5, 'GF_V': 0.70, 'GC_V': 2.00},
        'Comerciantes Unidos': {'PJ_L': 5, 'GF_L': 1.30, 'GC_L': 1.40, 'PJ_V': 5, 'GF_V': 1.00, 'GC_V': 2.30},
        'Sport Boys':          {'PJ_L': 5, 'GF_L': 1.10, 'GC_L': 1.30, 'PJ_V': 5, 'GF_V': 0.60, 'GC_V': 2.00},
        'UTC':                 {'PJ_L': 5, 'GF_L': 1.40, 'GC_L': 1.20, 'PJ_V': 5, 'GF_V': 0.70, 'GC_V': 1.90},
        'FC Cajamarca':        {'PJ_L': 5, 'GF_L': 1.10, 'GC_L': 1.50, 'PJ_V': 5, 'GF_V': 0.60, 'GC_V': 2.10},
        'César Vallejo':       {'PJ_L': 5, 'GF_L': 1.20, 'GC_L': 1.30, 'PJ_V': 5, 'GF_V': 0.70, 'GC_V': 1.80},
        'Alianza Atlético':    {'PJ_L': 5, 'GF_L': 1.10, 'GC_L': 1.00, 'PJ_V': 5, 'GF_V': 0.50, 'GC_V': 1.70},
        'Juan Pablo II':       {'PJ_L': 5, 'GF_L': 0.90, 'GC_L': 1.80, 'PJ_V': 5, 'GF_V': 0.50, 'GC_V': 2.40},
        'Deportivo Garcilaso': {'PJ_L': 5, 'GF_L': 1.30, 'GC_L': 1.20, 'PJ_V': 5, 'GF_V': 0.80, 'GC_V': 1.90}
    }
    prom_gf_l = sum(e['GF_L'] for e in db_dinamica.values()) / len(db_dinamica)
    prom_gc_l = sum(e['GC_L'] for e in db_dinamica.values()) / len(db_dinamica)
    return db_dinamica, prom_gf_l, prom_gc_l

db_equipos, prom_gf_l, prom_gc_l = cargar_y_calcular_estadisticas()

# ==========================================
# 2. CALENDARIO DE PARTIDOS (JORNADAS)
# ==========================================
calendario_fechas = {
    'Jornada 7': [
        ('Comerciantes Unidos', 'FC Cajamarca'),
        ('Los Chankas', 'Juan Pablo II'),
        ('UTC', 'Universitario'),
        ('Alianza Lima', 'Deportivo Garcilaso'),
        ('Sporting Cristal', 'Melgar')
    ],
    'Jornada 8': [
        ('Universitario', 'Sport Boys'),
        ('Cienciano', 'Sporting Cristal'),
        ('Alianza Lima', 'Cienciano'),
        ('Atlético Grau', 'Los Chankas')
    ]
}

# ==========================================
# 3. INTERFAZ VISUAL (DISEÑO PRO)
# ==========================================
st.set_page_config(page_title="LIGA 1 - PREDICTOR ESTADÍSTICO", page_icon="🏆", layout="centered")

st.title("🏆 LIGA 1 - PREDICTOR ESTADÍSTICO")
st.write("Algoritmo predictivo con sistema visual de detección de alertas y favoritos.")

# Selector de Jornada
jornada_seleccionada = st.selectbox("📅 Selecciona la Fecha del Torneo:", list(calendario_fechas.keys()))

st.markdown("---")

# Procesar y pintar todos los partidos en bloque
for local, visita in calendario_fechas[jornada_seleccionada]:
    
    # --- CÁLCULOS MATEMÁTICOS DE POISSON ---
    fuerza_of_l = db_equipos[local]['GF_L'] / prom_gf_l
    fuerza_def_v = db_equipos[visita]['GC_V'] / prom_gf_l
    lambda_local = fuerza_of_l * fuerza_def_v * prom_gf_l
    
    fuerza_of_v = db_equipos[visita]['GF_V'] / prom_gc_l
    fuerza_def_l = db_equipos[local]['GC_L'] / prom_gc_l
    lambda_visita = fuerza_of_v * fuerza_def_l * prom_gc_l
    
    # Calcular matriz de probabilidades (Goles de 0 a 5)
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
                
    # Normalizar para mostrar 100%
    total = prob_l + prob_e + prob_v
    pct_l = round((prob_l / total) * 100, 1)
    pct_e = round((prob_e / total) * 100, 1)
    pct_v = round((prob_v / total) * 100, 1)
    
    # Determinar si es FIJA (Favorito dominante > 80%)
    es_fija = pct_l >= 80.0 or pct_v >= 80.0
    color_borde = "#28a745" if es_fija else "#007bff"
    
    # --- RENDERIZADO HTML SEGURO ---
    badge_fija = '<span style="background-color:#ffeeba; color:#b55d00; padding:4px 10px; border-radius:12px; font-weight:bold; font-size:12px; float:right;">🔥 FIJA</span>' if es_fija else ''
    
    # Construcción de plantilla reemplazando valores sin usar f-strings problemáticos
    html_template = """
    <div style="border-left: 5px solid COLOR_BORDE; background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;">
        BADGE_FIJA
        <h3 style="margin-top:0; color:#1e293b; font-size:18px;">🏟️ LOCAL vs VISITA</h3>
        <div style="display: flex; gap: 10px; margin-top: 12px; margin-bottom: 15px; flex-wrap: wrap;">
            <span style="background-color: #d4edda; color: #155724; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size:14px;">🟢 Local: PCT_L%</span>
            <span style="background-color: #fff3cd; color: #856404; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size:14px;">🟡 Empate: PCT_E%</span>
            <span style="background-color: #cce5ff; color: #004085; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size:14px;">🔵 Visita: PCT_V%</span>
        </div>
        <div style="background-color: #e2e8f0; padding: 8px 12px; border-radius: 6px; display: inline-block; color: #334155; font-size:14px;">
            Resultado calculado: <strong>LOCAL MARCADOR_L - MARCADOR_V VISITA</strong>
        </div>
    </div>
    """
    
    card_render = html_template\
        .replace("COLOR_BORDE", str(color_borde))\
        .replace("BADGE_FIJA", str(badge_fija))\
        .replace("LOCAL", str(local))\
        .replace("VISITA", str(visita))\
        .replace("PCT_L", str(pct_l))\
        .replace("PCT_E", str(pct_e))\
        .replace("PCT_V", str(pct_v))\
        .replace("MARCADOR_L", str(marcador_exacto[0]))\
        .replace("MARCADOR_V", str(marcador_exacto[1]))

    st.markdown(card_render, unsafe_allowed_html=True)
