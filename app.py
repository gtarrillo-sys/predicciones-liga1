import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# =========================================================
# BASE DE DATOS SEGMENTADA POR CONTEXTO Y JERARQUÍA
# =========================================================
def obtener_base_respaldo_peru_avanzada():
    # GF_L_vs_Costa / GC_L_vs_Costa: Cuando recibe a un costeño
    # GF_L_vs_Altura / GC_L_vs_Altura: Cuando recibe a uno de altura
    # GF_V_en_Costa / GC_V_en_Costa: Cuando visita la costa
    # GF_V_en_Altura / GC_V_en_Altura: Cuando visita la altura
    # Jerarquia: True (Fuerza ofensiva invariable de visita)
    return {
        'Universitario': {
            'GF_L_vs_Costa': 2.50, 'GC_L_vs_Costa': 0.40, 'GF_L_vs_Altura': 2.30, 'GC_L_vs_Altura': 0.60,
            'GF_V_en_Costa': 1.60, 'GC_V_en_Costa': 0.80, 'GF_V_en_Altura': 1.10, 'GC_V_en_Altura': 1.00,
            'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': True
        },
        'Alianza Lima': {
            'GF_L_vs_Costa': 2.30, 'GC_L_vs_Costa': 0.50, 'GF_L_vs_Altura': 2.10, 'GC_L_vs_Altura': 0.70,
            'GF_V_en_Costa': 1.50, 'GC_V_en_Costa': 0.80, 'GF_V_en_Altura': 1.20, 'GC_V_en_Altura': 0.90,
            'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': True
        },
        'Sporting Cristal': {
            'GF_L_vs_Costa': 2.90, 'GC_L_vs_Costa': 0.70, 'GF_L_vs_Altura': 2.70, 'GC_L_vs_Altura': 0.90,
            'GF_V_en_Costa': 1.80, 'GC_V_en_Costa': 1.00, 'GF_V_en_Altura': 1.30, 'GC_V_en_Altura': 1.20,
            'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': True
        },
        'Sport Boys': {
            'GF_L_vs_Costa': 1.60, 'GC_L_vs_Costa': 1.10, 'GF_L_vs_Altura': 1.30, 'GC_L_vs_Altura': 1.30,
            'GF_V_en_Costa': 0.80, 'GC_V_en_Costa': 1.80, 'GF_V_en_Altura': 0.50, 'GC_V_en_Altura': 2.20,
            'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': False
        },
        'Melgar': {
            'GF_L_vs_Costa': 2.40, 'GC_L_vs_Costa': 0.60, 'GF_L_vs_Altura': 1.90, 'GC_L_vs_Altura': 0.80,
            'GF_V_en_Costa': 1.20, 'GC_V_en_Costa': 1.10, 'GF_V_en_Altura': 1.00, 'GC_V_en_Altura': 1.30,
            'Geografia': 'Altura', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': False
        },
        'Comerciantes Unidos': {
            'GF_L_vs_Costa': 1.80, 'GC_L_vs_Costa': 1.00, 'GF_L_vs_Altura': 1.40, 'GC_L_vs_Altura': 1.20,
            'GF_V_en_Costa': 0.90, 'GC_V_en_Costa': 2.10, 'GF_V_en_Altura': 0.80, 'GC_V_en_Altura': 1.70,
            'Geografia': 'Altura', 'Pasto': 'Sintetico', 'Acceso': 'Dificil', 'Jerarquia': False
        },
        'FC Cajamarca': {
            'GF_L_vs_Costa': 1.60, 'GC_L_vs_Costa': 1.00, 'GF_L_vs_Altura': 1.40, 'GC_L_vs_Altura': 1.20,
            'GF_V_en_Costa': 0.85, 'GC_V_en_Costa': 1.90, 'GF_V_en_Altura': 0.75, 'GC_V_en_Altura': 1.60,
            'Geografia': 'Altura', 'Pasto': 'Sintetico', 'Acceso': 'Normal', 'Jerarquia': False
        },
        'Los Chankas': {
            'GF_L_vs_Costa': 2.20, 'GC_L_vs_Costa': 0.80, 'GF_L_vs_Altura': 1.80, 'GC_L_vs_Altura': 1.00,
            'GF_V_en_Costa': 0.70, 'GC_V_en_Costa': 2.00, 'GF_V_en_Altura': 0.70, 'GC_V_en_Altura': 1.60,
            'Geografia': 'Altura', 'Pasto': 'Natural', 'Acceso': 'Dificil', 'Jerarquia': False
        },
        'Juan Pablo II College': {
            'GF_L_vs_Costa': 1.50, 'GC_L_vs_Costa': 1.10, 'GF_L_vs_Altura': 1.30, 'GC_L_vs_Altura': 1.30,
            'GF_V_en_Costa': 0.80, 'GC_V_en_Costa': 1.80, 'GF_V_en_Altura': 0.70, 'GC_V_en_Altura': 2.00,
            'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': False
        },
        'UTC Cajamarca': {
            'GF_L_vs_Costa': 1.80, 'GC_L_vs_Costa': 0.90, 'GF_L_vs_Altura': 1.40, 'GC_L_vs_Altura': 1.10,
            'GF_V_en_Costa': 0.75, 'GC_V_en_Costa': 2.00, 'GF_V_en_Altura': 0.65, 'GC_V_en_Altura': 1.70,
            'Geografia': 'Altura', 'Pasto': 'Sintetico', 'Acceso': 'Normal', 'Jerarquia': False
        },
        'Deportivo Garcilaso': {
            'GF_L_vs_Costa': 1.90, 'GC_L_vs_Costa': 0.90, 'GF_L_vs_Altura': 1.60, 'GC_L_vs_Altura': 1.10,
            'GF_V_en_Costa': 0.90, 'GC_V_en_Costa': 1.80, 'GF_V_en_Altura': 0.80, 'GC_V_en_Altura': 1.50,
            'Geografia': 'Altura', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': False
        },
        'Deportivo Moquegua': {
            'GF_L_vs_Costa': 1.40, 'GC_L_vs_Costa': 1.10, 'GF_L_vs_Altura': 1.30, 'GC_L_vs_Altura': 1.20,
            'GF_V_en_Costa': 0.75, 'GC_V_en_Costa': 2.10, 'GF_V_en_Altura': 0.70, 'GC_V_en_Altura': 1.90,
            'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': False
        },
        'Alianza Atlético': {
            'GF_L_vs_Costa': 1.50, 'GC_L_vs_Costa': 0.85, 'GF_L_vs_Altura': 1.30, 'GC_L_vs_Altura': 0.95,
            'GF_V_en_Costa': 0.80, 'GC_V_en_Costa': 1.60, 'GF_V_en_Altura': 0.70, 'GC_V_en_Altura': 1.40,
            'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': False
        },
        'ADT Tarma': {
            'GF_L_vs_Costa': 2.10, 'GC_L_vs_Costa': 0.60, 'GF_L_vs_Altura': 1.80, 'GC_L_vs_Altura': 0.70,
            'GF_V_en_Costa': 0.90, 'GC_V_en_Costa': 1.50, 'GF_V_en_Altura': 0.70, 'GC_V_en_Altura': 1.70,
            'Geografia': 'Altura', 'Pasto': 'Natural', 'Acceso': 'Dificil', 'Jerarquia': False
        },
        'Sport Huancayo': {
            'GF_L_vs_Costa': 1.85, 'GC_L_vs_Costa': 0.75, 'GF_L_vs_Altura': 1.55, 'GC_L_vs_Altura': 0.85,
            'GF_V_en_Costa': 0.80, 'GC_V_en_Costa': 1.80, 'GF_V_en_Altura': 0.70, 'GC_V_en_Altura': 1.60,
            'Geografia': 'Altura', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': False
        },
        'César Vallejo': {
            'GF_L_vs_Costa': 1.60, 'GC_L_vs_Costa': 1.00, 'GF_L_vs_Altura': 1.40, 'GC_L_vs_Altura': 1.20,
            'GF_V_en_Costa': 0.90, 'GC_V_en_Costa': 1.60, 'GF_V_en_Altura': 0.80, 'GC_V_en_Altura': 1.90,
            'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': False
        },
        'Cienciano': {
            'GF_L_vs_Costa': 1.95, 'GC_L_vs_Costa': 0.80, 'GF_L_vs_Altura': 1.65, 'GC_L_vs_Altura': 0.90,
            'GF_V_en_Costa': 1.00, 'GC_V_en_Costa': 1.30, 'GF_V_en_Altura': 0.80, 'GC_V_en_Altura': 1.50,
            'Geografia': 'Altura', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': False
        },
        'Cusco FC': {
            'GF_L_vs_Costa': 2.00, 'GC_L_vs_Costa': 0.70, 'GF_L_vs_Altura': 1.80, 'GC_L_vs_Altura': 0.80,
            'GF_V_en_Costa': 0.85, 'GC_V_en_Costa': 1.60, 'GF_V_en_Altura': 0.75, 'GC_V_en_Altura': 1.40,
            'Geografia': 'Altura', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': False
        },
        'Atlético Grau': {
            'GF_L_vs_Costa': 1.70, 'GC_L_vs_Costa': 0.75, 'GF_L_vs_Altura': 1.60, 'GC_L_vs_Altura': 0.85,
            'GF_V_en_Costa': 1.10, 'GC_V_en_Costa': 1.20, 'GF_V_en_Altura': 0.90, 'GC_V_en_Altura': 1.40,
            'Geografia': 'Costa', 'Pasto': 'Natural', 'Acceso': 'Normal', 'Jerarquia': False
        }
    }

def obtener_partidos_peru():
    return [
        {"local": "Comerciantes Unidos", "visita": "FC Cajamarca", "fecha": "Viernes", "hora": "15:00"},
        {"local": "Los Chankas", "visita": "Juan Pablo II College", "fecha": "Sábado", "hora": "13:00"},
        {"local": "UTC Cajamarca", "visita": "Universitario", "fecha": "Sábado", "hora": "15:30"},
        {"local": "Alianza Lima", "visita": "Deportivo Garcilaso", "fecha": "Sábado", "hora": "19:30"},
        {"local": "Deportivo Moquegua", "visita": "Alianza Atlético", "fecha": "Domingo", "hora": "11:00"},
        {"local": "ADT Tarma", "visita": "Sport Huancayo", "fecha": "Domingo", "hora": "13:15"},
        {"local": "Sport Boys", "visita": "Sporting Cristal", "fecha": "Domingo", "hora": "15:30"},
        {"local": "Cienciano", "visita": "Cusco FC", "fecha": "Domingo", "hora": "19:00"},
        {"local": "Atlético Grau", "visita": "Melgar", "fecha": "Lunes", "hora": "15:00"}
    ]

# INTERFAZ STREAMLIT
st.set_page_config(page_title="Liga 1 Predictor V2", page_icon="🇵🇪", layout="centered")
st.title("🇵🇪 LIGA 1 METRIC PRO — CONTEXTO & JERARQUÍA")
st.caption("Evolución: Datos segmentados por origen geográfico y Plus de Jerarquía Histórica.")

db_equipos = obtener_base_respaldo_peru_avanzada()
partidos_jornada = obtener_partidos_peru()

for partido in partidos_jornada:
    local, visita = partido['local'], partido['visita']
    if local not in db_equipos or visita not in db_equipos: continue
        
    geo_l = db_equipos[local]['Geografia']
    geo_v = db_equipos[visita]['Geografia']
    
    # SELECCIÓN DINÁMICA DE LA MUESTRA SEGÚN GEOGRAFÍA DEL RIVAL
    if geo_v == 'Costa':
        gf_local_segmentado = db_equipos[local]['GF_L_vs_Costa']
        gc_local_segmentado = db_equipos[local]['GC_L_vs_Costa']
    else:
        gf_local_segmentado = db_equipos[local]['GF_L_vs_Altura']
        gc_local_segmentado = db_equipos[local]['GC_L_vs_Altura']
        
    if geo_l == 'Costa':
        gf_visita_segmentado = db_equipos[visita]['GF_V_en_Costa']
        gc_visita_segmentado = db_equipos[visita]['GC_V_en_Costa']
    else:
        gf_visita_segmentado = db_equipos[visita]['GF_V_en_Altura']
        gc_visita_segmentado = db_equipos[visita]['GC_V_en_Altura']

    # INYECCIÓN DEL FACTOR JERARQUÍA (U, Alianza, Cristal)
    plus_jerarquia_ataque = 1.0
    alertas = []
    
    if db_equipos[visita]['Jerarquia']:
        plus_jerarquia_ataque = 1.15  # Plus de 15% de poder ofensivo porque saldrán a proponer sí o sí
        alertas.append(f"👑 **Efecto Jerarquía:** {visita} es un equipo grande. Su propuesta táctica externa es ofensiva e invariable.")

    # APLICACIÓN DE AGRESIVIDAD GEOGRÁFICA NATURAL
    if geo_l == 'Altura' and geo_v == 'Costa' and not db_equipos[visita]['Jerarquia']:
        gf_visita_segmentado *= 0.75  # Castigo de aire completo a equipos chicos de costa
        alertas.append(f"🏔️🥵 **Shock Hipóxico (Altura):** {visita} sufre la falta de oxígeno frente a {local}.")
    elif geo_l == 'Altura' and geo_v == 'Costa' and db_equipos[visita]['Jerarquia']:
        gf_visita_segmentado *= 0.90  # Un grande de costa se ahoga, pero lo compensa parcialmente con plantel
        alertas.append(f"🏔️🛡️ **Altura vs Jerarquía:** {visita} expuesto a la altura, pero mitiga daño por peso de plantel.")

    if db_equipos[local]['Pasto'] == 'Sintetico':
        alertas.append("👟 **Césped Artificial:** Bote rápido controlado por el local.")

    # CÁLCULO DE LAMBDAS CON VARIABLES SEGMENTADAS
    lambda_local = (gf_local_segmentado + gc_visita_segmentado) / 2.0
    lambda_visita = ((gf_visita_segmentado * plus_jerarquia_ataque) + gc_local_segmentado) / 2.0

    # POISSON
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
    
    with st.container(border=True):
        st.caption(f"📅 {partido['fecha']} — ⏰ {partido['hora']}")
        titulo_fija = " 🔥 FIJA DE VALOR" if pct_l >= 70.0 or pct_v >= 70.0 else ""
        st.markdown(f"### 🏟️ {local} vs {visita}{titulo_fija}")
        
        for a in alertas: st.markdown(a)
        
        st.markdown("**📊 Probabilidades de Resultado:**")
        col1, col2, col3 = st.columns(3)
        col1.success(f"🟢 Local: {pct_l}%")
        col2.warning(f"🟡 Empate: {pct_e}%")
        col3.info(f"🔵 Visita: {pct_v}%")
        
        st.markdown(f"**⚽ Goles (Línea de 2.5):** Over {pct_over}% | Under {pct_under_f}%")
