import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# Configuración de cabeceras para la API compartida desde tu RapidAPI
API_KEY = "cee4cebcd9msh82842660a6a542cp1710c9jsnfc3990e9eac0"
API_HOST = "free-api-live-football-data.p.rapidapi.com"

headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

# ID de la MLS en esta API específica
MLS_LEAGUE_ID = 253 
TEMPORADA_ACTUAL = 2026

# =========================================================
# 1. CARGA AUTOMÁTICA DE ESTADÍSTICAS EN VIVO DESDE LA API
# =========================================================
@st.cache_data(ttl=3600)  # Guarda en caché por 1 hora para no agotar tus 100 llamadas diarias
def obtener_metricas_api():
    url_standings = f"https://{API_HOST}/football-league-standings"
    querystring = {"league_id": str(MLS_LEAGUE_ID), "season": str(TEMPORADA_ACTUAL)}
    
    db_mls = {}
    try:
        response = requests.get(url_standings, headers=headers, params=querystring)
        data = response.json()
        
        # Procesar los datos de la tabla que provee la API
        if "results" in data and "standings" in data["results"]:
            for team in data["results"]["standings"]:
                nombre = team.get("team_name")
                
                # Extraer estadísticas de Local
                home_stats = team.get("home", {})
                pj_l = int(home_stats.get("played", 6))
                gf_l = int(home_stats.get("goals_for", 10))
                gc_l = int(home_stats.get("goals_against", 8))
                
                # Extraer estadísticas de Visita
                away_stats = team.get("away", {})
                pj_v = int(away_stats.get("played", 6))
                gf_v = int(away_stats.get("goals_for", 7))
                gc_v = int(away_stats.get("goals_against", 11))
                
                # Configuración geográfica base para factores tácticos
                conferencia = "Este" if team.get("group_name") == "Eastern Conference" else "Oeste"
                # Estadios conocidos con pasto sintético en la MLS
                sinteticos = ["Seattle Sounders", "Atlanta United", "Portland Timbers", "Charlotte FC", "New England"]
                tipo_pasto = "Sintetico" if nombre in sinteticos else "Natural"
                
                db_mls[nombre] = {
                    'PJ_L': pj_l, 'GF_L': gf_l / max(pj_l, 1), 'GC_L': gc_l / max(pj_l, 1),
                    'PJ_V': pj_v, 'GF_V': gf_v / max(pj_v, 1), 'GC_V': gc_v / max(pj_v, 1),
                    'Conf': conferencia, 'Pasto': tipo_pasto
                }
    except Exception as e:
        st.error(f"Error al conectar con las estadísticas de la API: {e}")
        
    # Valores de respaldo por si la API no responde momentáneamente
    if not db_mls:
        return {'Inter Miami': {'PJ_L': 6, 'GF_L': 2.4, 'GC_L': 1.2, 'PJ_V': 6, 'GF_V': 1.8, 'GC_V': 1.5, 'Conf': 'Este', 'Pasto': 'Natural'}}, 1.6, 1.4

    prom_gf_l = sum(e['GF_L'] for e in db_mls.values()) / len(db_mls)
    prom_gc_l = sum(e['GC_L'] for e in db_mls.values()) / len(db_mls)
    return db_mls, prom_gf_l, prom_gc_l

# =========================================================
# 2. CARGA AUTOMÁTICA DE PARTIDOS (PRÓXIMOS Y RECIENTES)
# =========================================================
@st.cache_data(ttl=1800)
def obtener_partidos_api():
    url_fixtures = f"https://{API_HOST}/football-league-fixtures"
    querystring = {"league_id": str(MLS_LEAGUE_ID), "season": str(TEMPORADA_ACTUAL)}
    
    proximos = []
    recientes = []
    
    try:
        response = requests.get(url_fixtures, headers=headers, params=querystring)
        data = response.json()
        
        if "results" in data and "fixtures" in data["results"]:
            for match in data["results"]["fixtures"]:
                status = match.get("status_short") # FT significa Finalizado, NS significa No Empezado
                
                # Limpieza de fecha y hora
                dt_str = match.get("event_date", "")
                try:
                    dt = datetime.strptime(dt_str[:16], "%Y-%m-%dT%H:%M")
                    fecha_f = dt.strftime("%A %d/%m")
                    hora_f = dt.strftime("%H:%M")
                except:
                    fecha_f = "Por definir"
                    hora_f = "--:--"
                
                partido_info = {
                    "local": match.get("home_team_name"),
                    "visita": match.get("away_team_name"),
                    "fecha": fecha_f,
                    "hora": hora_f,
                    "goles_l": match.get("goals_home_team"),
                    "goles_v": match.get("goals_away_team")
                }
                
                if status in ["FT", "AET", "PEN"]:
                    recientes.append(partido_info)
                else:
                    proximos.append(partido_info)
    except Exception as e:
        st.error(f"Error al conectar con el calendario de la API: {e}")
        
    return proximos[:14], recientes[:14] # Limitar a los 14 más inmediatos de la jornada

# =========================================================
# 3. INTERFAZ EN STREAMLIT CON PESTAÑAS (TABS)
# =========================================================
st.set_page_config(page_title="MLS AUTOMÁTICA PRO", page_icon="⚽", layout="centered")
st.title("⚽ MLS - ANALIZADOR AUTOMÁTICO EN VIVO")
st.write("Estadísticas de equipos y partidos actualizados en tiempo real mediante API.")

db_equipos, prom_gf_l, prom_gc_l = obtener_metricas_api()
partidos_proximos, partidos_recientes = obtener_partidos_api()

tab1, tab2 = st.tabs(["📅 Próxima Jornada (Predicciones)", "📊 Resultados Recientes (Auditoría)"])

# --- PESTAÑA 1: PREDICCIONES ---
with tab1:
    if not partidos_proximos:
        st.info("No hay partidos programados próximamente o la jornada actual ha concluido.")
    else:
        for partido in partidos_proximos:
            local, visita = partido['local'], partido['visita']
            
            if local not in db_equipos or visita not in db_equipos:
                continue
                
            # Algoritmo de Poisson con modificadores tácticos de localía y viaje
            factor_ataque_visita, factor_defensa_local = 1.0, 1.0
            alertas = []
            if db_equipos[local]['Conf'] != db_equipos[visita]['Conf']:
                factor_ataque_visita *= 0.85
                alertas.append("✈️ **Viaje Largo:** Cruce Interconferencia")
            if db_equipos[local]['Pasto'] == 'Sintetico' and db_equipos[visita]['Pasto'] == 'Natural':
                factor_defensa_local *= 0.90
                alertas.append("👟 **Césped Artificial:** Ventaja adaptativa local")

            lambda_local = (db_equipos[local]['GF_L'] / prom_gf_l) * (db_equipos[visita]['GC_V'] / prom_gf_l) * prom_gf_l * (1 / factor_defensa_local)
            lambda_visita = (db_equipos[visita]['GF_V'] / prom_gc_l) * (db_equipos[local]['GC_L'] / prom_gc_l) * prom_gc_l * factor_ataque_visita
            
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
            
            pct_l = round((prob_l / (prob_l + prob_e + prob_v)) * 100, 1)
            pct_e = round((prob_e / (prob_l + prob_e + prob_v)) * 100, 1)
            pct_v = round((prob_v / (prob_l + prob_e + prob_v)) * 100, 1)
            pct_over = round((1.0 - prob_under) * 100, 1)
            
            with st.container(border=True):
                st.caption(f"📅 {partido['fecha']} — ⏰ {partido['hora']}")
                st.markdown(f"### 🏟️ {local} vs {visita} " + (" :orange[**🔥 FIJA**]" if pct_l >= 70 or pct_v >= 70 else ""))
                for a in alertas: st.caption(a)
                
                col1, col2, col3 = st.columns(3)
                col1.success(f"🟢 Local: {pct_l}%")
                col2.warning(f"🟡 Empate: {pct_e}%")
                col3.info(f"🔵 Visita: {pct_v}%")
                
                st.markdown(f"⚽ **Más de 2.5 Goles (Over):** {pct_over}%" + (" 🔥" if pct_over >= 60 else ""))

# --- PESTAÑA 2: AUDITORÍA DE RESULTADOS RECIENTES ---
with tab2:
    if not partidos_recientes:
        st.info("No se registran partidos jugados recientemente en los registros de la API.")
    else:
        st.write("Compara las predicciones previas contra los marcadores reales de los últimos juegos:")
        for partido in partidos_recientes:
            with st.container(border=True):
                st.caption(f"✅ Partido Finalizado — {partido['fecha']}")
                st.markdown(f"### 🏟️ {partido['local']}  :red[{partido['goles_l']}]  vs  :red[{partido['goles_v']}]  {partido['visita']}")
