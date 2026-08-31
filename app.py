import datetime
import requests
from bs4 import BeautifulSoup
import streamlit as st

# =====================================================================
# CONFIGURACIÓN DE PÁGINA WEB (STREAMLIT)
# =====================================================================
st.set_page_config(page_title="Sistema de Apuestas 2.0", page_icon="🤖", layout="centered")

st.title("🤖 Sistema de Apuestas 2.0")
st.markdown("### Filtro Avanzado: Geografía + Tabla + Radar Financiero")
st.markdown("---")

# =====================================================================
# BASE DE DATOS INTEGRADA
# =====================================================================
PALABRAS_CRITICAS = ["sueldos", "deuda", "safap", "paro", "no concentran", "licencias", "resta de puntos", "huelga"]

DATA_GEOGRAFICA = {
    "Alianza Lima": {"ciudad": "Lima", "tipo": "Llano", "factor_local": 1.2},
    "Universitario": {"ciudad": "Lima", "tipo": "Llano", "factor_local": 1.25},
    "Deportivo Garcilaso": {"ciudad": "Cusco", "tipo": "Altura Extremada", "factor_local": 1.45},
    "Chankas CYC": {"ciudad": "Andahuaylas", "tipo": "Altura Extremada", "factor_local": 1.4},
    "Melgar": {"ciudad": "Arequipa", "tipo": "Altura Media", "factor_local": 1.3},
    "Cusco": {"ciudad": "Cusco", "tipo": "Altura Extremada", "factor_local": 1.4},
    "Cienciano": {"ciudad": "Cusco", "tipo": "Altura Extremada", "factor_local": 1.4},
    "Alianza Atlético": {"ciudad": "Sullana", "tipo": "Calor Extremo", "factor_local": 1.35},
    "Sport Boys": {"ciudad": "Callao", "tipo": "Llano/Humedad", "factor_local": 1.15},
    "Sporting Cristal": {"ciudad": "Lima", "tipo": "Llano", "factor_local": 1.2},
    "Comerciantes Unidos": {"ciudad": "Cutervo", "tipo": "Altura Media", "factor_local": 1.25},
    "Colegio Juan Pablo II": {"ciudad": "Chongoyape", "tipo": "Calor", "factor_local": 1.2},
    "Atlético Grau": {"ciudad": "Piura", "tipo": "Calor Extremo", "factor_local": 1.35},
    "Deporte Huancayo": {"ciudad": "Huancayo", "tipo": "Altura Extremada", "factor_local": 1.4},
    "CD Moquegua": {"ciudad": "Moquegua", "tipo": "Altura Media/Calor", "factor_local": 1.25},
    "FC Cajamarca": {"ciudad": "Cajamarca", "tipo": "Altura Media", "factor_local": 1.25},
    "ADT": {"ciudad": "Tarma", "tipo": "Altura Extremada", "factor_local": 1.45},
    "UTC": {"ciudad": "Cajamarca", "tipo": "Altura Media", "factor_local": 1.25}
}

TABLA_ACUMULADA = {
    1: "Alianza Lima", 2: "Universitario", 3: "Deportivo Garcilaso", 
    4: "Chankas CYC", 5: "Melgar", 6: "Cusco", 7: "Cienciano", 
    8: "Alianza Atlético", 9: "Sport Boys", 10: "Sporting Cristal",
    11: "Comerciantes Unidos", 12: "Colegio Juan Pablo II", 13: "Atlético Grau",
    14: "Deporte Huancayo", 15: "CD Moquegua", 16: "FC Cajamarca",
    17: "ADT", 18: "UTC"
}

# =====================================================================
# FUNCIONES LÓGICAS
# =====================================================================
def buscar_puesto_tabla(equipo):
    for puesto, nombre in TABLA_ACUMULADA.items():
        if equipo == nombre: return puesto
    return 99

def escanear_crisis_financiera(equipo):
    url_fuente = "https://www.ovacion.pe/rss"
    alertas_encontradas = []
    try:
        response = requests.get(url_fuente, timeout=5)
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')
        for item in items:
            texto = (item.title.text.lower() if item.title else "") + " " + (item.description.text.lower() if item.description else "")
            if equipo.lower() in texto:
                for palabra in PALABRAS_CRITICAS:
                    if palabra in texto:
                        alertas_encontradas.append(f"🚨 Noticia Crítica: '{item.title.text}'")
                        break
    except:
        pass
    return list(set(alertas_encontradas))

# =====================================================================
# INTERFAZ DE USUARIO INTERACTIVA
# =====================================================================
lista_equipos = sorted(list(DATA_GEOGRAFICA.keys()))

col1, col2 = st.columns(2)
with col1:
    local = st.selectbox("Selecciona Equipo Local:", lista_equipos, index=lista_equipos.index("Atlético Grau"))
with col2:
    visita = st.selectbox("Selecciona Equipo Visitante:", lista_equipos, index=lista_equipos.index("Melgar"))

if st.button("⚡ ANALIZAR PARTIDO CON SISTEMA 2.0", use_container_width=True):
    st.markdown("---")
    st.subheader(f"📊 Resultado del Análisis: {local} vs {visita}")
    
    # Capa 1: Tabla
    st.write("#### 🛡️ Capa 1: Contexto de Tabla y Presión")
    p_local = buscar_puesto_tabla(local)
    p_visita = buscar_puesto_tabla(visita)
    alertas_tabla = False
    
    if p_local >= 15:
        st.warning(f"⚠️ {local} (Puesto {p_local}) está en zona de descenso. Partido de alta fricción/tarjetas.")
        alertas_tabla = True
    elif p_local <= 3:
        st.info(f"🔥 {local} (Puesto {p_local}) pelea el título. Presión alta por ganar en casa.")
        
    if p_visita >= 15:
        st.warning(f"⚠️ {visita} (Puesto {p_visita}) pelea el descenso. Se espera bus atrás.")
        alertas_tabla = True
        
    if not alertas_tabla and p_local > 3:
        st.success("✅ Presión de tabla moderada. Flujo de juego limpio esperado.")

    # Capa 2: Finanzas
    st.write("#### 🕵️‍♂️ Capa 2: Escáner Financiero en Tiempo Real")
    with st.spinner("Buscando deudas o huelgas en internet..."):
        alertas_finanzas = escanear_crisis_financiera(local) + escanear_crisis_financiera(visita)
    
    crisis_activa = False
    if alertas_finanzas:
        crisis_activa = True
        for alerta in alertas_finanzas:
            st.error(alerta)
    else:
        st.success("✅ Filtro Limpio: Sin reportes de deudas ni huelgas activas.")

    # Capa 3: Veredicto
    st.write("#### 🧠 Capa 3: Sugerencia del Algoritmo")
    if crisis_activa:
        st.error("❌ APUESTA BLOQUEADA: El riesgo extra-cancha por problemas económicos es muy alto.")
    else:
        geo_local = DATA_GEOGRAFICA[local]
        if "Altura" in geo_local["tipo"] or "Calor" in geo_local["tipo"]:
            if p_visita <= 4:
                st.info(f"🎯 **Sugerencia:** Ambos Anotan (Sí) O Más de 1.5 Goles.\n\n*Sustento:* {local} tiene la ventaja de {geo_local['tipo']}, pero {visita} va arriba en la tabla y está obligado a proponer.")
            else:
                st.info(f"🎯 **Sugerencia:** Ganador Seco {local}.\n\n*Sustento:* La ventaja climática ({geo_local['tipo']}) es determinante contra un rival de zona media/baja.")
        else:
            st.info("🎯 **Sugerencia:** Total de Goles: Más de 1.5 O Doble Oportunidad Local.\n\n*Sustento:* Condiciones estándar. Con planteles estables, se juega bajo inercia normal.")
