import datetime
import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd

# =====================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA WEB (STREAMLIT)
# =====================================================================
st.set_page_config(
    page_title="Sistema Predictivo Liga 1 Pro", 
    page_icon="🤖", 
    layout="centered"
)

st.title("🤖 Sistema de Apuestas Profesional 2.0")
st.markdown("### Motor de Predicción: Geografía + Tabla Acumulada + Factor Racha + Radar Financiero")
st.markdown("---")

# =====================================================================
# 2. CARGA DE BASE DE DATOS INTELIGENTE DESDE EXCEL
# =====================================================================
PALABRAS_CRITICAS = ["sueldos", "deuda", "safap", "paro", "no concentran", "licencias", "resta de puntos", "huelga"]

@st.cache_data
def cargar_base_de_datos():
    try:
        # Leer las hojas del archivo Excel generado
        df_tabla = pd.read_excel("liga1_data.xlsx", sheet_name="Tabla_Posiciones")
        df_geo = pd.read_excel("liga1_data.xlsx", sheet_name="Data_Geografica")
        
        # Mapeo de Puesto -> Club en la Tabla Acumulada
        tabla_dict = dict(zip(df_tabla["Puesto"], df_tabla["Club"]))
        
        # Mapeo de variables Geográficas y de Racha Reciente
        geo_dict = {}
        for _, row in df_geo.iterrows():
            geo_dict[row["Club"]] = {
                "ciudad": row["Ciudad"],
                "tipo": row["Tipo_Clima"],
                "factor_local": float(row["Factor_Local"])
            }
            
        # Cruzar la racha (calculada en base a los últimos 5 partidos del link de Google)
        # Asignamos un factor por defecto si no se edita, o el valor real del Excel
        racha_dict = dict(zip(df_geo["Club"], df_geo["Racha"] if "Racha" in df_geo.columns else [3]*18))
        
        return tabla_dict, geo_dict, racha_dict
    except Exception as e:
        st.error(f"❌ Error crítico al cargar 'liga1_data.xlsx'. Asegúrate de que el archivo Excel esté subido en tu repositorio de GitHub. Detalle: {e}")
        return {}, {}, {}

TABLA_ACUMULADA, DATA_GEOGRAFICA, FACTOR_RACHA = cargar_base_de_datos()

# =====================================================================
# 3. FUNCIONES LÓGICAS Y RASTREADORES
# =====================================================================
def obtener_puesto_acumulado(equipo):
    for puesto, nombre in TABLA_ACUMULADA.items():
        if equipo == nombre: 
            return puesto
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
                        alertas_encontradas.append(f"🚨 Alerta Ovación: '{item.title.text}'")
                        break
    except:
        pass
    return list(set(alertas_encontradas))

# =====================================================================
# 4. INTERFAZ DE USUARIO INTERACTIVA (MENÚS DESPLEGABLES)
# =====================================================================
if DATA_GEOGRAFICA and TABLA_ACUMULADA:
    lista_equipos = sorted(list(DATA_GEOGRAFICA.keys()))

    col1, col2 = st.columns(2)
    with col1:
        local = st.selectbox("Selecciona Equipo Local:", lista_equipos, index=lista_equipos.index("Deportivo Garcilaso") if "Deportivo Garcilaso" in lista_equipos else 0)
    with col2:
        visita = st.selectbox("Selecciona Equipo Visitante:", lista_equipos, index=lista_equipos.index("Universitario") if "Universitario" in lista_equipos else 0)

    if st.button("⚡ INICIAR PROCESAMIENTO MATEMÁTICO", use_container_width=True):
        st.markdown("---")
        st.subheader(f"📊 Informe del Partido: {local} vs {visita}")
        
        # OBTENCIÓN DE VARIABLES MÉTRICAS
        p_local = obtener_puesto_acumulado(local)
        p_visita = obtener_puesto_acumulado(visita)
        r_local = FACTOR_RACHA.get(local, 3)
        r_visita = FACTOR_RACHA.get(visita, 3)
        geo_local = DATA_GEOGRAFICA[local]

        # -----------------------------------------------------------------
        # CAPA 1: CONTEXTO BIOLÓGICO DE LA TABLA ACUMULADA
        # -----------------------------------------------------------------
        st.write("#### 🛡️ Capa 1: Presión por Objetivos (Tabla Acumulada)")
        
        # Diagnóstico Local
        if p_local <= 4:
            st.info(f"🏆 {local} (Puesto {p_local}) defiende zona de Copa Libertadores. Motivación económica máxima.")
        elif p_local <= 8:
            st.info(f"🎟️ {local} (Puesto {p_local}) se encuentra en puestos de Copa Sudamericana.")
        elif p_local >= 16:
            st.warning(f"⚠️ {local} (Puesto {p_local}) está en ZONA DE DESCENSO DIRECTO. Desesperación alta, juego físico severo.")
            
        # Diagnóstico Visitante
        if p_visita <= 4:
            st.info(f"🏆 {visita} (Puesto {p_visita}) está obligado a proponer afuera para mantener cupo a Libertadores.")
        elif p_visita <= 8:
            st.info(f"🎟️ {visita} (Puesto {p_visita}) busca consolidar su clasificación a Sudamericana.")
        elif p_visita >= 16:
            st.warning(f"⚠️ {visita} (Puesto {p_visita}) pelea el DESCENSO. Planteará un bloque defensivo muy bajo (bus atrás).")

        # -----------------------------------------------------------------
        # CAPA 2: ANÁLISIS DE MOMENTO ANÍMICO (ÚLTIMOS 5 PARTIDOS)
        # -----------------------------------------------------------------
        st.write("#### 📈 Capa 2: Termómetro de Forma Reciente (Racha)")
        
        if r_local >= 4:
            st.success(f"🟩 {local} viene en inercia ganadora (Racha: {r_local}/5). Confianza alta que disminuye el cansancio.")
        elif r_local <= 2:
            st.error(f"🟥 {local} arrastra una crisis de resultados (Racha: {r_local}/5). Vulnerabilidad al recibir el primer gol.")
            
        if r_visita >= 4:
            st.success(f"🟩 {visita} llega con un ritmo competitivo sólido (Racha: {r_visita}/5).")
        elif r_visita <= 2:
            st.error(f"🟥 {visita} está golpeado anímicamente (Racha: {r_visita}/5). Propensión a errores defensivos forzados.")

        # -----------------------------------------------------------------
        # CAPA 3: RADAR FINANCIERO EN TIEMPO REAL
        # -----------------------------------------------------------------
        st.write("#### 🕵️‍♂️ Capa 3: Filtro Extra-Cancha (Problemas de Pagos)")
        with st.spinner("Escaneando diarios deportivos y alertas de huelgas..."):
            alertas_finanzas = escanear_crisis_financiera(local) + escanear_crisis_financiera(visita)
        
        crisis_activa = False
        if alertas_finanzas:
            crisis_activa = True
            for alerta in alertas_finanzas:
                st.error(alerta)
        else:
            st.success("✅ Filtro Financiero Limpio: Sin deudas ni huelgas reportadas en las últimas horas.")

        # -----------------------------------------------------------------
        # CAPA 4: VERDICTO DEL ALGORITMO INTEGRADO
        # -----------------------------------------------------------------
        st.write("#### 🧠 Capa 4: Sugerencia Final del Sistema")
        
        if crisis_activa:
            st.error("❌ APUESTA BLOQUEADA: Alto peligro institucional. Los problemas de sueldos rompen cualquier lógica deportiva.")
        else:
            # Algoritmo de decisión cruzando Clima + Puestos de Descenso/Copas + Rachas
            es_clima_extremo = "Altura" in geo_local["tipo"] or "Calor" in geo_local["tipo"]
            
            if es_clima_extremo:
                # Caso: Local fuerte en clima contra visita obligada por Libertadores o Descenso
                if p_visita <= 4 or p_visita >= 16:
                    # Si la visita viene jugando bien (buena racha), peleará el partido
                    if r_visita >= 4:
                        st.info(f"🎯 **Sugerencia Óptima:** Ambos Anotan (Sí) O Más de 1.5 Goles.\n\n*Sustento:* {local} explota su geografía ({geo_local['tipo']}), pero {visita} se juega la vida por objetivos críticos en el Acumulado (Puesto {p_visita}) y llega con una excelente racha de confianza ({r_visita}/5) para hacer daño.")
                    else:
                        st.info(f"🎯 **Sugerencia Óptima:** Ganador Seco {local} (Local).\n\n*Sustento:* El clima de {geo_local['ciudad']} ({geo_local['tipo']}) sumado a la pésima racha anímica de la visita ({r_visita}/5) causará un desgaste físico y psicológico irreversible en el segundo tiempo.")
                else:
                    st.info(f"🎯 **Sugerencia Óptima:** Doble Oportunidad: Ganador Local o Empate.\n\n*Sustento:* Ventaja geográfica estable frente a un rival en zona media sin presiones críticas.")
            else:
                # Partidos en el Llano (Lima, Callao, Trujillo, etc.)
                if r_local >= 4 and r_visita >= 4:
                    st.info("🎯 **Sugerencia Óptima:** Ambos Anotan (Sí) O Más de 2.5 Goles.\n\n*Sustento:* Choque de poderes en condiciones climáticas neutras. Ambos equipos vienen con las rachas encendidas e inercias de ataque altas.")
                else:
                    st.info("🎯 **Sugerencia Óptima:** Menos de 3.5 Goles O Más de 1.5 Goles en total.\n\n*Sustento:* Dinámica regular de juego en llano sin factores externos de distorsión.")
else:
    st.info("💡 Por favor, sube el archivo 'liga1_data.xlsx' a la raíz de tu repositorio de GitHub para inicializar los módulos predictivos.")
