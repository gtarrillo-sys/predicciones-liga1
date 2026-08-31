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

st.title("🤖 Sistema de Apuestas Profesional 3.0")
st.markdown("### Motor de Predicción Extremo: Geografía + Canchas + Desgaste por Traslado + Radar Financiero")
st.markdown("---")

# =====================================================================
# 2. CARGA DE BASE DE DATOS DESDE EXCEL
# =====================================================================
PALABRAS_CRITICAS = ["sueldos", "deuda", "safap", "paro", "no concentran", "licencias", "resta de puntos", "huelga"]

def cargar_base_de_datos():
    try:
        df_tabla = pd.read_excel("liga1_data.xlsx", sheet_name="Tabla_Posiciones")
        df_geo = pd.read_excel("liga1_data.xlsx", sheet_name="Data_Geografica")
        
        lista_partidos_fecha = []
        nombre_jornada = "Jornada Actual"
        
        try:
            df_partidos = pd.read_excel("liga1_data.xlsx", sheet_name="Partidos_Fecha")
            if "Jornada" in df_partidos.columns and not df_partidos.empty:
                nombre_jornada = str(df_partidos["Jornada"].iloc[0])
            
            for _, row in df_partidos.iterrows():
                lista_partidos_fecha.append({
                    "texto": f"⚽ {row['Local']} vs {row['Visitante']}",
                    "local": row['Local'],
                    "visita": row['Visitante']
                })
        except Exception as err_pestaña:
            st.warning(f"⚠️ Nota: No se pudo leer la pestaña 'Partidos_Fecha'. Detalle: {err_pestaña}")
        
        tabla_dict = dict(zip(df_tabla["Puesto"], df_tabla["Club"]))
        
        geo_dict = {}
        for _, row in df_geo.iterrows():
            geo_dict[row["Club"]] = {
                "ciudad": row["Ciudad"],
                "tipo": row["Tipo_Clima"],
                "factor_local": float(row["Factor_Local"]),
                "cancha": str(row["Tipo_Cancha"]) if "Tipo_Cancha" in df_geo.columns else "Natural",
                "traslado_complejo": int(row["Traslado_Complejo"]) if "Traslado_Complejo" in df_geo.columns else 0
            }
            
        racha_dict = dict(zip(df_geo["Club"], df_geo["Racha"] if "Racha" in df_geo.columns else [3]*18))
        
        return tabla_dict, geo_dict, racha_dict, lista_partidos_fecha, nombre_jornada
    except Exception as e:
        st.error(f"❌ Error crítico general al cargar 'liga1_data.xlsx'. Detalle: {e}")
        return {}, {}, {}, [], "Error"

TABLA_ACUMULADA, DATA_GEOGRAFICA, FACTOR_RACHA, PARTIDOS_PROGRAMADOS, NOMBRE_JORNADA = cargar_base_de_datos()

# =====================================================================
# 3. FUNCIONES LÓGICAS Y RASTREADORES
# =====================================================================
def obtener_puesto_acumulado(equipo):
    for puesto, nombre in TABLA_ACUMULADA.items():
        if equipo.lower() in nombre.lower() or nombre.lower() in equipo.lower(): 
            return puesto
    return 99

def mapear_nombre_estandar(equipo_raw):
    for nombre_real in DATA_GEOGRAFICA.keys():
        if equipo_raw.lower() in nombre_real.lower() or nombre_real.lower() in equipo_raw.lower():
            return nombre_real
    return equipo_raw

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
# 4. INTERFAZ DE USUARIO INTERACTIVA
# =====================================================================
if DATA_GEOGRAFICA and TABLA_ACUMULADA:
    lista_equipos = sorted(list(DATA_GEOGRAFICA.keys()))

    idx_local_def = 0
    idx_visita_def = 1 if len(lista_equipos) > 1 else 0
    
    if PARTIDOS_PROGRAMADOS:
        st.write(f"#### 🗓️ Análisis Planificado: **{NOMBRE_JORNADA}**")
        opciones_partidos = [p["texto"] for p in PARTIDOS_PROGRAMADOS] + ["🔄 Hacer un Cruce Manual / Libre"]
        partido_seleccionado = st.selectbox("Selecciona uno de los 9 partidos de la fecha:", opciones_partidos)
        
        if partido_seleccionado != "🔄 Hacer un Cruce Manual / Libre":
            for p in PARTIDOS_PROGRAMADOS:
                if p["texto"] == partido_seleccionado:
                    local_std = mapear_nombre_estandar(p["local"])
                    visita_std = mapear_nombre_estandar(p["visita"])
                    
                    if local_std in lista_equipos: idx_local_def = lista_equipos.index(local_std)
                    if visita_std in lista_equipos: idx_visita_def = lista_equipos.index(visita_std)
                    break
        st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        local = st.selectbox("Selecciona Equipo Local:", lista_equipos, index=idx_local_def)
    with col2:
        visita = st.selectbox("Selecciona Equipo Visitante:", lista_equipos, index=idx_visita_def)

    if st.button("⚡ INICIAR PROCESAMIENTO MATEMÁTICO", use_container_width=True):
        st.markdown("---")
        st.subheader(f"📊 Informe del Partido: {local} vs {visita}")
        
        p_local = obtener_puesto_acumulado(local)
        p_visita = obtener_puesto_acumulado(visita)
        r_local = FACTOR_RACHA.get(local, 3)
        r_visita = FACTOR_RACHA.get(visita, 3)
        geo_local = DATA_GEOGRAFICA[local]
        geo_visita = DATA_GEOGRAFICA[visita]

        # CAPA 1: TABLA ACUMULADA
        st.write("#### 🛡️ Capa 1: Presión por Objetivos (Tabla Acumulada)")
        if p_local <= 4:
            st.info(f"🏆 {local} (Puesto {p_local}) defiende zona de Copa Libertadores.")
        elif p_local <= 8:
            st.info(f"🎟️ {local} (Puesto {p_local}) se encuentra en puestos de Copa Sudamericana.")
        elif p_local >= 16:
            st.warning(f"⚠️ {local} (Puesto {p_local}) está en ZONA DE DESCENSO DIRECTO.")
            
        if p_visita <= 4:
            st.info(f"🏆 {visita} (Puesto {p_visita}) está obligado a proponer afuera.")
        elif p_visita <= 8:
            st.info(f"🎟️ {visita} (Puesto {p_visita}) busca consolidar su clasificación.")
        elif p_visita >= 16:
            st.warning(f"⚠️ {visita} (Puesto {p_visita}) pelea el DESCENSO.")

        # CAPA 2: RACHA Y ESTADO FÍSICO (NUEVO TRASLADO + CANCHA)
        st.write("#### 📈 Capa 2: Estado de Forma y Alertas de Desgaste")
        if r_local >= 4:
            st.success(f"🟩 {local} viene en inercia ganadora (Racha: {r_local}/5).")
        elif r_local <= 2:
            st.error(f"🟥 {local} arrastra una crisis de resultados (Racha: {r_local}/5).")
            
        if r_visita >= 4:
            st.success(f"🟩 {visita} llega con un ritmo competitivo sólido (Racha: {r_visita}/5).")
        elif r_visita <= 2:
            st.error(f"🟥 {visita} está golpeado anímicamente (Racha: {r_visita}/5).")

        # Alertas de canchas y traslados
        if geo_local["cancha"] == "Sintético":
            st.warning(f"🏟️ ALERTA DE GRAMADO: {local} juega en Grass Sintético. El balón corre más rápido y causa fatiga articular severa en equipos no acostumbrados.")
        
        if geo_local["traslado_complejo"] == 1:
            st.warning(f"🚌 ALERTA DE TRASLADO: Llegar a la ciudad de {local} ({geo_local['ciudad']}) requiere viajes largos por carretera o escalas pesadas. {visita} sufrirá un desgaste físico extra por el viaje.")

        # CAPA 3: FILTRO FINANCIERO
        st.write("#### 🕵️‍♂️ Capa 3: Filtro Extra-Cancha (Problemas de Pagos)")
        with st.spinner("Escaneando diarios deportivos..."):
            alertas_finanzas = escanear_crisis_financiera(local) + escanear_crisis_financiera(visita)
        
        crisis_activa = False
        if alertas_finanzas:
            crisis_activa = True
            for alerta in alertas_finanzas:
                st.error(alerta)
        else:
            st.success("✅ Filtro Financiero Limpio: Sin deudas ni huelgas reportadas.")

        # =====================================================================
        # CAPA 4: SUGERENCIA FINAL DEL SISTEMA EXTREMA
        # =====================================================================
        st.write("#### 🧠 Capa 4: Sugerencia Final del Sistema")
        
        if crisis_activa:
            st.error("❌ APUESTA BLOQUEADA: Alto peligro institucional por deudas.")
        else:
            es_clima_extremo = "Altura" in geo_local["tipo"] or "Calor" in geo_local["tipo"]
            
            # --- NUEVO CÁLCULO DE GOLES CON TRASLADO Y CANCHAS ---
            # Si el viaje es destructivo o es altura brava, los equipos bajan el ritmo al final (Menos goles)
            if (r_local <= 2 and r_visita <= 2) or ("Altura" in geo_local["tipo"] and geo_local["traslado_complejo"] == 1 and r_visita <= 2):
                goles_prediccion = "Menos de 2.5 Goles en el partido (Baja anotación)"
                sustento_goles = "El desgaste del traslado pesado a la altura hará que la visita dosifique energías, priorizando defenderse en bloque bajo."
            # Si el local vuela y la visita viene liquidada o el pasto sintético acelera el juego de ataque
            elif (r_local >= 4) or (r_visita <= 1) or (geo_local["cancha"] == "Sintético" and r_local >= 3):
                goles_prediccion = "Más de 2.5 Goles (Tendencia Alta / Goleada Probable) 🔥"
                sustento_goles = "El dinamismo de la cancha o el colapso físico defensivo de la visita por el viaje/clima facilitará transiciones rápidas y goles."
            else:
                goles_prediccion = "Más de 1.5 Goles totales (Línea segura)"
                sustento_goles = "Dinámica estándar donde las ventajas espaciales permitirán movimientos en las áreas."

            # --- NUEVO CÁLCULO DE RESULTADO CON FACTOR LOGÍSTICO ---
            # Si se junta Calor/Altura + Traslado matador o pasto sintético raro, la ventaja local es casi destructiva
            if es_clima_extremo or geo_local["cancha"] == "Sintético" or geo_local["traslado_complejo"] == 1:
                if p_visita >= 16 or r_visita <= 2:
                    sug_resultado = f"Ganador Seco {local} (Local) 🔥 - Fija por Desgaste Extremo"
                else:
                    sug_resultado = "Doble Oportunidad: Ganador Local o Empate"
            else:
                if r_local >= 4 and r_visita >= 4:
                    sug_resultado = "Ambos Anotan (Sí)"
                elif p_local >= 16 or p_visita >= 16:
                    sug_resultado = "Doble Oportunidad: Local o Visitante"
                else:
                    sug_resultado = "Doble Oportunidad: Ganador Local o Empate"

            # --- MOSTRAR CUADRO FINAL EN PANTALLA ---
            texto_final = f"🎯 **Pronóstico de Resultado:** {sug_resultado}\n\n⚽ **Predicción de Goles:** {goles_prediccion}\n\n*Sustento Logístico:* {local} explota su localía en cancha tipo {geo_local['cancha']} con clima {geo_local['tipo']}. El rival ({visita}) arrastra el impacto del factor traslado (Nivel: {geo_local['traslado_complejo']}) y su posición en el acumulado (Puesto {p_visita})."
            st.info(texto_final)
else:
    st.info("💡 Por favor, sube el archivo 'liga1_data.xlsx' para inicializar los módulos.")
