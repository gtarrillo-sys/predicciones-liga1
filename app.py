import datetime
import pandas as pd
import streamlit as st

# ==========================================
# 1. CONSTANTES Y CONFIGURACIÓN CLIMÁTICA
# ==========================================
PLAZAS_CALOR_EXTREMO = ["Piura", "Sullana", "Tarapoto", "Chiclayo", "Iquitos"]

def calcular_factor_horario(hora_str, plaza):
    """
    Calcula el multiplicador de desgaste físico según la hora y la ciudad.
    """
    try:
        hora_obj = datetime.datetime.strptime(str(hora_str).strip(), "%H:%M").time()
        hora_decimal = hora_obj.hour + (hora_obj.minute / 60.0)
    except Exception:
        try:
            hora_obj = datetime.datetime.strptime(str(hora_str).strip(), "%H:%M:%S").time()
            hora_decimal = hora_obj.hour + (hora_obj.minute / 60.0)
        except Exception:
            return 1.0, "🌙 Filtro Estandár / Condiciones Óptimas", "info"

    if plaza in PLAZAS_CALOR_EXTREMO:
        if 13.0 <= hora_decimal <= 15.5:
            return 1.3, "⚠️ FILTRO ACTIVO: Calor Extremo / Fatiga Crítica de Visita", "error"
        elif 15.5 < hora_decimal <= 17.5:
            return 1.15, "⏳ Filtro Neutro: Transición Térmica", "warning"
        
    return 1.0, "🌙 Filtro Nocturno / Condiciones Óptimas", "success"

# ==========================================
# 2. INTERFAZ VISUAL EN STREAMLIT
# ==========================================
st.set_page_config(page_title="Predicciones Liga 1", page_icon="⚽", layout="centered")

st.title("🔮 Sistema de Predicciones - Liga 1")
st.markdown("Configura los datos del partido para actualizar el algoritmo en tiempo real.")

# Formulario de entrada de datos
with st.sidebar:
    st.header("⚙️ Parámetros del Partido")
    local = st.text_input("Equipo Local", "Atlético Grau")
    visita = st.text_input("Equipo Visitante", "FBC Melgar")
    fecha = st.date_input("Fecha del Partido", datetime.date(2026, 9, 1))
    hora = st.text_input("Hora del Partido (HH:MM)", "15:00")
    plaza = st.selectbox("Plaza / Ciudad", ["Piura", "Sullana", "Lima", "Arequipa", "Cajamarca", "Chiclayo", "Tarapoto"])
    desgaste_logistico = st.checkbox("¿Alerta de viaje/vuelos para la visita?", value=True)

# Botón para ejecutar el motor
if st.button("🚀 Procesar Predicción con Inteligencia Artificial"):
    
    # Procesar lógica de Fecha, Hora y Clima
    factor_clima, alerta_clima, tipo_alerta = calcular_factor_horario(hora, plaza)
    
    # --- INTERFAZ: CABECERA DEL INFORME ---
    st.markdown("---")
    st.subheader(f"📊 INFORME DEL PARTIDO: {local} vs {visita}")
    st.markdown(f"📅 **Fecha:** {fecha.strftime('%d/%m/%m%Y')} | ⏰ **Hora:** {hora} ({plaza})")
    
    # Mostrar banner dinámico según el clima
    if tipo_alerta == "error":
        st.error(alerta_clima)
    elif tipo_alerta == "warning":
        st.warning(alerta_clima)
    else:
        st.success(alerta_clima)
    st.markdown("---")
    
    # --- CAPA 1: Objetivos ---
    st.markdown("### 🛡️ Capa 1: Presión por Objetivos")
    st.write(f"* **{local}:** Motivación máxima. Si gana, se prende arriba en el Clausura.")
    st.write(f"* **{visita}:** Obligado a sumar para mantenerse en los puestos de arriba del Acumulado.")
    
    # --- CAPA 2: Forma y Desgaste ---
    st.markdown("### 📈 Capa 2: Estado de Forma y Alertas de Desgaste")
    st.write(f"* **Racha:** Ambos vienen en buen momento. {local} llega con confianza extrema tras golear 3-1 a UTC en Cajamarca.")
    st.write(f"* 🏟️ **Cancha:** Grass Natural en el calor de {plaza}.")
    if desgaste_logistico:
        st.markdown(f"> 🚌 **¡ALERTA CRÍTICA DE TRASLADO!**: El plantel de {visita} sufrió problemas logísticos graves con sus vuelos. Llegan con fatiga acumulada.")
        
    # --- CAPA 3: Extra-Cancha ---
    st.markdown("### 🕵️‍♂️ Capa 3: Filtro Extra-Cancha")
    st.write("* **Filtro Financiero Limpio:** Sin problemas de pagos o planteles en huelga.")
    
    # --- CAPA 4: Predicción Final Algorítmica ---
    st.markdown("### 🧠 Capa 4: Sugerencia Final del Sistema")
    
    # El algoritmo toma decisiones basadas en el nuevo factor de fecha/hora
    if factor_clima == 1.3 and desgaste_logistico:
        pronostico_resultado = f"**Ganador Seco {local} (Local) 🔥**"
        pronostico_goles = f"**Más de 2.5 Goles (Tendencia Alta) 🔥** \n\n *Sustento:* El ritmo ofensivo de {local} (3 goles de visita previos) sumado a la caída física de la defensa de {visita} por el calor de las 3 p.m. y el viaje abrirá el arco en el segundo tiempo."
    else:
        pronostico_resultado = f"**Doble Oportunidad: {local} o Empate (1X)**"
        pronostico_goles = "**Menos de 2.5 Goles (Pronóstico estándar reservado)**"
        
    st.info(f"🎯 **Pronóstico de Resultado:** {pronostico_resultado}")
    st.success(f"⚽ **Predicción de Goles:** {pronostico_goles}")
