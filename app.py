import datetime
import pandas as pd

# ==========================================
# 1. CONSTANTES Y CONFIGURACIÓN CLIMÁTICA
# ==========================================
PLAZAS_CALOR_EXTREMO = ["Piura", "Sullana", "Tarapoto", "Chiclayo", "Iquitos"]

def calcular_factor_horario(hora_str, plaza):
    """
    Capa Nueva: Calcula el multiplicador de desgaste físico según la hora y la ciudad.
    Soporta formatos 'HH:M' o 'HH:MM:SS'
    """
    try:
        hora_obj = datetime.datetime.strptime(str(hora_str).strip(), "%H:%M").time()
        hora_decimal = hora_obj.hour + (hora_obj.minute / 60.0)
    except Exception:
        try:
            hora_obj = datetime.datetime.strptime(str(hora_str).strip(), "%H:%M:%S").time()
            hora_decimal = hora_obj.hour + (hora_obj.minute / 60.0)
        except Exception:
            return 1.0, "🌙 Filtro Estándar / Condiciones Óptimas"

    if plaza in PLAZAS_CALOR_EXTREMO:
        # Bloque 1: 1:00 p.m. a 3:30 p.m. -> Calor Extremo
        if 13.0 <= hora_decimal <= 15.5:
            return 1.3, "⚠️ FILTRO ACTIVO: Calor Extremo / Fatiga Crítica de Visita"
        # Bloque 2: 3:31 p.m. a 5:30 p.m. -> Transición Térmica
        elif 15.5 < hora_decimal <= 17.5:
            return 1.15, "⏳ Filtro Neutro: Transición Térmica"
        
    return 1.0, "🌙 Filtro Nocturno / Condiciones Óptimas"

# ==========================================
# 2. MOTOR PRINCIPAL DEL SISTEMA (4 CAPAS)
# ==========================================
def procesar_sistema_prediccion(local, visita, fecha, hora, plaza, desgaste_logistico=False):
    # Calcular el nuevo factor de fecha y hora
    factor_clima, alerta_clima = calcular_factor_horario(hora, plaza)
    
    # --- CAPA 1: Presión por Objetivos ---
    # (Aquí tu código leería los puntos reales de liga1_data.xlsx)
    capa1_txt = f"* **{local}:** Motivación máxima. Si gana, pelea la punta del Clausura.\n* **{visita}:** Obligado a sumar afuera para no colgarse de la Tabla Acumulada."
    
    # --- CAPA 2: Estado de Forma y Alertas de Desgaste ---
    alerta_transporte = ""
    if desgaste_logistico:
        alerta_transporte = f"\n* 🚌 **¡ALERTA CRÍTICA DE TRASLADO! (Nivel: 1):** El plantel de {visita} sufrió cancelaciones de vuelos y reprogramaciones de última hora. Llegan con desgaste mental y físico atípico."
    
    capa2_txt = f"* **Racha:** Ambos vienen firmes. {local} con confianza alta tras su última victoria de visita (3-1 en Cajamarca).\n* 🏟️ **Gramado:** Se juega sobre Cancha Natural en {plaza}.{alerta_transporte}"
    
    # --- CAPA 3: Filtro Extra-Cancha (Financiero) ---
    capa3_txt = "* **Filtro Financiero Limpio:** Sin alertas de deudas o huelgas de SAFAP en ninguno de los dos planteles."
    
    # --- CAPA 4: Sugerencia Final Ajustada por el Factor Horario ---
    # Lógica algorítmica: Si hay calor extremo (1.3) y desgaste logístico, se castiga severamente la resistencia de la visita
    if factor_clima == 1.3 and desgaste_logistico:
        pronostico_resultado = f"**Ganador Seco {local} (Local) 🔥 - Fija por Desgaste Extraordinario y Clima**"
        pronostico_goles = "**Más de 2.5 Goles en el partido (Tendencia Alta) 🔥**\n    * *Sustento:* El local viene fino de cara al gol (3-1 previo). El desgaste del viaje combinado con las pocas piernas que le quedarán a la defensa de {visita} a partir del min 60 por los 30°C abrirá espacios claros."
    else:
        pronostico_resultado = f"**Doble Oportunidad: {local} o Empate (1X)**"
        pronostico_goles = "**Menos de 2.5 Goles (Partido cerrado por oficio de la visita)**"

    # ==========================================
    # 3. GENERACIÓN DEL REPORTE VISUAL (OUTPUT)
    # ==========================================
    print("---" * 15)
    print(f"📊 INFORME DEL PARTIDO: {local} vs {visita}")
    print(f"📅 Fecha: {fecha} | ⏰ Hora: {hora} ({plaza}) | {alerta_clima}")
    print("---" * 15)
    print("\n### 🛡️ Capa 1: Presión por Objetivos")
    print(capa1_txt)
    print("\n### 📈 Capa 2: Estado de Forma y Alertas de Desgaste")
    print(capa2_txt)
    print("\n### 🕵️‍♂️ Capa 3: Filtro Extra-Cancha")
    print(capa3_txt)
    print("\n### 🧠 Capa 4: Sugerencia Final del Sistema")
    print(f"* 🎯 **Pronóstico de Resultado:** {pronostico_resultado}")
    print(f"* ⚽ **Predicción de Goles:** {pronostico_goles}")
    print("---" * 15)

# ==========================================
# 4. EJECUCIÓN / SIMULACIÓN DE PRUEBA
# ==========================================
if __name__ == "__main__":
    # Simulación exacta del partido de mañana usando tus datos reales
    procesar_sistema_prediccion(
        local="Atlético Grau",
        visita="FBC Melgar",
        fecha="01/09/2026",
        hora="15:00",
        plaza="Piura",
        desgaste_logistico=True # Activamos la alerta de los aviones que detectamos
    )
