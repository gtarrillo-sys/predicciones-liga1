import datetime
import requests
from bs4 import BeautifulSoup

# =====================================================================
# 1. CONFIGURACIÓN DE BASE DE DATOS DEL SISTEMA
# =====================================================================

# Palabras clave para el Radar Financiero y de Crisis Extra-cancha
PALABRAS_CRITICAS = ["sueldos", "deuda", "safap", "paro", "no concentran", "licencias", "resta de puntos", "huelga"]

# Diccionario Geográfico de la Liga 1 (Estadio, Altura/Clima, Factor de Ventaja)
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

# Tabla Acumulada Oficial (Actualizable cada jornada)
TABLA_ACUMULADA = {
    1: "Alianza Lima", 2: "Universitario", 3: "Deportivo Garcilaso", 
    4: "Chankas CYC", 5: "Melgar", 6: "Cusco", 7: "Cienciano", 
    8: "Alianza Atlético", 9: "Sport Boys", 10: "Sporting Cristal",
    11: "Comerciantes Unidos", 12: "Colegio Juan Pablo II", 13: "Atlético Grau",
    14: "Deporte Huancayo", 15: "CD Moquegua", 16: "FC Cajamarca",
    17: "ADT", 18: "UTC"
}

# =====================================================================
# 2. MÓDULOS DE PROCESAMIENTO (REGLAS DE NEGOCIO)
# =====================================================================

def buscar_puesto_tabla(equipo):
    """Busca la posición exacta de un equipo en la tabla mapeada."""
    for puesto, nombre in TABLA_ACUMULADA.items():
        if equipo.lower() in nombre.lower():
            return puesto
    return 99

def escanear_crisis_financiera(equipo):
    """Escanea portales de noticias deportivos (RSS) buscando alertas financieras del equipo."""
    fuentes_rss = [
        "https://www.ovacion.pe/rss",
        # Puedes agregar más urls de feeds RSS aquí (Rpp, Líbero, etc.)
    ]
    alertas_encontradas = []
    
    for url in fuentes_rss:
        try:
            response = requests.get(url, timeout=8)
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            for item in items:
                titulo = item.title.text.lower() if item.title else ""
                descripcion = item.description.text.lower() if item.description else ""
                contenido_noticia = titulo + " " + descripcion
                
                # Si se menciona al equipo en la noticia
                if equipo.lower() in contenido_noticia:
                    for palabra in PALABRAS_CRITICAS:
                        if palabra in contenido_noticia:
                            alertas_encontradas.append(f"🚨 CRISIS DETECTADA: '{palabra.upper()}' hallado en: '{item.title.text}'")
                            break
        except Exception:
            # Si una fuente RSS se cae, el sistema continúa con el resto
            pass
            
    return list(set(alertas_encontradas))

def calcular_pesos_y_pronostico(local, visita, alertas_tabla, crisis_activa):
    """Cruza todas las capas y genera la recomendación final del sistema."""
    print("\n[🧠 CAPA DE DECISIÓN FINAL: ALGORITMO INTEGRADO]")
    
    if crisis_activa:
        print("❌ VEREDICTO: APUESTA BLOQUEADA COMPLETAMENTE.")
        print("💡 Motivo: El riesgo extra-cancha por deudas/huelgas destruye cualquier probabilidad lógica.")
        return

    geo_local = DATA_GEOGRAFICA.get(local, {"tipo": "Desconocido", "factor_local": 1.0})
    puesto_local = buscar_puesto_tabla(local)
    puesto_visita = buscar_puesto_tabla(visita)
    
    # Lógica de Recomendación Automatizada
    if "Altura Extremada" in geo_local["tipo"] or "Calor Extremo" in geo_local["tipo"]:
        if puesto_visita <= 4:
            # Visita arriba en la tabla, no se va a regalar a pesar de la geografía dura
            print(f"🎯 SUGERENCIA: Ambos Equipos Anotan (Sí) O Más de 1.5 Goles.")
            print(f"📝 Sustento: {local} tiene la ventaja climática ({geo_local['tipo']}), pero {visita} va superior en la tabla y está obligado a proponer y buscar goles.")
        else:
            print(f"🎯 SUGERENCIA: Ganador Seco {local} ó {local} + Más de 1.5 goles.")
            print(f"📝 Sustento: La geografía extrema ({geo_local['tipo']}) favorece al local, y la visita no tiene la urgencia o jerarquía alta en la tabla para contrarrestarlo.")
    else:
        # Partidos en el llano o condiciones estables
        print("🎯 SUGERENCIA: Total de Goles: Más de 1.5 O Doble Oportunidad Local/Empate.")
        print("📝 Sustento: Condiciones de juego estables y planteles financieramente limpios. Se juega por inercia futbolística estándar.")

# =====================================================================
# 3. INTERFAZ DE CONTROL (EJECUCIÓN PRINCIPAL)
# =====================================================================

def ejecutar_protocolo_sistema_2_0(local, visita):
    print("=" * 70)
    print(f"🤖 RUNNING SISTEMA DE APUESTAS 2.0 - LIGA 1 PERÚ")
    print(f"Control de Jornada - Hora de consulta: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 70)
    
    # --- CAPA 1: EVALUACIÓN DE TABLA Y CONTEXTO PSICOLÓGICO ---
    print("\n[📊 CAPA 1: ANALIZANDO TABLA ACUMULADA Y CONTEXTO]")
    p_local = buscar_puesto_tabla(local)
    p_visita = buscar_puesto_tabla(visita)
    alertas_tabla = False
    
    if p_local == 99 or p_visita == 99:
        print("⚠️ Advertencia: Uno de los equipos no fue localizado correctamente en el mapa de la tabla.")
    
    if p_local >= 15:
        print(f"⚠️ Alerta de Fricción: {local} (Puesto {p_local}) está en ZONA DE DESCENSO. Alto riesgo de tarjetas rojas.")
        alertas_tabla = True
    elif p_local <= 3:
        print(f"🔥 Alerta de Ansiedad: {local} (Puesto {p_local}) pelea el LIDERATO. Presión alta en los primeros 20 minutos.")
        
    if p_visita >= 15:
        print(f"⚠️ Alerta de Resistencia: {visita} (Puesto {p_visita}) pelea el DESCENSO. Plantamiento ultra-defensivo esperado.")
        alertas_tabla = True
        
    if not alertas_tabla:
        print("✅ Contexto de tabla regular. Sin distorsiones extremas por baja presión.")

    # --- CAPA 2: RADAR AUTOMATIZADO DE CRISIS ECONÓMICA ---
    print("\n[🕵️‍♂️ CAPA 2: EJECUTANDO RADAR DE CRISIS FINANCIERA (SCRAPING)]")
    print(f"Buscando anomalías institucionales para {local} y {visita}...")
    
    alertas_finanzas = escanear_crisis_financiera(local) + escanear_crisis_financiera(visita)
    crisis_activa = False
    
    if alertas_finanzas:
        crisis_activa = True
        for alerta in alertas_finanzas:
            print(alerta)
    else:
        print("✅ Filtro Limpio: No se detectaron huelgas, deudas ni problemas de planillas vigentes.")

    # --- CAPA 3: CRUCE Y GENERACIÓN DE RECOMENDACIÓN ---
    calcular_pesos_y_pronostico(local, visita, alertas_tabla, crisis_activa)
    print("=" * 70 + "\n")

# =====================================================================
# ZONA DE PRUEBA EN VIVO
# =====================================================================
if __name__ == "__main__":
    # Probamos el partido del cierre: Grau vs Melgar
    ejecutar_protocolo_sistema_2_0("Atlético Grau", "Melgar")
