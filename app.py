# =====================================================================
# 4. INTERFAZ DE USUARIO INTERACTIVA
# =====================================================================
if DATA_GEOGRAFICA and TABLA_ACUMULADA:
    lista_equipos = sorted(list(DATA_GEOGRAFICA.keys()))

    idx_local_def = 0
    idx_visita_def = 1 if len(lista_equipos) > 1 else 0
    
    if PARTIDOS_PROGRAMADOS:
        st.write(f"#### 🗓️ Análisis Planificado: **{NOMBRE_JORNADA}**")
        
        # Ponemos los 9 partidos primero para que el primero de la lista salga por defecto
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

        # CAPA 1: TABLA ACUMULADA
        st.write("#### 🛡️ Capa 1: Presión por Objetivos (Tabla Acumulada)")
        if p_local <= 4:
            st.info(f"🏆 {local} (Puesto {p_local}) defiende zona de Copa Libertadores. Motivación económica máxima.")
        elif p_local <= 8:
            st.info(f"🎟️ {local} (Puesto {p_local}) se encuentra en puestos de Copa Sudamericana.")
        elif p_local >= 16:
            st.warning(f"⚠️ {local} (Puesto {p_local}) está en ZONA DE DESCENSO DIRECTO. Desesperación alta, juego físico severo.")
            
        if p_visita <= 4:
            st.info(f"🏆 {visita} (Puesto {p_visita}) está obligado a proponer afuera para mantener cupo a Libertadores.")
        elif p_visita <= 8:
            st.info(f"🎟️ {visita} (Puesto {p_visita}) busca consolidar su clasificación a Sudamericana.")
        elif p_visita >= 16:
            st.warning(f"⚠️ {visita} (Puesto {p_visita}) pelea el DESCENSO. Planteará un bloque defensivo muy bajo (bus atrás).")

        # CAPA 2: RACHA
        st.write("#### 📈 Capa 2: Termómetro de Forma Reciente (Racha)")
        if r_local >= 4:
            st.success(f"🟩 {local} viene en inercia ganadora (Racha: {r_local}/5). Confianza alta que disminuye el cansancio.")
        elif r_local <= 2:
            st.error(f"🟥 {local} arrastra una crisis de resultados (Racha: {r_local}/5). Vulnerabilidad al recibir el primer gol.")
            
        if r_visita >= 4:
            st.success(f"🟩 {visita} llega con un ritmo competitivo sólido (Racha: {r_visita}/5).")
        elif r_visita <= 2:
            st.error(f"🟥 {visita} está golpeado anímicamente (Racha: {r_visita}/5). Propensión a errores defensivos forzados.")

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
            st.success("✅ Filtro Financiero Limpio: Sin deudas ni huelgas reportadas en las últimas horas.")

        # CAPA 4: SUGERENCIA FINAL
        st.write("#### 🧠 Capa 4: Sugerencia Final del Sistema")
        if crisis_activa:
            st.error("❌ APUESTA BLOQUEADA: Alto peligro institucional. Los problemas de sueldos rompen cualquier lógica deportiva.")
        else:
            es_clima_extremo = "Altura" in geo_local["tipo"] or "Calor" in geo_local["tipo"]
            if es_clima_extremo:
                if p_visita <= 4 or p_visita >= 16:
                    if r_visita >= 4:
                        st.info(f"🎯 **Sugerencia Óptima:** Ambos Anotan (Sí) O Más de 1.5 Goles.\n\n*Sustento:* {local} explota su geografía ({geo_local['tipo']}), pero {visita} se juega la vida por objetivos críticos en el Acumulado (Puesto {p_visita}) y llega con una excelente racha de confianza ({r_visita}/5) para hacer daño.")
                    else:
                        st.info(f"🎯 **Sugerencia Óptima:** Ganador Seco {local} (Local).\n\n*Sustento:* El clima de {geo_local['ciudad']} ({geo_local['tipo']}) sumado a la pésima racha anímica de la visita ({r_visita}/5) causará un desgaste físico y psicológico irreversible en el segundo tiempo.")
                else:
                    st.info(f"🎯 **Sugerencia Óptima:** Doble Oportunidad: Ganador Local o Empate.\n\n*Sustento:* Ventaja geográfica estable frente a un rival en zona media sin presiones críticas.")
            else:
                if r_local >= 4 and r_visita >= 4:
                    st.info("🎯 **Sugerencia Óptima:** Ambos Anotan (Sí) O Más de 2.5 Goles.\n\n*Sustento:* Choque de poderes en condiciones climáticas neutras. Ambos equipos vienen con las rachas encendidas e inercias de ataque altas.")
                else:
                    st.info("🎯 **Sugerencia Óptima:** Menos de 3.5 Goles O Más de 1.5 Goles en total.\n\n*Sustento:* Dinámica regular de juego en llano sin factores externos de distorsión.")
