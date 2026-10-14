    else:
        st.info("No se encontró información en Historial_H2H.")

with tab5:
    st.subheader("🧾 Control de hojas y registros")
    resumen = []
    for hoja in HOJAS_ESPERADAS:
        df = datos.get(hoja, pd.DataFrame())
        resumen.append({
            "Hoja": hoja, "Filas": len(df), "Columnas": len(df.columns),
            "Estado": "Cargada" if hoja in datos["hojas_disponibles"] else "No encontrada",
        })
    st.dataframe(pd.DataFrame(resumen), use_container_width=True, hide_index=True)
    st.write("Columnas detectadas por hoja:")
    for hoja in HOJAS_ESPERADAS:
        with st.expander(hoja):
            st.write(list(datos.get(hoja, pd.DataFrame()).columns))
    st.write("Promedios base detectados en Formulas:")
    st.json({
        "Promedio goles local": datos["prior_local"],
        "Promedio goles visita": datos["prior_visita"],
    })
    for hoja in ["Resultados_Apertura", "Resultados_Clausura"]:
        dfh = datos.get(hoja, pd.DataFrame())
        if "fecha" not in dfh.columns:
            st.error(f"{hoja}: no se detectó la columna Fecha.")
        else:
            fechas = pd.to_datetime(dfh["fecha"], errors="coerce")
            validas = int(fechas.notna().sum())
            st.write(f"**{hoja}** — fechas válidas: {validas}/{len(dfh)}")
            if validas:
                st.write(f"Rango: {fechas.min().strftime('%d/%m/%Y')} → {fechas.max().strftime('%d/%m/%Y')}")

    st.success(
        "Control temporal activo: el histórico se corta estrictamente con fecha anterior al partido. "
        "La jornada queda como control secundario cuando no existe fecha."
    )

    control_prom = validar_promedios_clausura()
    st.write("**Control de coherencia de Prom_GF/GC de Resultados_Clausura**")
    st.json({
        "filas_clausura": control_prom["filas"],
        "filas_comparables": control_prom["filas_comparables"],
        "filas_con_diferencia": control_prom["filas_con_diferencia"],
        "diferencia_maxima": round(control_prom["diferencia_maxima"], 6)
            if np.isfinite(control_prom["diferencia_maxima"]) else None,
    })
    if control_prom["filas_con_diferencia"] > 0:
        st.warning(
            "Las columnas Prom_GF/GC almacenadas en Resultados_Clausura no son "
            "confiables como entrada directa: se detectaron diferencias frente al "
            "cálculo cronológico por fecha. El modelo las recalcula internamente "
            "con los equipos de la fila actual y solo utiliza las columnas Prom_* "
            "del Excel como referencia de auditoría."
        )
    else:
        st.success("Las columnas Prom_GF/GC coinciden con el cálculo cronológico del modelo.")

    st.info(
        "Metodología Clausura: Prom_GF/GC se calcula acumulando Apertura y Clausura previos al partido. "
        "La lambda base sigue ((K×N)/D2) para local y ((M×L)/D6) para visitante. "
        "Los factores de altitud se aplican usando la columna J (Altitud_Visita), "
        "tal como está escrito en las fórmulas del Excel actualizado."
    )

with tab6:
    st.subheader("🔥 Tensión competitiva por partido")
    st.caption(
        "La lectura usa la columna Situación Actual y el puesto de la Tabla Acumulada. "
        "No declara matemáticamente eliminado a ningún equipo: para ello se requiere "
        "simular todos los calendarios restantes y aplicar el reglamento completo."
    )
    if df_pronosticos.empty:
        st.info("No hay partidos para evaluar.")
    else:
        cols = [
            "Día", "Hora", "Local", "Visita", "Objetivo local", "Objetivo visita",
            "Presión competitiva", "Índice presión (0-1)", "Pronóstico 1X2",
        ]
        st.dataframe(
            df_pronosticos[cols].style.apply(estilizar_texto, axis=1),
            use_container_width=True, hide_index=True,
        )

    st.markdown("#### Lectura de objetivos según la tabla acumulada")
    if not tabla_acumulada.empty:
        tabla_tension = tabla_acumulada.copy()
        col_equipo = primera_columna(tabla_tension, ["equipo", "club"])
        col_rango = primera_columna(tabla_tension, ["rango", "pos"])
        col_situacion = primera_columna(tabla_tension, ["situacion_actual"])
        if col_equipo:
            tabla_tension["Objetivo estimado"] = tabla_tension.apply(
                lambda r: objetivo_equipo(r), axis=1
            )
            tabla_tension["Urgencia estimada (0-1)"] = tabla_tension.apply(
                lambda r: round(urgencia_equipo(r), 2), axis=1
            )
            columnas = [c for c in [
                col_rango, col_equipo, "pj", "pts", "dg", col_situacion,
                "Objetivo estimado", "Urgencia estimada (0-1)"
            ] if c and c in tabla_tension.columns]
            st.dataframe(
                tabla_tension[columnas].sort_values(col_rango if col_rango else col_equipo),
                use_container_width=True, hide_index=True
            )
    st.warning(
        "La presión psicológica no tiene un efecto universal: algunos equipos elevan "
        "su rendimiento y otros se vuelven más conservadores. Por eso el índice se "
        "presenta como señal contextual y no debe interpretarse como una probabilidad "
        "de ganar ni como una garantía de clasificación o descenso."
    )



