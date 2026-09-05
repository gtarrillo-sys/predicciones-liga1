import pandas as pd
import streamlit as st

# Configuración inicial de la página
st.set_page_config(
    page_title="Sistema de Predicciones Liga 1 2026",
    page_icon="⚽",
    layout="wide",
)


# --- 1. CARGA DE DATOS DESDE EXCEL ---
@st.cache_data
def cargar_datos():
  excel_path = "Liga1_2026.xlsx"

  # Carga de hojas
  df_geo = pd.read_excel(excel_path, sheet_name="Data_Geografica")
  df_resultados = pd.read_excel(excel_path, sheet_name="Resultados_Clausura")

  # Importante: dtype=str fuerza la lectura exacta del texto en las celdas
  df_proximos = pd.read_excel(
      excel_path, sheet_name="Partidos_Fecha", dtype=str
  )

  df_clausura = pd.read_excel(excel_path, sheet_name="Tabla_Clausura")
  df_acumulado = pd.read_excel(excel_path, sheet_name="Tabla_Acumulada")

  # Limpiar espacios en blanco en los nombres de las columnas
  for df in [df_geo, df_resultados, df_proximos, df_clausura, df_acumulado]:
    df.columns = df.columns.astype(str).str.strip()

  return df_geo, df_resultados, df_proximos, df_clausura, df_acumulado


# Cargar los datasets
df_geo, df_resultados, df_proximos, df_clausura, df_acumulado = cargar_datos()

# --- 2. ENCABEZADO ---
st.title("⚽ Sistema de Predicciones Liga 1 2026")
st.subheader("Modelo Estadístico para la Liga 1 Peruana")

# Pestañas principales
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Pronóstico Individual por Partido",
    "🧢 Resumen de la Jornada",
    "🏆 Tabla Clausura",
    "📊 Tabla Acumulada",
    "🗺️ Data Geográfica",
])


# --- 3. PESTAÑA 1: PRONÓSTICO INDIVIDUAL POR PARTIDO ---
with tab1:
  st.header("🔍 Análisis Detallado")

  col_jornada, col_partido = st.columns(2)

  with col_jornada:
    lista_jornadas = df_proximos["Jornada"].dropna().unique().tolist()
    jornada_sel = st.selectbox(
        "📅 Selecciona la Fecha:",
        lista_jornadas,
        key="sb_jornada_tab1",
    )

  # Filtrar los partidos correspondientes a la jornada seleccionada
  df_jornada = df_proximos[df_proximos["Jornada"] == jornada_sel].copy()

  # Crear una columna de etiqueta para el selector (Local vs Visita)
  df_jornada["Partido_Label"] = (
      df_jornada["Local"].astype(str) + " vs " + df_jornada["Visita"].astype(str)
  )

  with col_partido:
    partido_sel = st.selectbox(
        "⚔️ Selecciona el Partido:",
        df_jornada["Partido_Label"].tolist(),
        key="sb_partido_tab1",
    )

  # Fila seleccionada
  row_match = df_jornada[
      df_jornada["Partido_Label"] == partido_sel
  ].iloc[0]

  # --- PROCESAMIENTO Y DEPURACIÓN DE FECHA / DIA / HORA ---
  f_val = str(row_match.get("Fecha", "")).strip()
  d_val = str(row_match.get("Dia", "")).strip()
  h_val = str(row_match.get("Hora", "")).strip()

  # Formatear la Fecha uniendo las columnas 'Dia' y 'Fecha'
  if f_val and f_val.lower() != "nan":
    fecha_limpia = f_val.split(" ")[0]  # Limpia tiempo si viniera adjunto
    if d_val and d_val.lower() != "nan":
      fecha_str = f"{d_val} {fecha_limpia}"
    else:
      fecha_str = fecha_limpia
  else:
    fecha_str = "Fecha por confirmar"

  # Formatear la Hora
  if h_val and h_val.lower() != "nan":
    hora_str = h_val.split(" ")[-1][:5]
  else:
    hora_str = ""

  info_horario = (
      f"📅 {fecha_str} - 🕒 {hora_str}" if hora_str else f"📅 {fecha_str}"
  )

  # Datos del estadio y altitud del Local
  estadio_str = str(row_match.get("Estadio", "")).strip()
  alt_local = row_match.get("Altitud_Local", "0")

  # Cabecera descriptiva del partido seleccionado
  st.subheader(
      f"🏟️ {partido_sel} | {str(row_match.get('Ciudad', ''))} ({alt_local}"
      " msnm)"
  )
  st.caption(info_horario)

  st.write("---")

  # --- SIMULACIÓN / MUESTRA DE PROBABILIDADES ---
  # Nota: Reemplazar estas variables por la salida real de tu modelo Poisson / Regresión
  prob_local = 0.157
  prob_empate = 0.196
  prob_visita = 0.646

  cuota_local = round(1 / prob_local, 2) if prob_local > 0 else 0
  cuota_empate = round(1 / prob_empate, 2) if prob_empate > 0 else 0
  cuota_visita = round(1 / prob_visita, 2) if prob_visita > 0 else 0

  c1, c2, c3 = st.columns(3)

  with c1:
    st.write(f"**Gana {row_match['Local']}**")
    st.markdown(f"### {prob_local*100:.1f}%")
    st.caption(f"↑ Cuota Justa: {cuota_local}")

  with c2:
    st.write("**Empate**")
    st.markdown(f"### {prob_empate*100:.1f}%")
    st.caption(f"↑ Cuota Justa: {cuota_empate}")

  with c3:
    st.write(f"**Gana {row_match['Visita']}**")
    st.markdown(f"### {prob_visita*100:.1f}%")
    st.caption(f"↑ Cuota Justa: {cuota_visita}")


# --- 4. PESTAÑA 2: RESUMEN DE LA JORNADA ---
with tab2:
  st.header("🧢 Resumen de la Jornada")
  jornada_resumen = st.selectbox(
      "Selecciona la Jornada a revisar:",
      lista_jornadas,
      key="sb_jornada_tab2",
  )
  df_res = df_proximos[df_proximos["Jornada"] == jornada_resumen].copy()

  # Formatear la columna combinada de Fecha/Día para la tabla global
  df_res["Fecha_Display"] = df_res.apply(
      lambda r: f"{r['Dia']} {str(r['Fecha']).split(' ')[0]}"
      if pd.notna(r.get("Dia"))
      else str(r.get("Fecha")),
      axis=1,
  )

  columnas_mostrar = [
      "Fecha_Display",
      "Hora",
      "Local",
      "Visita",
      "Estadio",
      "Ciudad",
      "Altitud_Local",
  ]
  cols_presentes = [c for c in columnas_mostrar if c in df_res.columns]

  st.dataframe(df_res[cols_presentes], use_container_width=True)


# --- 5. PESTAÑA 3: TABLA CLAUSURA ---
with tab3:
  st.header("🏆 Tabla de Posiciones - Clausura")
  st.dataframe(df_clausura, use_container_width=True)


# --- 6. PESTAÑA 4: TABLA ACUMULADA ---
with tab4:
  st.header("📊 Tabla Acumulada")
  st.dataframe(df_acumulado, use_container_width=True)


# --- 7. PESTAÑA 5: DATA GEOGRÁFICA ---
with tab5:
  st.header("🗺️ Información Geográfica y Altitudes")
  st.dataframe(df_geo, use_container_width=True)
  
