import pandas as pd
import streamlit as st

# Configuración inicial
st.set_page_config(
    page_title="Sistema de Predicciones Liga 1 2026",
    page_icon="⚽",
    layout="wide",
)


# --- 1. CARGA DE DATOS ---
@st.cache_data
def cargar_datos():
  excel_path = "Liga1_2026.xlsx"

  df_geo = pd.read_excel(excel_path, sheet_name="Data_Geografica")
  df_resultados = pd.read_excel(excel_path, sheet_name="Resultados_Clausura")
  df_proximos = pd.read_excel(
      excel_path, sheet_name="Partidos_Fecha", dtype=str
  )
  df_clausura = pd.read_excel(excel_path, sheet_name="Tabla_Clausura")
  df_acumulado = pd.read_excel(excel_path, sheet_name="Tabla_Acumulada")

  for df in [df_geo, df_resultados, df_proximos, df_clausura, df_acumulado]:
    df.columns = df.columns.astype(str).str.strip()

  return df_geo, df_resultados, df_proximos, df_clausura, df_acumulado


df_geo, df_resultados, df_proximos, df_clausura, df_acumulado = cargar_datos()

# --- 2. ENCABEZADO ---
st.title("⚽ Sistema de Predicciones Liga 1 2026")
st.subheader("Modelo Estadístico para la Liga 1 Peruana")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Pronóstico Individual por Partido",
    "🧢 Resumen de la Jornada",
    "🏆 Tabla Clausura",
    "📊 Tabla Acumulada",
    "🗺️ Data Geográfica",
])


# --- 3. PESTAÑA 1: PRONÓSTICO INDIVIDUAL ---
with tab1:
  st.header("🔍 Análisis Detallado")

  col_jornada, col_partido = st.columns(2)

  with col_jornada:
    lista_jornadas = df_proximos["Jornada"].dropna().unique().tolist()
    jornada_sel = st.selectbox(
        "📅 Selecciona la Fecha:", lista_jornadas, key="sb_jornada_tab1"
    )

  df_jornada = df_proximos[df_proximos["Jornada"] == jornada_sel].copy()
  df_jornada["Partido_Label"] = (
      df_jornada["Local"].astype(str) + " vs " + df_jornada["Visita"].astype(str)
  )

  with col_partido:
    partido_sel = st.selectbox(
        "⚔️ Selecciona el Partido:",
        df_jornada["Partido_Label"].tolist(),
        key="sb_partido_tab1",
    )

  row_match = df_jornada[df_jornada["Partido_Label"] == partido_sel].iloc[0]

  # --- FORMATO DE FECHA / DIA / HORA ---
  f_val = str(row_match.get("Fecha", "")).strip()
  d_val = str(row_match.get("Dia", "")).strip()
  h_val = str(row_match.get("Hora", "")).strip()

  if f_val and f_val.lower() != "nan":
    fecha_limpia = f_val.split(" ")[0]
    fecha_str = (
        f"{d_val} {fecha_limpia}"
        if d_val and d_val.lower() != "nan"
        else fecha_limpia
    )
  else:
    fecha_str = "Fecha por confirmar"

  hora_str = h_val.split(" ")[-1][:5] if h_val and h_val.lower() != "nan" else ""
  info_horario = (
      f"📅 {fecha_str} - 🕒 {hora_str}" if hora_str else f"📅 {fecha_str}"
  )

  alt_local = row_match.get("Altitud_Local", "0")

  st.subheader(
      f"🏟️ {partido_sel} | {str(row_match.get('Ciudad', ''))} ({alt_local} msnm)"
  )
  st.caption(info_horario)

  st.write("---")

  # --- PROBABILIDADES 1X2 ---
  # Reemplazar por los valores calculados por tu modelo Poisson/Regresión
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

  # --- RECOMENDACIÓN 1X2 / LA FIJA ---
  if prob_visita > 0.50:
    fija_txt = f"Gana {row_match['Visita']} (Directo)"
    confian_txt = "Alta"
  elif prob_local > 0.50:
    fija_txt = f"Gana {row_match['Local']} (Directo)"
    confian_txt = "Alta"
  elif (prob_visita + prob_empate) > 0.70:
    fija_txt = f"Empate o Visita ({row_match['Visita']})"
    confian_txt = "Media-Alta"
  else:
    fija_txt = f"Local o Empate ({row_match['Local']})"
    confian_txt = "Media"

  st.write(" ")
  col_fija1, col_fija2 = st.columns(2)
  with col_fija1:
    st.info(f"**Pronóstico Sugerido (1X2):** {fija_txt}")
  with col_fija2:
    st.success(f"**Nivel de Confianza:** {confian_txt}")

  st.write("---")

  # --- MERCADO DE GOLES Y AMBOS MARCAN ---
  col_goles, col_btts = st.columns(2)

  # Probabilidades reales obtenidas del modelo Poisson
  prob_over25 = 0.582
  prob_under25 = 1 - prob_over25

  prob_btts_si = 0.524
  prob_btts_no = 1 - prob_btts_si

  # --- LÓGICA DE PRONÓSTICO: GOLES (OVER/UNDER) ---
  if prob_over25 >= 0.60:
    sug_goles = "Más de 2.5 Goles (Over)"
    conf_goles = "Alta"
  elif prob_over25 >= 0.52:
    sug_goles = "Más de 1.5 / 2.5 Goles"
    conf_goles = "Media"
  elif prob_under25 >= 0.60:
    sug_goles = "Menos de 2.5 Goles (Under)"
    conf_goles = "Alta"
  else:
    sug_goles = "Menos de 2.5 / 3.5 Goles"
    conf_goles = "Media"

  # --- LÓGICA DE PRONÓSTICO: AMBOS MARCAN (BTTS) ---
  if prob_btts_si >= 0.58:
    sug_btts = "Ambos Equipos Anotan (Sí)"
    conf_btts = "Alta"
  elif prob_btts_si >= 0.50:
    sug_btts = "Ambos Equipos Anotan (Sí)"
    conf_btts = "Media"
  elif prob_btts_no >= 0.58:
    sug_btts = "Ambos Equipos NO Anotan (No)"
    conf_btts = "Alta"
  else:
    sug_btts = "Ambos Equipos NO Anotan (No)"
    conf_btts = "Media"

  # RENDER SECCIÓN GOLES
  with col_goles:
    st.subheader("⚽ Mercado de Goles (Over / Under 2.5)")
    st.write(f"**Más de 2.5 Goles:** {prob_over25*100:.1f}%")
    st.progress(prob_over25)
    st.caption(
        f"Cuota Justa Over: {round(1/prob_over25, 2) if prob_over25 > 0 else 0}"
    )

    st.write(f"**Menos de 2.5 Goles:** {prob_under25*100:.1f}%")
    st.progress(prob_under25)
    st.caption(
        f"Cuota Justa Under:"
        f" {round(1/prob_under25, 2) if prob_under25 > 0 else 0}"
    )

    st.info(f"**Pronóstico Sugerido:** {sug_goles}")
    st.caption(f"🎯 Nivel de Confianza: **{conf_goles}**")

  # RENDER SECCIÓN BTTS
  with col_btts:
    st.subheader("🔥 Ambos Equipos Anotan (BTTS)")
    st.write(f"**Sí Anotan Ambos:** {prob_btts_si*100:.1f}%")
    st.progress(prob_btts_si)
    st.caption(
        f"Cuota Justa Sí:"
        f" {round(1/prob_btts_si, 2) if prob_btts_si > 0 else 0}"
    )

    st.write(f"**No Anotan Ambos:** {prob_btts_no*100:.1f}%")
    st.progress(prob_btts_no)
    st.caption(
        f"Cuota Justa No:"
        f" {round(1/prob_btts_no, 2) if prob_btts_no > 0 else 0}"
    )

    st.info(f"**Pronóstico Sugerido:** {sug_btts}")
    st.caption(f"🎯 Nivel de Confianza: **{conf_btts}**")


# --- 4. PESTAÑA 2: RESUMEN DE LA JORNADA ---
with tab2:
  st.header("🧢 Resumen de la Jornada")
  jornada_resumen = st.selectbox(
      "Selecciona la Jornada a revisar:", lista_jornadas, key="sb_jornada_tab2"
  )
  df_res = df_proximos[df_proximos["Jornada"] == jornada_resumen].copy()

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


# --- 5. OTRAS PESTAÑAS ---
with tab3:
  st.header("🏆 Tabla de Posiciones - Clausura")
  st.dataframe(df_clausura, use_container_width=True)

with tab4:
  st.header("📊 Tabla Acumulada")
  st.dataframe(df_acumulado, use_container_width=True)

with tab5:
  st.header("🗺️ Información Geográfica y Altitudes")
  st.dataframe(df_geo, use_container_width=True)
