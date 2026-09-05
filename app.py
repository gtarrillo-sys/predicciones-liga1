import numpy as np
import pandas as pd
from scipy.stats import poisson
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


# --- 2. MOTOR MATEMÁTICO: MODELO DIXON-COLES ---
def tau_dixon_coles(x, y, lambda_param, mu_param, rho=-0.13):
  """Ajuste de correlación de Dixon-Coles para marcadores bajos (0-0, 1-0, 0-1, 1-1).

  rho: parámetro de dependencia (típicamente entre -0.10 y -0.15)
  """
  if x == 0 and y == 0:
    return 1.0 - (lambda_param * mu_param * rho)
  elif x == 0 and y == 1:
    return 1.0 + (lambda_param * rho)
  elif x == 1 and y == 0:
    return 1.0 + (mu_param * rho)
  elif x == 1 and y == 1:
    return 1.0 - rho
  else:
    return 1.0


def calcular_dixon_coles(
    equipo_local, equipo_visita, df_tabla, df_geo_info, rho=-0.13
):
  """Calcula la matriz de probabilidades de marcadores ajustada por Dixon-Coles

  considerando fuerza relativa y factor de altitud.
  """
  promedio_goles_liga = 1.35
  home_advantage = 1.22  # Parámetro gamma de localía base en Dixon-Coles

  # Obtener fuerza de ataque y defensa desde la tabla
  def get_fuerza(equipo):
    row = df_tabla[
        df_tabla.iloc[:, 0].astype(str).str.contains(equipo, case=False, na=False)
    ]
    if not row.empty:
      try:
        pj = float(row.iloc[0, 1]) if float(row.iloc[0, 1]) > 0 else 1
        gf = float(row.iloc[0, 5])
        gc = float(row.iloc[0, 6])
        return (gf / pj), (gc / pj)
      except:
        return 1.3, 1.1
    return 1.3, 1.1

  att_loc, def_loc = get_fuerza(equipo_local)
  att_vis, def_vis = get_fuerza(equipo_visita)

  # Factor de Altitud (Efecto geográfico sobre la tasa esperada)
  row_geo = df_geo_info[
      df_geo_info.iloc[:, 0]
      .astype(str)
      .str.contains(equipo_local, case=False, na=False)
  ]
  altitud = 0
  if not row_geo.empty:
    try:
      altitud = float(row_geo.iloc[0].get("Altitud", 0))
    except:
      altitud = 0

  factor_altitud = 1.0 + (altitud / 8500.0)

  # Tasas de intensidad esperada Dixon-Coles (lambda y mu)
  lambda_local = max(
      0.3,
      att_loc
      * (def_vis / promedio_goles_liga)
      * home_advantage
      * factor_altitud,
  )
  mu_visita = max(0.2, att_vis * (def_loc / promedio_goles_liga))

  max_goles = 9
  matriz_prob = np.zeros((max_goles, max_goles))

  # Construcción de la matriz con el ajuste tau
  for x in range(max_goles):
    for y in range(max_goles):
      p_x = poisson.pmf(x, lambda_local)
      p_y = poisson.pmf(y, mu_visita)
      tau = tau_dixon_coles(x, y, lambda_local, mu_visita, rho)
      matriz_prob[x, y] = max(0.0, p_x * p_y * tau)

  # Normalización de la matriz
  matriz_prob /= matriz_prob.sum()

  # 1X2
  prob_empate = float(np.trace(matriz_prob))
  prob_local = float(np.tril(matriz_prob, -1).sum())
  prob_visita = float(np.triu(matriz_prob, 1).sum())

  # Over / Under 2.5
  prob_under25 = 0.0
  for i in range(max_goles):
    for j in range(max_goles):
      if i + j < 2.5:
        prob_under25 += matriz_prob[i, j]
  prob_over25 = 1.0 - prob_under25

  # Ambos Anotan (BTTS)
  prob_no_btts = (
      matriz_prob[0, :].sum()
      + matriz_prob[:, 0].sum()
      - matriz_prob[0, 0]
  )
  prob_btts_si = 1.0 - prob_no_btts

  return prob_local, prob_empate, prob_visita, prob_over25, prob_btts_si


# --- 3. ENCABEZADO Y PESTAÑAS ---
st.title("⚽ Sistema de Predicciones Liga 1 2026")
st.subheader("Modelo Estadístico Dixon-Coles para la Liga 1 Peruana")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Pronóstico Individual por Partido",
    "🧢 Resumen de la Jornada",
    "🏆 Tabla Clausura",
    "📊 Tabla Acumulada",
    "🗺️ Data Geográfica",
])


# --- 4. PESTAÑA 1: PRONÓSTICO INDIVIDUAL ---
with tab1:
  st.header("🔍 Análisis Detallado (Ajuste Dixon-Coles)")

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

  equipo_local = str(row_match["Local"]).strip()
  equipo_visita = str(row_match["Visita"]).strip()

  # CÁLCULO DIXON-COLES EN TIEMPO REAL
  prob_local, prob_empate, prob_visita, prob_over25, prob_btts_si = (
      calcular_dixon_coles(equipo_local, equipo_visita, df_clausura, df_geo)
  )

  prob_under25 = 1.0 - prob_over25
  prob_btts_no = 1.0 - prob_btts_si

  # --- FORMATO FECHA Y LUGAR ---
  f_val = str(row_match.get("Fecha", "")).strip()
  d_val = str(row_match.get("Dia", "")).strip()
  h_val = str(row_match.get("Hora", "")).strip()

  fecha_limpia = (
      f_val.split(" ")[0] if f_val and f_val.lower() != "nan" else "Por confirmar"
  )
  fecha_str = (
      f"{d_val} {fecha_limpia}"
      if d_val and d_val.lower() != "nan"
      else fecha_limpia
  )
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

  # --- VISUALIZACIÓN PROBABILIDADES 1X2 ---
  cuota_local = round(1 / prob_local, 2) if prob_local > 0 else 0
  cuota_empate = round(1 / prob_empate, 2) if prob_empate > 0 else 0
  cuota_visita = round(1 / prob_visita, 2) if prob_visita > 0 else 0

  c1, c2, c3 = st.columns(3)
  with c1:
    st.write(f"**Gana {equipo_local}**")
    st.markdown(f"### {prob_local*100:.1f}%")
    st.caption(f"↑ Cuota Justa: {cuota_local}")

  with c2:
    st.write("**Empate**")
    st.markdown(f"### {prob_empate*100:.1f}%")
    st.caption(f"↑ Cuota Justa: {cuota_empate}")

  with c3:
    st.write(f"**Gana {equipo_visita}**")
    st.markdown(f"### {prob_visita*100:.1f}%")
    st.caption(f"↑ Cuota Justa: {cuota_visita}")

  # --- LA FIJA ---
  if prob_local > 0.50:
    fija_txt = f"Gana {equipo_local} (Directo)"
    confian_txt = "Alta"
  elif prob_visita > 0.50:
    fija_txt = f"Gana {equipo_visita} (Directo)"
    confian_txt = "Alta"
  elif (prob_local + prob_empate) > 0.68:
    fija_txt = f"Local o Empate ({equipo_local})"
    confian_txt = "Media-Alta"
  elif (prob_visita + prob_empate) > 0.68:
    fija_txt = f"Empate o Visita ({equipo_visita})"
    confian_txt = "Media-Alta"
  else:
    fija_txt = "Empate o Doble Opción Local"
    confian_txt = "Media"

  st.write(" ")
  col_fija1, col_fija2 = st.columns(2)
  with col_fija1:
    st.info(f"**Pronóstico Sugerido (1X2):** {fija_txt}")
  with col_fija2:
    st.success(f"**Nivel de Confianza:** {confian_txt}")

  st.write("---")

  # --- RECOMENDACIONES DE GOLES Y BTTS ---
  if prob_over25 >= 0.58:
    sug_goles = "Más de 2.5 Goles (+2.5)"
    conf_goles = "Alta"
  elif prob_over25 >= 0.50:
    sug_goles = "Más de 1.5 Goles (+1.5)"
    conf_goles = "Media"
  elif prob_under25 >= 0.58:
    sug_goles = "Menos de 2.5 Goles (-2.5)"
    conf_goles = "Alta"
  else:
    sug_goles = "Menos de 3.5 Goles (-3.5)"
    conf_goles = "Media"

  if prob_btts_si >= 0.56:
    sug_btts = "Ambos Equipos Anotan (Sí)"
    conf_btts = "Alta"
  elif prob_btts_si >= 0.48:
    sug_btts = "Ambos Equipos Anotan (Sí)"
    conf_btts = "Media"
  else:
    sug_btts = "Ambos Equipos NO Anotan (No)"
    conf_btts = "Media"

  col_goles, col_btts = st.columns(2)

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


# --- 5. OTRAS PESTAÑAS ---
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

with tab3:
  st.header("🏆 Tabla de Posiciones - Clausura")
  st.dataframe(df_clausura, use_container_width=True)

with tab4:
  st.header("📊 Tabla Acumulada")
  st.dataframe(df_acumulado, use_container_width=True)

with tab5:
  st.header("🗺️ Información Geográfica y Altitudes")
  st.dataframe(df_geo, use_container_width=True)
