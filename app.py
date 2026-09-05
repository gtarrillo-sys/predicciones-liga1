import numpy as np
import pandas as pd
from scipy.stats import poisson
import streamlit as st

# Configuración inicial de la página
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


# --- 2. FUNCIONES AUXILIARES DE EXTRACCIÓN Y CONDICIÓN ---
def obtener_fuerza_especifica(
    equipo_local, equipo_visita, df_resultados_hist, df_tabla
):
  """Calcula de forma precisa el ataque y defensa del LOCAL jugando de LOCAL

  y del VISITANTE jugando de VISITANTE usando los partidos disputados.
  """
  nombre_loc = str(equipo_local).strip().lower()
  nombre_vis = str(equipo_visita).strip().lower()

  # Búsqueda dinámica de columnas en Resultados_Clausura
  c_loc = next(
      (c for c in df_resultados_hist.columns if "local" in c.lower()), None
  )
  c_vis = next(
      (c for c in df_resultados_hist.columns if "visita" in c.lower()), None
  )
  c_gloc = next(
      (
          c
          for c in df_resultados_hist.columns
          if "goles_l" in c.lower() or "gl" in c.lower() or "goles_local" in c.lower()
      ),
      None,
  )
  c_gvis = next(
      (
          c
          for c in df_resultados_hist.columns
          if "goles_v" in c.lower() or "gv" in c.lower() or "goles_visita" in c.lower()
      ),
      None,
  )

  # Valores base de rescate
  att_loc, def_loc = 1.35, 1.35
  att_vis, def_vis = 1.35, 1.35

  # Rendimiento de FC Cajamarca / Local en CASA
  if c_loc and c_gloc and c_gvis:
    p_local = df_resultados_hist[
        df_resultados_hist[c_loc].astype(str).str.lower().str.contains(nombre_loc)
    ]
    if not p_local.empty:
      try:
        gf = p_local[c_gloc].astype(float).sum()
        gc = p_local[c_gvis].astype(float).sum()
        pj = len(p_local)
        if pj > 0:
          att_loc = gf / pj
          def_loc = gc / pj
      except Exception:
        pass

  # Rendimiento de Cienciano / Visita FUERA DE CASA
  if c_vis and c_gloc and c_gvis:
    p_visita = df_resultados_hist[
        df_resultados_hist[c_vis].astype(str).str.lower().str.contains(nombre_vis)
    ]
    if not p_visita.empty:
      try:
        gf = p_visita[c_gvis].astype(float).sum()
        gc = p_visita[c_gloc].astype(float).sum()
        pj = len(p_visita)
        if pj > 0:
          att_vis = gf / pj
          def_vis = gc / pj
      except Exception:
        pass

  return att_loc, def_loc, att_vis, def_vis


def obtener_altitud(equipo_nombre, df_geo_info):
  """Obtiene la altitud en msnm del equipo desde la pestaña geográfica."""
  nombre_search = str(equipo_nombre).strip().lower()
  col_eq = df_geo_info.columns[0]

  row = df_geo_info[
      df_geo_info[col_eq].astype(str).str.lower().str.contains(nombre_search)
  ]

  if not row.empty:
    col_alt = next(
        (c for c in df_geo_info.columns if "altitud" in c.lower()), None
    )
    if col_alt:
      try:
        return float(row[col_alt].values[0])
      except Exception:
        pass
  return 0.0


# --- 3. MOTOR MATEMÁTICO: MODELO DIXON-COLES ---
def tau_dixon_coles(x, y, lambda_param, mu_param, rho=-0.13):
  """Función Tau para corregir marcadores de baja anotación en Dixon-Coles."""
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
    equipo_local,
    equipo_visita,
    df_tabla,
    df_resultados_hist,
    df_geo_info,
    rho=-0.13,
):
  promedio_goles_liga = 1.35

  # 1. Obtención de métricas desglosadas por Local / Visita específica
  att_loc, def_loc, att_vis, def_vis = obtener_fuerza_especifica(
      equipo_local, equipo_visita, df_resultados_hist, df_tabla
  )

  # 2. Factor Geográfico Relativo (Diferencial de Altitud)
  alt_loc = obtener_altitud(equipo_local, df_geo_info)
  alt_vis = obtener_altitud(equipo_visita, df_geo_info)

  dif_altitud = max(0.0, alt_loc - alt_vis)
  factor_altitud = 1.0 + (dif_altitud / 10000.0)

  # 3. Ventaja de Localía Dinámica (no regala ventaja si el local defiende mal)
  home_advantage = 1.15 if def_loc <= att_loc else 1.02

  # 4. Intensidades esperadas de Gol (Lambda para Local, Mu para Visita)
  lambda_local = max(
      0.3,
      (att_loc * (def_vis / promedio_goles_liga))
      * home_advantage
      * factor_altitud,
  )
  mu_visita = max(0.3, (att_vis * (def_loc / promedio_goles_liga)))

  # 5. Construcción de la matriz con corrección Dixon-Coles
  max_goles = 9
  matriz_prob = np.zeros((max_goles, max_goles))

  for x in range(max_goles):
    for y in range(max_goles):
      p_x = poisson.pmf(x, lambda_local)
      p_y = poisson.pmf(y, mu_visita)
      tau = tau_dixon_coles(x, y, lambda_local, mu_visita, rho)
      matriz_prob[x, y] = max(0.0, p_x * p_y * tau)

  total_p = matriz_prob.sum()
  if total_p > 0:
    matriz_prob /= total_p

  # Cálculo de Resultados 1X2
  prob_empate = float(np.trace(matriz_prob))
  prob_local = float(np.tril(matriz_prob, -1).sum())
  prob_visita = float(np.triu(matriz_prob, 1).sum())

  # Mercado de Goles Over / Under 2.5
  prob_under25 = sum(
      matriz_prob[i, j]
      for i in range(max_goles)
      for j in range(max_goles)
      if i + j < 2.5
  )
  prob_over25 = 1.0 - prob_under25

  # Mercado Ambos Anotan (BTTS)
  prob_no_btts = (
      matriz_prob[0, :].sum()
      + matriz_prob[:, 0].sum()
      - matriz_prob[0, 0]
  )
  prob_btts_si = 1.0 - prob_no_btts

  return prob_local, prob_empate, prob_visita, prob_over25, prob_btts_si


# --- 4. ENCABEZADO Y PESTAÑAS ---
st.title("⚽ Sistema de Predicciones Liga 1 2026")
st.subheader("Modelo Estadístico Dixon-Coles para la Liga 1 Peruana")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Pronóstico Individual por Partido",
    "🧢 Resumen de la Jornada",
    "🏆 Tabla Clausura",
    "📊 Tabla Acumulada",
    "🗺️ Data Geográfica",
])


# --- 5. PESTAÑA 1: PRONÓSTICO INDIVIDUAL ---
with tab1:
  st.header("🔍 Análisis Detallado (Modelo Dixon-Coles)")

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

  # CÁLCULO DIXON-COLES EN TIEMPO REAL CON EXCEL
  prob_local, prob_empate, prob_visita, prob_over25, prob_btts_si = (
      calcular_dixon_coles(
          equipo_local,
          equipo_visita,
          df_clausura,
          df_resultados,
          df_geo,
      )
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

  # --- VISUALIZACIÓN PROBABILIDADES 1X2 Y CUOTAS JUSTAS ---
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

  # --- PRONÓSTICO SUGERIDO (LA FIJA) ---
  if prob_local > 0.50:
    fija_txt = f"Gana {equipo_local} (Directo)"
    confian_txt = "Alta"
  elif prob_visita > 0.50:
    fija_txt = f"Gana {equipo_visita} (Directo)"
    confian_txt = "Alta"
  elif (prob_local + prob_empate) > 0.65 and prob_local > prob_visita:
    fija_txt = f"Local o Empate ({equipo_local})"
    confian_txt = "Media-Alta"
  elif (prob_visita + prob_empate) > 0.65 and prob_visita > prob_local:
    fija_txt = f"Empate o Visita ({equipo_visita})"
    confian_txt = "Media-Alta"
  else:
    fija_txt = "Empate o Doble Opción Visita"
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


# --- 6. PESTAÑAS SECUNDARIAS ---
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
