import re
import numpy as np
import pandas as pd
from scipy.stats import poisson
import streamlit as st

st.set_page_config(
    page_title="Sistema de Predicciones Liga 1 2026",
    page_icon="⚽",
    layout="wide",
)


@st.cache_data(ttl=60)
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


def normalizar_texto(texto):
  if not isinstance(texto, str):
    return ""
  txt = texto.lower().strip()
  txt = re.sub(
      r"\b(club|fc|cd|atletico|atlético|deportivo|asociacion|asociación|college)\b",
      "",
      txt,
  )
  return re.sub(r"\s+", " ", txt).strip()


def coincidencia_equipo(nombre1, nombre2):
  n1 = normalizar_texto(nombre1)
  n2 = normalizar_texto(nombre2)
  if not n1 or not n2:
    return False

  if n1 in n2 or n2 in n1:
    return True

  palabras_1 = set(n1.split())
  palabras_2 = set(n2.split())
  return len(palabras_1.intersection(palabras_2)) > 0


def obtener_fuerza_ponderada(
    equipo_local, equipo_visita, df_resultados_hist, df_tabla_acumulada
):
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

  # Medias de liga por defecto (prior bayesiano)
  att_loc, def_loc = 1.30, 1.10
  att_vis, def_vis = 1.15, 1.20

  if c_loc and c_vis and c_gloc and c_gvis:
    p_loc = df_resultados_hist[
        df_resultados_hist[c_loc]
        .astype(str)
        .apply(lambda x: coincidencia_equipo(equipo_local, x))
    ].copy()

    p_vis = df_resultados_hist[
        df_resultados_hist[c_vis]
        .astype(str)
        .apply(lambda x: coincidencia_equipo(equipo_visita, x))
    ].copy()

    # Promedio suavizado con prior (evita distorsión por pocos partidos)
    if not p_loc.empty:
      g_favor = p_loc[c_gloc].astype(float).clip(upper=2.5).mean()
      g_contra = p_loc[c_gvis].astype(float).clip(upper=2.5).mean()
      att_loc = (g_favor + 1.30) / 2.0
      def_loc = (g_contra + 1.10) / 2.0

    if not p_vis.empty:
      g_favor = p_vis[c_gvis].astype(float).clip(upper=2.5).mean()
      g_contra = p_vis[c_gloc].astype(float).clip(upper=2.5).mean()
      att_vis = (g_favor + 1.15) / 2.0
      def_vis = (g_contra + 1.20) / 2.0

  # BÚSQUEDA Y AJUSTE POR TABLA ACUMULADA
  pts_loc, pts_vis = None, None
  if not df_tabla_acumulada.empty:
    col_eq = df_tabla_acumulada.columns[0]
    col_pts = next(
        (c for c in df_tabla_acumulada.columns if "pt" in c.lower() or "puntos" in c.lower()),
        None,
    )

    if col_pts:
      for _, row in df_tabla_acumulada.iterrows():
        nombre_tabla = str(row[col_eq])
        if (
            pts_loc is None
            and coincidencia_equipo(equipo_local, nombre_tabla)
        ):
          try:
            pts_loc = float(row[col_pts])
          except Exception:
            pass
        if (
            pts_vis is None
            and coincidencia_equipo(equipo_visita, nombre_tabla)
        ):
          try:
            pts_vis = float(row[col_pts])
          except Exception:
            pass

  # REGLA DE JERARQUÍA REALISTA
  if pts_loc is not None and pts_vis is not None:
    dif_pts = pts_vis - pts_loc
    # Ajuste exponencial moderado según la diferencia en la Acumulada
    factor_ajuste = 1.0 + (dif_pts * 0.035)

    if dif_pts > 0:  # Visitante superior en rendimiento anual
      att_vis *= factor_ajuste
      def_vis /= factor_ajuste
      att_loc /= factor_ajuste
      def_loc *= factor_ajuste
    elif dif_pts < 0:  # Local superior
      factor_loc = 1.0 + (abs(dif_pts) * 0.035)
      att_loc *= factor_loc
      def_loc /= factor_loc
      att_vis /= factor_loc
      def_vis *= factor_loc

  return max(0.70, att_loc), max(0.70, def_loc), max(0.70, att_vis), max(0.70, def_vis)


def obtener_altitud(equipo_nombre, df_geo_info):
  col_eq = df_geo_info.columns[0]
  mask = df_geo_info[col_eq].astype(str).apply(
      lambda x: coincidencia_equipo(equipo_nombre, x)
  )
  row = df_geo_info[mask]
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


def tau_dixon_coles(x, y, lambda_param, mu_param, rho=-0.11):
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
    rho=-0.11,
):
  promedio_goles_liga = 1.25

  att_loc, def_loc, att_vis, def_vis = obtener_fuerza_ponderada(
      equipo_local, equipo_visita, df_resultados_hist, df_tabla
  )

  alt_loc = obtener_altitud(equipo_local, df_geo_info)
  alt_vis = obtener_altitud(equipo_visita, df_geo_info)

  dif_altitud = max(0.0, alt_loc - alt_vis)
  factor_altitud = 1.0 + (dif_altitud / 12000.0)

  home_advantage = 1.06

  lambda_local = (
      att_loc * (def_vis / promedio_goles_liga)
  ) * home_advantage * factor_altitud
  mu_visita = att_vis * (def_loc / promedio_goles_liga)

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

  prob_local = float(np.tril(matriz_prob, -1).sum())
  prob_empate = float(np.trace(matriz_prob))
  prob_visita = float(np.triu(matriz_prob, 1).sum())

  prob_under25 = sum(
      matriz_prob[i, j]
      for i in range(max_goles)
      for j in range(max_goles)
      if i + j < 2.5
  )
  prob_over25 = 1.0 - prob_under25

  prob_btts_si = float(matriz_prob[1:, 1:].sum())
  prob_btts_no = 1.0 - prob_btts_si

  return prob_local, prob_empate, prob_visita, prob_over25, prob_btts_si


# INTERFAZ STREAMLIT
st.title("⚽ Sistema de Predicciones Liga 1 2026")
st.subheader("Modelo Calibrado (Regresión a la Media + Peso de Acumulada)")

if st.sidebar.button("🔄 Recargar Datos del Excel"):
  st.cache_data.clear()
  st.rerun()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Pronóstico Individual por Partido",
    "🧢 Resumen de la Jornada",
    "🏆 Tabla Clausura",
    "📊 Tabla Acumulada",
    "🗺️ Data Geográfica",
])

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

  equipo_local = str(row_match["Local"]).strip()
  equipo_visita = str(row_match["Visita"]).strip()

  prob_local, prob_empate, prob_visita, prob_over25, prob_btts_si = (
      calcular_dixon_coles(
          equipo_local,
          equipo_visita,
          df_acumulado,
          df_resultados,
          df_geo,
      )
  )

  prob_under25 = 1.0 - prob_over25
  prob_btts_no = 1.0 - prob_btts_si

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

  if prob_local > 0.48:
    fija_txt = f"Gana {equipo_local} (Directo)"
    confian_txt = "Alta"
  elif prob_visita > 0.48:
    fija_txt = f"Gana {equipo_visita} (Directo)"
    confian_txt = "Alta"
  elif (prob_visita + prob_empate) > 0.60 and prob_visita > prob_local:
    fija_txt = f"Empate o Visita ({equipo_visita})"
    confian_txt = "Media-Alta"
  elif (prob_local + prob_empate) > 0.60 and prob_local > prob_visita:
    fija_txt = f"Local o Empate ({equipo_local})"
    confian_txt = "Media-Alta"
  else:
    fija_txt = "Doble Opción o Sin Apuesta"
    confian_txt = "Media"

  st.write(" ")
  col_fija1, col_fija2 = st.columns(2)
  with col_fija1:
    st.info(f"**Pronóstico Sugerido (1X2):** {fija_txt}")
  with col_fija2:
    st.success(f"**Nivel de Confianza:** {confian_txt}")

  st.write("---")

  if prob_over25 >= 0.52:
    sug_goles = "Más de 2.5 Goles (+2.5)"
    conf_goles = "Alta" if prob_over25 >= 0.58 else "Media"
  elif prob_over25 >= 0.45:
    sug_goles = "Más de 1.5 Goles (+1.5)"
    conf_goles = "Media"
  else:
    sug_goles = "Menos de 2.5 Goles (-2.5)"
    conf_goles = "Alta" if prob_under25 >= 0.58 else "Media"

  if prob_btts_si >= 0.50:
    sug_btts = "Ambos Equipos Anotan (Sí)"
    conf_btts = "Alta" if prob_btts_si >= 0.58 else "Media"
  else:
    sug_btts = "Ambos Equipos NO Anotan (No)"
    conf_btts = "Alta" if prob_btts_no >= 0.58 else "Media"

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
