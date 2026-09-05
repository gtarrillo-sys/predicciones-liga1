import math
import numpy as np
import pandas as pd
import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="Predicciones Liga 1 2026", page_icon="⚽", layout="wide"
)

st.title("⚽ Sistema de Predicciones Liga 1 2026")
st.subheader("Modelo Estadístico para la Liga 1 Peruana")


# 1. Función Poisson base
def pmf_poisson(k, lambd):
  if lambd <= 0:
    return 1.0 if k == 0 else 0.0
  return (lambd**k * math.exp(-lambd)) / math.factorial(k)


# 2. Factor Tau de Dixon-Coles
def tau_dixon_coles(x, y, lambda_l, lambda_v, rho=-0.12):
  if x == 0 and y == 0:
    return 1.0 - (lambda_l * lambda_v * rho)
  elif x == 1 and y == 0:
    return 1.0 + (lambda_v * rho)
  elif x == 0 and y == 1:
    return 1.0 + (lambda_l * rho)
  elif x == 1 and y == 1:
    return 1.0 - rho
  else:
    return 1.0


# Carga de datos desde Excel
@st.cache_data
def cargar_datos():
  excel_path = "Liga1_2026.xlsx"
  df_geo = pd.read_excel(excel_path, sheet_name="Data_Geografica")
  df_resultados = pd.read_excel(excel_path, sheet_name="Resultados_Clausura")
  df_proximos = pd.read_excel(excel_path, sheet_name="Partidos_Fecha")
  df_clausura = pd.read_excel(excel_path, sheet_name="Tabla_Clausura")
  df_acumulado = pd.read_excel(excel_path, sheet_name="Tabla_Acumulada")
  return df_geo, df_resultados, df_proximos, df_clausura, df_acumulado


# Botón lateral para refrescar cache
with st.sidebar:
  st.header("⚙️ Opciones")
  if st.button("🔄 Recargar Datos del Excel"):
    st.cache_data.clear()
    st.rerun()

try:
  df_geo, df_resultados, df_proximos, df_clausura, df_acumulado = (
      cargar_datos()
  )

  tab1, tab2, tab3, tab4, tab5 = st.tabs([
      "🎯 Pronóstico Individual por Partido",
      "🔮 Resumen de la Jornada",
      "🏆 Tabla Clausura",
      "📊 Tabla Acumulada",
      "🗺️ Data Geográfica",
  ])

  # --- CÁLCULO DE FUERZAS CON RECENCIA ---
  CANT_PARTIDOS_RECIENTES = 45  # Últimas 5 jornadas aprox.
  df_reciente = (
      df_resultados.tail(CANT_PARTIDOS_RECIENTES)
      if len(df_resultados) > CANT_PARTIDOS_RECIENTES
      else df_resultados
  )

  prom_xg_loc = df_reciente["xG_Local"].mean()
  prom_xg_vis = df_reciente["xG_Visita"].mean()
  equipos = pd.unique(df_resultados[["Local", "Visita"]].values.ravel())

  fuerzas = {}
  for eq in equipos:
    df_l = df_reciente[df_reciente["Local"] == eq]
    df_v = df_reciente[df_reciente["Visita"] == eq]

    if len(df_l) < 2:
      df_l = df_resultados[df_resultados["Local"] == eq]
    if len(df_v) < 2:
      df_v = df_resultados[df_resultados["Visita"] == eq]

    att_l = (
        df_l["xG_Local"].mean() / prom_xg_loc if len(df_l) > 0 else 1.0
    )
    def_l = (
        df_l["xG_Visita"].mean() / prom_xg_vis if len(df_l) > 0 else 1.0
    )
    att_v = (
        df_v["xG_Visita"].mean() / prom_xg_vis if len(df_v) > 0 else 1.0
    )
    def_v = (
        df_v["xG_Local"].mean() / prom_xg_loc if len(df_l) > 0 else 1.0
    )

    fuerzas[eq] = {
        "Att_Loc": att_l,
        "Def_Loc": def_l,
        "Att_Vis": att_v,
        "Def_Vis": def_v,
    }

  # --- TAB 1: PRONÓSTICO INDIVIDUAL ---
  with tab1:
    st.header("🔍 Análisis Detallado")

    jornadas_disponibles = list(df_proximos["Jornada"].unique())
    col_j1, col_j2 = st.columns([1, 2])

    with col_j1:
      jornada_sel = st.selectbox("📅 Selecciona la Fecha:", jornadas_disponibles)

    df_partidos_jornada = df_proximos[df_proximos["Jornada"] == jornada_sel]
    partidos_lista = (
        df_partidos_jornada["Local"] + " vs " + df_partidos_jornada["Visita"]
    ).tolist()

    with col_j2:
      partido_sel = st.selectbox("⚔️ Selecciona el Partido:", partidos_lista)

    row_match = df_partidos_jornada[
        (df_partidos_jornada["Local"] + " vs " + df_partidos_jornada["Visita"])
        == partido_sel
    ].iloc[0]

    loc, vis = row_match["Local"], row_match["Visita"]
    alt_l, alt_v = row_match["Altitud_Local"], row_match["Altitud_Visita"]

    # EXTRAER FECHA Y HORA FORMATO SEGURO
    val_fecha = row_match.get("Fecha", None)
    val_hora = row_match.get("Hora", None)

    if pd.notna(val_fecha):
      if isinstance(val_fecha, (pd.Timestamp, pd.DatetimeIndex)):
        fecha_str = val_fecha.strftime("%d/%m/%Y")
      else:
        fecha_str = str(val_fecha)
    else:
      fecha_str = "Fecha por confirmar"

    if pd.notna(val_hora):
      hora_str = str(val_hora)[:5]
    else:
      hora_str = ""

    info_horario = (
        f"📅 {fecha_str} - 🕒 {hora_str}" if hora_str else f"📅 {fecha_str}"
    )

    f_l = fuerzas.get(
        loc, {"Att_Loc": 1.0, "Def_Loc": 1.0, "Att_Vis": 1.0, "Def_Vis": 1.0}
    )
    f_v = fuerzas.get(
        vis, {"Att_Loc": 1.0, "Def_Loc": 1.0, "Att_Vis": 1.0, "Def_Vis": 1.0}
    )

    lambda_l = f_l["Att_Loc"] * f_v["Def_Vis"] * prom_xg_loc
    lambda_v = f_v["Att_Vis"] * f_l["Def_Loc"] * prom_xg_vis

    if alt_l >= 2000 and alt_v < 500:
      lambda_l *= 1.15
      lambda_v *= 0.85

    lambda_v = max(lambda_v, 0.92)
    lambda_l = max(lambda_l, 1.05)

    # MATRIZ DIXON-COLES
    matriz = np.zeros((7, 7))
    for i in range(7):
      for j in range(7):
        p_base = pmf_poisson(i, lambda_l) * pmf_poisson(j, lambda_v)
        tau = tau_dixon_coles(i, j, lambda_l, lambda_v, rho=-0.12)
        matriz[i, j] = p_base * tau

    matriz = matriz / np.sum(matriz)

    prob_l = np.sum(np.tril(matriz, -1))
    prob_e = np.sum(np.diag(matriz))
    prob_v = np.sum(np.triu(matriz, 1))

    prob_btts_si = np.sum(matriz[1:, 1:])
    prob_btts_no = 1.0 - prob_btts_si

    total_goles = np.indices((7, 7))[0] + np.indices((7, 7))[1]
    prob_over_15 = np.sum(matriz[total_goles > 1.5])
    prob_over_25 = np.sum(matriz[total_goles > 2.5])
    prob_over_35 = np.sum(matriz[total_goles > 3.5])

    st.markdown("---")
    st.markdown(
        f"### 🏟️ **{loc}** vs **{vis}** | *{row_match['Ciudad']} ({alt_l}"
        f" msnm)*\n##### {info_horario}"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"Gana {loc}",
        f"{prob_l*100:.1f}%",
        f"Cuota Justa: {1/prob_l:.2f}" if prob_l > 0 else "-",
    )
    c2.metric(
        "Empate",
        f"{prob_e*100:.1f}%",
        f"Cuota Justa: {1/prob_e:.2f}" if prob_e > 0 else "-",
    )
    c3.metric(
        f"Gana {vis}",
        f"{prob_v*100:.1f}%",
        f"Cuota Justa: {1/prob_v:.2f}" if prob_v > 0 else "-",
    )

    st.markdown("---")

    # EVALUACIÓN DE FIJO / ALTA PROBABILIDAD
    max_prob = max(prob_l, prob_v)
    equipo_favorito = loc if prob_l > prob_v else vis

    if max_prob >= 0.60:
      st.success(
          f"🔥 **PARTIDO FIJO / ALTA PROBABILIDAD:** Existe una probabilidad"
          f" muy alta ({max_prob*100:.1f}%) a favor de **{equipo_favorito}**."
      )
    else:
      st.warning(
          "⚠️ **PARTIDO PAREJO / EVALUATIVO:** Las probabilidades están"
          " divididas; se recomienda cautela en el mercado 1X2."
      )

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
      st.subheader("⚽ Marcador de Goles (Over / Under)")
      st.write(
          f"• **Más de 1.5 Goles (+1.5):** **{prob_over_15*100:.1f}%** |"
          f" Menos: **{(1-prob_over_15)*100:.1f}%**"
      )
      st.write(
          f"• **Más de 2.5 Goles (+2.5):** **{prob_over_25*100:.1f}%** |"
          f" Menos: **{(1-prob_over_25)*100:.1f}%**"
      )
      st.write(
          f"• **Más de 3.5 Goles (+3.5):** **{prob_over_35*100:.1f}%** |"
          f" Menos: **{(1-prob_over_35)*100:.1f}%**"
      )

      rec_goles = (
          "Más de 2.5 Goles"
          if prob_over_25 > 0.55
          else ("Menos de 2.5 Goles" if prob_over_25 < 0.45 else "Rango 2 goles")
      )
      st.info(f"💡 **Sugerencia de Línea:** {rec_goles}")

    with col_g2:
      st.subheader("🥊 ¿Ambos Equipos Anotan? (BTTS)")
      st.write(f"• **Sí anotan ambos:** **{prob_btts_si*100:.1f}%**")
      st.write(f"• **No anotan ambos:** **{prob_btts_no*100:.1f}%**")

      rec_btts = (
          "Ambos Anotan: SÍ"
          if prob_btts_si > 0.52
          else ("Ambos Anotan: NO" if prob_btts_no > 0.52 else "Neutro / Evaluativo")
      )
      st.info(f"💡 **Sugerencia BTTS:** {rec_btts}")

  # --- TAB 2: RESUMEN DE LA JORNADA ---
  with tab2:
    st.header("📊 Resumen Consolidado de la Fecha")
    jornada_tabla = st.selectbox(
        "Filtrar Fecha para Ver Matriz:",
        jornadas_disponibles,
        key="jornada_tab2",
    )

    df_jornada_comp = df_proximos[df_proximos["Jornada"] == jornada_tabla]
    resumen_list = []

    for _, row in df_jornada_comp.iterrows():
      l, v = row["Local"], row["Visita"]
      a_l, a_v = row["Altitud_Local"], row["Altitud_Visita"]

      f_val = row.get("Fecha", "")
      h_val = row.get("Hora", "")

      f_txt = str(f_val) if pd.notna(f_val) else ""
      h_txt = str(h_val)[:5] if pd.notna(h_val) else ""

      fl = fuerzas.get(
          l, {"Att_Loc": 1.0, "Def_Loc": 1.0, "Att_Vis": 1.0, "Def_Vis": 1.0}
      )
      fv = fuerzas.get(
          v, {"Att_Loc": 1.0, "Def_Loc": 1.0, "Att_Vis": 1.0, "Def_Vis": 1.0}
      )

      ll = fl["Att_Loc"] * fv["Def_Vis"] * prom_xg_loc
      lv = fv["Att_Vis"] * fl["Def_Loc"] * prom_xg_vis

      if a_l >= 2000 and a_v < 500:
        ll *= 1.15
        lv *= 0.85

      lv = max(lv, 0.92)
      ll = max(ll, 1.05)

      mat = np.zeros((7, 7))
      for i in range(7):
        for j in range(7):
          pb = pmf_poisson(i, ll) * pmf_poisson(j, lv)
          mat[i, j] = pb * tau_dixon_coles(i, j, ll, lv, -0.12)

      mat = mat / np.sum(mat)

      pl = np.sum(np.tril(mat, -1))
      pe = np.sum(np.diag(mat))
      pv = np.sum(np.triu(mat, 1))

      pbtts = np.sum(mat[1:, 1:])
      pover25 = np.sum(mat[total_goles > 2.5])

      resumen_list.append({
          "Partido": f"{l} vs {v}",
          "Fecha / Hora": f"{f_txt} {h_txt}".strip(),
          "Ciudad": row["Ciudad"],
          "Prob. Local": f"{pl*100:.1f}%",
          "Prob. Empate": f"{pe*100:.1f}%",
          "Prob. Visita": f"{pv*100:.1f}%",
          "Ambos Anotan (Sí)": f"{pbtts*100:.1f}%",
          "Más 2.5 Goles": f"{pover25*100:.1f}%",
          "Cuota Justa L": round(1 / pl, 2) if pl > 0 else "-",
      })

    st.dataframe(pd.DataFrame(resumen_list), use_container_width=True)

  # --- TAB 3, 4 Y 5 ---
  with tab3:
    st.header("Tabla de Posiciones - Torneo Clausura")
    st.dataframe(df_clausura, use_container_width=True)

  with tab4:
    st.header("Tabla Acumulada 2026")
    st.dataframe(df_acumulado, use_container_width=True)

  with tab5:
    st.header("Información Geográfica de Clubes")
    st.dataframe(df_geo, use_container_width=True)

except Exception as e:
  st.error(f"Error en la aplicación: {e}")
