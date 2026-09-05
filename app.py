import math
import numpy as np
import pandas as pd
import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="Predicciones Liga 1 2026", page_icon="⚽", layout="wide"
)

st.title("⚽ Sistema de Predicciones Liga 1 2026")
st.subheader("Modelo Estadístico de Poisson + Factor Geográfico")


# Función de Poisson
def pmf_poisson(k, lambd):
  if lambd <= 0:
    return 1.0 if k == 0 else 0.0
  return (lambd**k * math.exp(-lambd)) / math.factorial(k)


# Carga de datos
@st.cache_data
def cargar_datos():
  excel_path = "Liga1_2026.xlsx"
  df_geo = pd.read_excel(excel_path, sheet_name="Data_Geografica")
  df_resultados = pd.read_excel(excel_path, sheet_name="Resultados_Clausura")
  df_proximos = pd.read_excel(excel_path, sheet_name="Partidos_Fecha")
  df_clausura = pd.read_excel(excel_path, sheet_name="Tabla_Clausura")
  df_acumulado = pd.read_excel(excel_path, sheet_name="Tabla_Acumulada")
  return df_geo, df_resultados, df_proximos, df_clausura, df_acumulado


try:
  df_geo, df_resultados, df_proximos, df_clausura, df_acumulado = (
      cargar_datos()
  )

  tab1, tab2, tab3, tab4 = st.tabs([
      "🔮 Predicciones Próxima Fecha",
      "🏆 Tabla Clausura",
      "📊 Tabla Acumulada",
      "🗺️ Data Geográfica",
  ])

  # --- TAB 1: PREDICCIONES ---
  with tab1:
    st.header("Pronósticos, Probabilidades y Valor")

    # Promedios xG Liga
    prom_xg_loc = df_resultados["xG_Local"].mean()
    prom_xg_vis = df_resultados["xG_Visita"].mean()

    # Cálculo de Fuerzas
    equipos = pd.unique(df_resultados[["Local", "Visita"]].values.ravel())
    fuerzas = {}
    for eq in equipos:
      df_l = df_resultados[df_resultados["Local"] == eq]
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
          df_v["xG_Local"].mean() / prom_xg_loc if len(df_v) > 0 else 1.0
      )

      fuerzas[eq] = {
          "Att_Loc": att_l,
          "Def_Loc": def_l,
          "Att_Vis": att_v,
          "Def_Vis": def_v,
      }

    pred_list = []
    for idx, row in df_proximos.iterrows():
      loc, vis = row["Local"], row["Visita"]
      alt_l, alt_v = row["Altitud_Local"], row["Altitud_Visita"]

      f_l = fuerzas.get(
          loc,
          {"Att_Loc": 1.0, "Def_Loc": 1.0, "Att_Vis": 1.0, "Def_Vis": 1.0},
      )
      f_v = fuerzas.get(
          vis,
          {"Att_Loc": 1.0, "Def_Loc": 1.0, "Att_Vis": 1.0, "Def_Vis": 1.0},
      )

      lambda_l = f_l["Att_Loc"] * f_v["Def_Vis"] * prom_xg_loc
      lambda_v = f_v["Att_Vis"] * f_l["Def_Loc"] * prom_xg_vis

      # Factor Altitud (>2000m vs <500m)
      if alt_l >= 2000 and alt_v < 500:
        lambda_l *= 1.18
        lambda_v *= 0.82

      # Matriz de Probabilidades Poisson
      matriz = np.zeros((7, 7))
      for i in range(7):
        for j in range(7):
          matriz[i, j] = pmf_poisson(i, lambda_l) * pmf_poisson(j, lambda_v)

      prob_l = np.sum(np.tril(matriz, -1))
      prob_e = np.sum(np.diag(matriz))
      prob_v = np.sum(np.triu(matriz, 1))

      # Cálculo Over/Under 2.5 y Sugerencia de Goles
      prob_under_25 = sum(
          matriz[i, j]
          for i in range(7)
          for j in range(7)
          if (i + j) < 2.5
      )
      prob_over_25 = 1.0 - prob_under_25

      if prob_over_25 > 0.55:
        rec_goles = "Más de 2.5 Goles (+2.5)"
      elif prob_under_25 > 0.55:
        rec_goles = "Menos de 2.5 Goles (-2.5)"
      else:
        rec_goles = "Indefinido / Neutro"

      pred_list.append({
          "Jornada": row["Jornada"],
          "Partido": f"{loc} vs {vis}",
          "Ciudad": row["Ciudad"],
          "xG Est. Local": round(lambda_l, 2),
          "xG Est. Visita": round(lambda_v, 2),
          "Prob. Local (%)": round(prob_l * 100, 1),
          "Prob. Empate (%)": round(prob_e * 100, 1),
          "Prob. Visita (%)": round(prob_v * 100, 1),
          "Mercado Goles (2.5)": rec_goles,
          "Cuota Justa Local": round(1 / prob_l, 2) if prob_l > 0 else "-",
      })

    df_pred = pd.DataFrame(pred_list)
    st.dataframe(df_pred, use_container_width=True)

    # Detalle individual por partido
    st.divider()
    st.subheader("🔍 Análisis Detallado por Partido")
    partido_sel = st.selectbox(
        "Selecciona un partido para ver métricas:", df_pred["Partido"].unique()
    )

    det_row = df_pred[df_pred["Partido"] == partido_sel].iloc[0]
    rec_goles_sel = det_row["Mercado Goles (2.5)"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Prob. Local", f"{det_row['Prob. Local (%)']}%")
    col2.metric("Prob. Empate", f"{det_row['Prob. Empate (%)']}%")
    col3.metric("Prob. Visita", f"{det_row['Prob. Visita (%)']}%")

    st.info(f"⚽ **Sugerencia de Goles:** {rec_goles_sel}")

  # --- TAB 2: TABLA CLAUSURA ---
  with tab2:
    st.header("Tabla de Posiciones - Torneo Clausura")
    st.dataframe(df_clausura, use_container_width=True)

  # --- TAB 3: TABLA ACUMULADA ---
  with tab3:
    st.header("Tabla Acumulada 2026 (Copas y Descenso)")
    st.dataframe(df_acumulado, use_container_width=True)

  # --- TAB 4: DATA GEOGRÁFICA ---
  with tab4:
    st.header("Información Geográfica y de Clima")
    st.dataframe(df_geo, use_container_width=True)

except Exception as e:
  st.error(f"Error al cargar el sistema: {e}")
