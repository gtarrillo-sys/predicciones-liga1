import numpy as np
import pandas as pd
from scipy.stats import poisson

# ==========================================
# 1. CARGA Y LIMPIEZA DE DATOS
# ==========================================
file_path = "Liga1_2026.xlsx"

df_geo = pd.read_excel(file_path, sheet_name="Data_Geografica")
df_resultados = pd.read_excel(file_path, sheet_name="Resultados_Clausura")
df_proximos = pd.read_excel(file_path, sheet_name="Partidos_Fecha")
df_acumulado = pd.read_excel(file_path, sheet_name="Tabla_Acumulada")

# Corrección dinámica del Delta de Altitud en partidos programados
df_proximos["Delta_Altitud_msnm"] = (
    df_proximos["Altitud_Local"] - df_proximos["Altitud_Visita"]
).abs()


# ==========================================
# 2. MOTOR DE TABLAS DINÁMICAS (CLAUSURA Y ACUMULADO)
# ==========================================
def generar_tabla_posiciones(df_partidos):
  equipos = pd.unique(df_partidos[["Local", "Visita"]].values.ravel())
  stats = {
      eq: {
          "PJ": 0,
          "PG": 0,
          "PE": 0,
          "PP": 0,
          "GF": 0,
          "GC": 0,
          "DG": 0,
          "Pts": 0,
      }
      for eq in equipos
  }

  for _, row in df_partidos.iterrows():
    loc, vis = row["Local"], row["Visita"]
    gl, gv = int(row["Goles_Local"]), int(row["Goles_Visita"])

    stats[loc]["PJ"] += 1
    stats[vis]["PJ"] += 1
    stats[loc]["GF"] += gl
    stats[loc]["GC"] += gv
    stats[vis]["GF"] += gv
    stats[vis]["GC"] += gl

    if gl > gv:
      stats[loc]["PG"] += 1
      stats[loc]["Pts"] += 3
      stats[vis]["PP"] += 1
    elif gl < gv:
      stats[vis]["PG"] += 1
      stats[vis]["Pts"] += 3
      stats[loc]["PP"] += 1
    else:
      stats[loc]["PE"] += 1
      stats[loc]["Pts"] += 1
      stats[vis]["PE"] += 1
      stats[vis]["Pts"] += 1

  df_tabla = pd.DataFrame.from_dict(stats, orient="index")
  df_tabla["DG"] = df_tabla["GF"] - df_tabla["GC"]
  df_tabla = (
      df_tabla.reset_index()
      .rename(columns={"index": "Equipo"})
      .sort_values(
          by=["Pts", "DG", "GF"], ascending=[False, False, False]
      )
      .reset_index(drop=True)
  )
  df_tabla.index += 1
  df_tabla.index.name = "Pos"
  return df_tabla


tabla_clausura_calc = generar_tabla_posiciones(df_resultados)


# ==========================================
# 3. MODELO DE PREDICCIÓN POISSON CON FACTOR DE ALTITUD
# ==========================================
def calcular_fuerzas_ataque_defensa(df_partidos):
  prom_xg_local = df_partidos["xG_Local"].mean()
  prom_xg_visita = df_partidos["xG_Visita"].mean()

  fuerzas = {}
  equipos = pd.unique(df_partidos[["Local", "Visita"]].values.ravel())

  for eq in equipos:
    df_loc = df_partidos[df_partidos["Local"] == eq]
    df_vis = df_partidos[df_partidos["Visita"] == eq]

    att_loc = (
        df_loc["xG_Local"].mean() / prom_xg_local
        if len(df_loc) > 0
        else 1.0
    )
    def_loc = (
        df_loc["xG_Visita"].mean() / prom_xg_visita
        if len(df_loc) > 0
        else 1.0
    )

    att_vis = (
        df_vis["xG_Visita"].mean() / prom_xg_visita
        if len(df_vis) > 0
        else 1.0
    )
    def_vis = (
        df_vis["xG_Local"].mean() / prom_xg_local
        if len(df_vis) > 0
        else 1.0
    )

    fuerzas[eq] = {
        "Att_Loc": att_loc,
        "Def_Loc": def_loc,
        "Att_Vis": att_vis,
        "Def_Vis": def_vis,
    }

  return fuerzas, prom_xg_local, prom_xg_visita


fuerzas, prom_loc, prom_vis = calcular_fuerzas_ataque_defensa(df_resultados)


def predecir_partido(
    local, visita, altitud_loc, altitud_vis, max_goles=6
):
  # Base Poisson desde xG
  f_loc = fuerzas.get(
      local, {"Att_Loc": 1.0, "Def_Loc": 1.0, "Att_Vis": 1.0, "Def_Vis": 1.0}
  )
  f_vis = fuerzas.get(
      visita, {"Att_Loc": 1.0, "Def_Loc": 1.0, "Att_Vis": 1.0, "Def_Vis": 1.0}
  )

  lambda_loc = f_loc["Att_Loc"] * f_vis["Def_Vis"] * prom_loc
  lambda_vis = f_vis["Att_Vis"] * f_loc["Def_Loc"] * prom_vis

  # Factor de Altitud (Si el local está a >2000m y la visita viene de llano <500m)
  if altitud_loc >= 2000 and altitud_vis < 500:
    lambda_loc *= 1.18  # Boost local por altura
    lambda_vis *= 0.82  # Penalización a la visita por falta de aclimatación

  # Matriz de Probabilidades
  matriz = np.zeros((max_goles + 1, max_goles + 1))
  for i in range(max_goles + 1):
    for j in range(max_goles + 1):
      matriz[i, j] = poisson.pmf(i, lambda_loc) * poisson.pmf(j, lambda_vis)

  prob_local = np.sum(np.tril(matriz, -1))
  prob_empate = np.sum(np.diag(matriz))
  prob_visita = np.sum(np.triu(matriz, 1))

  return {
      "Lambda_Local": round(lambda_loc, 2),
      "Lambda_Visita": round(lambda_vis, 2),
      "Prob_Local (%)": round(prob_local * 100, 1),
      "Prob_Empate (%)": round(prob_empate * 100, 1),
      "Prob_Visita (%)": round(prob_visita * 100, 1),
      "Cuota_Valor_Local": round(1 / prob_local, 2)
      if prob_local > 0
      else None,
  }


# ==========================================
# 4. EJECUCIÓN DE PRONÓSTICOS PRÓXIMA FECHA
# ==========================================
resultados_pronosticos = []
for _, row in df_proximos.iterrows():
  pred = predecir_partido(
      row["Local"], row["Visita"], row["Altitud_Local"], row["Altitud_Visita"]
  )
  pred.update({
      "Jornada": row["Jornada"],
      "Local": row["Local"],
      "Visita": row["Visita"],
      "Ciudad": row["Ciudad"],
  })
  resultados_pronosticos.append(pred)

df_pronosticos = pd.DataFrame(resultados_pronosticos)
df_pronosticos = df_pronosticos[[
    "Jornada",
    "Local",
    "Visita",
    "Lambda_Local",
    "Lambda_Visita",
    "Prob_Local (%)",
    "Prob_Empate (%)",
    "Prob_Visita (%)",
    "Cuota_Valor_Local",
]]

print("=== PRONÓSTICOS DE LA JORNADA (MODELO POISSON + GEOGRAFÍA) ===")
print(df_pronosticos.to_string(index=False))
st.info(f"⚽ **Sugerencia de Goles:** {rec_goles}")
