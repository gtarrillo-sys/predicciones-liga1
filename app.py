import os
import re
import numpy as np
import pandas as pd
from scipy.stats import poisson
import streamlit as st

# =========================================================
# CONFIGURACIÓN DE LA PÁGINA STREAMLIT
# =========================================================
st.set_page_config(
    page_title="Predicción Liga 1 Perú - Dixon Coles",
    page_icon="⚽",
    layout="wide",
)

# =========================================================
# 1. DICCIONARIO Y NORMALIZACIÓN ROBUSTA DE NOMBRES
# =========================================================
DICCIONARIO_EQUIPOS = {
    # Juan Pablo II
    "juan pablo ii college": "colegio juan pablo ii",
    "juan pablo ii": "colegio juan pablo ii",
    "colegio juan pablo ii": "colegio juan pablo ii",
    # Los Chankas
    "los chankas": "los chankas",
    "chankas": "los chankas",
    "chankas cyc": "los chankas",
    "los chankas cyc": "los chankas",
    # Equipos de Lima y Costa
    "sporting cristal": "sporting cristal",
    "cristal": "sporting cristal",
    "alianza lima": "alianza lima",
    "universitario": "universitario",
    "universitario de deportes": "universitario",
    "sport boys": "sport boys",
    "alianza atletico": "alianza atletico",
    "alianza atlético": "alianza atletico",
    "atletico grau": "atletico grau",
    "atlético grau": "atletico grau",
    "comerciantes unidos": "comerciantes unidos",
    # Equipos de Altura
    "fbc melgar": "melgar",
    "melgar": "melgar",
    "deportivo garcilaso": "garcilaso",
    "garcilaso": "garcilaso",
    "cusco fc": "cusco",
    "cusco": "cusco",
    "deporte huancayo": "sport huancayo",
    "deportivo huancayo": "sport huancayo",
    "sport huancayo": "sport huancayo",
    "huancayo": "sport huancayo",
    "fc cajamarca": "ut c",
    "utc": "ut c",
    "ut c": "ut c",
    "utc cajamarca": "ut c",
    "adt": "adt",
    "adt de tarma": "adt",
}


def estandarizar_nombre(nombre):
  """Limpia el texto y busca la equivalencia en el diccionario central."""
  if not isinstance(nombre, str):
    return ""
  txt = nombre.lower().strip()
  txt = re.sub(
      r"\b(club|fc|cd|atletico|atlético|deportivo|asociacion|asociación)\b",
      "",
      txt,
  )
  txt = re.sub(r"\s+", " ", txt).strip()
  return DICCIONARIO_EQUIPOS.get(txt, txt)


# =========================================================
# 2. CARGA DE DATOS DESDE EXCEL
# =========================================================
def obtener_ruta_excel(nombre_archivo="Liga1_2026.xlsx"):
  """Obtiene la ruta absoluta del archivo Excel en la carpeta del script."""
  directorio_actual = os.path.dirname(os.path.abspath(__file__))
  return os.path.join(directorio_actual, nombre_archivo)


@st.cache_data(ttl=60)
def cargar_datos_excel(fuente_archivo):
  try:
    xls = pd.ExcelFile(fuente_archivo)

    # Carga de pestañas
    df_partidos = pd.read_excel(xls, sheet_name="Partidos_Fecha")
    df_tabla = pd.read_excel(xls, sheet_name="Tabla_Acumulada")
    df_geo = pd.read_excel(xls, sheet_name="Geo_Info")

    # Limpieza de columnas
    df_partidos.columns = df_partidos.columns.str.strip().str.lower()
    df_tabla.columns = df_tabla.columns.str.strip().str.lower()
    df_geo.columns = df_geo.columns.str.strip().str.lower()

    # Estandarización de nombres
    df_partidos["local_std"] = df_partidos["local"].apply(estandarizar_nombre)
    df_partidos["visita_std"] = df_partidos["visita"].apply(
        estandarizar_nombre
    )

    df_tabla["equipo_std"] = df_tabla["equipo"].apply(estandarizar_nombre)
    df_geo["equipo_std"] = df_geo["equipo"].apply(estandarizar_nombre)

    return df_partidos, df_tabla, df_geo, None
  except Exception as e:
    return None, None, None, str(e)


# =========================================================
# 3. AUXILIARES: FUERZA Y ALTITUD
# =========================================================
def obtener_fuerza_equipos(equipo_local, equipo_visita, df_tabla):
  row_loc = df_tabla[df_tabla["equipo_std"] == equipo_local]
  row_vis = df_tabla[df_tabla["equipo_std"] == equipo_visita]

  att_loc = row_loc["gf"].values[0] / max(1, row_loc["pj"].values[0]) if not row_loc.empty and "gf" in row_loc.columns else 1.25
  def_loc = row_loc["gc"].values[0] / max(1, row_loc["pj"].values[0]) if not row_loc.empty and "gc" in row_loc.columns else 1.15

  att_vis = row_vis["gf"].values[0] / max(1, row_vis["pj"].values[0]) if not row_vis.empty and "gf" in row_vis.columns else 1.10
  def_vis = row_vis["gc"].values[0] / max(1, row_vis["pj"].values[0]) if not row_vis.empty and "gc" in row_vis.columns else 1.30

  return att_loc, def_loc, att_vis, def_vis


def obtener_altitud(equipo, df_geo):
  row = df_geo[df_geo["equipo_std"] == equipo]
  if not row.empty and "altitud" in row.columns:
    return float(row["altitud"].values[0])
  return 150.0


# =========================================================
# 4. MOTOR MODELO DIXON-COLES
# =========================================================
def tau_dixon_coles(x, y, lambda_local, mu_visita, rho=-0.11):
  if x == 0 and y == 0:
    return max(0.0001, 1.0 - (lambda_local * mu_visita * rho))
  elif x == 1 and y == 0:
    return max(0.0001, 1.0 + (mu_visita * rho))
  elif x == 0 and y == 1:
    return max(0.0001, 1.0 + (lambda_local * rho))
  elif x == 1 and y == 1:
    return max(0.0001, 1.0 - rho)
  else:
    return 1.0


def calcular_dixon_coles(
    equipo_local, equipo_visita, df_tabla, df_geo_info, rho=-0.11
):
  att_loc, def_loc, att_vis, def_vis = obtener_fuerza_equipos(
      equipo_local, equipo_visita, df_tabla
  )

  alt_loc = obtener_altitud(equipo_local, df_geo_info)
  alt_vis = obtener_altitud(equipo_visita, df_geo_info)

  dif_altitud = max(0.0, alt_loc - alt_vis)
  home_advantage = 1.15 if alt_loc < 1000 else 1.28

  factor_def_visita = 1.0 + (dif_altitud / 6500.0) if alt_loc >= 2000 else 1.0

  lambda_local = max(
      1.05, (att_loc * (def_vis / 1.20)) * home_advantage * factor_def_visita
  )

  penalizacion_visita_altura = 0.78 if alt_loc >= 2500 else 1.0
  mu_visita = max(0.55, att_vis * (def_loc / 1.20) * penalizacion_visita_altura)

  max_goles = 8
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

  prob_under25 = float(
      sum(
          matriz_prob[i, j]
          for i in range(max_goles)
          for j in range(max_goles)
          if i + j < 2.5
      )
  )
  prob_over25 = 1.0 - prob_under25
  prob_btts_si = float(matriz_prob[1:, 1:].sum())

  return (
      prob_local,
      prob_empate,
      prob_visita,
      prob_over25,
      prob_btts_si,
      matriz_prob,
  )


# =========================================================
# 5. INTERFAZ STREAMLIT
# =========================================================
st.title("⚽ Modelo de Predicción Liga 1 Perú")
st.caption("Ajustado por Dixon-Coles y Factor de Altitud Real")

# Opción en barra lateral para subir el archivo manualmente
st.sidebar.header("📁 Archivo de Datos")
archivo_subido = st.sidebar.file_uploader(
    "Subir archivo Excel (Liga1_2026.xlsx)", type=["xlsx"]
)

# Búsqueda local de Liga1_2026.xlsx
ruta_local = obtener_ruta_excel("Liga1_2026.xlsx")

if archivo_subido is not None:
  fuente_excel = archivo_subido
elif os.path.exists(ruta_local):
  fuente_excel = ruta_local
else:
  fuente_excel = None

# Botón para recargar la memoria caché
if st.sidebar.button("🔄 Recargar Datos"):
  st.cache_data.clear()
  st.rerun()

if fuente_excel is None:
  st.error(
      "❌ No se encontró el archivo 'Liga1_2026.xlsx' en la carpeta del"
      " proyecto."
  )
  st.info(
      "👉 Por favor, arrastra y suelta tu archivo 'Liga1_2026.xlsx' en el"
      " cargador de la barra lateral izquierda."
  )
else:
  df_partidos, df_tabla, df_geo, error = cargar_datos_excel(fuente_excel)

  if error:
    st.error(f"Error al procesar la estructura del Excel: {error}")
  else:
    fechas_disponibles = sorted(df_partidos["fecha"].unique())
    fecha_sel = st.sidebar.selectbox(
        "Seleccionar Fecha de la Liga 1:", fechas_disponibles
    )

    df_f = df_partidos[df_partidos["fecha"] == fecha_sel]

    st.subheader(f"Pronósticos para la Fecha {fecha_sel}")

    for idx, row in df_f.iterrows():
      eq_loc_std = row["local_std"]
      eq_vis_std = row["visita_std"]

      nombre_loc_orig = row["local"]
      nombre_vis_orig = row["visita"]

      p_loc, p_emp, p_vis, p_over, p_btts, matriz = calcular_dixon_coles(
          eq_loc_std, eq_vis_std, df_tabla, df_geo
      )

      # Sugerencia de apuesta
      if p_loc >= 0.40 and (p_loc - p_vis) >= 0.08:
        fija_txt = f"Gana {nombre_loc_orig} (Directo)"
        confian_txt = "Alta" if p_loc >= 0.52 else "Media-Alta"
      elif p_vis >= 0.40 and (p_vis - p_loc) >= 0.08:
        fija_txt = f"Gana {nombre_vis_orig} (Directo)"
        confian_txt = "Alta" if p_vis >= 0.52 else "Media-Alta"
      elif (p_loc + p_emp) >= 0.62:
        fija_txt = f"Local o Empate ({nombre_loc_orig})"
        confian_txt = "Media"
      elif (p_vis + p_emp) >= 0.62:
        fija_txt = f"Empate o Visita ({nombre_vis_orig})"
        confian_txt = "Media"
      else:
        fija_txt = "Doble Opción / Partido Abierto"
        confian_txt = "Media"

      with st.expander(f"📌 {nombre_loc_orig} vs {nombre_vis_orig}"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
          st.metric(
              label=f"Victoria {nombre_loc_orig}", value=f"{p_loc * 100:.1f}%"
          )
          st.caption(f"Cuota Justa: {1 / max(p_loc, 0.001):.2f}")

        with col2:
          st.metric(label="Empate", value=f"{p_emp * 100:.1f}%")
          st.caption(f"Cuota Justa: {1 / max(p_emp, 0.001):.2f}")

        with col3:
          st.metric(
              label=f"Victoria {nombre_vis_orig}", value=f"{p_vis * 100:.1f}%"
          )
          st.caption(f"Cuota Justa: {1 / max(p_vis, 0.001):.2f}")

        with col4:
          st.write(f"**Sugerencia:** {fija_txt}")
          st.write(f"**Nivel Confianza:** {confian_txt}")
          st.write(f"**Over 2.5:** {p_over * 100:.1f}%")
          st.write(f"**Ambos Anotan:** {p_btts * 100:.1f}%")
