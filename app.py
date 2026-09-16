import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO CSS CORPORATIVO
# ==============================================================================
st.set_page_config(
    page_title="Predicción Liga 1 Perú - Quipus Data",
    page_icon="⚽",
    layout="wide",
)

st.markdown(
    """
    <style>
    table, th, td {
        border: 1px solid #9ca592 !important;
        border-collapse: collapse !important;
        text-align: center !important;
        white-space: normal !important;
        word-break: break-word !important;
    }
    th {
        background-color: #7b8870 !important; /* Verde corporativo Quipus Data */
        color: white !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Búsqueda flexible del logo corporativo
logo_encontrado = None
for posible_nombre in [
    "logo.png",
    "logo_quipus.png",
    "quipus_logo.png",
    "logo-quipus.png",
]:
  if os.path.exists(posible_nombre):
    logo_encontrado = posible_nombre
    break

if logo_encontrado:
  st.image(logo_encontrado, width=260)
else:
  st.markdown("### 📊 Quipus Data")

st.title("⚽ Modelo de Predicción Liga 1 Perú")

# ==============================================================================
# 2. FUNCIONES MATEMÁTICAS, MOMENTUM Y RECOMENDACIÓN COMBINADA
# ==============================================================================


def tau(x, y, lambda_, mu, rho):
  if x == 0 and y == 0:
    return 1.0 - lambda_ * mu * rho
  elif x == 0 and y == 1:
    return 1.0 + lambda_ * rho
  elif x == 1 and y == 0:
    return 1.0 + mu * rho
  elif x == 1 and y == 1:
    return 1.0 - rho
  else:
    return 1.0


def matriz_dixon_coles(lambda_, mu, rho=0.13, max_goles=6):
  matriz = np.zeros((max_goles, max_goles))
  for i in range(max_goles):
    for j in range(max_goles):
      prob = poisson.pmf(i, lambda_) * poisson.pmf(j, mu)
      adj = tau(i, j, lambda_, mu, rho)
      matriz[i, j] = max(0.0, prob * adj)

  total = np.sum(matriz)
  if total > 0:
    matriz /= total
  return matriz


def calcular_tasa_ponderada(df_historial, equipo, es_local=True):
  col_l = buscar_columna(df_historial, ["local", "equipo_local"])
  col_v = buscar_columna(
      df_historial, ["visita", "visitante", "equipo_visita"]
  )
  col_gl = buscar_columna(df_historial, ["gl", "goles_local", "goles local"])
  col_gv = buscar_columna(df_historial, ["gv", "goles_visita", "goles visita"])

  if not col_l or not col_v or not col_gl or not col_gv:
    return 1.3

  if es_local:
    partidos = df_historial[df_historial[col_l] == equipo].tail(3)
    if len(partidos) == 0:
      return 1.3
    pesos = [0.5, 0.3, 0.2][: len(partidos)]
    pesos = [p / sum(pesos) for p in pesos]
    return float(np.sum(partidos[col_gl].values * pesos))
  else:
    partidos = df_historial[df_historial[col_v] == equipo].tail(3)
    if len(partidos) == 0:
      return 1.1
    pesos = [0.5, 0.3, 0.2][: len(partidos)]
    pesos = [p / sum(pesos) for p in pesos]
    return float(np.sum(partidos[col_gv].values * pesos))


def obtener_marcador_y_recomendacion(
    matriz, p_loc, p_emp, p_vis, loc, vis, p_over25, p_btts
):
  if p_loc >= p_vis and p_loc >= p_emp:
    ganador = "Local"
    equipo_ganador = loc
  elif p_vis > p_loc and p_vis >= p_emp:
    ganador = "Visita"
    equipo_ganador = vis
  else:
    ganador = "Empate"
    equipo_ganador = "Empate"

  if ganador != "Empate":
    rec = f"Gana {equipo_ganador}"
    if p_over25 >= 50.0:
      rec += " + Más de 2.5"
    elif p_btts >= 50.0:
      rec += " + Ambos marcan"
  else:
    rec = "Empate"

  max_p = -1.0
  best_i, best_j = 1, 0

  for i in range(6):
    for j in range(6):
      es_valido = False
      if ganador == "Local" and i > j:
        es_valido = True
      elif ganador == "Visita" and j > i:
        es_valido = True
      elif ganador == "Empate" and i == j:
        es_valido = True

      if es_valido:
        es_over = i + j >= 3
        if (p_over25 >= 50.0 and es_over) or (p_over25 < 50.0 and not es_over):
          peso = matriz[i, j] * 1.5
        else:
          peso = matriz[i, j]

        if peso > max_p:
          max_p = peso
          best_i, best_j = i, j

  if max_p == -1.0:
    for i in range(6):
      for j in range(6):
        if ganador == "Local" and i > j and matriz[i, j] > max_p:
          max_p = matriz[i, j]
          best_i, best_j = i, j
        elif ganador == "Visita" and j > i and matriz[i, j] > max_p:
          max_p = matriz[i, j]
          best_i, best_j = i, j
        elif ganador == "Empate" and i == j and matriz[i, j] > max_p:
          max_p = matriz[i, j]
          best_i, best_j = i, j

  return f"{best_i} - {best_j}", rec


# ==============================================================================
# 3. CARGA INTELIGENTE Y PROCESAMIENTO
# ==============================================================================
@st.cache_data
def cargar_excel(filepath):
  xls = pd.ExcelFile(filepath)
  hojas = xls.sheet_names

  hoja_partidos = None
  for hoja in ["Partidos_Fecha", "Resultados_Clausura", "Resultados_Apertura"]:
    if hoja in hojas:
      hoja_partidos = hoja
      break

  if not hoja_partidos:
    for hoja in hojas:
      temp_df = pd.read_excel(filepath, sheet_name=hoja)
      temp_cols = [str(c).lower().strip() for c in temp_df.columns]
      if any(k in temp_cols for k in ["local", "visita", "visitante"]):
        hoja_partidos = hoja
        break

  if not hoja_partidos:
    hoja_partidos = hojas[0]

  df_partidos = pd.read_excel(filepath, sheet_name=hoja_partidos)
  df_partidos.columns = df_partidos.columns.astype(str).str.strip()

  df_historia = None
  if "Resultados_Clausura" in hojas:
    df_historia = pd.read_excel(filepath, sheet_name="Resultados_Clausura")
  elif "Resultados_Apertura" in hojas:
    df_historia = pd.read_excel(filepath, sheet_name="Resultados_Apertura")
  else:
    df_historia = df_partidos

  return {"partidos": df_partidos, "historia": df_historia}


def buscar_columna(df, palabras_clave):
  for col in df.columns:
    col_clean = str(col).lower().strip()
    for clave in palabras_clave:
      if clave in col_clean:
        return col
  return None


def calcular_tasas_equipo(datos, equipo):
  df = datos["historia"].copy()
  col_local = buscar_columna(df, ["local", "equipo_local"])
  col_visita = buscar_columna(df, ["visita", "visitante", "equipo_visita"])
  col_gl = buscar_columna(df, ["gl", "goles_local", "goles local"])
  col_gv = buscar_columna(df, ["gv", "goles_visita", "goles visita"])

  if not col_local or not col_visita or not col_gl or not col_gv:
    return 1.3, 1.2

  locales = df[df[col_local] == equipo]
  visitas = df[df[col_visita] == equipo]

  goles_favor = locales[col_gl].sum() + visitas[col_gv].sum()
  goles_contra = locales[col_gv].sum() + visitas[col_gl].sum()
  total_partidos = len(locales) + len(visitas)

  if total_partidos == 0:
    return 1.3, 1.2

  return goles_favor / total_partidos, goles_contra / total_partidos


def obtener_factores_contextual(datos, local, visita):
  return 1.15, 0.88, 0.13


def obtener_ajuste_situacional_completo(local, visita, hora):
  plazas_calor = [
      "Alianza Atlético",
      "Atlético Grau",
      "Comerciantes Unidos",
      "Union Comercio",
  ]
  plazas_altura = [
      "Cienciano",
      "Cusco FC",
      "Deportivo Garcilaso",
      "Sport Huancayo",
      "FC Cajamarca",
      "ADT",
      "Melgar",
  ]
  grandes_jerarquia = ["Universitario", "Sporting Cristal", "Alianza Lima"]

  f_loc, f_vis = 1.0, 1.0
  hora_num = 15
  if pd.notna(hora) and hora is not None:
    try:
      if hasattr(hora, "hour"):
        hora_num = hora.hour
      else:
        hora_num = int(str(hora).split(":")[0])
    except Exception:
      hora_num = 15

  visita_es_altura = visita in plazas_altura
  visita_es_calor = visita in plazas_calor

  if local in plazas_calor and not visita_es_calor:
    if hora_num in [11, 12, 13]:
      f_loc *= 1.12
      f_vis *= 0.80
    elif hora_num in [14, 15]:
      f_loc *= 1.05
      f_vis *= 0.88
  elif local in plazas_altura and not visita_es_altura:
    if hora_num in [11, 12, 13]:
      f_loc *= 1.15
      f_vis *= 0.78
    elif hora_num in [14, 15]:
      f_loc *= 1.08
      f_vis *= 0.86
    elif hora_num >= 18:
      f_loc *= 1.04
      f_vis *= 0.92

  if visita in grandes_jerarquia:
    f_vis *= 1.10

  return f_loc, f_vis


# ==============================================================================
# 4. RENDERIZADO DE TABLAS (VISTA RESUMIDA Y VISTA DETALLADA)
# ==============================================================================
EXCEL_PATH = "Liga1_2026.xlsx"

if os.path.exists(EXCEL_PATH):
  try:
    datos = cargar_excel(EXCEL_PATH)
    partidos_df = datos["partidos"]

    col_local = buscar_columna(partidos_df, ["local", "equipo_local", "loc"])
    col_visita = buscar_columna(
        partidos_df, ["visita", "visitante", "equipo_visita", "vis"]
    )
    col_jornada = buscar_columna(partidos_df, ["jornada", "fecha_liga"])
    col_fecha = buscar_columna(partidos_df, ["fecha", "date"])
    col_hora = buscar_columna(partidos_df, ["hora", "time"])

    if not col_local or not col_visita:
      st.error("❌ No se pudieron identificar las columnas de equipos.")
      st.stop()

    if col_jornada:
      jornadas = sorted(partidos_df[col_jornada].dropna().unique())
      st.sidebar.header("⚽ Selección de Jornada")
      jornada_sel = st.sidebar.selectbox("Seleccionar Jornada", jornadas)
      partidos_fecha = partidos_df[
          partidos_df[col_jornada] == jornada_sel
      ].copy()
    else:
      partidos_fecha = partidos_df.copy()

    resultados = []
    for _, row in partidos_fecha.iterrows():
      loc, vis = str(row[col_local]), str(row[col_visita])
      fecha_p = (
          row[col_fecha] if col_fecha and col_fecha in row else "Por definir"
      )

      hora_raw = row[col_hora] if col_hora and col_hora in row else None
      if pd.notna(hora_raw) and hora_raw is not None:
        if hasattr(hora_raw, "strftime"):
          hora_str = hora_raw.strftime("%H:%M")
        else:
          hora_str = str(hora_raw)[:5]
      else:
        hora_str = "--:--"

      gf_loc, ga_loc = calcular_tasas_equipo(datos, loc)
      gf_vis, ga_vis = calcular_tasas_equipo(datos, vis)

      tasa_rec_loc = calcular_tasa_ponderada(
          datos["historia"], loc, es_local=True
      )
      tasa_rec_vis = calcular_tasa_ponderada(
          datos["historia"], vis, es_local=False
      )

      f_loc, f_vis, rho_dc = obtener_factores_contextual(datos, loc, vis)
      f_sit_loc, f_sit_vis = obtener_ajuste_situacional_completo(
          loc, vis, hora_raw
      )

      l_loc = max(0.4, ((tasa_rec_loc + ga_vis) / 2.0) * f_loc * f_sit_loc)
      m_vis = max(0.3, ((tasa_rec_vis + ga_loc) / 2.0) * f_vis * f_sit_vis)

      matriz = matriz_dixon_coles(l_loc, m_vis, rho=rho_dc)

      p_loc = float(np.sum(np.tril(matriz, -1)))
      p_emp = float(np.sum(np.diag(matriz)))
      p_vis = float(np.sum(np.triu(matriz, 1)))

      p_over25 = float(
          1.0
          - sum(
              matriz[i, j] for i in range(6) for j in range(6) if i + j <= 2
          )
      )
      p_btts = float(
          sum(matriz[i, j] for i in range(1, 6) for j in range(1, 6))
      )

      marcador, rec = obtener_marcador_y_recomendacion(
          matriz,
          p_loc * 100,
          p_emp * 100,
          p_vis * 100,
          loc,
          vis,
          p_over25 * 100,
          p_btts * 100,
      )

      fecha_limpia = (
          str(fecha_p)[:10] if pd.notna(fecha_p) else "Por definir"
      )
      if "-" in fecha_limpia and len(fecha_limpia.split("-")[0]) == 4:
        partes = fecha_limpia.split("-")
        if len(partes) == 3:
          fecha_limpia = f"{partes[2]}/{partes[1]}/{partes[0]}"

      resultados.append({
          "Fecha": fecha_limpia,
          "Hora": hora_str,
          "Local": loc,
          "Visita": vis,
          "% Local": round(p_loc * 100, 1),
          "% Empate": round(p_emp * 100, 1),
          "% Visita": round(p_vis * 100, 1),
          "Más de 2.5 goles": round(p_over25 * 100, 1),
          "Ambos marcan: Sí": round(p_btts * 100, 1),
          "Marcador_Modal": marcador,
          "Recomendacion": rec,
      })

    df_pronosticos = pd.DataFrame(resultados)

    max_prob = df_pronosticos[["% Local", "% Empate", "% Visita"]].max(axis=1)
    df_pronosticos.insert(
        0, "🔥", ["🔥" if p >= 60.0 else "➖" for p in max_prob]
    )

    # Creamos un DataFrame exclusivo para la vista resumida (Celular)
    df_resumido = df_pronosticos[
        ["🔥", "Fecha", "Hora", "Local", "Visita", "Marcador_Modal", "Recomendacion"]
    ].copy()

    def resaltar_texto_porcentajes(row):
      styles = [""] * len(row)
      estilo_texto_verde = "color: #2e6930; font-weight: bold;"

      p_local, p_empate, p_visita = (
          row["% Local"],
          row["% Empate"],
          row["% Visita"],
      )
      max_val = max(p_local, p_empate, p_visita)

      if p_local == max_val and p_local >= 60.0:
        styles[row.index.get_loc("% Local")] = estilo_texto_verde
      elif p_empate == max_val and p_empate >= 60.0:
        styles[row.index.get_loc("% Empate")] = estilo_texto_verde
      elif p_visita == max_val and p_visita >= 60.0:
        styles[row.index.get_loc("% Visita")] = estilo_texto_verde

      if row["Más de 2.5 goles"] >= 50.0:
        styles[row.index.get_loc("Más de 2.5 goles")] = estilo_texto_verde

      if row["Ambos marcan: Sí"] >= 50.0:
        styles[row.index.get_loc("Ambos marcan: Sí")] = estilo_texto_verde

      if max_val >= 70.0:
        styles[row.index.get_loc("Recomendacion")] = estilo_texto_verde

      return styles

    def resaltar_texto_resumen(row):
      styles = [""] * len(row)
      estilo_texto_verde = "color: #2e6930; font-weight: bold;"
      # Si la recomendación contiene texto fuerte o podemos evaluar el max original
      # Para mantenerlo simple, si la celda de recomendación es larga o queremos destacarla:
      if "Gana" in str(row["Recomendacion"]):
        styles[row.index.get_loc("Recomendacion")] = estilo_texto_verde
      return styles

    # Estilos para ambas tablas
    estilo_tabla_completa = df_pronosticos.style.apply(
        resaltar_texto_porcentajes, axis=1
    ).format(
        "{:.1f}",
        subset=[
            "% Local",
            "% Empate",
            "% Visita",
            "Más de 2.5 goles",
            "Ambos marcan: Sí",
        ],
    )

    estilo_tabla_resumida = df_resumido.style.apply(
        resaltar_texto_resumen, axis=1
    )

    st.subheader("📋 Resumen de pronósticos")

    # Pestañas para elegir entre Vista Móvil (Resumida) y Vista PC (Completa)
    pestana_celular, pestana_pc = st.tabs(
        ["📱 Vista Resumida (Ideal para Celular)", "💻 Vista Completa (Detallada)"]
    )

    with pestana_celular:
      st.markdown(
          "*Vista ligera optimizada con los datos clave para teléfonos móviles.*"
      )
      st.dataframe(
          estilo_tabla_resumida, use_container_width=True, hide_index=True
      )

    with pestana_pc:
      st.markdown(
          "*Vista extendida con porcentajes completos de probabilidad y goles.*"
      )
      st.dataframe(
          estilo_tabla_completa, use_container_width=True, hide_index=True
      )

  except Exception as e:
    st.error(f"❌ Error al procesar el archivo Excel: {str(e)}")
else:
  st.error(f"❌ No se encontró el archivo '{EXCEL_PATH}'.")
