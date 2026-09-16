import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA STREAMLIT Y LOGO CORPORATIVO
# ==============================================================================
st.set_page_config(
    page_title="Predicción Liga 1 Perú - Quipus Data",
    page_icon="⚽",
    layout="wide"
)

# Renderizar Logo y Título Corporativo
if os.path.exists("logo.png"):
    st.image("logo.png", width=260)

st.title("⚽ Modelo de Predicción Liga 1 Perú")

# ==============================================================================
# 2. FUNCIONES MATEMÁTICAS Y DIXON-COLES
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

def obtener_marcador_modal(matriz, p_loc, p_emp, p_vis):
    if p_loc > p_vis and p_loc > p_emp:
        max_p = -1.0
        best_i, best_j = 1, 0
        for i in range(6):
            for j in range(6):
                if i > j and matriz[i, j] > max_p:
                    max_p = matriz[i, j]
                    best_i, best_j = i, j
        return f"{best_i} - {best_j}"

    elif p_vis > p_loc and p_vis > p_emp:
        max_p = -1.0
        best_i, best_j = 0, 1
        for i in range(6):
            for j in range(6):
                if j > i and matriz[i, j] > max_p:
                    max_p = matriz[i, j]
                    best_i, best_j = i, j
        return f"{best_i} - {best_j}"

    else:
        max_p = -1.0
        best_i, best_j = 1, 1
        for i in range(6):
            if matriz[i, i] > max_p:
                max_p = matriz[i, i]
                best_i, best_j = i, i
        return f"{best_i} - {best_j}"

# ==============================================================================
# 3. FUNCIONES DE PROCESAMIENTO Y AJUSTES SITUACIONALES
# ==============================================================================
@st.cache_data
def cargar_excel(filepath):
    xls = pd.ExcelFile(filepath)
    hojas = xls.sheet_names
    
    # Cargar pestaña de partidos o primera pestaña
    hoja_partidos = "partidos" if "partidos" in hojas else hojas[0]
    df_partidos = pd.read_excel(filepath, sheet_name=hoja_partidos)
    df_partidos.columns = df_partidos.columns.str.strip().str.lower()
    
    if "fecha" in df_partidos.columns:
        df_partidos["fecha"] = pd.to_datetime(df_partidos["fecha"], errors="coerce")

    return {"partidos": df_partidos}

def calcular_tasas_equipo(datos, equipo, antes_de=None):
    df = datos["partidos"].copy()
    if antes_de is not None and "fecha" in df.columns:
        df = df[df["fecha"] < antes_de]

    # Filtrar jugados con goles válidos
    jugados = df[df["gl"].notna() & df["gv"].notna()]

    locales = jugados[jugados["local"] == equipo]
    visitas = jugados[jugados["visita"] == equipo]

    goles_favor = locales["gl"].sum() + visitas["gv"].sum()
    goles_contra = locales["gv"].sum() + visitas["gl"].sum()
    total_partidos = len(locales) + len(visitas)

    if total_partidos == 0:
        return 1.3, 1.2

    return goles_favor / total_partidos, goles_contra / total_partidos

def obtener_factores_contextual(datos, local, visita):
    return 1.15, 0.88, 0.13

def obtener_ajuste_situacional_completo(local, visita, hora, objetivo_loc, objetivo_vis):
    """
    Ajuste de intensidades considerando:
    1. Hora + Clima + Cruce de Plazas
    2. Jerarquía Ofensiva de los Grandes
    3. Tensión por Objetivos de Cierre del Clausura
    """
    plazas_calor = ["Alianza Atlético", "Atlético Grau", "Comerciantes Unidos", "Union Comercio"]
    plazas_altura = ["Cienciano", "Cusco FC", "Deportivo Garcilaso", "Sport Huancayo", "FC Cajamarca", "ADT", "Melgar"]
    grandes_jerarquia = ["Universitario", "Sporting Cristal", "Alianza Lima"]

    f_loc, f_vis = 1.0, 1.0

    # Determinar hora entera
    hora_num = 15
    if pd.notna(hora) and hora is None:
        try:
            if hasattr(hora, 'hour'):
                hora_num = hora.hour
            else:
                hora_num = int(str(hora).split(':')[0])
        except Exception:
            hora_num = 15

    # A. CLIMA / HORARIO / CRUCE DE ORIGEN
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

    # B. JERARQUÍA OFENSIVA DE LOS GRANDES COMO VISITANTE
    if visita in grandes_jerarquia:
        f_vis *= 1.10

    # C. TENSIÓN Y PRESION POR OBJETIVOS DEL CLAUSURA
    obj_loc = str(objetivo_loc).lower() if pd.notna(objetivo_loc) else ""
    obj_vis = str(objetivo_vis).lower() if pd.notna(objetivo_vis) else ""

    if ("titulo" in obj_loc or "copa" in obj_loc) and ("titulo" in obj_vis or "copa" in obj_vis):
        f_loc *= 1.08
        f_vis *= 1.08
    elif "descenso" in obj_loc and "descenso" in obj_vis:
        f_loc *= 0.88
        f_vis *= 0.88
    elif "descenso" in obj_loc and "nada" in obj_vis:
        f_loc *= 1.12
        f_vis *= 0.90
    elif "descenso" in obj_vis and "nada" in obj_loc:
        f_vis *= 1.10
        f_loc *= 0.92

    return f_loc, f_vis

# ==============================================================================
# 4. EJECUCIÓN DEL PANEL STREAMLIT (CARGA AUTOMÁTICA DESDE GITHUB)
# ==============================================================================
EXCEL_PATH = "Liga1_2026.xlsx"  # Archivo en la raíz del repositorio

if os.path.exists(EXCEL_PATH):
    try:
        datos = cargar_excel(EXCEL_PATH)
        partidos_df = datos["partidos"]

        # Selector de Jornada en la barra lateral
        if "jornada" in partidos_df.columns:
            jornadas = sorted(partidos_df["jornada"].dropna().unique())
            st.sidebar.header("⚽ Selección de Jornada")
            jornada_sel = st.sidebar.selectbox("Seleccionar Jornada", jornadas)
            partidos_fecha = partidos_df[partidos_df["jornada"] == jornada_sel].copy()
        else:
            partidos_fecha = partidos_df.copy()

        resultados = []
        for _, row in partidos_fecha.iterrows():
            loc, vis = str(row["local"]), str(row["visita"])
            fecha_p = row.get("fecha", None)

            # Formato de Hora
            hora_raw = row.get("hora", None)
            if pd.notna(hora_raw) and hora_raw is not None:
                if hasattr(hora_raw, 'strftime'):
                    hora_str = hora_raw.strftime("%H:%M")
                else:
                    hora_str = str(hora_raw)[:5]
            else:
                hora_str = "--:--"

            # Objetivos del Clausura si existen en el Excel
            obj_loc = row.get("objetivo_local", "normal")
            obj_vis = row.get("objetivo_visita", "normal")

            # Cálculo de tasas de gol
            gf_loc, ga_loc = calcular_tasas_equipo(datos, loc, antes_de=fecha_p)
            gf_vis, ga_vis = calcular_tasas_equipo(datos, vis, antes_de=fecha_p)

            # Factores contextuales y situacionales
            f_loc, f_vis, rho_dc = obtener_factores_contextual(datos, loc, vis)
            f_sit_loc, f_sit_vis = obtener_ajuste_situacional_completo(loc, vis, hora_raw, obj_loc, obj_vis)

            # Intensidades ajustadas
            l_loc = max(0.4, ((gf_loc + ga_vis) / 2.0) * f_loc * f_sit_loc)
            m_vis = max(0.3, ((gf_vis + ga_loc) / 2.0) * f_vis * f_sit_vis)

            matriz = matriz_dixon_coles(l_loc, m_vis, rho=rho_dc)

            p_loc = float(np.sum(np.tril(matriz, -1)))
            p_emp = float(np.sum(np.diag(matriz)))
            p_vis = float(np.sum(np.triu(matriz, 1)))

            p_over25 = float(
                1.0 - sum(matriz[i, j] for i in range(6) for j in range(6) if i + j <= 2)
            )
            p_btts = float(
                sum(matriz[i, j] for i in range(1, 6) for j in range(1, 6))
            )

            marcador = obtener_marcador_modal(matriz, p_loc, p_emp, p_vis)
            rec = (
                f"Gana {loc}"
                if p_loc > p_vis and p_loc > p_emp
                else (f"Gana {vis}" if p_vis > p_loc and p_vis > p_emp else "Empate")
            )

            resultados.append(
                {
                    "Fecha": (
                        fecha_p.strftime("%Y-%m-%d") if pd.notna(fecha_p) else "Por definir"
                    ),
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
                }
            )

        # Convertir a DataFrame
        df_pronosticos = pd.DataFrame(resultados)

        # 1. Agregar columna 🔥 a partidos calientes (>= 50%)
        max_prob = df_pronosticos[["% Local", "% Empate", "% Visita"]].max(axis=1)
        df_pronosticos.insert(0, "🔥", ["🔥" if p >= 50.0 else "➖" for p in max_prob])

        # 2. Función para resaltar el TEXTO de los porcentajes en verde
        def resaltar_texto_porcentajes(row):
            styles = [""] * len(row)
            # Estilo verde fuerte para el texto sin pintar el fondo de la celda
            estilo_texto_verde = "color: #1e7e34; font-weight: bold;"

            p_local, p_empate, p_visita = row["% Local"], row["% Empate"], row["% Visita"]
            max_val = max(p_local, p_empate, p_visita)

            if p_local == max_val:
                styles[row.index.get_loc("% Local")] = estilo_texto_verde
            elif p_empate == max_val:
                styles[row.index.get_loc("% Empate")] = estilo_texto_verde
            elif p_visita == max_val:
                styles[row.index.get_loc("% Visita")] = estilo_texto_verde

            if row["Más de 2.5 goles"] >= 50.0:
                styles[row.index.get_loc("Más de 2.5 goles")] = estilo_texto_verde

            if row["Ambos marcan: Sí"] >= 50.0:
                styles[row.index.get_loc("Ambos marcan: Sí")] = estilo_texto_verde

            return styles

        # 3. Estilizado visual corporativo Quipus Data
        estilo_tabla = (
            df_pronosticos.style.apply(resaltar_texto_porcentajes, axis=1)
            .format("{:.1f}", subset=["% Local", "% Empate", "% Visita", "Más de 2.5 goles", "Ambos marcan: Sí"])
            .set_properties(
                **{
                    "border-color": "#e0e2dc",
                    "font-family": "sans-serif",
                    "text-align": "center",
                }
            )
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#9ca592"),
                            ("color", "#ffffff"),
                            ("font-weight", "bold"),
                            ("text-align", "center"),
                            ("padding", "8px"),
                        ],
                    },
                    {
                        "selector": "tbody tr:hover",
                        "props": [("background-color", "#f4f5f2 !important")],
                    },
                ]
            )
        )

        # 4. Renderizado en pantalla
        st.subheader("📋 Resumen de pronósticos")
        st.dataframe(estilo_tabla, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error al procesar el archivo del repositorio: {str(e)}")
else:
    st.error(f"❌ No se encontró el archivo '{EXCEL_PATH}' en la raíz del repositorio de GitHub.")
