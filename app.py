import base64
import os
import numpy as np
import pandas as pd
from scipy.stats import poisson
import streamlit as st

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ENCABEZADO VISUAL
# ==============================================================================
st.set_page_config(
    page_title="Modelo de Predicción Liga 1 Perú - Quipus Data",
    page_icon="⚽",
    layout="wide",
)

LOGO_PATH = "logo_quipus.png"

if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div style="text-align: left; margin-bottom: 10px;">
            <img src="data:image/png;base64,{logo_b64}" style="max-width: 380px; width: 100%; height: auto; border-radius: 4px;">
        </div>
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 25px;">
            <span style="font-size: 1.8rem;">⚽</span>
            <h2 style="margin: 0; padding: 0; font-size: 1.8rem; color: #1E293B; font-weight: 700;">Modelo de Predicción Liga 1 Perú</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("⚽ Modelo de Predicción Liga 1 Perú - Quipus Data")

# ==============================================================================
# 2. PROCESAMIENTO Y LIMPIEZA DE DATOS
# ==============================================================================
PRIOR_GF = 1.3
PRIOR_GA = 1.1


def normalizar_nombre(nombre):
    if not isinstance(nombre, str):
        return ""
    import unicodedata

    n = nombre.strip().lower()
    n = unicodedata.normalize("NFD", n).encode("ascii", "ignore").decode("utf-8")
    mapeo = {
        "alianza atletico": "alianza atletico",
        "alianza lima": "alianza lima",
        "atletico grau": "atletico grau",
        "cantolao": "academia cantolao",
        "cajamarca": "fc cajamarca",
        "comerciantes unidos": "comerciantes unidos",
        "cusco": "cusco fc",
        "deportivo garcilaso": "deportivo garcilaso",
        "garcilaso": "deportivo garcilaso",
        "juan pablo ii": "juan pablo ii college",
        "melgar": "fbc melgar",
        "sport boys": "sport boys",
        "sport huancayo": "sport huancayo",
        "sporting cristal": "sporting cristal",
        "universitario": "universitario",
        "utc": "utc cajamarca",
        "chankas": "los chankas",
        "los chankas": "los chankas",
        "cienciano": "cienciano",
        "adt": "adt",
    }
    return mapeo.get(n, n)


def estandarizar_columnas(df):
    df.columns = [
        str(c)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        for c in df.columns
    ]
    renombres = {
        "equipo_local": "local",
        "equipo_visitante": "visita",
        "visitante": "visita",
        "goles_local": "gl",
        "goles_visitante": "gv",
        "goles_visita": "gv",
    }
    df = df.rename(columns=renombres)

    if "local" in df.columns:
        df["local_std"] = df["local"].apply(normalizar_nombre)
    if "visita" in df.columns:
        df["visita_std"] = df["visita"].apply(normalizar_nombre)
    if "equipo" in df.columns:
        df["equipo_std"] = df["equipo"].apply(normalizar_nombre)
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


@st.cache_data
def cargar_excel(fuente):
    xls = pd.ExcelFile(fuente)
    hojas = xls.sheet_names

    requeridas = [
        "Partidos_Fecha",
        "Resultados_Apertura",
        "Resultados_Clausura",
        "Tabla_Acumulada",
    ]
    faltantes = [h for h in requeridas if h not in hojas]
    if faltantes:
        raise ValueError(f"Faltan hojas obligatorias: {faltantes}")

    partidos = estandarizar_columnas(pd.read_excel(xls, sheet_name="Partidos_Fecha"))
    apertura = estandarizar_columnas(pd.read_excel(xls, sheet_name="Resultados_Apertura"))
    clausura = estandarizar_columnas(pd.read_excel(xls, sheet_name="Resultados_Clausura"))
    acumulada = estandarizar_columnas(pd.read_excel(xls, sheet_name="Tabla_Acumulada"))

    tabla_clausura = (
        estandarizar_columnas(pd.read_excel(xls, sheet_name="Tabla_Clausura"))
        if "Tabla_Clausura" in hojas
        else pd.DataFrame()
    )
    h2h = (
        estandarizar_columnas(pd.read_excel(xls, sheet_name="Historial_H2H"))
        if "Historial_H2H" in hojas
        else pd.DataFrame()
    )
    geo = (
        estandarizar_columnas(pd.read_excel(xls, sheet_name="Data_Geografica"))
        if "Data_Geografica" in hojas
        else pd.DataFrame()
    )

    if "xg_local" in apertura.columns and "xg_visita" in apertura.columns:
        apertura["xg_local_clean"] = (
            apertura["xg_local"].astype(str).str.replace(",", ".").astype(float)
        )
        apertura["xg_visita_clean"] = (
            apertura["xg_visita"].astype(str).str.replace(",", ".").astype(float)
        )

    return {
        "partidos": partidos,
        "apertura": apertura,
        "clausura": clausura,
        "acumulada": acumulada,
        "tabla_clausura": tabla_clausura,
        "geo": geo,
        "h2h": h2h,
    }


# ==============================================================================
# 3. FACTORES GEOGRÁFICOS, LOGÍSTICA Y CLÁSICOS
# ==============================================================================
def obtener_factores_contextual(datos, local, visita):
    geo = datos.get("geo", pd.DataFrame())
    h2h = datos.get("h2h", pd.DataFrame())

    eq_loc = normalizar_nombre(local)
    eq_vis = normalizar_nombre(visita)

    # Variables por defecto
    mult_loc = 1.15  # Ventaja de localía estándar
    mult_vis = 1.00
    rho_ajustado = -0.08

    # 1. Factores Geográficos y Logísticos
    if not geo.empty and "equipo_std" in geo.columns:
        info_loc = geo[geo["equipo_std"] == eq_loc]
        info_vis = geo[geo["equipo_std"] == eq_vis]

        if not info_loc.empty and not info_vis.empty:
            alt_loc = float(info_loc.iloc[0].get("altitud", 0))
            alt_vis = float(info_vis.iloc[0].get("altitud", 0))
            cancha_loc = str(info_loc.iloc[0].get("cancha", "natural")).lower()
            cancha_vis = str(info_vis.iloc[0].get("cancha", "natural")).lower()
            traslado = float(info_vis.iloc[0].get("horas_traslado", 0))

            # Shock de altura (si el local es de altura y la visita de costa)
            delta_alt = alt_loc - alt_vis
            if delta_alt > 1500:
                mult_loc += 0.15
                mult_vis -= 0.18
            elif delta_alt > 2000:
                mult_loc += 0.25
                mult_vis -= 0.25

            # Penalización Gras Sintético
            if "sintetic" in cancha_loc and "sintetic" not in cancha_vis:
                mult_vis -= 0.10

            # Penalización por viaje complejo (ej. bus > 3 hrs tras vuelo a Cutervo)
            if traslado >= 3.0:
                mult_vis -= 0.08

    # 2. Detección de Clásicos / Derbis
    if not h2h.empty:
        match_h2h = h2h[
            ((h2h["local_std"] == eq_loc) & (h2h["visita_std"] == eq_vis))
            | ((h2h["local_std"] == eq_vis) & (h2h["visita_std"] == eq_loc))
        ]
        if not match_h2h.empty:
            tipo_rivalidad = str(
                match_h2h.iloc[0].get("tipo_rivalidad", "normal")
            ).lower()
            if "clasico" in tipo_rivalidad or "derbi" in tipo_rivalidad:
                mult_loc *= 0.70  # Dilución de ventaja local en clásicos
                rho_ajustado = -0.15  # Compresión matemática hacia empates de bajo score

    return mult_loc, max(0.5, mult_vis), rho_ajustado


# ==============================================================================
# 4. MOTOR PONDERADO DUAL Y DIXON-COLES
# ==============================================================================
def calcular_tasas_equipo(datos, equipo, antes_de=None, ventana_clausura=5):
    eq = normalizar_nombre(equipo)
    ap = datos.get("apertura", pd.DataFrame()).copy()
    cl = datos.get("clausura", pd.DataFrame()).copy()

    # Apertura (60% Gol + 40% xG)
    ap_eq = ap[(ap["local_std"] == eq) | (ap["visita_std"] == eq)].copy()
    if not ap_eq.empty:
        gf_ap, ga_ap = [], []
        for _, r in ap_eq.iterrows():
            es_local = r["local_std"] == eq
            gl, gv = float(r["gl"]), float(r["gv"])
            xg_l = float(r.get("xg_local_clean", gl)) if pd.notna(r.get("xg_local_clean")) else gl
            xg_v = float(r.get("xg_visita_clean", gv)) if pd.notna(r.get("xg_visita_clean")) else gv

            if es_local:
                gf_ap.append(0.60 * gl + 0.40 * xg_l)
                ga_ap.append(0.60 * gv + 0.40 * xg_v)
            else:
                gf_ap.append(0.60 * gv + 0.40 * xg_v)
                ga_ap.append(0.60 * gl + 0.40 * xg_l)
        media_gf_ap, media_ga_ap = np.mean(gf_ap), np.mean(ga_ap)
    else:
        media_gf_ap, media_ga_ap = PRIOR_GF, PRIOR_GA

    # Clausura (Racha 5 partidos)
    if antes_de is not None and pd.notna(antes_de) and "fecha" in cl.columns:
        cl = cl[cl["fecha"].isna() | (cl["fecha"] < antes_de)]

    cl_eq = cl[(cl["local_std"] == eq) | (cl["visita_std"] == eq)].copy()
    if not cl_eq.empty:
        if "fecha" in cl_eq.columns and cl_eq["fecha"].notna().any():
            cl_eq = cl_eq.sort_values("fecha")
        cl_eq = cl_eq.tail(ventana_clausura)

        gf_cl, ga_cl = [], []
        for _, r in cl_eq.iterrows():
            if r["local_std"] == eq:
                gf_cl.append(float(r["gl"]))
                ga_cl.append(float(r["gv"]))
            else:
                gf_cl.append(float(r["gv"]))
                ga_cl.append(float(r["gl"]))
        media_gf_cl, media_ga_cl = np.mean(gf_cl), np.mean(ga_cl)
    else:
        media_gf_cl, media_ga_cl = media_gf_ap, media_ga_ap

    # Ponderación 35% Apertura / 65% Clausura
    gf_final = 0.35 * media_gf_ap + 0.65 * media_gf_cl
    ga_final = 0.35 * media_ga_ap + 0.65 * media_ga_cl

    return float(gf_final), float(ga_final)


def matriz_dixon_coles(lambda_loc, mu_vis, rho=-0.08, max_goles=5):
    m = np.zeros((max_goles + 1, max_goles + 1))
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            p = poisson.pmf(i, lambda_loc) * poisson.pmf(j, mu_vis)
            if i == 0 and j == 0:
                tau = 1.0 - lambda_loc * mu_vis * rho
            elif i == 1 and j == 0:
                tau = 1.0 + lambda_loc * rho
            elif i == 0 and j == 1:
                tau = 1.0 + mu_vis * rho
            elif i == 1 and j == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0
            m[i, j] = max(0.0, p * tau)
    suma = np.sum(m)
    return m / suma if suma > 0 else m


def obtener_marcador_modal(m, p_loc, p_emp, p_vis):
    exp_gl = float(np.sum(np.arange(m.shape[0])[:, None] * m))
    exp_gv = float(np.sum(np.arange(m.shape[1])[None, :] * m))

    gl, gv = int(np.round(exp_gl)), int(np.round(exp_gv))

    if p_loc > p_vis and p_loc > p_emp:
        if gl <= gv:
            gl = gv + 1
    elif p_vis > p_loc and p_vis > p_emp:
        if gv <= gl:
            gv = gl + 1

    return f"{gl} - {gv}"


# ==============================================================================
# 5. EJECUCIÓN DEL PANEL STREAMLIT
# ==============================================================================
st.sidebar.header("📂 Carga de Datos")
archivo_excel = st.sidebar.file_uploader("Subir Excel Liga 1", type=["xlsx"])

if archivo_excel:
    try:
        datos = cargar_excel(archivo_excel)
        st.success("✅ Excel cargado y procesado con éxito.")

        partidos_df = datos["partidos"]
        if "jornada" in partidos_df.columns:
            jornadas = sorted(partidos_df["jornada"].dropna().unique())
            jornada_sel = st.sidebar.selectbox("Seleccionar Jornada", jornadas)
            partidos_fecha = partidos_df[partidos_df["jornada"] == jornada_sel].copy()
        else:
            partidos_fecha = partidos_df.copy()

        resultados = []
        for _, row in partidos_fecha.iterrows():
            loc, vis = str(row["local"]), str(row["visita"])
            fecha_p = row.get("fecha", None)

            gf_loc, ga_loc = calcular_tasas_equipo(datos, loc, antes_de=fecha_p)
            gf_vis, ga_vis = calcular_tasas_equipo(datos, vis, antes_de=fecha_p)

            # Factores contextuales
            f_loc, f_vis, rho_dc = obtener_factores_contextual(datos, loc, vis)

            # Cálculo de intensidades
            l_loc = max(0.4, ((gf_loc + ga_vis) / 2.0) * f_loc)
            m_vis = max(0.3, ((gf_vis + ga_loc) / 2.0) * f_vis)

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
                    "Local": loc,
                    "Visita": vis,
                    "% Local": round(p_loc * 100, 1),
                    "% Empate": round(p_emp * 100, 1),
                    "% Visita": round(p_vis * 100, 1),
                    "% +2.5": round(p_over25 * 100, 1),
                    "% BTTS Sí": round(p_btts * 100, 1),
                    "Marcador_Modal": marcador,
                    "Recomendacion": rec,
                }
            )

# Reemplazo de las líneas 391 y 392:
    df_pronosticos = pd.DataFrame(resultados)
    
    # 1. Agregar columna 🔥 a partidos calientes (>= 50%)
    max_prob = df_pronosticos[['% Local', '% Empate', '% Visita']].max(axis=1)
    df_pronosticos.insert(0, '🔥', ['🔥' if p >= 50.0 else '➖' for p in max_prob])

    # 2. Función para pintar de verde pastel la celda ganadora
    def resaltar_max_probabilidad(row):
        styles = [''] * len(row)
        p_local, p_empate, p_visita = row['% Local'], row['% Empate'], row['% Visita']
        max_val = max(p_local, p_empate, p_visita)
        estilo_verde = 'background-color: #d4edda; color: #155724; font-weight: bold;'
        
        if p_local == max_val:
            styles[row.index.get_loc('% Local')] = estilo_verde
        elif p_empate == max_val:
            styles[row.index.get_loc('% Empate')] = estilo_verde
        elif p_visita == max_val:
            styles[row.index.get_loc('% Visita')] = estilo_verde
        return styles

    # 3. Aplicar estilos estilo Quipus Data
    estilo_tabla = df_pronosticos.style\
        .apply(resaltar_max_probabilidad, axis=1)\
        .set_properties(**{
            'border-color': '#e0e2dc',
            'font-family': 'sans-serif',
            'text-align': 'center'
        })\
        .set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#9ca592'),
                ('color', '#ffffff'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '8px')
            ]},
            {'selector': 'tbody tr:hover', 'props': [
                ('background-color', '#f4f5f2 !important')
            ]}
        ])

    st.subheader("📋 Resumen de pronósticos")
    st.dataframe(estilo_tabla, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"Error al procesar el modelo: {str(e)}")
else:
    st.info("👈 Por favor, sube tu archivo Excel en la barra lateral para generar el panel.")
