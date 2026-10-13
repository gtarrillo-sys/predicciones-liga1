import base64
import os
import numpy as np
import pandas as pd
from scipy.stats import poisson
import streamlit as st

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS UNIFICADOS
# ==============================================================================
st.set_page_config(
    page_title="Modelo de Predicción Liga 1 Perú - Quipus Data",
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

LOGO_PATH = "logo_quipus.png"
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div style="text-align: left; margin-bottom: 10px;">
            <img src="data:image/png;base64,{logo_b64}" style="max-width: 280px; width: 100%; height: auto;">
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("### 📊 Quipus Data")

st.title("⚽ Modelo de Predicción Liga 1 Perú")

# ==============================================================================
# 2. CARGA Y LECTURA COMPLETA DE TODAS LAS HOJAS DEL EXCEL
# ==============================================================================
def normalizar(nombre):
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
        "fc cajamarca": "fc cajamarca",
        "comerciantes unidos": "comerciantes unidos",
        "cusco": "cusco fc",
        "cusco fc": "cusco fc",
        "deportivo garcilaso": "deportivo garcilaso",
        "garcilaso": "deportivo garcilaso",
        "juan pablo ii": "juan pablo ii college",
        "juan pablo ii college": "juan pablo ii college",
        "melgar": "fbc melgar",
        "fbc melgar": "fbc melgar",
        "sport boys": "sport boys",
        "sport huancayo": "sport huancayo",
        "sporting cristal": "sporting cristal",
        "universitario": "universitario",
        "utc": "utc cajamarca",
        "utc cajamarca": "utc cajamarca",
        "chankas": "chankas cyc",
        "chankas cyc": "chankas cyc",
        "los chankas": "chankas cyc",
        "cienciano": "cienciano",
        "adt": "adt",
        "cd moquegua": "cd moquegua",
        "moquegua": "cd moquegua",
    }
    return mapeo.get(n, n)

def estandarizar(df):
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        for c in df.columns
    ]
    renombres = {
        "equipo_local": "local", "equipo_visitante": "visita", "visitante": "visita",
        "goles_local": "gl", "goles_visitante": "gv", "goles_visita": "gv", "club": "equipo"
    }
    df = df.rename(columns=renombres)
    if "local" in df.columns:
        df["local_std"] = df["local"].apply(normalizar)
    if "visita" in df.columns:
        df["visita_std"] = df["visita"].apply(normalizar)
    if "equipo" in df.columns:
        df["equipo_std"] = df["equipo"].apply(normalizar)
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df

@st.cache_data
def cargar_excel_definitivo(filepath):
    xls = pd.ExcelFile(filepath)
    hojas = xls.sheet_names

    partidos = estandarizar(pd.read_excel(xls, "Partidos_Fecha")) if "Partidos_Fecha" in hojas else pd.DataFrame()
    apertura = estandarizar(pd.read_excel(xls, "Resultados_Apertura")) if "Resultados_Apertura" in hojas else pd.DataFrame()
    clausura = estandarizar(pd.read_excel(xls, "Resultados_Clausura")) if "Resultados_Clausura" in hojas else pd.DataFrame()
    acumulada = estandarizar(pd.read_excel(xls, "Tabla_Acumulada")) if "Tabla_Acumulada" in hojas else pd.DataFrame()
    tabla_clausura = estandarizar(pd.read_excel(xls, "Tabla_Clausura")) if "Tabla_Clausura" in hojas else pd.DataFrame()
    h2h = estandarizar(pd.read_excel(xls, "Historial_H2H")) if "Historial_H2H" in hojas else pd.DataFrame()
    geo = estandarizar(pd.read_excel(xls, "Data_Geografica")) if "Data_Geografica" in hojas else pd.DataFrame()
    formulas = pd.read_excel(xls, "Formulas") if "Formulas" in hojas else pd.DataFrame()

    for df_xg in [apertura, clausura]:
        if "xg_local" in df_xg.columns and "xg_visita" in df_xg.columns:
            df_xg["xg_local_clean"] = df_xg["xg_local"].astype(str).str.replace(",", ".").astype(float)
            df_xg["xg_visita_clean"] = df_xg["xg_visita"].astype(str).str.replace(",", ".").astype(float)

    prior_gf, prior_ga = 1.6824, 1.0815
    if not formulas.empty:
        try:
            prior_gf = float(formulas.iloc[0, 3])
            prior_ga = float(formulas.iloc[4, 3])
        except Exception:
            pass

    return {
        "partidos": partidos, "apertura": apertura, "clausura": clausura,
        "acumulada": acumulada, "tabla_clausura": tabla_clausura,
        "geo": geo, "h2h": h2h, "prior_gf": prior_gf, "prior_ga": prior_ga
    }

# ==============================================================================
# 3. MOTOR DE PRESIÓN SITUACIONAL Y DIXON-COLES
# ==============================================================================
def obtener_factores_contextuales(datos, local, visita, row_partido):
    geo = datos.get("geo", pd.DataFrame())
    h2h = datos.get("h2h", pd.DataFrame())
    acumulada = datos.get("acumulada", pd.DataFrame())

    eq_loc = normalizar(local)
    eq_vis = normalizar(visita)

    mult_loc, mult_vis, rho = 1.15, 0.88, -0.08

    # A) Datos Geográficos (Data_Geografica)
    if not geo.empty and "equipo_std" in geo.columns:
        info_loc = geo[geo["equipo_std"] == eq_loc]
        info_vis = geo[geo["equipo_std"] == eq_vis]

        if not info_loc.empty and not info_vis.empty:
            alt_loc = float(info_loc.iloc[0].get("altitud_msnm", 0))
            alt_vis = float(info_vis.iloc[0].get("altitud_msnm", 0))
            clima_loc = str(info_loc.iloc[0].get("tipo_clima", "")).lower()
            clima_vis = str(info_vis.iloc[0].get("tipo_clima", "")).lower()
            cancha_loc = str(info_loc.iloc[0].get("tipo_cancha", "natural")).lower()
            cancha_vis = str(info_vis.iloc[0].get("tipo_cancha", "natural")).lower()

            delta_alt = float(row_partido.get("delta_altitud_msnm", alt_loc - alt_vis))
            if delta_alt > 2000:
                mult_loc *= 1.20
                mult_vis *= 0.75
            elif delta_alt > 1200:
                mult_loc *= 1.10
                mult_vis *= 0.85

            if "calor" in clima_loc and "calor" not in clima_vis:
                mult_loc *= 1.08
                mult_vis *= 0.88

            if "sintetic" in cancha_loc and "sintetic" not in cancha_vis:
                mult_vis *= 0.90

    # B) Días de Descanso y Horarios (Partidos_Fecha)
    desc_loc = float(row_partido.get("dias_descanso_local", 7))
    desc_vis = float(row_partido.get("dias_descanso_visita", 7))
    if desc_loc < 4:
        mult_loc *= 0.93
    if desc_vis < 4:
        mult_vis *= 0.92

    hora_val = row_partido.get("hora", None)
    if pd.notna(hora_val):
        try:
            h_int = int(str(hora_val).split(":")[0]) if ":" in str(hora_val) else getattr(hora_val, "hour", 15)
            if h_int in [11, 12, 13] and mult_loc > 1.1:
                mult_loc *= 1.06
                mult_vis *= 0.92
        except Exception:
            pass

    # C) Motor de Presión Situacional (Tabla_Acumulada - Últimas 5 Fechas)
    if not acumulada.empty and "equipo_std" in acumulada.columns:
        row_loc_acc = acumulada[acumulada["equipo_std"] == eq_loc]
        row_vis_acc = acumulada[acumulada["equipo_std"] == eq_vis]

        pj_loc = float(row_loc_acc.iloc[0].get("pj", 28)) if not row_loc_acc.empty else 28
        if pj_loc >= 28:
            sit_loc = str(row_loc_acc.iloc[0].get("situacion_actual", "")).lower() if not row_loc_acc.empty else ""
            sit_vis = str(row_vis_acc.iloc[0].get("situacion_actual", "")).lower() if not row_vis_acc.empty else ""

            # Zona Alta / Torneos Internacionales
            if any(k in sit_loc for k in ["libertadores", "sudamericana", "titulo"]):
                mult_loc *= 1.10
            # Zona Baja / Peligro de Descenso
            elif any(k in sit_loc for k in ["peligro", "riesgo", "descenso"]):
                mult_loc *= 1.12
            # Zona Media
            elif "zona media" in sit_loc:
                mult_loc *= 0.92

            if any(k in sit_vis for k in ["libertadores", "sudamericana", "titulo"]):
                mult_vis *= 1.08
            elif any(k in sit_vis for k in ["peligro", "riesgo", "descenso"]):
                mult_vis *= 1.06
            elif "zona media" in sit_vis:
                mult_vis *= 0.90

    # D) Clásicos e Historial Directo (Historial_H2H)
    if not h2h.empty:
        grandes = ["alianza lima", "universitario", "sporting cristal"]
        if eq_loc in grandes and eq_vis in grandes:
            mult_loc *= 0.85
            mult_vis *= 0.95
            rho = -0.15

    # Retorno garantizado de un diccionario con las variables requeridas
    return {
        "f_loc": float(mult_loc),
        "f_vis": max(0.4, float(mult_vis)),
        "rho": float(rho)
    }

def calcular_tasas_equipo(datos, equipo, antes_de=None):
    eq = normalizar(equipo)
    ap = datos.get("apertura", pd.DataFrame()).copy()
    cl = datos.get("clausura", pd.DataFrame()).copy()
    prior_gf = datos.get("prior_gf", 1.6824)
    prior_ga = datos.get("prior_ga", 1.0815)

    ap_eq = ap[(ap["local_std"] == eq) | (ap["visita_std"] == eq)].copy() if not ap.empty else pd.DataFrame()
    if not ap_eq.empty:
        gf_ap, ga_ap = [], []
        for _, r in ap_eq.iterrows():
            es_loc = r["local_std"] == eq
            gl, gv = float(r["gl"]), float(r["gv"])
            xgl = float(r.get("xg_local_clean", gl)) if pd.notna(r.get("xg_local_clean")) else gl
            xgv = float(r.get("xg_visita_clean", gv)) if pd.notna(r.get("xg_visita_clean")) else gv

            if es_loc:
                gf_ap.append(0.60 * gl + 0.40 * xgl)
                ga_ap.append(0.60 * gv + 0.40 * xgv)
            else:
                gf_ap.append(0.60 * gv + 0.40 * xgv)
                ga_ap.append(0.60 * gl + 0.40 * xgl)
        m_gf_ap, m_ga_ap = np.mean(gf_ap), np.mean(ga_ap)
    else:
        m_gf_ap, m_ga_ap = prior_gf, prior_ga

    if antes_de is not None and pd.notna(antes_de) and "fecha" in cl.columns:
        cl = cl[cl["fecha"].isna() | (cl["fecha"] < antes_de)]

    cl_eq = cl[(cl["local_std"] == eq) | (cl["visita_std"] == eq)].copy() if not cl.empty else pd.DataFrame()
    if not cl_eq.empty:
        cl_eq = cl_eq.tail(5)
        gf_cl, ga_cl = [], []
        for _, r in cl_eq.iterrows():
            es_loc = r["local_std"] == eq
            gl, gv = float(r["gl"]), float(r["gv"])
            xgl = float(r.get("xg_local_clean", gl)) if pd.notna(r.get("xg_local_clean")) else gl
            xgv = float(r.get("xg_visita_clean", gv)) if pd.notna(r.get("xg_visita_clean")) else gv

            if es_loc:
                gf_cl.append(0.50 * gl + 0.50 * xgl)
                ga_cl.append(0.50 * gv + 0.50 * xgv)
            else:
                gf_cl.append(0.50 * gv + 0.50 * xgv)
                ga_cl.append(0.50 * gl + 0.50 * xgl)
        m_gf_cl, m_ga_cl = np.mean(gf_cl), np.mean(ga_cl)
    else:
        m_gf_cl, m_ga_cl = m_gf_ap, m_ga_ap

    gf_final = 0.35 * m_gf_ap + 0.65 * m_gf_cl
    ga_final = 0.35 * m_ga_ap + 0.65 * m_ga_cl
    return float(gf_final), float(ga_final)

def matriz_dixon_coles(lambda_loc, mu_vis, rho=-0.08, max_goles=6):
    m = np.zeros((max_goles, max_goles))
    for i in range(max_goles):
        for j in range(max_goles):
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
    s = np.sum(m)
    return m / s if s > 0 else m

def obtener_marcador_y_recomendacion(matriz, p_loc, p_emp, p_vis, loc, vis, p_over25, p_btts):
    if p_loc >= p_vis and p_loc >= p_emp:
        ganador, eq_ganador, max_p = "Local", loc, p_loc
    elif p_vis > p_loc and p_vis >= p_emp:
        ganador, eq_ganador, max_p = "Visita", vis, p_vis
    else:
        ganador, eq_ganador, max_p = "Empate", "Empate", p_emp

    # 1. Definición estricta de recomendación
    if ganador == "Empate" or max_p < 42.0:
        rec = "Empate / Partido Cerrado"
    else:
        rec = f"Gana {eq_ganador}"
        if p_over25 >= 52.0 and p_over25 >= p_btts:
            rec += " + Más de 2.5"
        elif p_btts >= 52.0:
            rec += " + Ambos marcan"

    # 2. Marcador modal 100% coherente con la recomendación
    if rec == "Empate / Partido Cerrado":
        best_i, best_j = (1, 1) if p_btts >= 48.0 else (0, 0)
    else:
        necesita_btts = "Ambos marcan" in rec
        necesita_over = "Más de 2.5" in rec
        best_i, best_j, max_peso = (2, 1) if ganador == "Local" else (1, 2), -1.0

        for i in range(6):
            for j in range(6):
                valido = (ganador == "Local" and i > j) or (ganador == "Visita" and j > i)
                if necesita_btts:
                    valido = valido and (i >= 1 and j >= 1)
                if necesita_over:
                    valido = valido and (i + j >= 3)

                if valido and matriz[i, j] > max_peso:
                    max_peso = matriz[i, j]
                    best_i, best_j = i, j

    return f"{best_i} - {best_j}", rec

# ==============================================================================
# 4. INTERFAZ Y DESPLIEGUE EN STREAMLIT
# ==============================================================================
EXCEL_PATH = "Liga1_2026.xlsx"
dias_espanol = {"Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles", "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"}

def aplicar_estilos_tabla(row):
    styles = [""] * len(row)
    estilo_verde = "color: #2e6930; font-weight: bold;"
    estilo_rojo = "color: #c0392b; font-weight: bold;"

    if row["Recomendacion"] == "Empate / Partido Cerrado":
        styles[row.index.get_loc("Marcador_Modal")] = estilo_rojo
        styles[row.index.get_loc("Recomendacion")] = estilo_rojo
    else:
        styles[row.index.get_loc("Recomendacion")] = estilo_verde

    for col in ["% Local", "% Empate", "% Visita", "Más de 2.5 goles", "Ambos marcan: Sí"]:
        if col in row.index and isinstance(row[col], (int, float)) and row[col] >= 55.0:
            styles[row.index.get_loc(col)] = estilo_verde

    return styles

if os.path.exists(EXCEL_PATH):
    try:
        datos = cargar_excel_definitivo(EXCEL_PATH)
        partidos_df = datos["partidos"]

        st.sidebar.header("⚽ Selección de Jornada")
        if "jornada" in partidos_df.columns:
            jornadas = sorted(partidos_df["jornada"].dropna().unique())
            jornada_sel = st.sidebar.selectbox("Seleccionar Jornada", jornadas)
            partidos_fecha = partidos_df[partidos_df["jornada"] == jornada_sel].copy()
        else:
            partidos_fecha = partidos_df.copy()

        st.sidebar.markdown("---")
        st.sidebar.header("📊 Opciones de Visualización")
        tipo_vista = st.sidebar.radio("Vista de la tabla", ["Vista Resumida", "Vista Completa"])

        resultados = []
        for _, row in partidos_fecha.iterrows():
            loc, vis = str(row["local"]), str(row["visita"])
            fecha_p = row.get("fecha", None)

            nombre_dia, hora_str = "Por definir", "--:--"
            if pd.notna(fecha_p):
                dt_val = pd.to_datetime(fecha_p)
                nombre_dia = dias_espanol.get(dt_val.strftime("%A"), "Por definir")

            hora_raw = row.get("hora", None)
            if pd.notna(hora_raw):
                hora_str = hora_raw.strftime("%H:%M") if hasattr(hora_raw, "strftime") else str(hora_raw)[:5]

            gf_loc, ga_loc = calcular_tasas_equipo(datos, loc, antes_de=fecha_p)
            gf_vis, ga_vis = calcular_tasas_equipo(datos, vis, antes_de=fecha_p)

            # Obtención totalmente segura a través de diccionario
            factores = obtener_factores_contextuales(datos, loc, vis, row)
            f_loc = factores["f_loc"]
            f_vis = factores["f_vis"]
            rho_dc = factores["rho"]

            l_loc = max(0.4, ((gf_loc + ga_vis) / 2.0) * f_loc)
            m_vis = max(0.3, ((gf_vis + ga_loc) / 2.0) * f_vis)

            matriz = matriz_dixon_coles(l_loc, m_vis, rho=rho_dc)

            p_loc = float(np.sum(np.tril(matriz, -1))) * 100
            p_emp = float(np.sum(np.diag(matriz))) * 100
            p_vis = float(np.sum(np.triu(matriz, 1))) * 100

            p_over25 = float(1.0 - sum(matriz[i, j] for i in range(6) for j in range(6) if i + j <= 2)) * 100
            p_btts = float(sum(matriz[i, j] for i in range(1, 6) for j in range(1, 6))) * 100

            marcador, rec = obtener_marcador_y_recomendacion(
                matriz, p_loc, p_emp, p_vis, loc, vis, p_over25, p_btts
            )

            resultados.append({
                "Día": nombre_dia,
                "Hora": hora_str,
                "Local": loc,
                "Visita": vis,
                "% Local": round(p_loc, 1),
                "% Empate": round(p_emp, 1),
                "% Visita": round(p_vis, 1),
                "Más de 2.5 goles": round(p_over25, 1),
                "Ambos marcan: Sí": round(p_btts, 1),
                "Marcador_Modal": marcador,
                "Recomendacion": rec,
            })

        df_pronosticos = pd.DataFrame(resultados)
        max_prob = df_pronosticos[["% Local", "% Empate", "% Visita"]].max(axis=1)
        df_pronosticos.insert(0, "🔥", ["🔥" if p >= 58.0 else "➖" for p in max_prob])

        tab1, tab2, tab3 = st.tabs(["📋 Pronósticos Jornada", "🏆 Tabla Acumulada", "🥇 Tabla Clausura"])

        with tab1:
            st.subheader("📋 Resumen de pronósticos")
            if tipo_vista == "Vista Resumida":
                st.markdown("*Vista ligera optimizada con los días de la semana y datos clave para móviles.*")
                df_resumido = df_pronosticos[["🔥", "Día", "Hora", "Local", "Visita", "Marcador_Modal", "Recomendacion"]]
                st.dataframe(df_resumido.style.apply(aplicar_estilos_tabla, axis=1), use_container_width=True, hide_index=True)
            else:
                st.markdown("*Vista extendida con porcentajes completos de probabilidad y goles.*")
                st.dataframe(
                    df_pronosticos.style.apply(aplicar_estilos_tabla, axis=1).format("{:.1f}", subset=["% Local", "% Empate", "% Visita", "Más de 2.5 goles", "Ambos marcan: Sí"]),
                    use_container_width=True,
                    hide_index=True,
                )

        with tab2:
            st.subheader("🏆 Tabla Acumulada Liga 1 (Incentivos de Presión)")
            st.dataframe(datos["acumulada"], use_container_width=True)

        with tab3:
            st.subheader("🥇 Tabla Posiciones Clausura")
            st.dataframe(datos["tabla_clausura"], use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error al procesar el modelo: {str(e)}")
else:
    st.error(f"❌ No se encontró el archivo '{EXCEL_PATH}'.")
