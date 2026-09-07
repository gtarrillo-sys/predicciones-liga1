import os
import re
import unicodedata
import numpy as np
import pandas as pd
from scipy.stats import poisson
import streamlit as st

# =========================================================
# CONFIGURACIÓN DE LA PÁGINA STREAMLIT Y ESTILOS
# =========================================================
st.set_page_config(
    page_title="Predicción Liga 1 Perú - Dixon Coles",
    page_icon="⚽",
    layout="wide",
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e9ecef;
    }
    .suggestion-box-blue {
        background-color: #e8f0fe;
        color: #1a73e8;
        padding: 12px 16px;
        border-radius: 8px;
        font-weight: 500;
        margin-top: 10px;
    }
    .suggestion-box-green {
        background-color: #e6f4ea;
        color: #137333;
        padding: 12px 16px;
        border-radius: 8px;
        font-weight: 500;
        margin-top: 10px;
    }
    .sub-caption {
        color: #5f6368;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# 1. NORMALIZACIÓN DE TEXTO Y DICCIONARIO DE EQUIPOS
# =========================================================
def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode("utf-8")
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)
    return " ".join(texto.lower().split())

DICCIONARIO_EQUIPOS = {
    "juan pablo ii college": "colegio juan pablo ii",
    "juan pablo ii": "colegio juan pablo ii",
    "colegio juan pablo ii": "colegio juan pablo ii",
    "los chankas": "los chankas",
    "chankas": "los chankas",
    "chankas cyc": "los chankas",
    "sporting cristal": "sporting cristal",
    "cristal": "sporting cristal",
    "alianza lima": "alianza lima",
    "universitario": "universitario",
    "universitario de deportes": "universitario",
    "sport boys": "sport boys",
    "alianza atletico": "alianza atletico",
    "atletico grau": "atletico grau",
    "comerciantes unidos": "comerciantes unidos",
    "fbc melgar": "melgar",
    "melgar": "melgar",
    "deportivo garcilaso": "garcilaso",
    "garcilaso": "garcilaso",
    "cusco fc": "cusco",
    "cusco": "cusco",
    "deporte huancayo": "sport huancayo",
    "sport huancayo": "sport huancayo",
    "fc cajamarca": "ut c",
    "utc cajamarca": "ut c",
    "utc": "ut c",
    "ut c": "ut c",
    "adt": "adt",
    "cienciano": "cienciano",
    "cd moquegua": "cd moquegua",
    "moquegua": "cd moquegua",
}

ALTITUDES_DEFAULT = {
    "adt": (3050, "Tarma"),
    "cienciano": (3360, "Cusco"),
    "cusco": (3360, "Cusco"),
    "garcilaso": (3360, "Cusco"),
    "sport huancayo": (3250, "Huancayo"),
    "los chankas": (2920, "Andahuaylas"),
    "ut c": (2750, "Cajamarca"),
    "comerciantes unidos": (2620, "Cutervo"),
    "melgar": (2335, "Arequipa"),
    "universitario": (150, "Lima"),
    "alianza lima": (150, "Lima"),
    "sporting cristal": (150, "Lima"),
    "sport boys": (10, "Callao"),
    "atletico grau": (50, "Piura"),
    "alianza atletico": (50, "Sullana"),
    "colegio juan pablo ii": (150, "Chongoyape"),
    "cd moquegua": (1410, "Moquegua"),
}

def estandarizar_nombre(nombre):
    clean_txt = normalizar_texto(nombre)
    clean_txt = re.sub(r"\b(club|fc|cd|atletico|deportivo|asociacion)\b", "", clean_txt)
    clean_txt = " ".join(clean_txt.split())
    return DICCIONARIO_EQUIPOS.get(clean_txt, clean_txt)

def resolver_columna_club(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    posibles = ["club", "equipo", "nombre", "team", "clubes", "equipos"]
    for pos in posibles:
        if pos in df.columns:
            df = df.rename(columns={pos: "club"})
            break
    if "club" not in df.columns:
        df["club"] = df.iloc[:, 0]
    return df

# =========================================================
# 2. CARGA Y GUARDA DE DATOS
# =========================================================
def obtener_ruta_excel(nombre_archivo="Liga1_2026.xlsx"):
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directorio_actual, nombre_archivo)

@st.cache_data(ttl=60)
def cargar_datos_excel(fuente_archivo):
    try:
        xls = pd.ExcelFile(fuente_archivo)
        pestanas = xls.sheet_names

        df_partidos = pd.read_excel(xls, sheet_name="Partidos_Fecha")
        df_partidos.columns = df_partidos.columns.str.strip().str.lower()
        
        # Extracción limpia de la columna 'jornada' sin confundirse con 'fecha'
        if "jornada" in df_partidos.columns:
            df_partidos["jornada_num"] = df_partidos["jornada"].astype(str).str.extract(r"(\d+)")[0].fillna("8")
        elif "fecha_num" in df_partidos.columns:
            df_partidos["jornada_num"] = df_partidos["fecha_num"].astype(str)
        else:
            df_partidos["jornada_num"] = "8"

        if "fecha" in df_partidos.columns:
            df_partidos["fecha_str"] = df_partidos["fecha"].astype(str)
        else:
            df_partidos["fecha_str"] = "2026-09-06"

        if "hora" not in df_partidos.columns:
            df_partidos["hora"] = "15:00"

        df_partidos["local_std"] = df_partidos["local"].apply(estandarizar_nombre)
        df_partidos["visita_std"] = df_partidos["visita"].apply(estandarizar_nombre)

        if "goles_local" not in df_partidos.columns:
            df_partidos["goles_local"] = np.nan
        if "goles_visita" not in df_partidos.columns:
            df_partidos["goles_visita"] = np.nan
        if "jugado" not in df_partidos.columns:
            df_partidos["jugado"] = False

        if "Data_Geografica" in pestanas:
            df_geo = pd.read_excel(xls, sheet_name="Data_Geografica")
            df_geo = resolver_columna_club(df_geo)
            df_geo["equipo_std"] = df_geo["club"].apply(estandarizar_nombre)
        else:
            df_geo = pd.DataFrame([
                {"equipo_std": k, "altitud": v[0], "ciudad": v[1]} for k, v in ALTITUDES_DEFAULT.items()
            ])

        df_acumulada = pd.read_excel(xls, sheet_name="Tabla_Acumulada")
        df_acumulada = resolver_columna_club(df_acumulada)
        df_acumulada["equipo_std"] = df_acumulada["club"].apply(estandarizar_nombre)

        if "Tabla_Clausura" in pestanas:
            df_clausura = pd.read_excel(xls, sheet_name="Tabla_Clausura")
            df_clausura = resolver_columna_club(df_clausura)
            df_clausura["equipo_std"] = df_clausura["club"].apply(estandarizar_nombre)
        else:
            df_clausura = df_acumulada.copy()

        return df_partidos, df_acumulada, df_clausura, df_geo, None
    except Exception as e:
        return None, None, None, None, str(e)

def guardar_cambios_excel(df_partidos, df_acumulada, df_clausura, df_geo, ruta):
    try:
        with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
            df_partidos.to_excel(writer, sheet_name="Partidos_Fecha", index=False)
            df_acumulada.to_excel(writer, sheet_name="Tabla_Acumulada", index=False)
            df_clausura.to_excel(writer, sheet_name="Tabla_Clausura", index=False)
            df_geo.to_excel(writer, sheet_name="Data_Geografica", index=False)
        return True, "Cambios guardados con éxito en el archivo Excel."
    except Exception as e:
        return False, str(e)

# =========================================================
# 3. CÁLCULOS MODELO DIXON-COLES
# =========================================================
def obtener_fuerza_equipos(equipo_local, equipo_visita, df_tabla):
    df_t = df_tabla.copy()
    row_loc = df_t.loc[df_t["equipo_std"] == equipo_local]
    row_vis = df_t.loc[df_t["equipo_std"] == equipo_visita]

    att_loc = float(row_loc["gf"].values[0]) / max(1, float(row_loc["pj"].values[0])) if not row_loc.empty and "gf" in row_loc.columns else 1.25
    def_loc = float(row_loc["gc"].values[0]) / max(1, float(row_loc["pj"].values[0])) if not row_loc.empty and "gc" in row_loc.columns else 1.15

    att_vis = float(row_vis["gf"].values[0]) / max(1, float(row_vis["pj"].values[0])) if not row_vis.empty and "gf" in row_vis.columns else 1.10
    def_vis = float(row_vis["gc"].values[0]) / max(1, float(row_vis["pj"].values[0])) if not row_vis.empty and "gc" in row_vis.columns else 1.30

    return att_loc, def_loc, att_vis, def_vis

def obtener_info_geo(equipo, df_geo):
    df_g = df_geo.copy()
    row = df_g.loc[df_g["equipo_std"] == equipo]
    if not row.empty:
        alt = float(row["altitud"].values[0]) if "altitud" in row.columns else ALTITUDES_DEFAULT.get(equipo, (150, "Lima"))[0]
        ciudad = str(row["ciudad"].values[0]) if "ciudad" in row.columns else ALTITUDES_DEFAULT.get(equipo, (150, "Lima"))[1]
        return alt, ciudad
    info = ALTITUDES_DEFAULT.get(equipo, (150, "Lima"))
    return float(info[0]), info[1]

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

def calcular_dixon_coles(equipo_local, equipo_visita, df_tabla, df_geo_info, rho=-0.11):
    att_loc, def_loc, att_vis, def_vis = obtener_fuerza_equipos(equipo_local, equipo_visita, df_tabla)
    alt_loc, ciudad_loc = obtener_info_geo(equipo_local, df_geo_info)
    alt_vis, _ = obtener_info_geo(equipo_visita, df_geo_info)

    dif_altitud = max(0.0, alt_loc - alt_vis)
    home_advantage = 1.15 if alt_loc < 1000 else 1.28
    factor_def_visita = 1.0 + (dif_altitud / 6500.0) if alt_loc >= 2000 else 1.0

    lambda_local = max(1.05, (att_loc * (def_vis / 1.20)) * home_advantage * factor_def_visita)
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

    prob_under25 = float(sum(matriz_prob[i, j] for i in range(max_goles) for j in range(max_goles) if i + j < 2.5))
    prob_over25 = 1.0 - prob_under25
    prob_btts_si = float(matriz_prob[1:, 1:].sum())
    prob_btts_no = 1.0 - prob_btts_si

    return prob_local, prob_empate, prob_visita, prob_over25, prob_under25, prob_btts_si, prob_btts_no, alt_loc, ciudad_loc

# =========================================================
# 4. INTERFAZ STREAMLIT
# =========================================================
st.title("⚽ Modelo de Predicción Liga 1 Perú")
st.caption("Ajustado por Dixon-Coles y Factor de Altitud Real")

st.sidebar.header("📁 Archivo de Datos")
archivo_subido = st.sidebar.file_uploader("Subir archivo Excel (Liga1_2026.xlsx)", type=["xlsx"])
ruta_local = obtener_ruta_excel("Liga1_2026.xlsx")

fuente_excel = archivo_subido if archivo_subido is not None else (ruta_local if os.path.exists(ruta_local) else None)

if st.sidebar.button("🔄 Recargar Datos"):
    st.cache_data.clear()
    st.rerun()

if fuente_excel is None:
    st.error("❌ No se encontró el archivo 'Liga1_2026.xlsx'.")
else:
    df_partidos, df_acum, df_claus, df_geo, error = cargar_datos_excel(fuente_excel)

    if error:
        st.error(f"Error al procesar la estructura del Excel: {error}")
    else:
        tabla_ref = st.sidebar.radio("Tabla de Rendimiento:", ("Tabla Acumulada", "Tabla Clausura"), index=0)
        df_tabla_act = df_acum if tabla_ref == "Tabla Acumulada" else df_claus

        # Listado dinámico de jornadas
        jornadas_raw = sorted(df_partidos["jornada_num"].unique(), key=lambda x: int(x) if str(x).isdigit() else 0)
        idx_defecto = jornadas_raw.index("8") if "8" in jornadas_raw else 0
        jornada_sel = st.sidebar.selectbox(
            "Seleccionar Jornada:",
            options=jornadas_raw,
            format_func=lambda x: f"Jornada {x}",
            index=idx_defecto
        )

        # Filtrar limpiando el índice
        df_f = df_partidos[df_partidos["jornada_num"].astype(str) == str(jornada_sel)].copy()
        df_f = df_f.reset_index(drop=True)

        tab_partidos, tab_actualizar = st.tabs([f"📊 Pronósticos Jornada {jornada_sel}", "📝 Actualizar Resultados y Marcadores"])

        with tab_partidos:
            st.subheader(f"Jornada {jornada_sel} - Partidos y Pronósticos ({tabla_ref})")

            for idx in range(len(df_f)):
                row = df_f.iloc[idx]
                eq_loc_std = row["local_std"]
                eq_vis_std = row["visita_std"]

                nombre_loc = str(row["local"]).strip()
                nombre_vis = str(row["visita"]).strip()
                fecha_partido = str(row.get("fecha_str", "2026-09-06")).split(" ")[0]
                hora_partido = str(row.get("hora", "15:00"))

                p_loc, p_emp, p_vis, p_over, p_under, p_btts_si, p_btts_no, altitud, ciudad = calcular_dixon_coles(
                    eq_loc_std, eq_vis_std, df_tabla_act, df_geo
                )

                if p_loc >= 0.40 and (p_loc - p_vis) >= 0.08:
                    sug_1x2 = f"Gana {nombre_loc}"
                    conf_1x2 = "Alta" if p_loc >= 0.52 else "Media-Alta"
                elif p_vis >= 0.40 and (p_vis - p_loc) >= 0.08:
                    sug_1x2 = f"Gana {nombre_vis}"
                    conf_1x2 = "Alta" if p_vis >= 0.52 else "Media-Alta"
                elif (p_loc + p_emp) >= 0.62:
                    sug_1x2 = f"Empate o Visita ({nombre_vis})" if p_vis > p_loc else f"Local o Empate ({nombre_loc})"
                    conf_1x2 = "Media-Alta"
                else:
                    sug_1x2 = "Doble Opción"
                    conf_1x2 = "Media"

                sug_goles = "Más de 2.5 Goles (+2.5)" if p_over > p_under else "Menos de 2.5 Goles (-2.5)"
                conf_goles = "Alta" if max(p_over, p_under) >= 0.60 else "Media"

                sug_btts = "Ambos Equipos SÍ Anotan (Sí)" if p_btts_si > p_btts_no else "Ambos Equipos NO Anotan (No)"
                conf_btts = "Media" if max(p_btts_si, p_btts_no) < 0.60 else "Alta"

                with st.container():
                    st.markdown(f"### 🏟️ {nombre_loc} vs {nombre_vis} | {ciudad} ({int(altitud)} msnm)")
                    st.caption(f"📅 Fecha: {fecha_partido} - 🕒 Hora: {hora_partido}")

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write(f"**Gana {nombre_loc}**")
                        st.title(f"{p_loc * 100:.1f}%")
                        st.caption(f"↑ Cuota Justa: {1 / max(p_loc, 0.001):.2f}")
                    with c2:
                        st.write("**Empate**")
                        st.title(f"{p_emp * 100:.1f}%")
                        st.caption(f"↑ Cuota Justa: {1 / max(p_emp, 0.001):.2f}")
                    with c3:
                        st.write(f"**Gana {nombre_vis}**")
                        st.title(f"{p_vis * 100:.1f}%")
                        st.caption(f"↑ Cuota Justa: {1 / max(p_vis, 0.001):.2f}")

                    s1, s2 = st.columns([2, 1])
                    with s1:
                        st.markdown(f"<div class='suggestion-box-blue'><b>Pronóstico Sugerido (1X2):</b> {sug_1x2}</div>", unsafe_allow_html=True)
                    with s2:
                        st.markdown(f"<div class='suggestion-box-green'><b>Nivel de Confianza:</b> {conf_1x2}</div>", unsafe_allow_html=True)

                    st.write("")
                    col_goles, col_btts = st.columns(2)

                    with col_goles:
                        st.markdown("#### ⚽ Mercado de Goles (Over / Under 2.5)")
                        st.write(f"**Más de 2.5 Goles:** {p_over * 100:.1f}%")
                        st.progress(float(p_over))
                        st.caption(f"Cuota Justa Over: {1 / max(p_over, 0.001):.2f}")

                        st.write(f"**Menos de 2.5 Goles:** {p_under * 100:.1f}%")
                        st.progress(float(p_under))
                        st.caption(f"Cuota Justa Under: {1 / max(p_under, 0.001):.2f}")

                        st.markdown(f"<div class='suggestion-box-blue'><b>Pronóstico Sugerido:</b> {sug_goles}</div>", unsafe_allow_html=True)
                        st.caption(f"🎯 Nivel de Confianza: {conf_goles}")

                    with col_btts:
                        st.markdown("#### 🔥 Ambos Equipos Anotan (BTTS)")
                        st.write(f"**Sí Anotan Ambos:** {p_btts_si * 100:.1f}%")
                        st.progress(float(p_btts_si))
                        st.caption(f"Cuota Justa Sí: {1 / max(p_btts_si, 0.001):.2f}")

                        st.write(f"**No Anotan Ambos:** {p_btts_no * 100:.1f}%")
                        st.progress(float(p_btts_no))
                        st.caption(f"Cuota Justa No: {1 / max(p_btts_no, 0.001):.2f}")

                        st.markdown(f"<div class='suggestion-box-blue'><b>Pronóstico Sugerido:</b> {sug_btts}</div>", unsafe_allow_html=True)
                        st.caption(f"🎯 Nivel de Confianza: {conf_btts}")

                    st.divider()

        with tab_actualizar:
            st.subheader(f"⚙️ Panel de Actualización de Marcadores - Jornada {jornada_sel}")
            st.info("Ingresa los marcadores reales para actualizar la base de datos.")

            for idx in range(len(df_f)):
                row = df_f.iloc[idx]
                c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 3, 2])
                with c1:
                    st.write(f"**{row['local']}**")
                with c2:
                    val_loc = int(row["goles_local"]) if pd.notnull(row["goles_local"]) else 0
                    g_loc = st.number_input(f"GL_{idx}", min_value=0, max_value=15, value=val_loc, key=f"gl_{idx}", label_visibility="collapsed")
                with c3:
                    val_vis = int(row["goles_visita"]) if pd.notnull(row["goles_visita"]) else 0
                    g_vis = st.number_input(f"GV_{idx}", min_value=0, max_value=15, value=val_vis, key=f"gv_{idx}", label_visibility="collapsed")
                with c4:
                    st.write(f"**{row['visita']}**")
                with c5:
                    jugado = st.checkbox("Jugado", value=bool(row["jugado"]), key=f"jug_{idx}")

                # Actualizar DataFrame global por coincidencia de fila
                match_mask = (df_partidos["jornada_num"].astype(str) == str(jornada_sel)) & \
                             (df_partidos["local"] == row["local"]) & \
                             (df_partidos["visita"] == row["visita"])
                df_partidos.loc[match_mask, "goles_local"] = g_loc
                df_partidos.loc[match_mask, "goles_visita"] = g_vis
                df_partidos.loc[match_mask, "jugado"] = jugado

            st.write("")
            if st.button("💾 Guardar Marcadores y Recalcular Excel"):
                ok, msg = guardar_cambios_excel(df_partidos, df_acum, df_claus, df_geo, ruta_local)
                if ok:
                    st.success(msg)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"Error al guardar los datos: {msg}")
