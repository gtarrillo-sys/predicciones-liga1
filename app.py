import math
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# 1. CONFIGURACIÓN Y CARGA DE HOJAS DEL EXCEL
# =========================================================
st.set_page_config(
    page_title="Predicciones Liga 1 - Modelo Híbrido Integral",
    page_icon="⚽",
    layout="wide",
)

PLAZAS_CALOR = [
    "Piura",
    "Sullana",
    "Tarapoto",
    "Chiclayo",
    "Iquitos",
    "Chongoyape",
]
PLAZAS_SINTETICAS = [
    "Juliaca",
    "Andahuaylas",
    "Trujillo",
    "Nueva Cajamarca",
    "Cajamarca",
    "Cutervo",
]


@st.cache_data
def cargar_datos():
    try:
        xls = pd.ExcelFile("liga1_data.xlsx")

        # Cargar hojas principales
        df_partidos = (
            pd.read_excel(
                xls,
                "Partidos_Fecha"
                if "Partidos_Fecha" in xls.sheet_names
                else xls.sheet_names[0],
            )
            if xls
            else pd.DataFrame()
        )

        df_acumulada = (
            pd.read_excel(xls, "Tabla_Posiciones_Acumulado")
            if "Tabla_Posiciones_Acumulado" in xls.sheet_names
            else (
                pd.read_excel(xls, "Tabla_Acumulada")
                if "Tabla_Acumulada" in xls.sheet_names
                else pd.DataFrame()
            )
        )

        df_clausura = (
            pd.read_excel(xls, "Tabla_Posiciones_Clausura")
            if "Tabla_Posiciones_Clausura" in xls.sheet_names
            else pd.DataFrame()
        )

        df_resultados = (
            pd.read_excel(xls, "Resultados_Clausura")
            if "Resultados_Clausura" in xls.sheet_names
            else df_partidos
        )

        # Limpieza de nombres de columnas
        for df in [df_partidos, df_acumulada, df_clausura, df_resultados]:
            if not df.empty:
                df.columns = df.columns.astype(str).str.strip()

        return df_partidos, df_acumulada, df_clausura, df_resultados
    except Exception:
        return None, None, None, None


df_partidos, df_acumulada, df_clausura, df_resultados = cargar_datos()


# =========================================================
# 2. FUNCIONES DE EXTRACCIÓN Y CÁLCULO DE RACHA
# =========================================================
def obtener_posicion(df_tabla, equipo_nombre):
    """Obtiene la posición según la columna 'Rango' o la coincidencia de nombre"""
    if df_tabla.empty:
        return 10

    busqueda = (
        "utc" if "utc" in equipo_nombre.lower() else equipo_nombre.lower()
    )

    col_club = next(
        (
            c
            for c in df_tabla.columns
            if any(k in str(c).lower() for k in ["club", "equipo"])
        ),
        df_tabla.columns[1] if len(df_tabla.columns) > 1 else df_tabla.columns[0],
    )

    coincidencias = df_tabla[
        df_tabla[col_club]
        .astype(str)
        .str.lower()
        .str.contains(busqueda, na=False)
    ]

    if not coincidencias.empty:
        # Priorizar columna 'Rango' o 'Pos'
        for c in ["Rango", "Pos", "Puesto", "N°"]:
            if c in df_tabla.columns:
                try:
                    return int(coincidencias.iloc[0][c])
                except Exception:
                    pass
        return coincidencias.index[0] + 1

    return 10


def calcular_racha_desde_resultados(df_res, equipo_nombre, jornada_limite=8):
    """Calcula los puntos de los últimos 5 partidos jugados desde la hoja Resultados_Clausura"""
    if df_res.empty:
        return 7

    busqueda = (
        "utc" if "utc" in equipo_nombre.lower() else equipo_nombre.lower()
    )

    # Identificar columnas de Local, Visita y Goles/Resultados
    col_loc = next(
        (c for c in df_res.columns if "local" in str(c).lower()), None
    )
    col_vis = next(
        (c for c in df_res.columns if "visita" in str(c).lower()), None
    )
    col_jor = next(
        (
            c
            for c in df_res.columns
            if any(k in str(c).lower() for k in ["jornada", "fecha"])
        ),
        None,
    )

    if not col_loc or not col_vis:
        return 7

    # Filtrar partidos jugados por el equipo antes o durante la jornada límite
    df_equipo = df_res[
        (df_res[col_loc].astype(str).str.lower().str.contains(busqueda, na=False))
        | (
            df_res[col_vis]
            .astype(str)
            .str.lower()
            .str.contains(busqueda, na=False)
        )
    ].copy()

    if col_jor and col_jor in df_equipo.columns:
        # Convertir jornada a número si es posible
        df_equipo["Jornada_Num"] = (
            df_equipo[col_jor]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
            .fillna(0)
        )
        df_equipo = df_equipo[df_equipo["Jornada_Num"] < jornada_limite]
        df_equipo = df_equipo.sort_values(by="Jornada_Num", ascending=False)

    # Tomar los últimos 5 partidos jugados
    ultimos_5 = df_equipo.head(5)
    puntos = 0

    col_g_loc = next(
        (
            c
            for c in df_res.columns
            if "goles_local" in str(c).lower() or "goles local" in str(c).lower()
        ),
        None,
    )
    col_g_vis = next(
        (
            c
            for c in df_res.columns
            if "goles_visita" in str(c).lower() or "goles visita" in str(c).lower()
        ),
        None,
    )

    for _, row in ultimos_5.iterrows():
        es_local = (
            busqueda in str(row[col_loc]).lower()
        )

        # Si están los goles explícitos
        if col_g_loc and col_g_vis and pd.notnull(row[col_g_loc]):
            try:
                gl = int(row[col_g_loc])
                gv = int(row[col_g_vis])

                if gl == gv:
                    puntos += 1
                elif es_local and gl > gv:
                    puntos += 3
                elif not es_local and gv > gl:
                    puntos += 3
                continue
            except Exception:
                pass

        # Si hay una columna de Resultado (G, E, P / V, E, D)
        col_res = next(
            (
                c
                for c in df_res.columns
                if "resultado" in str(c).lower() or "res" in str(c).lower()
            ),
            None,
        )
        if col_res:
            val = str(row[col_res]).strip().lower()
            if "gan" in val or "v" in val:
                puntos += 3
            elif "emp" in val or "e" in val:
                puntos += 1

    return puntos if len(ultimos_5) > 0 else 7


# =========================================================
# 3. MOTORES MATEMÁTICOS
# =========================================================
def calcular_poisson(lambda_l, lambda_v, max_g=5):
    mat = np.zeros((max_g + 1, max_g + 1))
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p_i = ((lambda_l**i) * math.exp(-lambda_l)) / math.factorial(i)
            p_j = ((lambda_v**j) * math.exp(-lambda_v)) / math.factorial(j)
            mat[i][j] = p_i * p_j

    p_loc = np.sum(np.tril(mat, -1))
    p_emp = np.sum(np.diag(mat))
    p_vis = np.sum(np.triu(mat, 1))
    p_bts = sum(
        mat[i][j] for i in range(1, max_g + 1) for j in range(1, max_g + 1)
    )

    return p_loc, p_emp, p_vis, p_bts


def funcion_sigmoide(z):
    return 1.0 / (1.0 + math.exp(-z))


def modelo_regresion_logistica(features):
    (
        lambda_l,
        lambda_v,
        clima_3pm,
        goleada,
        desgaste,
        sintetica,
        urg_l,
        urg_v,
        racha_l,
        racha_v,
        pos_ac_l,
        pos_ac_v,
        pos_cl_l,
        pos_cl_v,
    ) = features

    factor_racha = (racha_l - racha_v) / 15.0

    # Posición Ponderada: 60% Acumulado + 40% Clausura
    pos_efectiva_l = (0.60 * pos_ac_l) + (0.40 * pos_cl_l)
    pos_efectiva_v = (0.60 * pos_ac_v) + (0.40 * pos_cl_v)

    f_pos_l = (18 - pos_efectiva_l + 1) / 18.0
    f_pos_v = (18 - pos_efectiva_v + 1) / 18.0
    dif_posicion = f_pos_l - f_pos_v

    z_1x2 = (
        0.15
        + (0.30 * (lambda_l - lambda_v))
        - (0.25 * clima_3pm)
        - (0.25 * goleada)
        + (0.40 * desgaste)
        + (0.35 * sintetica)
        + (0.15 * (urg_l - urg_v))
        + (0.50 * factor_racha)
        + (0.45 * dif_posicion)
    )

    z_bts = (
        -0.10
        + (0.25 * (lambda_l + lambda_v))
        - (0.50 * clima_3pm)
        - (0.35 * goleada)
        - (0.20 * desgaste)
    )

    return funcion_sigmoide(z_1x2), funcion_sigmoide(z_bts)


# =========================================================
# 4. INTERFAZ streamlit
# =========================================================
st.title("⚽ Predicciones Liga 1 - Análisis Multicapa Integral")

if df_partidos is not None and not df_partidos.empty:
    with st.sidebar:
        st.header("🗓️ Selección de Partido")
        jornadas = df_partidos["Jornada"].dropna().unique()
        jornada_sel = st.selectbox("Seleccionar Jornada", jornadas)

        # Extraer número numérico de jornada
        try:
            num_jornada = int(
                "".join(filter(str.isdigit, str(jornada_sel)))
            )
        except Exception:
            num_jornada = 8

        df_jornada = df_partidos[
            df_partidos["Jornada"] == jornada_sel
        ].copy()
        df_jornada["Duelo"] = (
            df_jornada["Local"].astype(str)
            + " vs "
            + df_jornada["Visita"].astype(str)
        )
        partido_sel = st.selectbox(
            "Seleccionar Duelo", df_jornada["Duelo"].unique()
        )

        fila = df_jornada[df_jornada["Duelo"] == partido_sel].iloc[0]
        local = str(fila.get("Local", "Local"))
        visita = str(fila.get("Visita", "Visita"))
        hora = str(fila.get("Hora", "15:15"))
        plaza = str(fila.get("Ciudad", "Sullana"))
        estadio = str(
            fila.get("Estadio", "Estadio Campeones del 36")
        )

        # Lectura de Posiciones
        pos_ac_l_auto = obtener_posicion(df_acumulada, local)
        pos_ac_v_auto = obtener_posicion(df_acumulada, visita)

        pos_cl_l_auto = obtener_posicion(df_clausura, local)
        pos_cl_v_auto = obtener_posicion(df_clausura, visita)

        # Cálculo de Racha desde Resultados_Clausura
        pts_racha_l_auto = calcular_racha_desde_resultados(
            df_resultados, local, num_jornada
        )
        pts_racha_v_auto = calcular_racha_desde_resultados(
            df_resultados, visita, num_jornada
        )

        st.markdown("---")
        st.header("🏆 Posiciones en Tablas")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.caption("Tabla Acumulada")
            pos_ac_l = st.number_input(
                f"Acum. {local}", 1, 18, int(pos_ac_l_auto)
            )
            pos_ac_v = st.number_input(
                f"Acum. {visita}", 1, 18, int(pos_ac_v_auto)
            )
        with col_t2:
            st.caption("Tabla Clausura")
            pos_cl_l = st.number_input(
                f"Claus. {local}", 1, 18, int(pos_cl_l_auto)
            )
            pos_cl_v = st.number_input(
                f"Claus. {visita}", 1, 18, int(pos_cl_v_auto)
            )

        st.markdown("---")
        st.header("🔥 Racha de los Últimos 5 Partidos")
        pts_racha_l = st.slider(
            f"Pts Últimos 5 - {local}", 0, 15, int(pts_racha_l_auto)
        )
        pts_racha_v = st.slider(
            f"Pts Últimos 5 - {visita}", 0, 15, int(pts_racha_v_auto)
        )

        st.markdown("---")
        st.header("⚙️ Promedios de Goles (λ)")
        lambda_l = st.number_input(
            f"Prom. Goles Local ({local})", 0.5, 4.0, 2.33, 0.05
        )
        lambda_v = st.number_input(
            f"Prom. Goles Visita ({visita})", 0.1, 4.0, 0.67, 0.05
        )

        st.markdown("---")
        st.header("⚙️ Entorno")
        es_sintetica_auto = any(
            p.lower() in plaza.lower() for p in PLAZAS_SINTETICAS
        )
        tipo_cancha = st.selectbox(
            "Gramado",
            ["Natural", "Sintético"],
            index=1 if es_sintetica_auto else 0,
        )

        urgencia_l = st.slider("Urgencia Local", 1, 5, 4)
        urgencia_v = st.slider("Urgencia Visita", 1, 5, 3)

        vino_de_golear = st.checkbox(
            "Local viene de golear", value=True if "Alianza" in local else False
        )
        desgaste_logistico = st.checkbox(
            "Alerta de viaje en Visita",
            value=True if "UTC" in visita else False,
        )

    # Detección Climática y de Terreno
    try:
        h_dec = int(str(hora).split(":")[0]) + int(str(hora).split(":")[1]) / 60.0
    except Exception:
        h_dec = 15.25

    es_calor_extremo = (
        1
        if (
            any(p.lower() in plaza.lower() for p in PLAZAS_CALOR)
            and 13.0 <= h_dec <= 15.5
        )
        else 0
    )
    es_sintetica = 1 if tipo_cancha == "Sintético" else 0
    es_goleada = 1 if vino_de_golear else 0
    es_desgaste = 1 if desgaste_logistico else 0

    p_loc_poi, p_emp_poi, p_vis_poi, p_bts_poi = calcular_poisson(
        lambda_l, lambda_v
    )
    features = [
        lambda_l,
        lambda_v,
        es_calor_extremo,
        es_goleada,
        es_desgaste,
        es_sintetica,
        urgencia_l,
        urgencia_v,
        pts_racha_l,
        pts_racha_v,
        pos_ac_l,
        pos_ac_v,
        pos_cl_l,
        pos_cl_v,
    ]
    p_1x2_log, p_bts_log = modelo_regresion_logistica(features)

    prob_1x_final = ((p_loc_poi + p_emp_poi) + p_1x2_log) / 2.0
    prob_bts_final = (p_bts_poi + p_bts_log) / 2.0

    # Despliegue en Pantalla
    st.subheader(f"📊 INFORME DETALLADO: {local} vs {visita}")
    st.markdown(
        f"📍 **Plaza:** {plaza} | ⏰ **Hora:** {hora} hrs | 🏟️ **Estadio:** {estadio} ({tipo_cancha})"
    )

    if es_calor_extremo:
        st.error(
            "⚠️ **FILTRO CLIMÁTICO ACTIVO:** Calor Extremo detectado en la plaza. El ritmo tiende a desacelerar en el 2do tiempo."
        )

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("### 🏆 Capa 1: Posición y Momentum")
        st.write(
            f"* **Posición Acumulada (60%):** {local} ({pos_ac_l}°) vs {visita} ({pos_ac_v}°)"
        )
        st.write(
            f"* **Posición Clausura (40%):** {local} ({pos_cl_l}°) vs {visita} ({pos_cl_v}°)"
        )
        st.write(
            f"* **Racha Reciente (Calculada):** {local} ({pts_racha_l} pts) vs {visita} ({pts_racha_v} pts)"
        )
        st.write(
            f"* **Nivel de Urgencia:** Local {urgencia_l}/5 | Visita {urgencia_v}/5"
        )

        st.markdown("### 📈 Capa 2: Cancha y Logística")
        st.write(
            f"* **Racha Local:** {'Viene de golear / Racha positiva' if es_goleada else 'Forma estándar'}"
        )
        st.write(
            f"* **Estado Visita:** {'Desgaste de viaje' if es_desgaste else 'Traslado normal'}"
        )

    with col_c2:
        st.markdown("### 🧮 Capa 3: Métricas Híbridas Integradas")
        st.write(f"* **Poisson (1X):** {((p_loc_poi + p_emp_poi)*100):.1f}%")
        st.write(
            f"* **Regresión Logística (1X Ponderado):** {(p_1x2_log*100):.1f}%"
        )
        st.write(f"* **Poisson (Ambos Anotan):** {(p_bts_poi*100):.1f}%")
        st.write(
            f"* **Regresión Logística (Ambos Anotan):** {(p_bts_log*100):.1f}%"
        )

    st.markdown("---")
    st.markdown("### 🧠 Capa 4: Sugerencia Final del Sistema")

    if prob_1x_final >= 0.60:
        if prob_1x_final >= 0.72:
            rec_resultado = f"**Gana {local} Seco** (Dominio de posición ponderada, racha reciente y localía)."
        else:
            rec_resultado = f"**Doble Oportunidad: {local} o Empate (1X)** (Excelente respaldo estadístico)."

        if es_calor_extremo:
            rec_goles = "**Menos de 2.5 Goles / Ambos Anotan: NO** (Ajuste por desaceleración climática)."
        else:
            rec_goles = (
                "**Más de 1.5 Goles**"
                if prob_bts_final > 0.50
                else "**Ambos Anotan: NO**"
            )
    else:
        rec_resultado = (
            f"**Doble Oportunidad: {visita} o Empate (X2)** / Pronóstico reservado."
        )
        rec_goles = "**Menos de 2.5 Goles**"

    st.success(f"🎯 **Sugerencia de Resultado:** {rec_resultado}")
    st.info(f"⚽ **Sugerencia de Goles:** {rec_goles}")

else:
    st.error("No se pudo cargar la información desde `liga1_data.xlsx`.")
