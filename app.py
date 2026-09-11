import os
import re
import io
import unicodedata
import numpy as np
import pandas as pd
from scipy.stats import poisson
import streamlit as st

# =========================================================
# LIGA 1 PERÚ - MODELO V3
# Dixon-Coles + Elo + Forma + Altitud + Rivalidad + H2H
# =========================================================

st.set_page_config(
    page_title="Predicción Liga 1 Perú - V3",
    page_icon="⚽",
    layout="wide",
)

st.markdown("""
<style>
.suggestion-box-blue{background:#e8f0fe;color:#1a73e8;padding:12px 16px;border-radius:8px;font-weight:500;margin-top:10px}
.suggestion-box-green{background:#e6f4ea;color:#137333;padding:12px 16px;border-radius:8px;font-weight:500;margin-top:10px}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 1. NORMALIZACIÓN
# =========================================================
def normalizar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = unicodedata.normalize("NFD", str(texto)).encode("ascii","ignore").decode("utf-8")
    texto = re.sub(r"[^a-zA-Z0-9\s]", " ", texto.lower())
    return " ".join(texto.split())

# IMPORTANTE:
# FC Cajamarca y UTC son equipos DISTINTOS.
# Atlético Grau y Alianza Atlético también son DISTINTOS.
DICCIONARIO_EQUIPOS = {
    "alianza lima":"alianza lima", "alianza":"alianza lima",
    "universitario":"universitario",
    "universitario de deportes":"universitario",
    "sporting cristal":"sporting cristal", "cristal":"sporting cristal",
    "sport boys":"sport boys",
    "atletico grau":"atletico grau", "grau":"atletico grau",
    "alianza atletico":"alianza atletico",
    "alianza atletico sullana":"alianza atletico",
    "cusco fc":"cusco", "cusco":"cusco",
    "cienciano":"cienciano",
    "deportivo garcilaso":"garcilaso", "garcilaso":"garcilaso",
    "fbc melgar":"melgar", "melgar":"melgar",
    "deporte huancayo":"sport huancayo", "sport huancayo":"sport huancayo",
    "los chankas":"los chankas", "chankas":"los chankas",
    "chankas cyc":"los chankas",
    "fc cajamarca":"fc cajamarca", "fbc cajamarca":"fc cajamarca",
    "utc cajamarca":"utc", "utc":"utc",
    "universidad tecnica de cajamarca":"utc", "ut c":"utc",
    "comerciantes unidos":"comerciantes unidos",
    "adt":"adt", "asociacion deportiva tarma":"adt",
    "cd moquegua":"cd moquegua", "deportivo moquegua":"cd moquegua",
    "moquegua":"cd moquegua",
    "colegio juan pablo ii":"colegio juan pablo ii",
    "juan pablo ii":"colegio juan pablo ii",
    "juan pablo ii college":"colegio juan pablo ii",
}

ALTITUDES_DEFAULT = {
    "adt":(3050,"Tarma"), "cienciano":(3360,"Cusco"),
    "cusco":(3360,"Cusco"), "garcilaso":(3360,"Cusco"),
    "sport huancayo":(3250,"Huancayo"), "los chankas":(2920,"Andahuaylas"),
    "utc":(2750,"Cajamarca"), "fc cajamarca":(2750,"Cajamarca"),
    "comerciantes unidos":(2620,"Cutervo"), "melgar":(2335,"Arequipa"),
    "universitario":(150,"Lima"), "alianza lima":(150,"Lima"),
    "sporting cristal":(150,"Lima"), "sport boys":(10,"Callao"),
    "atletico grau":(50,"Piura"), "alianza atletico":(50,"Sullana"),
    "colegio juan pablo ii":(150,"Chongoyape"), "cd moquegua":(1410,"Moquegua")
}

def estandarizar_nombre(nombre):
    txt = normalizar_texto(nombre)
    if txt in DICCIONARIO_EQUIPOS:
        return DICCIONARIO_EQUIPOS[txt]
    # No eliminamos "atletico" antes del diccionario.
    txt = re.sub(r"\b(club|futbol club|fc|cd|fbc|deportivo|asociacion)\b"," ",txt)
    txt = " ".join(txt.split())
    return DICCIONARIO_EQUIPOS.get(txt,txt)

def resolver_columna(df,candidatos,destino):
    for c in candidatos:
        if c in df.columns:
            return df.rename(columns={c:destino})
    return df

def resolver_columna_club(df):
    df=df.copy()
    df.columns=[normalizar_texto(c) for c in df.columns]
    return resolver_columna(df,["club","equipo","nombre","team","clubes","equipos"],"club")

# =========================================================
# 2. RIVALIDADES
# =========================================================
RIVALIDADES = {
    frozenset(["atletico grau","alianza atletico"]):"Clásico Piurano",
    frozenset(["cusco","cienciano"]):"Clásico Cusqueño",
    frozenset(["fc cajamarca","utc"]):"Clásico Cajamarquino",
    frozenset(["fc cajamarca","comerciantes unidos"]):"Rivalidad Cajamarquina",
    frozenset(["utc","comerciantes unidos"]):"Rivalidad Cajamarquina",
}

def obtener_rivalidad(a,b):
    return RIVALIDADES.get(frozenset([a,b]))

# =========================================================
# 3. PREPARACIÓN DE DATOS
# =========================================================
def preparar_partidos(df,temporada_default=None):
    df=df.copy()
    df.columns=[normalizar_texto(c) for c in df.columns]
    df=resolver_columna(df,["local","equipo local","local team"],"local")
    df=resolver_columna(df,["visitante","visita","equipo visitante","away","away team"],"visita")
    df=resolver_columna(df,["gl","goles local","goles_local","goles"],"goles_local")
    df=resolver_columna(df,["gv","goles visita","goles_visitante","goles_visita"],"goles_visita")
    df=resolver_columna(df,["fecha","date"],"fecha")
    df=resolver_columna(df,["temporada","season","ano"],"temporada")

    if "local" not in df.columns or "visita" not in df.columns:
        return pd.DataFrame()

    if "fecha" not in df.columns: df["fecha"]=pd.NaT
    df["fecha"]=pd.to_datetime(df["fecha"],errors="coerce",dayfirst=True)
    if "temporada" not in df.columns: df["temporada"]=temporada_default
    else: df["temporada"]=df["temporada"].fillna(temporada_default)

    for c in ["goles_local","goles_visita"]:
        if c not in df.columns: df[c]=np.nan
        df[c]=pd.to_numeric(df[c],errors="coerce")

    df["local_std"]=df["local"].apply(estandarizar_nombre)
    df["visita_std"]=df["visita"].apply(estandarizar_nombre)
    df["jugado_calc"]=(
        df["goles_local"].notna() & df["goles_visita"].notna() &
        df["local_std"].ne("") & df["visita_std"].ne("")
    )
    return df

def cargar_datos_excel(fuente):
    try:
        xls=pd.ExcelFile(fuente)
        hojas=xls.sheet_names

        if "Partidos_Fecha" not in hojas:
            raise ValueError("Falta la hoja 'Partidos_Fecha'.")

        dfp=pd.read_excel(xls,sheet_name="Partidos_Fecha")
        dfp.columns=[normalizar_texto(c) for c in dfp.columns]

        if "jornada" in dfp.columns:
            dfp["jornada_num"]=dfp["jornada"].astype(str).str.extract(r"(\d+)")[0].fillna("1")
        elif "fecha_num" in dfp.columns:
            dfp["jornada_num"]=dfp["fecha_num"].astype(str)
        else:
            dfp["jornada_num"]="1"

        if "fecha" in dfp.columns:
            dfp["fecha"]=pd.to_datetime(dfp["fecha"],errors="coerce",dayfirst=True)
            dfp["fecha_str"]=dfp["fecha"].dt.strftime("%Y-%m-%d")
        else:
            dfp["fecha"]=pd.NaT; dfp["fecha_str"]=""

        if "hora" not in dfp.columns: dfp["hora"]="15:00"
        dfp["local_std"]=dfp["local"].apply(estandarizar_nombre)
        dfp["visita_std"]=dfp["visita"].apply(estandarizar_nombre)
        if "goles_local" not in dfp.columns: dfp["goles_local"]=np.nan
        if "goles_visita" not in dfp.columns: dfp["goles_visita"]=np.nan
        if "jugado" not in dfp.columns:
            dfp["jugado"]=dfp["goles_local"].notna() & dfp["goles_visita"].notna()

        if "Data_Geografica" in hojas:
            geo=resolver_columna_club(pd.read_excel(xls,sheet_name="Data_Geografica"))
            geo["equipo_std"]=geo["club"].apply(estandarizar_nombre)
        else:
            geo=pd.DataFrame([{"equipo_std":k,"altitud":v[0],"ciudad":v[1]} for k,v in ALTITUDES_DEFAULT.items()])

        if "Tabla_Acumulada" in hojas:
            acum=resolver_columna_club(pd.read_excel(xls,sheet_name="Tabla_Acumulada"))
            acum["equipo_std"]=acum["club"].apply(estandarizar_nombre)
        else: acum=pd.DataFrame()

        if "Tabla_Clausura" in hojas:
            claus=resolver_columna_club(pd.read_excel(xls,sheet_name="Tabla_Clausura"))
            claus["equipo_std"]=claus["club"].apply(estandarizar_nombre)
        else: claus=pd.DataFrame()

        resultados=[]
        for hoja in ["Resultados_Apertura","Resultados_Clausura"]:
            if hoja in hojas:
                z=preparar_partidos(pd.read_excel(xls,sheet_name=hoja),2026)
                if not z.empty: resultados.append(z)

        res=pd.concat(resultados,ignore_index=True) if resultados else pd.DataFrame()
        if not res.empty:
            res=res[res["jugado_calc"]].sort_values("fecha",na_position="last").reset_index(drop=True)

        if "Historial_H2H" in hojas:
            h2h=preparar_partidos(pd.read_excel(xls,sheet_name="Historial_H2H"))
            h2h=h2h[h2h["jugado_calc"]].sort_values("fecha",na_position="last").reset_index(drop=True)
            estado=f"Historial_H2H cargado: {len(h2h)} enfrentamientos."
        else:
            h2h=pd.DataFrame()
            estado="No existe 'Historial_H2H'. El sistema funcionará sin H2H."

        return dfp,acum,claus,geo,res,h2h,estado,None
    except Exception as e:
        return None,None,None,None,None,None,None,str(e)

# =========================================================
# 4. ELO
# =========================================================
def calcular_elo_historico(df,k=24.0,elo_inicial=1500.0,ventaja_local=55.0):
    if df is None or df.empty: return {}
    ratings={}
    for _,r in df.sort_values("fecha",na_position="last").iterrows():
        a,b=r["local_std"],r["visita_std"]
        ra=ratings.get(a,elo_inicial); rb=ratings.get(b,elo_inicial)
        exp=1/(1+10**(-((ra+ventaja_local)-rb)/400))
        gl,gv=float(r["goles_local"]),float(r["goles_visita"])
        resultado=1.0 if gl>gv else 0.5 if gl==gv else 0.0
        cambio=k*(resultado-exp)
        ratings[a]=ra+cambio
        ratings[b]=rb-cambio
    return ratings

# =========================================================
# 5. FORMA
# =========================================================
def forma_equipo(equipo,df,n=5):
    base={"partidos":0,"puntos":0.0,"ppg":1.0,"gf":0.0,"gc":0.0}
    if df is None or df.empty: return base
    z=df[(df["local_std"]==equipo)|(df["visita_std"]==equipo)].sort_values("fecha").tail(n)
    if z.empty: return base
    pts=gf=gc=0.0
    for _,r in z.iterrows():
        if r["local_std"]==equipo: f,c=r["goles_local"],r["goles_visita"]
        else: f,c=r["goles_visita"],r["goles_local"]
        gf+=f; gc+=c
        pts+=3 if f>c else 1 if f==c else 0
    return {"partidos":len(z),"puntos":pts,"ppg":pts/len(z),"gf":gf/len(z),"gc":gc/len(z)}

# =========================================================
# 6. FUERZA ATAQUE/DEFENSA
# =========================================================
def medias_liga(df):
    if df is None or df.empty: return 1.35,1.10
    return max(float(df["goles_local"].mean()),0.80),max(float(df["goles_visita"].mean()),0.60)

def fuerza_equipos(local,visita,tabla,res):
    ml,mv=medias_liga(res)

    def desde_tabla(eq):
        if tabla is None or tabla.empty or "equipo_std" not in tabla.columns: return None
        r=tabla[tabla["equipo_std"]==eq]
        if r.empty: return None
        cols={normalizar_texto(c):c for c in tabla.columns}
        pjcol=cols.get("pj"); gfcol=cols.get("gf"); gccol=cols.get("gc")
        if not all([pjcol,gfcol,gccol]): return None
        pj=pd.to_numeric(r.iloc[0][pjcol],errors="coerce")
        gf=pd.to_numeric(r.iloc[0][gfcol],errors="coerce")
        gc=pd.to_numeric(r.iloc[0][gccol],errors="coerce")
        if pd.isna(pj) or pj<=0 or pd.isna(gf) or pd.isna(gc): return None
        return float(gf/pj),float(gc/pj)

    def desde_res(eq):
        if res is None or res.empty: return ml,mv
        z=res[(res["local_std"]==eq)|(res["visita_std"]==eq)]
        if z.empty: return ml,mv
        gf=z.loc[z["local_std"]==eq,"goles_local"].sum()+z.loc[z["visita_std"]==eq,"goles_visita"].sum()
        gc=z.loc[z["local_std"]==eq,"goles_visita"].sum()+z.loc[z["visita_std"]==eq,"goles_local"].sum()
        return float(gf/len(z)),float(gc/len(z))

    a=desde_tabla(local) or desde_res(local)
    b=desde_tabla(visita) or desde_res(visita)
    return a[0],a[1],b[0],b[1],ml,mv

# =========================================================
# 7. GEOGRAFÍA
# =========================================================
def geo_equipo(eq,geo):
    if geo is not None and not geo.empty and "equipo_std" in geo.columns:
        r=geo[geo["equipo_std"]==eq]
        if not r.empty:
            alt=pd.to_numeric(r.iloc[0].get("altitud",np.nan),errors="coerce")
            ciudad=str(r.iloc[0].get("ciudad",ALTITUDES_DEFAULT.get(eq,(150,"Lima"))[1]))
            if pd.notna(alt): return float(alt),ciudad
    return ALTITUDES_DEFAULT.get(eq,(150,"Lima"))

# =========================================================
# 8. H2H
# =========================================================
def peso_recencia(fecha,ref):
    if pd.isna(fecha) or pd.isna(ref): return 0.35
    anos=max(0,(pd.Timestamp(ref)-pd.Timestamp(fecha)).days)/365.25
    if anos<=2: return 1.00
    if anos<=4: return 0.75
    if anos<=6: return 0.50
    if anos<=8: return 0.30
    return 0.15

def h2h_info(local,visita,df,fecha_ref,max_partidos=10):
    base={"n":0,"local_w":0,"draw":0,"visit_w":0,"indice":0.5,"same_n":0,"same_indice":0.5}
    if df is None or df.empty: return base,pd.DataFrame()

    z=df.copy()
    if pd.notna(fecha_ref):
        z=z[z["fecha"].isna()|(z["fecha"]<pd.Timestamp(fecha_ref))]
    z=z[
        ((z["local_std"]==local)&(z["visita_std"]==visita))|
        ((z["local_std"]==visita)&(z["visita_std"]==local))
    ].sort_values("fecha",na_position="last").tail(max_partidos)

    if z.empty: return base,z

    ref=pd.Timestamp(fecha_ref) if pd.notna(fecha_ref) else pd.Timestamp.today()
    wl=wd=wv=0.0
    sw=sd=sl=st=0.0

    for _,r in z.iterrows():
        p=peso_recencia(r["fecha"],ref)
        gl,gv=float(r["goles_local"]),float(r["goles_visita"])
        if r["local_std"]==local:
            a,b=gl,gv
            mismo=True
        else:
            a,b=gv,gl
            mismo=False
        if a>b: wl+=p
        elif a==b: wd+=p
        else: wv+=p
        if mismo:
            st+=p
            if a>b: sw+=p
            elif a==b: sd+=p
            else: sl+=p

    total=wl+wd+wv
    indice=(wl+0.5*wd)/total if total else 0.5
    same=(sw+0.5*sd)/st if st else 0.5

    # Conteo sin ponderar, para mostrar al usuario.
    lw=dw=vw=0
    for _,r in z.iterrows():
        if r["local_std"]==local:
            a,b=r["goles_local"],r["goles_visita"]
        else:
            a,b=r["goles_visita"],r["goles_local"]
        if a>b: lw+=1
        elif a==b: dw+=1
        else: vw+=1

    base.update({"n":len(z),"local_w":lw,"draw":dw,"visit_w":vw,
                 "indice":float(indice),"same_n":int(z["local_std"].eq(local).sum()),
                 "same_indice":float(same)})
    return base,z

# =========================================================
# 9. DIXON-COLES + ELO + FORMA + ALTITUD + H2H
# =========================================================
def tau_dc(x,y,lam,mu,rho=-0.11):
    if x==0 and y==0: return max(0.0001,1-lam*mu*rho)
    if x==1 and y==0: return max(0.0001,1+mu*rho)
    if x==0 and y==1: return max(0.0001,1+lam*rho)
    if x==1 and y==1: return max(0.0001,1-rho)
    return 1.0

def matriz_dc(lam,mu,rho=-0.11,n=9):
    m=np.zeros((n,n))
    for x in range(n):
        for y in range(n):
            m[x,y]=max(0,poisson.pmf(x,lam)*poisson.pmf(y,mu)*tau_dc(x,y,lam,mu,rho))
    s=m.sum()
    return m/s if s>0 else m

def calcular_modelo(local,visita,tabla,res,geo,h2h,fecha_ref,rho=-0.11):
    attl,defl,attv,defv,ml,mv=fuerza_equipos(local,visita,tabla,res)

    # Fuerza relativa.
    al=np.clip(attl/max(ml,0.8),0.60,1.60)
    dl=np.clip(defl/max(mv,0.8),0.60,1.60)
    av=np.clip(attv/max(mv,0.6),0.60,1.60)
    dv=np.clip(defv/max(ml,0.8),0.60,1.60)

    # Elo.
    elo=calcular_elo_historico(res)
    el=elo.get(local,1500.0); ev=elo.get(visita,1500.0)
    de=el-ev

    rival=obtener_rivalidad(local,visita)
    de_aj=de*(0.90 if rival else 1.0)
    fe_l=np.exp(np.clip(de_aj,-400,400)/400*0.14)
    fe_v=np.exp(np.clip(-de_aj,-400,400)/400*0.14)

    # Forma últimos 5.
    fl=foma=forma_equipo(local,res,5)
    fv=forma_equipo(visita,res,5)
    dforma=fl["ppg"]-fv["ppg"]
    ff_l=np.exp(np.clip(dforma,-3,3)*(0.040 if rival else 0.055))
    ff_v=np.exp(np.clip(-dforma,-3,3)*(0.040 if rival else 0.055))

    # Altitud.
    altl,ciudad=geo_equipo(local,geo)
    altv,ciudadv=geo_equipo(visita,geo)
    dif=max(0,altl-altv)
    factor_alt_l=1.0
    factor_alt_v=1.0
    if altl>=2000:
        factor_alt_l*=1+min(0.10,dif/8000)
        factor_alt_v*=1-min(0.18,dif/6500)
    factor_alt_v=max(0.82,factor_alt_v)
    home_adv=1.08 if altl<2000 else 1.12

    # Goles esperados.
    lam=ml*al*dv*home_adv*fe_l*ff_l*factor_alt_l
    mu=mv*av*dl*fe_v*ff_v*factor_alt_v
    lam=float(np.clip(lam,0.25,3.80))
    mu=float(np.clip(mu,0.20,3.20))

    m=matriz_dc(lam,mu,rho,9)
    p1=float(np.tril(m,-1).sum())
    px=float(np.trace(m))
    p2=float(np.triu(m,1).sum())

    # H2H: máximo 12%; 16% en rivalidades.
    hh,hh_rows=h2h_info(local,visita,h2h,fecha_ref,10)
    ph=0.12*min(1,hh["n"]/10)
    if rival: ph=min(0.16,ph*1.25)

    indice=hh["indice"]
    if hh["same_n"]>=2:
        indice=0.70*indice+0.30*hh["same_indice"]

    hv=np.array([0.50+0.50*(indice-0.50),0.50,0.50-0.50*(indice-0.50)])
    hv=hv/hv.sum()
    base=np.array([p1,px,p2])
    final=(1-ph)*base+ph*hv
    final=final/final.sum()
    p1,px,p2=final.tolist()

    over=float(1-sum(m[i,j] for i in range(9) for j in range(9) if i+j<=2))
    under=1-over
    btts=float(m[1:,1:].sum())
    nbtts=1-btts
    idx=np.unravel_index(np.argmax(m),m.shape)

    return {
        "p_local":p1,"p_empate":px,"p_visita":p2,
        "over":over,"under":under,"btts":btts,"no_btts":nbtts,
        "lambda":lam,"mu":mu,"elo_local":el,"elo_visita":ev,"dif_elo":de,
        "forma_local":fl,"forma_visita":fv,"alt_local":altl,"alt_visita":altv,
        "ciudad_local":ciudad,"ciudad_visita":ciudadv,
        "rivalidad":rival,"h2h":hh,"peso_h2h":ph,
        "marcador_modal":f"{idx[0]}-{idx[1]}","prob_modal":float(m[idx])
    }

def cuota(p): return 1/max(float(p),0.001)

def sugerencia(p1,px,p2,nl,nv):
    opciones=[(p1,f"Gana {nl}"),(px,"Empate"),(p2,f"Gana {nv}")]
    opciones.sort(reverse=True)
    if opciones[0][0]>=0.55: return opciones[0][1],"Alta"
    if opciones[0][0]>=0.45 and opciones[0][0]-opciones[1][0]>=0.08:
        return opciones[0][1],"Media-Alta"
    if p1+px>=0.67 and p1>=p2: return f"Local o Empate ({nl})","Media-Alta"
    if p2+px>=0.67 and p2>=p1: return f"Empate o Visita ({nv})","Media-Alta"
    return "Doble Opción","Media"

# =========================================================
# 10. EXCEL
# =========================================================
def exportar_excel(p,a,c,g,h,res):
    b=io.BytesIO()
    with pd.ExcelWriter(b,engine="openpyxl") as w:
        p.to_excel(w,sheet_name="Partidos_Fecha",index=False)
        if a is not None and not a.empty: a.to_excel(w,sheet_name="Tabla_Acumulada",index=False)
        if c is not None and not c.empty: c.to_excel(w,sheet_name="Tabla_Clausura",index=False)
        if g is not None and not g.empty: g.to_excel(w,sheet_name="Data_Geografica",index=False)
        if res is not None and not res.empty: res.to_excel(w,sheet_name="Resultados_Consolidados",index=False)
        if h is not None and not h.empty: h.to_excel(w,sheet_name="Historial_H2H",index=False)
        else: pd.DataFrame(columns=["Fecha","Local","Visitante","GL","GV","Temporada"]).to_excel(w,sheet_name="Historial_H2H",index=False)
    b.seek(0)
    return b

def guardar_excel(p,a,c,g,h,res,ruta):
    try:
        with open(ruta,"wb") as f:
            f.write(exportar_excel(p,a,c,g,h,res).getvalue())
        return True,"Guardado correctamente."
    except Exception as e:
        return False,str(e)

# =========================================================
# 11. CARGA
# =========================================================
st.title("⚽ Modelo de Predicción Liga 1 Perú")
st.caption("V3: Dixon-Coles + Elo + Forma reciente + Altitud + Rivalidad + Historial H2H")

st.sidebar.header("📁 Archivo de Datos")
upload=st.sidebar.file_uploader("Subir Liga1_2026.xlsx",type=["xlsx"])
ruta=os.path.join(os.path.dirname(os.path.abspath(__file__)),"Liga1_2026.xlsx")
fuente=upload if upload is not None else (ruta if os.path.exists(ruta) else None)

if st.sidebar.button("🔄 Recargar Datos"):
    st.session_state.clear()
    st.rerun()

if fuente is None:
    st.error("❌ No se encontró Liga1_2026.xlsx. Sube el archivo desde el menú lateral.")
    st.stop()

if "cargado" not in st.session_state or upload is not None:
    datos=cargar_datos_excel(fuente)
    if datos[-1] is not None:
        st.error(datos[-1]); st.stop()
    p,a,c,g,res,h,estado,_=datos
    st.session_state.update(df_partidos=p,df_acum=a,df_claus=c,df_geo=g,df_resultados=res,df_h2h=h,h2h_estado=estado,cargado=True)

# =========================================================
# 12. SIDEBAR
# =========================================================
opciones=[]
if not st.session_state.df_acum.empty: opciones.append("Tabla Acumulada")
if not st.session_state.df_claus.empty: opciones.append("Tabla Clausura")
if not opciones: opciones=["Resultados históricos"]

tabla_ref=st.sidebar.radio("Tabla de Rendimiento:",opciones)
tabla=st.session_state.df_acum if tabla_ref=="Tabla Acumulada" else st.session_state.df_claus

jornadas=sorted(st.session_state.df_partidos["jornada_num"].astype(str).unique(),key=lambda x:int(x) if x.isdigit() else 0)
idx=jornadas.index("8") if "8" in jornadas else len(jornadas)-1
jornada=st.sidebar.selectbox("Seleccionar Jornada:",jornadas,index=max(0,idx),format_func=lambda x:f"Jornada {x}")

rho=st.sidebar.slider("Rho Dixon-Coles",-0.20,0.00,-0.11,0.01)
st.sidebar.info(st.session_state.h2h_estado)
st.sidebar.caption("H2H: últimos 10 enfrentamientos anteriores al partido. Peso máximo 12%; 16% en rivalidades.")

dfj=st.session_state.df_partidos[st.session_state.df_partidos["jornada_num"].astype(str)==str(jornada)].reset_index(drop=True)

tab1,tab2,tab3=st.tabs([f"📊 Pronósticos Jornada {jornada}","📝 Actualizar Resultados","🔎 Diagnóstico"])

# =========================================================
# 13. PRONÓSTICOS
# =========================================================
with tab1:
    st.subheader(f"Jornada {jornada} - Pronósticos ({tabla_ref})")

    for i,row in dfj.iterrows():
        nl=str(row["local"]).strip(); nv=str(row["visita"]).strip()
        r=calcular_modelo(row["local_std"],row["visita_std"],tabla,
                          st.session_state.df_resultados,st.session_state.df_geo,
                          st.session_state.df_h2h,row.get("fecha",pd.NaT),rho)

        p1,px,p2=r["p_local"],r["p_empate"],r["p_visita"]
        sug,conf=sugerencia(p1,px,p2,nl,nv)
        fg="Alta" if max(r["over"],r["under"])>=0.65 else "Media"
        fb="Alta" if max(r["btts"],r["no_btts"])>=0.65 else "Media"
        fecha=row.get("fecha",pd.NaT)
        fecha_txt=fecha.strftime("%Y-%m-%d") if pd.notna(fecha) else "No registrada"

        st.markdown(f"### 🏟️ {nl} vs {nv} | {r['ciudad_local']} ({r['alt_local']:.0f} msnm)")
        st.caption(f"📅 {fecha_txt}  |  🕒 {row.get('hora','15:00')}")

        c1,c2,c3=st.columns(3)
        with c1:
            st.write(f"**Gana {nl}**"); st.title(f"{p1*100:.1f}%"); st.caption(f"Cuota justa: {cuota(p1):.2f}")
        with c2:
            st.write("**Empate**"); st.title(f"{px*100:.1f}%"); st.caption(f"Cuota justa: {cuota(px):.2f}")
        with c3:
            st.write(f"**Gana {nv}**"); st.title(f"{p2*100:.1f}%"); st.caption(f"Cuota justa: {cuota(p2):.2f}")

        x1,x2=st.columns([2,1])
        with x1:
            st.markdown(f"<div class='suggestion-box-blue'><b>Pronóstico 1X2:</b> {sug}</div>",unsafe_allow_html=True)
        with x2:
            st.markdown(f"<div class='suggestion-box-green'><b>Confianza:</b> {conf}</div>",unsafe_allow_html=True)

        m1,m2,m3,m4=st.columns(4)
        with m1:
            st.metric("Elo Local",f"{r['elo_local']:.0f}"); st.caption(f"Visita: {r['elo_visita']:.0f}")
        with m2:
            st.metric("Forma Local",f"{r['forma_local']['ppg']:.2f}"); st.caption(f"Visita: {r['forma_visita']['ppg']:.2f} pts/pj")
        with m3:
            hh=r["h2h"]; st.metric("H2H usados",hh["n"])
            st.caption(f"{hh['local_w']} G · {hh['draw']} E · {hh['visit_w']} G")
        with m4:
            st.metric("Peso H2H",f"{r['peso_h2h']*100:.1f}%")
            st.caption("Ajuste limitado")

        if r["rivalidad"]:
            st.info(f"🔥 **{r['rivalidad']}**: se modera el peso de la diferencia de fuerza.")

        if r["h2h"]["n"]:
            st.write(f"**H2H últimos {r['h2h']['n']}:** {r['h2h']['local_w']} victorias {nl} · {r['h2h']['draw']} empates · {r['h2h']['visit_w']} victorias {nv}.")
            st.caption(f"H2H con {nl} como local: {r['h2h']['same_n']}.")

        g1,g2=st.columns(2)
        with g1:
            st.markdown("#### ⚽ Over / Under 2.5")
            st.write(f"Más de 2.5: **{r['over']*100:.1f}%**"); st.progress(float(r["over"]))
            st.caption(f"Cuota justa: {cuota(r['over']):.2f}")
            st.write(f"Menos de 2.5: **{r['under']*100:.1f}%**"); st.progress(float(r["under"]))
            st.caption(f"Cuota justa: {cuota(r['under']):.2f}")
            st.markdown(f"<div class='suggestion-box-blue'><b>Sugerido:</b> {'Más de 2.5' if r['over']>=r['under'] else 'Menos de 2.5'}</div>",unsafe_allow_html=True)
            st.caption(f"Confianza: {fg}")
        with g2:
            st.markdown("#### 🔥 Ambos equipos anotan")
            st.write(f"Sí: **{r['btts']*100:.1f}%**"); st.progress(float(r["btts"]))
            st.caption(f"Cuota justa: {cuota(r['btts']):.2f}")
            st.write(f"No: **{r['no_btts']*100:.1f}%**"); st.progress(float(r["no_btts"]))
            st.caption(f"Cuota justa: {cuota(r['no_btts']):.2f}")
            st.markdown(f"<div class='suggestion-box-blue'><b>Sugerido:</b> {'BTTS Sí' if r['btts']>=r['no_btts'] else 'BTTS No'}</div>",unsafe_allow_html=True)
            st.caption(f"Confianza: {fb}")

        st.write(f"**🎯 Marcador modal:** {r['marcador_modal']} ({r['prob_modal']*100:.1f}%)")
        st.caption(f"λ Local={r['lambda']:.2f} | μ Visita={r['mu']:.2f}")
        st.divider()

# =========================================================
# 14. ACTUALIZACIÓN
# =========================================================
with tab2:
    st.subheader(f"⚙️ Actualizar Jornada {jornada}")

    with st.form(f"form_{jornada}"):
        cambios=[]
        for i,row in dfj.iterrows():
            c1,c2,c3,c4,c5=st.columns([3,1,1,3,2])
            with c1: st.write(f"**{row['local']}**")
            with c2:
                gl=st.number_input("GL",0,15,int(row["goles_local"]) if pd.notna(row["goles_local"]) else 0,key=f"gl_{jornada}_{i}",label_visibility="collapsed")
            with c3:
                gv=st.number_input("GV",0,15,int(row["goles_visita"]) if pd.notna(row["goles_visita"]) else 0,key=f"gv_{jornada}_{i}",label_visibility="collapsed")
            with c4: st.write(f"**{row['visita']}**")
            with c5:
                jug=st.checkbox("Jugado",bool(row["jugado"]),key=f"jug_{jornada}_{i}")
            cambios.append((row["local"],row["visita"],gl,gv,jug))
        enviar=st.form_submit_button("💾 Aplicar y Guardar")

    if enviar:
        for loc,vis,gl,gv,jug in cambios:
            mask=(st.session_state.df_partidos["jornada_num"].astype(str)==str(jornada))&(st.session_state.df_partidos["local"]==loc)&(st.session_state.df_partidos["visita"]==vis)
            st.session_state.df_partidos.loc[mask,"goles_local"]=gl
            st.session_state.df_partidos.loc[mask,"goles_visita"]=gv
            st.session_state.df_partidos.loc[mask,"jugado"]=jug

        nuevos=preparar_partidos(st.session_state.df_partidos)
        jugados=nuevos[nuevos["jugado_calc"]].copy()
        historicos=st.session_state.df_resultados.copy()
        if historicos is None or historicos.empty:
            st.session_state.df_resultados=jugados
        else:
            # Reemplaza resultados de partidos que ya estaban en la jornada.
            claves=set(zip(jugados["fecha"],jugados["local_std"],jugados["visita_std"]))
            historicos=historicos[~historicos.apply(lambda x:(x["fecha"],x["local_std"],x["visita_std"]) in claves,axis=1)]
            st.session_state.df_resultados=pd.concat([historicos,jugados],ignore_index=True).sort_values("fecha").reset_index(drop=True)

        if os.path.exists(ruta) and upload is None:
            ok,msg=guardar_excel(st.session_state.df_partidos,st.session_state.df_acum,st.session_state.df_claus,st.session_state.df_geo,st.session_state.df_h2h,st.session_state.df_resultados,ruta)
            st.success("✅ Datos guardados en Liga1_2026.xlsx." if ok else f"⚠️ {msg}")
        else:
            st.success("✅ Sesión actualizada.")
        st.rerun()

    st.markdown("#### 📥 Copia de seguridad")
    st.download_button(
        "Descargar Liga1_2026_Actualizado.xlsx",
        data=exportar_excel(st.session_state.df_partidos,st.session_state.df_acum,st.session_state.df_claus,st.session_state.df_geo,st.session_state.df_h2h,st.session_state.df_resultados),
        file_name="Liga1_2026_Actualizado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =========================================================
# 15. DIAGNÓSTICO
# =========================================================
with tab3:
    st.subheader("🔎 Diagnóstico")
    a1,a2,a3,a4=st.columns(4)
    with a1: st.metric("Partidos históricos",len(st.session_state.df_resultados))
    with a2: st.metric("H2H",len(st.session_state.df_h2h))
    with a3:
        equipos=set(st.session_state.df_resultados["local_std"])|set(st.session_state.df_resultados["visita_std"]) if not st.session_state.df_resultados.empty else set()
        st.metric("Equipos",len(equipos))
    with a4: st.metric("Equipos con Elo",len(calcular_elo_historico(st.session_state.df_resultados)))

    st.markdown("### 📚 Historial H2H")
    if st.session_state.df_h2h.empty:
        st.warning("No hay H2H cargados. La hoja debe llamarse Historial_H2H.")
        st.code("Fecha | Local | Visitante | GL | GV | Temporada")
    else:
        cols=[c for c in ["fecha","local","visita","goles_local","goles_visita","temporada"] if c in st.session_state.df_h2h.columns]
        st.dataframe(st.session_state.df_h2h[cols].tail(20),use_container_width=True,hide_index=True)

    st.markdown("### ⚠️ Verificación de nombres")
    nombres=sorted(set(st.session_state.df_partidos["local"].astype(str))|set(st.session_state.df_partidos["visita"].astype(str)))
    ver=pd.DataFrame([{"Nombre original":n,"Nombre estándar":estandarizar_nombre(n)} for n in nombres])
    st.dataframe(ver,use_container_width=True,hide_index=True)
    st.info("Verifica especialmente: FC Cajamarca ≠ UTC y Atlético Grau ≠ Alianza Atlético.")
