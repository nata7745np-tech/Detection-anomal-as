"""
=============================================================
  DETECCIÓN DE ANOMALÍAS CON ISOLATION FOREST
  AgroSmart Colombia
  
  Curso: Ciencia de Datos
  Herramienta: Python + Streamlit + scikit-learn
  Caso 5: Detección de Anomalías en Sensores Agrícolas
=============================================================

INSTRUCCIONES PARA EJECUTAR:
1. Instalar dependencias:
   pip install streamlit pandas scikit-learn plotly

2. Ejecutar desde la terminal de VS Code:
   streamlit run app_caso5_anomalias.py

3. Se abrirá automáticamente en el navegador
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Detección de Anomalías en Sensores Agrícolas",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Detección de Anomalías en Sensores Agrícolas")
st.markdown("""
**AgroSmart Colombia**  
Una empresa de agricultura de precisión monitorea cultivos en varias regiones del país. Quiere detectar lecturas anómalas de sensores que podrían indicar plagas, fallas de equipo o condiciones climáticas extremas.
""")
st.divider()

# ══════════════════════════════════════════════════════════
# SECCIÓN 1: CARGA DE DATOS
# ══════════════════════════════════════════════════════════
st.header("📂 1. Cargar datos")

col1, col2 = st.columns([2, 1])
with col1:
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
with col2:
    st.info("¿No tienes archivo? Activa el dataset de ejemplo abajo ↓")

usar_ejemplo = st.checkbox("✅ Usar dataset de ejemplo", value=True)

@st.cache_data
def cargar_ejemplo():
    return pd.read_csv("caso5_sensores_cultivos.csv")

def cargar_datos(archivo, usar_ejemplo):
    if archivo is not None:
        return pd.read_csv(archivo)
    elif usar_ejemplo:
        return cargar_ejemplo()
    return None

df = cargar_datos(archivo, usar_ejemplo)

if df is not None:
    st.success(f"✅ Datos cargados: **{len(df)} registros** | **{len(df.columns)} variables**")
    with st.expander("👀 Vista previa de los datos"):
        st.dataframe(df.head(10), use_container_width=True)
    with st.expander("📊 Estadísticas descriptivas"):
        st.dataframe(df.describe(), use_container_width=True)
else:
    st.warning("⚠️ Carga un archivo o activa el dataset de ejemplo para continuar.")
    st.stop()

# ══════════════════════════════════════════════════════════
# SECCIÓN 2: CONFIGURACIÓN DEL MODELO
# ══════════════════════════════════════════════════════════
st.divider()
st.header("⚙️ 2. Configurar el modelo de detección")

columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
columnas_numericas = [c for c in columnas_numericas
                      if not any(x in c.lower() for x in ['id', 'cod', 'codigo'])]

col1, col2 = st.columns(2)

with col1:
    vars_default = ['temperatura_celsius', 'humedad_relativa_pct', 'ph_suelo', 'luminosidad_lux', 'co2_ppm']
    vars_disponibles = [v for v in vars_default if v in columnas_numericas]
    variables_elegidas = st.multiselect(
        "📌 Variables para detectar anomalías",
        options=columnas_numericas,
        default=vars_disponibles if vars_disponibles else columnas_numericas[:4],
        help="Selecciona las variables numéricas que el modelo analizará"
    )

with col2:
    contaminacion = st.slider(
        "🎯 Proporción esperada de anomalías (%)",
        min_value=1, max_value=20, value=5,
        help="¿Qué porcentaje de los registros crees que son anómalos?"
    )
    st.caption(f"El modelo buscará aproximadamente {int(len(df) * contaminacion / 100)} anomalías")

if len(variables_elegidas) < 2:
    st.warning("⚠️ Selecciona al menos 2 variables para continuar.")
    st.stop()

# ══════════════════════════════════════════════════════════
# SECCIÓN 3: EJECUTAR EL MODELO
# ══════════════════════════════════════════════════════════
st.divider()
st.header("🤖 3. Ejecutar detección de anomalías")

if st.button("▶️ Detectar anomalías ahora", type="primary", use_container_width=True):

    with st.spinner("Analizando patrones en los datos..."):

        X = df[variables_elegidas].dropna()
        indices_validos = X.index

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        modelo = IsolationForest(
            contamination=contaminacion / 100,
            random_state=42,
            n_estimators=100
        )
        predicciones = modelo.fit_predict(X_scaled)
        scores = modelo.score_samples(X_scaled)

        df_resultado = df.loc[indices_validos].copy()
        df_resultado['anomalia'] = predicciones
        df_resultado['es_anomalia'] = df_resultado['anomalia'].apply(
            lambda x: '🚨 Anómalo' if x == -1 else '✅ Normal'
        )
        df_resultado['score_anomalia'] = scores
        df_resultado['score_anomalia'] = (
            (df_resultado['score_anomalia'] - df_resultado['score_anomalia'].min()) /
            (df_resultado['score_anomalia'].max() - df_resultado['score_anomalia'].min())
        ).round(4)

    n_anomalias = (df_resultado['anomalia'] == -1).sum()
    n_normales = (df_resultado['anomalia'] == 1).sum()

    st.success(f"✅ Análisis completado — Se detectaron **{n_anomalias} registros anómalos**")
    st.session_state['resultado'] = df_resultado
    st.session_state['variables'] = variables_elegidas
    st.session_state['n_anomalias'] = n_anomalias

# ══════════════════════════════════════════════════════════
# SECCIÓN 4: RESULTADOS Y VISUALIZACIONES
# ══════════════════════════════════════════════════════════
if 'resultado' in st.session_state:

    df_resultado = st.session_state['resultado']
    variables_viz = st.session_state['variables']
    n_anomalias = st.session_state['n_anomalias']
    n_normales = len(df_resultado) - n_anomalias
    pct = n_anomalias / len(df_resultado) * 100

    st.divider()
    st.header("📊 4. Resultados")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total registros analizados", f"{len(df_resultado):,}")
    with col2:
        st.metric("🚨 Registros anómalos", f"{n_anomalias:,}", f"{pct:.1f}% del total")
    with col3:
        st.metric("✅ Registros normales", f"{n_normales:,}")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Dispersión de anomalías",
        "📦 Distribución por variable",
        "🚨 Registros anómalos",
        "📋 Comparación de perfiles"
    ])

    colores_anomalia = {'✅ Normal': '#2196F3', '🚨 Anómalo': '#F44336'}

    with tab1:
        st.subheader("¿Dónde se ubican las anomalías?")
        if len(variables_viz) >= 3:
            fig = px.scatter_3d(
                df_resultado,
                x=variables_viz[0], y=variables_viz[1], z=variables_viz[2],
                color='es_anomalia',
                color_discrete_map=colores_anomalia,
                title="Anomalías en 3 dimensiones",
                symbol='es_anomalia',
                opacity=0.7
            )
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(
                df_resultado,
                x=variables_viz[0], y=variables_viz[1],
                color='es_anomalia',
                color_discrete_map=colores_anomalia,
                title="Anomalías detectadas"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        variable_sel = st.selectbox("Selecciona una variable:", variables_viz)
        fig = px.histogram(
            df_resultado, x=variable_sel,
            color='es_anomalia',
            color_discrete_map=colores_anomalia,
            barmode='overlay', opacity=0.7,
            title=f"Distribución de '{variable_sel}': normales vs anómalos",
            nbins=40
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Registros marcados como anómalos")
        df_anomalos = df_resultado[df_resultado['anomalia'] == -1].sort_values('score_anomalia')
        st.dataframe(df_anomalos.drop(columns=['anomalia']), use_container_width=True)

    with tab4:
        st.subheader("Comparación de promedios: normales vs anómalos")
        perfil = df_resultado.groupby('es_anomalia')[variables_viz].mean().round(2)
        st.dataframe(perfil, use_container_width=True)

        perfil_norm = (perfil - perfil.min()) / (perfil.max() - perfil.min() + 1e-9)
        fig = go.Figure()
        for i, grupo in enumerate(perfil_norm.index):
            vals = perfil_norm.loc[grupo].tolist()
            vals += [vals[0]]
            cats = variables_viz + [variables_viz[0]]
            color = '#F44336' if 'Anóm' in grupo else '#2196F3'
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself', name=grupo, line_color=color))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                         title="Perfil normalizado: normales vs anómalos", height=430)
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════
    # SECCIÓN 5: DESCARGAR
    # ══════════════════════════════════════════════════════
    st.divider()
    st.header("💾 5. Descargar resultados")
    csv_out = df_resultado.drop(columns=['anomalia']).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Descargar registros con etiqueta de anomalía (CSV)",
        data=csv_out,
        file_name="resultado_anomalias.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ══════════════════════════════════════════════════════
    # SECCIÓN 6: Retos
    # ══════════════════════════════════════════════════════

    st.divider()
    st.header("6. Retos")

    df_anomalos = df_resultado[df_resultado['anomalia'] == -1].copy()

    # Filtro sidebar por región
    if 'region' in df_resultado.columns:
        regiones = ['Todas'] + sorted(df_resultado['region'].unique().tolist())
        region_sel = st.sidebar.selectbox("Filtrar por región", regiones)
        if region_sel == 'Todas':
            df_filtrado = df_anomalos
        else:
            df_filtrado = df_anomalos[df_anomalos['region'] == region_sel]
    else:
        df_filtrado = df_anomalos
        region_sel = 'Todas'

    # Grafico de barras de anomalias por tipo de cultivo 
    st.subheader("Reto 1: Anomalías por tipo de cultivo")
    if 'tipo_cultivo' in df_filtrado.columns and len(df_filtrado) > 0:
        conteo_cultivo = df_filtrado['tipo_cultivo'].value_counts().reset_index()
        conteo_cultivo.columns = ['tipo_cultivo', 'cantidad']
        fig = px.bar(
            conteo_cultivo,
            x='tipo_cultivo', y='cantidad',
            title=f"Anomalías por tipo de cultivo - {region_sel}",
            color='cantidad',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay anomalías para mostrar en la región seleccionada")

    # Mostrar tabla filtrada por región 
    st.subheader("Reto 2: Filtrar por región")
    st.write(f"Mostrando región: {region_sel} - {len(df_filtrado)} anomalías")
    st.dataframe(df_filtrado.drop(columns=['anomalia']), use_container_width=True)

    # Evolucion temporal del sensor mas anómalo 
    st.subheader("Reto 3: Evolución temporal del sensor más anómalo")
    if len(df_anomalos) > 0:
        sensor_top = df_anomalos.sort_values('score_anomalia').iloc[0]
        variable_temporal = variables_viz[0]
        np.random.seed(42)
        fechas = pd.date_range(start='2026-01-01', periods=50, freq='D')
        serie = np.random.normal(df_resultado[variable_temporal].mean(), 
                                 df_resultado[variable_temporal].std(), 50)
        serie[-1] = sensor_top[variable_temporal]
        df_serie = pd.DataFrame({'fecha': fechas, variable_temporal: serie})
        fig = px.line(df_serie, x='fecha', y=variable_temporal,
                      title=f"Simulación temporal del sensor {sensor_top['sensor_id']}")
        fig.add_scatter(x=[df_serie['fecha'].iloc[-1]], y=[serie[-1]], 
                        mode='markers', marker=dict(color='red', size=12), name='Anomalía')
        st.plotly_chart(fig, use_container_width=True)

    # Mapa de colombia de puntos anómalos por región con plotly 
    st.subheader("Reto 4: Mapa de anomalías por región")
    if 'region' in df_anomalos.columns:
        coords = {
            'Antioquia': (6.2442, -75.5812),
            'Cundinamarca': (4.7110, -74.0721),
            'Valle del Cauca': (3.8509, -76.5320),
            'Tolima': (4.0925, -75.1545),
            'Huila': (2.5359, -75.5277),
            'Nariño': (1.2136, -77.2811),
            'Boyacá': (5.4545, -73.3620),
            'Santander': (6.6437, -73.6536),
            'Córdoba': (8.7479, -75.8814),
            'Meta': (4.1420, -73.6266)
        }
        df_mapa = df_anomalos.copy()
        df_mapa['lat'] = df_mapa['region'].apply(lambda r: coords.get(r, (4.5, -74.0))[0])
        df_mapa['lon'] = df_mapa['region'].apply(lambda r: coords.get(r, (4.5, -74.0))[1])
        conteo_region = df_mapa.groupby(['region', 'lat', 'lon']).size().reset_index(name='anomalias')
        fig_mapa = px.scatter_mapbox(conteo_region, lat='lat', lon='lon', size='anomalias',
                                     color='anomalias', hover_name='region',
                                     color_continuous_scale='Reds', size_max=40, zoom=4,
                                     center={'lat': 4.5, 'lon': -74.0}, 
                                     mapbox_style='open-street-map',
                                     title='Anomalías por región en Colombia')
        fig_mapa.update_layout(height=550)
        st.plotly_chart(fig_mapa, use_container_width=True)
