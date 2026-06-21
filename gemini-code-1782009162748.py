import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Simulador Anchoveta - Mar de Grau",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── FUNCIONES PRINCIPALES (Con cache para alta velocidad) ──────────────────────
@st.cache_data
def generar_grilla_base():
    SEED = 42
    rng = np.random.default_rng(SEED)
    RES = 0.25
    LAT_MIN, LAT_MAX = -18.0, -4.0
    LON_MIN, LON_MAX = -82.0, -70.0
    
    lat_vals = np.arange(LAT_MIN, LAT_MAX + RES, RES)
    lon_vals = np.arange(LON_MIN, LON_MAX + RES, RES)
    LON_G, LAT_G = np.meshgrid(lon_vals, lat_vals)
    lat_grid = LAT_G.ravel()
    lon_grid = LON_G.ravel()
    
    def lon_costa(lat):
        return -75.0 + 0.15 * (lat + 10)
        
    dist_costa = np.abs(lon_grid - lon_costa(lat_grid))
    mascara = dist_costa < 8.0
    
    lat_grid = lat_grid[mascara]
    lon_grid = lon_grid[mascara]
    N_CELLS = len(lat_grid)
    
    prof_celda = (50 + 120 * dist_costa[mascara] / 8.0 + 30 * np.abs(lat_grid + 10) / 8.0 + rng.exponential(20, N_CELLS))
    prof_celda = np.clip(prof_celda, 10, 700)
    
    def zona_lat(lat):
        if lat > -7:   return 0
        elif lat > -10: return 1
        elif lat > -13: return 2
        elif lat > -16: return 3
        else:           return 4
    zona_celda = np.array([zona_lat(la) for la in lat_grid])
    
    return lat_grid, lon_grid, dist_costa[mascara], prof_celda, zona_celda, N_CELLS

@st.cache_data
def simular_dataset(intensidad_enso):
    lat_grid, lon_grid, dist_costa, prof_celda, zona_celda, N_CELLS = generar_grilla_base()
    rng = np.random.default_rng(42)
    
    START, END = "2019-01-01", "2023-12-31"
    fechas_base = pd.date_range(START, END, freq="W")
    T = len(fechas_base)
    dia_año = fechas_base.dayofyear.values
    sin_anual = np.sin(2 * np.pi * dia_año / 365.25)
    cos_anual = np.cos(2 * np.pi * dia_año / 365.25)
    
    oni_mensual = {
        2019: [ 0.8,  0.8,  0.8,  0.6,  0.5,  0.6,  0.3,  0.1, -0.1, -0.2, -0.3, -0.5],
        2020: [-0.5, -0.4, -0.2,  0.1,  0.1, -0.2, -0.6, -0.9, -1.3, -1.3, -1.3, -1.2],
        2021: [-1.1, -1.0, -0.8, -0.6, -0.3, -0.1,  0.0, -0.2, -0.7, -1.0, -1.0, -1.0],
        2022: [-1.0, -1.1, -1.2, -1.2, -1.1, -1.1, -0.9, -0.9, -1.0, -1.5, -1.7, -1.0],
        2023: [-0.6, -0.4,  0.0,  0.5,  1.0,  1.5,  1.5,  1.9,  2.0,  2.1,  2.0,  2.0],
    }
    
    enso_t = np.array([oni_mensual[f.year][f.month - 1] for f in fechas_base], dtype=float)
    enso_t = enso_t * intensidad_enso
    
    noise = np.zeros(T)
    for i in range(1, T):
        noise[i] = 0.7 * noise[i-1] + rng.normal(0, 0.08)
    enso_t = np.clip(enso_t + noise, -2.5, 2.8)
    
    def ar1_series(mu_t, phi, sigma, rng, clip=None):
        x = np.zeros(T)
        x[0] = mu_t[0] + rng.normal(0, sigma)
        for i in range(1, T):
            x[i] = mu_t[i] + phi * (x[i-1] - mu_t[i-1]) + rng.normal(0, sigma)
        return np.clip(x, *clip) if clip else x

    sst_mu_t = 17.5 + 1.2*sin_anual - 0.5*cos_anual + 1.8*enso_t
    sst_t = ar1_series(sst_mu_t, phi=0.82, sigma=0.35, rng=rng, clip=(12, 27))
    
    LAG = 3
    sst_lag = np.concatenate([sst_t[:LAG][::-1], sst_t[:-LAG]])
    chl_mu_t = 6.5 - 2.0*sin_anual + 1.5*cos_anual - 1.2*enso_t - 0.12*(sst_lag - 17)
    chl_t = ar1_series(chl_mu_t, phi=0.75, sigma=0.45, rng=rng, clip=(0.1, 18))
    
    ox_mu_t = 4.5 - 0.18*sst_t - 0.5*enso_t - 0.4*sin_anual
    ox_t = ar1_series(ox_mu_t, phi=0.70, sigma=0.30, rng=rng, clip=(0.3, 7.0))
    
    sal_mu_t = 34.85 - 0.12*enso_t
    sal_t = ar1_series(sal_mu_t, phi=0.90, sigma=0.07, rng=rng, clip=(33.5, 36.0))
    
    vel_mu_t = 0.28 - 0.10*sin_anual + 0.04*enso_t
    vel_t = ar1_series(vel_mu_t, phi=0.65, sigma=0.05, rng=rng, clip=(0.01, 0.8))
    
    sst_spatial = (0.12 * (lat_grid + 11) - 0.8 * np.exp(-dist_costa / 2) + rng.normal(0, 0.3, N_CELLS))
    chl_hotspot = (2.0 * np.exp(-((lat_grid + 9.0)**2) / 8) + 1.5 * np.exp(-((lat_grid + 6.5)**2) / 5) + 1.0 * np.exp(-((lat_grid + 15.0)**2) / 6) - 0.3 * dist_costa + rng.normal(0, 0.4, N_CELLS))
    ox_spatial = (-0.2 * dist_costa + 0.3 * np.exp(-((lat_grid + 12)**2) / 10) + rng.normal(0, 0.2, N_CELLS))
    
    SST = np.clip(sst_t[:, None] + sst_spatial[None, :] + rng.normal(0, 0.4, (T, N_CELLS)), 12, 27)
    CHL = np.clip(chl_t[:, None] + chl_hotspot[None, :] + rng.normal(0, 0.5, (T, N_CELLS)), 0.1, 18)
    OX = np.clip(ox_t[:, None] + ox_spatial[None, :] + rng.normal(0, 0.25, (T, N_CELLS)), 0.3, 7.0)
    SAL = np.clip(sal_t[:, None] + rng.normal(0, 0.05, (T, N_CELLS)), 33.5, 36.0)
    VEL = np.clip(vel_t[:, None] + rng.normal(0, 0.04, (T, N_CELLS)), 0.01, 0.8)
    ENSO = np.tile(enso_t[:, None], (1, N_CELLS))
    PROF = np.tile(prof_celda[None, :], (T, 1))
    
    pref_sst = np.exp(-((SST - 16.5)**2) / (2 * 1.8**2))
    pref_chl = np.log1p(CHL) / np.log1p(18)
    pref_prof = np.exp(-((PROF - 55)**2) / (2 * 40**2))
    pref_enso = np.exp(-0.45 * ENSO)
    pref_ox = np.tanh(np.maximum(OX - 1.0, 0))
    pref_lat = np.exp(-((lat_grid[None, :] + 9.0)**2) / (2 * 4.0**2))
    pref_vel = np.exp(-((VEL - 0.2)**2) / (2 * 0.15**2))
    
    signal = (2.0 + 1.8 * pref_sst + 1.5 * pref_chl + 1.0 * pref_prof + 1.0 * pref_enso + 0.8 * pref_ox + 0.6 * pref_lat + 0.4 * pref_vel)
    ruido_target = (rng.normal(0, 0.50, (T, N_CELLS)) + (rng.random((T, N_CELLS)) < 0.05) * rng.normal(0, 1.2, (T, N_CELLS)))
    LOG_DENS = np.clip(signal + ruido_target, -0.5, 8.0)
    
    fechas_rep = np.repeat(fechas_base, N_CELLS)
    t_idx_rep = np.repeat(np.arange(T), N_CELLS)
    c_idx_rep = np.tile(np.arange(N_CELLS), T)
    
    df = pd.DataFrame({
        "fecha": fechas_rep,
        "semana_anio": fechas_rep.isocalendar().week.astype(int),
        "mes": fechas_rep.month,
        "anio": fechas_rep.year,
        "latitud": lat_grid[c_idx_rep].round(3),
        "longitud": lon_grid[c_idx_rep].round(3),
        "zona_pesca": zona_celda[c_idx_rep],
        "dist_costa_deg": dist_costa[c_idx_rep].round(3),
        "sst_c": SST[t_idx_rep, c_idx_rep].round(3),
        "clorofila_mg_m3": CHL[t_idx_rep, c_idx_rep].round(3),
        "profundidad_m": PROF[t_idx_rep, c_idx_rep].round(1),
        "salinidad_psu": SAL[t_idx_rep, c_idx_rep].round(3),
        "oxigeno_ml_l": OX[t_idx_rep, c_idx_rep].round(3),
        "enso_index": ENSO[t_idx_rep, c_idx_rep].round(3),
        "corriente_vel_ms": VEL[t_idx_rep, c_idx_rep].round(3),
        "log_densidad": LOG_DENS[t_idx_rep, c_idx_rep].round(4),
    })
    return df

# ─── INTERFAZ DE USUARIO (SIDEBAR) ────────────────────────────────────────────
st.sidebar.title("Configuración del Modelo")
st.sidebar.markdown("Modifica los parámetros para simular nuevos escenarios marinos.")

intensidad_enso = st.sidebar.slider("Multiplicador de Fuerza ENSO", 0.0, 2.0, 1.0, 0.1, 
                                     help="1.0 es el valor histórico estándar.")

df = simular_dataset(intensidad_enso)

fechas_disponibles = sorted(df['fecha'].unique())
fecha_seleccionada = st.sidebar.select_slider("Selecciona Semana para Visualizar Mapa", 
                                             options=fechas_disponibles, 
                                             format_func=lambda x: x.strftime('%Y-%m-%d'))

dict_zonas = {0: "Piura", 1: "Chimbote", 2: "Lima", 3: "Pisco-Ica", 4: "Ilo-Matarani"}
zonas_seleccionadas = st.sidebar.multiselect("Filtrar por Zona de Pesca", 
                                             options=[0,1,2,3,4], 
                                             default=[0,1,2,3,4], 
                                             format_func=lambda x: dict_zonas[x])

df_filtrado = df[df['zona_pesca'].isin(zonas_seleccionadas)]

# ─── CUERPO PRINCIPAL DE LA APP ───────────────────────────────────────────────
st.title("🐟 Generador y Dashboard Sintético de Anchoveta")
st.markdown("### Enfoque: Grilla Regular Raster 0.25° × 0.25° (Mar de Grau, Perú)")

col1, col2, col3, col4 = st.columns(4)
snap_semana = df_filtrado[df_filtrado.fecha == fecha_seleccionada]

with col1:
    st.metric("Total Registros", f"{len(df_filtrado):,}")
with col2:
    st.metric("Densidad Promedio (log)", f"{df_filtrado['log_densidad'].mean():.2f}")
with col3:
    st.metric("SST Promedio (°C)", f"{snap_semana['sst_c'].mean():.1f} °C" if not snap_semana.empty else "N/A")
with col4:
    enso_val = snap_semana['enso_index'].iloc[0] if not snap_semana.empty else 0
    st.metric("Índice ONI (ENSO)", f"{enso_val:.2f}")

col_mapa, col_ts = st.columns([1, 1.2])

with col_mapa:
    st.subheader(f"Mapa de Densidad: {fecha_seleccionada.strftime('%Y-%m-%d')}")
    if snap_semana.empty:
        st.warning("No hay datos para las zonas seleccionadas.")
    else:
        fig, ax = plt.subplots(figsize=(6, 7))
        sc = ax.scatter(snap_semana.longitud, snap_semana.latitud, c=snap_semana.log_densidad,
                        cmap="YlOrRd", s=40, vmin=-0.5, vmax=8.0, alpha=0.9)
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        plt.colorbar(sc, ax=ax, label="log(ton/km²)")
        st.pyplot(fig)

with col_ts:
    st.subheader("Serie Temporal Global vs ENSO")
    ts = df_filtrado.groupby("fecha")[["sst_c","log_densidad","enso_index"]].mean()
    
    fig_ts, ax_ts = plt.subplots(figsize=(7, 5.5))
    ax2 = ax_ts.twinx()
    ax_ts.plot(ts.index, ts.log_densidad, color="#1E88E5", lw=2, label="log_densidad (izq)")
    ax2.plot(ts.index, ts.enso_index, color="#8E24AA", lw=1.5, ls="--", alpha=0.7, label="ENSO (der)")
    ax2.axhline(0, color="gray", lw=0.6, ls=":")
    ax_ts.set_ylabel("log_densidad", color="#1E88E5")
    ax2.set_ylabel("ENSO index", color="#8E24AA")
    st.pyplot(fig_ts)

st.markdown("---")
st.subheader("📥 Descargar Datos Sintéticos")
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.dataframe(df_filtrado.head(10), use_container_width=True)
with col_dl2:
    st.markdown("Descarga el dataset estructurado en base a las opciones seleccionadas.")
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Dataset como CSV",
        data=csv,
        file_name="anchoveta_sintetico_grid.csv",
        mime="text/csv",
    )