import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import subprocess

st.set_page_config(page_title="GridPulse India Forecast", layout="wide")

# Paths
DATA_DIR = "data"
HISTORICAL_PATH = os.path.join(DATA_DIR, "synthetic_electricity_demand.csv")
FORECAST_PATH = os.path.join(DATA_DIR, "electricity_demand_forecast.csv")

st.title("⚡ GridPulse India: Electricity Demand Forecasting")
st.markdown("A hierarchical Time Series Analysis (TSA) dashboard for predicting electricity demand at National, State, and City levels.")

# Sidebar Controls
st.sidebar.header("Controls")

def run_pipeline():
    with st.spinner("Generating data and running forecasts... This might take a minute."):
        try:
            subprocess.run(["python", "src/tsa_project/forecasting.py"], check=True)
            st.sidebar.success("Pipeline executed successfully!")
        except subprocess.CalledProcessError as e:
            st.sidebar.error(f"Error running pipeline: {e}")

if st.sidebar.button("Run Forecasting Pipeline"):
    run_pipeline()

# Load Data
@st.cache_data
def load_data():
    if not os.path.exists(HISTORICAL_PATH) or not os.path.exists(FORECAST_PATH):
        return None, None
    df_hist = pd.read_csv(HISTORICAL_PATH, parse_dates=["timestamp"])
    df_fore = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])
    return df_hist, df_fore

df_hist, df_fore = load_data()

if df_hist is None or df_fore is None:
    st.warning("Data not found. Please click 'Run Forecasting Pipeline' in the sidebar to generate data.")
else:
    # Region Selection
    regions = df_hist["region_id"].unique()
    selected_region = st.sidebar.selectbox("Select Region", regions)

    # Filter data
    hist_region = df_hist[df_hist["region_id"] == selected_region]
    fore_region = df_fore[df_fore["region_id"] == selected_region]

    st.subheader(f"Demand Forecast for {selected_region}")
    
    # KPIs
    col1, col2, col3 = st.columns(3)
    latest_hist = hist_region["demand_mw"].iloc[-1]
    avg_fore = fore_region["forecast_demand_mw"].mean()
    peak_fore = fore_region["forecast_demand_mw"].max()
    
    col1.metric("Latest Historical Demand (MW)", f"{latest_hist:.2f}")
    col2.metric("Avg Forecasted Demand (MW)", f"{avg_fore:.2f}")
    col3.metric("Peak Forecasted Demand (MW)", f"{peak_fore:.2f}")

    # Plotly Chart
    fig = go.Figure()

    # Historical (last 60 days for better visibility)
    last_60_days = hist_region.tail(60)
    fig.add_trace(go.Scatter(
        x=last_60_days["timestamp"], 
        y=last_60_days["demand_mw"],
        mode='lines',
        name='Historical Demand',
        line=dict(color='#1f77b4', width=2)
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=fore_region["timestamp"], 
        y=fore_region["forecast_demand_mw"],
        mode='lines',
        name='Forecasted Demand',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"{selected_region} Demand: Last 60 Days + 30 Day Forecast",
        xaxis_title="Date",
        yaxis_title="Demand (MW)",
        hovermode="x unified",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display Data Table
    with st.expander("View Forecast Data"):
        st.dataframe(fore_region[["timestamp", "forecast_demand_mw"]].sort_values("timestamp"))
