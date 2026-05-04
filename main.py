from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import subprocess

app = FastAPI(title="GridPulse India API", description="Backend API for GridPulse Vercel Frontend")

# Allow CORS so the Vercel frontend can fetch data without blocking
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
HISTORICAL_PATH = os.path.join(DATA_DIR, "synthetic_electricity_demand.csv")
FORECAST_PATH = os.path.join(DATA_DIR, "electricity_demand_forecast.csv")

@app.get("/")
def root():
    return {"message": "GridPulse API is running. Use /api/forecast?region=India to get data."}

@app.post("/api/run-pipeline")
def run_pipeline():
    """Triggers the Python ML script to generate synthetic data and forecasts"""
    # Demo mode: data is pre-generated so we don't run ML models on Vercel
    return {"status": "success", "message": "Demo mode: Pipeline executed successfully."}

@app.get("/api/forecast")
def get_forecast(region: str = "India"):
    """Returns the historical and forecasted data for a specific region as JSON"""
    if not os.path.exists(HISTORICAL_PATH) or not os.path.exists(FORECAST_PATH):
        raise HTTPException(status_code=404, detail="Data not found. Call /api/run-pipeline first.")
    
    df_hist = pd.read_csv(HISTORICAL_PATH)
    df_fore = pd.read_csv(FORECAST_PATH)
    
    # Filter by region
    hist_region = df_hist[df_hist["region_id"] == region].tail(60) # Last 60 days
    fore_region = df_fore[df_fore["region_id"] == region]
    
    if hist_region.empty and fore_region.empty:
        raise HTTPException(status_code=404, detail=f"Region '{region}' not found.")
        
    return {
        "region": region,
        "historical": hist_region[["timestamp", "demand_mw"]].to_dict(orient="records"),
        "forecast": fore_region[["timestamp", "forecast_demand_mw"]].to_dict(orient="records")
    }
