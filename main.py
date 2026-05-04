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

from fastapi.responses import HTMLResponse

@app.get("/")
def root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GridPulse India Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    </head>
    <body class="bg-gray-900 text-white min-h-screen p-8">
        <div class="max-w-6xl mx-auto">
            <h1 class="text-4xl font-bold mb-2">⚡ GridPulse India</h1>
            <p class="text-gray-400 mb-8">Electricity Demand Forecasting Dashboard</p>
            
            <div class="mb-4">
                <label class="mr-2 font-bold">Select Region:</label>
                <select id="regionSelect" class="bg-gray-800 text-white p-2 rounded border border-gray-700" onchange="fetchData()">
                    <option value="India">India (National)</option>
                    <option value="Maharashtra">Maharashtra (State)</option>
                    <option value="Mumbai">Mumbai (City)</option>
                </select>
            </div>
            
            <div id="chart" class="w-full h-[600px] bg-gray-800 rounded-xl shadow-lg border border-gray-700 p-4"></div>
        </div>

        <script>
            async function fetchData() {
                const region = document.getElementById('regionSelect').value;
                const response = await fetch(`/api/forecast?region=${region}`);
                const data = await response.json();
                
                const hist_x = data.historical.map(d => d.timestamp);
                const hist_y = data.historical.map(d => d.demand_mw);
                
                const fore_x = data.forecast.map(d => d.timestamp);
                const fore_y = data.forecast.map(d => d.forecast_demand_mw);

                const trace1 = {
                    x: hist_x, y: hist_y, mode: 'lines', name: 'Historical Demand',
                    line: {color: '#3b82f6', width: 2}
                };
                
                const trace2 = {
                    x: fore_x, y: fore_y, mode: 'lines', name: 'Forecast',
                    line: {color: '#f59e0b', width: 2, dash: 'dash'}
                };

                const layout = {
                    title: `${region} Demand Forecast`,
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {color: '#fff'},
                    xaxis: {gridcolor: '#374151'},
                    yaxis: {title: 'Demand (MW)', gridcolor: '#374151'}
                };

                Plotly.newPlot('chart', [trace1, trace2], layout);
            }
            // Load data automatically on page load
            fetchData();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
