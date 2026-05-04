# Electricity Demand Forecasting by Region

This project demonstrates a Time Series Analysis (TSA) pipeline for forecasting electricity demand at multiple geographic levels across India:
- National: India grid load
- State: Maharashtra demand
- City: Mumbai demand

It uses synthetic demand data plus modern forecasting techniques:
- LSTM global forecasting model for shared temporal patterns
- Fourier terms for seasonality features
- Hierarchical aggregation/reconciliation for national/state/city forecasts
- `sktime` for hierarchical time series support and model evaluation

## Structure

- `src/tsa_project/data_generation.py`: synthetic electricity demand generation and engineered calendar features
- `src/tsa_project/forecasting.py`: global LSTM training, forecasting, hierarchical aggregation, and evaluation
- `assets/system_architecture.svg`: visual architecture diagram for the TSA system
- `presentation/generate_ppt.py`: script to generate the short PPT deck
- `requirements.txt`: Python dependencies

## Getting Started

1. Create a Python environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Run the synthetic data and modeling pipeline:

```bash
python src/tsa_project/forecasting.py
```

3. Generate the presentation deck:

```bash
python presentation/generate_ppt.py
```

## Project Goals

- Model electricity demand across a hierarchy of regions
- Use signal processing features (Fourier seasonality) and trends
- Apply global deep learning with shared representation across regions
- Demonstrate where TSA methods are used in data, modeling, and architecture layers

---

# Product Requirements Document (PRD): GridPulse India

## 1. Executive Summary
**GridPulse India** is a hierarchical Time Series Analysis (TSA) platform designed to forecast electricity demand across multiple geographic levels in India: National, State (Maharashtra), and City (Mumbai). The platform provides an interactive web dashboard for grid operators and energy analysts to visualize historical demand and accurate, AI-driven forecasts, enabling better resource planning and grid stability.

## 2. Problem Statement
Accurate electricity demand forecasting is critical for energy grid stability. Overestimating demand leads to wasted resources and increased costs, while underestimating demand results in blackouts and grid failures. Traditional forecasting models often fail to capture the complex, nested relationships between national grids and local consumption, as well as the intricate seasonality (daily, weekly, yearly) of electricity usage.

## 3. Target Audience
*   **Grid Operators:** Need reliable short-term and medium-term forecasts to balance supply and demand.
*   **Energy Analysts & Planners:** Require insights into consumption trends across different geographic hierarchies for infrastructure planning.
*   **State Electricity Boards:** Need localized data (e.g., Maharashtra, Mumbai) to manage state-level energy procurement and distribution.

## 4. Product Vision
To be the most reliable, interactive, and intelligent electricity demand forecasting tool for the Indian power sector, bridging the gap between advanced deep learning techniques and actionable operational insights.

## 5. Key Features & Requirements

### 5.1. Hierarchical Forecasting Engine
*   **Requirement:** The system must forecast demand at three distinct levels: National (India), State (Maharashtra), and City (Mumbai).
*   **Requirement:** The system must use hierarchical reconciliation techniques to ensure that state and city-level forecasts aggregate logically to the national forecast.

### 5.2. Advanced Machine Learning Pipeline
*   **Requirement:** Utilize a Global LSTM (Long Short-Term Memory) deep learning model to learn shared temporal patterns across all regions simultaneously.
*   **Requirement:** Incorporate advanced feature engineering, specifically Fourier terms, to capture complex seasonalities (weekly and annual cycles).

### 5.3. Interactive Web Dashboard (Streamlit / Next.js)
*   **Requirement:** Provide a user-friendly interface accessible via web browsers.
*   **Requirement:** Include a region selector allowing users to toggle between National, State, and City views.
*   **Requirement:** Display high-level Key Performance Indicators (KPIs) such as Latest Historical Demand, Average Forecast, and Peak Forecast.
*   **Requirement:** Render interactive, zoomable line charts displaying at least 60 days of historical data seamlessly connected to a 30-day forecast.

### 5.4. On-Demand Data Generation
*   **Requirement:** Allow users to manually trigger the data generation and forecasting pipeline directly from the UI to incorporate the latest synthetic/real data.
*   **Requirement:** Provide visual feedback (spinners, success/error messages) during the execution of the pipeline.

## 6. System Architecture
*   **Frontend & UI:** Next.js (Vercel) / Streamlit (Python)
*   **Visualizations:** Plotly Express / Recharts
*   **Machine Learning Core:** TensorFlow/Keras (LSTM), Scikit-Learn (Scaling), `sktime` (Hierarchical Reconciliation)
*   **Data Processing:** Pandas, NumPy
*   **Deployment:** Docker containerized backend, deployable to cloud platforms (Render, Railway).

## 7. Non-Functional Requirements
*   **Performance:** The UI must load within 2 seconds. The forecasting pipeline should execute in under 60 seconds.
*   **Scalability:** The backend must be Dockerized to ensure it can be easily scaled horizontally across cloud instances if user traffic increases.
*   **Usability:** The dashboard must support a dark mode theme for better viewing in operational control rooms.

## 8. Future Roadmap (V2)
*   **Live Data Integration:** Replace synthetic data generation with live APIs from the National Load Despatch Centre (NLDC) or similar grid data providers.
*   **Exogenous Variables:** Incorporate real-time weather data (temperature, humidity) as regressors in the LSTM model.
*   **Alerting System:** Implement email/SMS alerts when forecasted demand exceeds predefined grid capacity thresholds.
