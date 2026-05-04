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
