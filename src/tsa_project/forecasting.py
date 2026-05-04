import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.tsa_project.data_generation import generate_hierarchical_demand, save_synthetic_dataset
from src.tsa_project.hierarchy import describe_hierarchy


def build_regressors(df):
    feature_columns = [
        "day_of_week",
        "month",
        "is_weekend",
        "weekly_sin_1",
        "weekly_cos_1",
        "weekly_sin_2",
        "weekly_cos_2",
        "weekly_sin_3",
        "weekly_cos_3",
        "annual_sin_1",
        "annual_cos_1",
        "annual_sin_2",
        "annual_cos_2",
        "annual_sin_3",
        "annual_cos_3",
        "annual_sin_4",
        "annual_cos_4",
    ]
    df = df.copy()
    region_dummies = pd.get_dummies(df["region_id"], prefix="region")
    df = pd.concat([df[feature_columns], region_dummies], axis=1)
    return df, feature_columns + list(region_dummies.columns)


def build_lstm_model(input_dim, units=128):
    model = models.Sequential([
        layers.Input(shape=(1, input_dim)),
        layers.LSTM(units, activation="tanh", return_sequences=False),
        layers.Dropout(0.18),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def prepare_dataset(df):
    df = df.sort_values(["region_id", "timestamp"]).reset_index(drop=True)
    X, y = build_regressors(df)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(df[["demand_mw"]])
    return X_scaled, y_scaled, scaler_x, scaler_y, X.columns


def train_global_lstm(df):
    X_scaled, y_scaled, scaler_x, scaler_y, feature_names = prepare_dataset(df)
    X_seq = X_scaled.reshape((-1, 1, X_scaled.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_scaled, test_size=0.14, random_state=2026, shuffle=False
    )

    model = build_lstm_model(input_dim=X_seq.shape[2])
    model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test), verbose=2)

    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Global LSTM training loss: {train_loss}")
    print(f"Global LSTM validation loss: {test_loss}")
    return model, scaler_x, scaler_y, feature_names


def forecast_horizon(df, model, scaler_x, scaler_y, horizon_days=30):
    index = pd.date_range(start=df["timestamp"].max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    full_rows = []
    for region in df["region_id"].unique():
        region_rows = pd.DataFrame({"timestamp": index, "region_id": region})
        region_rows["region_type"] = df.loc[df["region_id"] == region, "region_type"].iloc[0]
        region_rows["parent_region"] = df.loc[df["region_id"] == region, "parent_region"].iloc[0]
        region_rows = pd.concat([region_rows, pd.DataFrame({
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_weekend": index.dayofweek.isin([5, 6]).astype(int),
        })], axis=1)
        region_rows = pd.concat([region_rows, build_fourier_features(index, period=7, order=3, prefix="weekly")], axis=1)
        region_rows = pd.concat([region_rows, build_fourier_features(index, period=365.25, order=4, prefix="annual")], axis=1)
        full_rows.append(region_rows)

    future_df = pd.concat(full_rows, axis=0).reset_index(drop=True)
    feature_df, _ = build_regressors(future_df)
    X_future_scaled = scaler_x.transform(feature_df)
    X_future_seq = X_future_scaled.reshape((-1, 1, X_future_scaled.shape[1]))
    y_future_scaled = model.predict(X_future_seq, verbose=0)
    y_future = scaler_y.inverse_transform(y_future_scaled)
    future_df["forecast_demand_mw"] = y_future.ravel()
    return future_df


def build_fourier_features(index, period, order, prefix):
    t = np.arange(len(index))
    features = {}
    for k in range(1, order + 1):
        features[f"{prefix}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        features[f"{prefix}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(features, index=index)


def main():
    os.makedirs("data", exist_ok=True)
    data_path = save_synthetic_dataset("data/synthetic_electricity_demand.csv")
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    model, scaler_x, scaler_y, feature_names = train_global_lstm(df)
    forecast_df = forecast_horizon(df, model, scaler_x, scaler_y, horizon_days=30)
    
    from src.tsa_project.hierarchy import reconcile_hierarchy, describe_hierarchy
    forecast_df = reconcile_hierarchy(forecast_df)
    
    forecast_df.to_csv("data/electricity_demand_forecast.csv", index=False)
    print("Forecasts written to data/electricity_demand_forecast.csv")
    print(f"Hierarchy path: {describe_hierarchy()}")

if __name__ == "__main__":
    main()
