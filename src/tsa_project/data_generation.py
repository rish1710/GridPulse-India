import numpy as np
import pandas as pd


def create_time_index(start_date="2018-01-01", periods=1825, freq="D"):
    return pd.date_range(start=start_date, periods=periods, freq=freq)


def build_fourier_features(index, period, order, prefix):
    t = np.arange(len(index))
    features = {}
    for k in range(1, order + 1):
        features[f"{prefix}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        features[f"{prefix}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(features, index=index)


def generate_electricity_series(index, base, trend_scale, weekly_amp, annual_amp, noise_scale, shift=0):
    t = np.arange(len(index))
    weekday = index.dayofweek
    day_of_year = index.dayofyear

    trend = trend_scale * (t / len(index))
    weekly = weekly_amp * np.sin(2 * np.pi * (weekday + shift) / 7)
    annual = annual_amp * np.sin(2 * np.pi * day_of_year / 365.25)
    noise = np.random.normal(scale=noise_scale, size=len(index))
    return base + trend + weekly + annual + noise


def map_regions():
    return [
        {"region_id": "India", "region_type": "national", "parent": None},
        {"region_id": "Maharashtra", "region_type": "state", "parent": "India"},
        {"region_id": "Mumbai", "region_type": "city", "parent": "Maharashtra"},
    ]


def generate_hierarchical_demand(start_date="2018-01-01", periods=1825, seed=42):
    np.random.seed(seed)
    index = create_time_index(start_date=start_date, periods=periods)
    regions = map_regions()

    data = []
    for region in regions:
        if region["region_id"] == "India":
            base = 23000
            trend_scale = 2000
            weekly_amp = 1600
            annual_amp = 2500
            noise_scale = 300
        elif region["region_id"] == "Maharashtra":
            base = 5200
            trend_scale = 600
            weekly_amp = 420
            annual_amp = 640
            noise_scale = 120
        elif region["region_id"] == "Mumbai":
            base = 1800
            trend_scale = 280
            weekly_amp = 220
            annual_amp = 340
            noise_scale = 80
        else:
            base = 1000
            trend_scale = 100
            weekly_amp = 80
            annual_amp = 120
            noise_scale = 40

        series = generate_electricity_series(
            index=index,
            base=base,
            trend_scale=trend_scale,
            weekly_amp=weekly_amp,
            annual_amp=annual_amp,
            noise_scale=noise_scale,
            shift=0,
        )

        features = pd.DataFrame({
            "timestamp": index,
            "region_id": region["region_id"],
            "region_type": region["region_type"],
            "parent_region": region["parent"],
            "demand_mw": series,
        })
        data.append(features)

    df = pd.concat(data, axis=0).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp", "region_id"]).reset_index(drop=True)
    df = add_temporal_features(df)
    return df


def add_temporal_features(df):
    df = df.copy()
    idx = df["timestamp"]
    df["day_of_week"] = idx.dt.dayofweek
    df["month"] = idx.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df = pd.concat([df, build_fourier_features(idx, period=7, order=3, prefix="weekly")], axis=1)
    df = pd.concat([df, build_fourier_features(idx, period=365.25, order=4, prefix="annual")], axis=1)
    return df


def save_synthetic_dataset(path="data/synthetic_electricity_demand.csv"):
    df = generate_hierarchical_demand()
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print("Generating synthetic electricity demand dataset...")
    path = save_synthetic_dataset()
    print(f"Data written to {path}")
