import numpy as np
import pandas as pd


def create_time_index(start_date="2018-01-01", periods=1825, freq="D"):
    """
    Creates a pandas DatetimeIndex for the time series.
    
    Args:
        start_date (str): The starting date for the index.
        periods (int): The number of periods (days) to generate.
        freq (str): The frequency of the data (default is 'D' for daily).
    """
    return pd.date_range(start=start_date, periods=periods, freq=freq)


def build_fourier_features(index, period, order, prefix):
    """
    Generates Fourier series features (sine and cosine waves) to model multiple seasonalities.
    
    Args:
        index (pd.DatetimeIndex): The time index of the series.
        period (float): The period of the seasonality (e.g., 7 for weekly, 365.25 for annual).
        order (int): The number of Fourier terms (sine/cosine pairs) to generate.
        prefix (str): Prefix for the generated column names.
        
    Returns:
        pd.DataFrame: A DataFrame containing the generated Fourier features.
    """
    t = np.arange(len(index))
    features = {}
    for k in range(1, order + 1):
        # Calculate sine and cosine components for the k-th harmonic
        features[f"{prefix}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        features[f"{prefix}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(features, index=index)


def generate_electricity_series(index, base, trend_scale, weekly_amp, annual_amp, noise_scale, shift=0):
    """
    Generates a synthetic electricity demand time series with trend, seasonality, and noise.
    """
    t = np.arange(len(index))
    weekday = index.dayofweek
    day_of_year = index.dayofyear

    # Linear trend over the time period
    trend = trend_scale * (t / len(index))
    # Weekly seasonality (sine wave based on day of the week)
    weekly = weekly_amp * np.sin(2 * np.pi * (weekday + shift) / 7)
    # Annual seasonality (sine wave based on day of the year)
    annual = annual_amp * np.sin(2 * np.pi * day_of_year / 365.25)
    # Random Gaussian noise
    noise = np.random.normal(scale=noise_scale, size=len(index))
    
    # Combine components to form the final series
    return base + trend + weekly + annual + noise


def map_regions():
    """
    Defines the hierarchical structure of regions for the demand dataset.
    Returns a list of dictionaries, each specifying a region's ID, type, and parent region.
    """
    return [
        {"region_id": "India", "region_type": "national", "parent": None},
        {"region_id": "Maharashtra", "region_type": "state", "parent": "India"},
        {"region_id": "Mumbai", "region_type": "city", "parent": "Maharashtra"},
    ]


def generate_hierarchical_demand(start_date="2018-01-01", periods=1825, seed=42):
    """
    Generates synthetic hierarchical electricity demand data for all defined regions.
    """
    np.random.seed(seed) # Ensure reproducibility
    index = create_time_index(start_date=start_date, periods=periods)
    regions = map_regions()

    data = []
    for region in regions:
        # Define base demand and seasonal amplitudes based on the region level
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
            # Default parameters for unknown regions
            base = 1000
            trend_scale = 100
            weekly_amp = 80
            annual_amp = 120
            noise_scale = 40

        # Generate the time series for the current region
        series = generate_electricity_series(
            index=index,
            base=base,
            trend_scale=trend_scale,
            weekly_amp=weekly_amp,
            annual_amp=annual_amp,
            noise_scale=noise_scale,
            shift=0,
        )

        # Create a DataFrame for the region's data
        features = pd.DataFrame({
            "timestamp": index,
            "region_id": region["region_id"],
            "region_type": region["region_type"],
            "parent_region": region["parent"],
            "demand_mw": series,
        })
        data.append(features)

    # Combine all region data into a single DataFrame
    df = pd.concat(data, axis=0).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp", "region_id"]).reset_index(drop=True)
    
    # Enrich the dataset with temporal features (day of week, month, etc.)
    df = add_temporal_features(df)
    return df


def add_temporal_features(df):
    """
    Adds time-based features (e.g., day of week, month, Fourier terms) to the DataFrame.
    These features are useful as predictors for machine learning models.
    """
    df = df.copy()
    idx = df["timestamp"]
    
    # Extract basic time features
    df["day_of_week"] = idx.dt.dayofweek
    df["month"] = idx.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Add Fourier features for weekly and annual seasonality
    df = pd.concat([df, build_fourier_features(idx, period=7, order=3, prefix="weekly")], axis=1)
    df = pd.concat([df, build_fourier_features(idx, period=365.25, order=4, prefix="annual")], axis=1)
    return df


def save_synthetic_dataset(path="data/synthetic_electricity_demand.csv"):
    """
    Generates the hierarchical demand dataset and saves it to a CSV file.
    """
    df = generate_hierarchical_demand()
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print("Generating synthetic electricity demand dataset...")
    path = save_synthetic_dataset()
    print(f"Data written to {path}")
