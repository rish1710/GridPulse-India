import pandas as pd
import numpy as np

try:
    from sktime.transformations.hierarchical.reconcile import Reconciler
except ImportError:
    Reconciler = None

def get_region_hierarchy():
    return {
        "India": {
            "Maharashtra": ["Mumbai"],
        }
    }

def describe_hierarchy():
    hierarchy = get_region_hierarchy()
    return "India → Maharashtra → Mumbai"

def reconcile_hierarchy(forecast_df):
    if Reconciler is None:
        raise ImportError(
            "sktime is required for hierarchical reconciliation. Please install the requirements."
        )
    
    print("Applying sktime hierarchical reconciliation (BottomUp)...")
    
    df = forecast_df.copy()
    
    # To use sktime, we need a MultiIndex: (country, state, city, timestamp)
    # The convention in sktime is that total levels are omitted or marked in a specific way, 
    # but the easiest way to reconcile is to provide the bottom level and use Aggregator, 
    # OR provide all levels and use Reconciler.
    # We will format it as a MultiIndex and apply Reconciler.
    
    def get_levels(row):
        if row["region_id"] == "Mumbai":
            return "India", "Maharashtra", "Mumbai"
        elif row["region_id"] == "Maharashtra":
            return "India", "Maharashtra", "__total"
        elif row["region_id"] == "India":
            return "India", "__total", "__total"
        return "__total", "__total", "__total"
    
    levels = df.apply(get_levels, axis=1)
    df["country"] = [x[0] for x in levels]
    df["state"] = [x[1] for x in levels]
    df["city"] = [x[2] for x in levels]
    
    # Setup sktime multi-index
    df = df.set_index(["country", "state", "city", "timestamp"])
    
    # We only reconcile the target variable
    y = df[["forecast_demand_mw"]].sort_index()
    
    # Apply sktime Reconciler
    reconciler = Reconciler(method="bu") # BottomUp
    reconciled_y = reconciler.fit_transform(y)
    
    # Convert back to standard dataframe
    reconciled_df = reconciled_y.reset_index()
    
    # Map back to region_id
    def get_region_id(row):
        if row["city"] != "__total":
            return row["city"]
        elif row["state"] != "__total":
            return row["state"]
        else:
            return row["country"]
            
    reconciled_df["region_id"] = reconciled_df.apply(get_region_id, axis=1)
    
    # Merge the reconciled forecasts back with the other columns if needed, 
    # but we just need timestamp, region_id, and forecast_demand_mw.
    final_df = reconciled_df[["timestamp", "region_id", "forecast_demand_mw"]].copy()
    
    # Merge region_type and parent_region back
    region_info = forecast_df[["region_id", "region_type", "parent_region"]].drop_duplicates()
    final_df = final_df.merge(region_info, on="region_id", how="left")
    
    return final_df
