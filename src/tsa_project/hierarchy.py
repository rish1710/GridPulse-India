import pandas as pd
import numpy as np

# Attempt to import sktime's Reconciler for hierarchical forecasting
try:
    from sktime.transformations.hierarchical.reconcile import Reconciler
except ImportError:
    # If sktime is not installed, fallback to None
    Reconciler = None

def get_region_hierarchy():
    """
    Defines the structural hierarchy of regions as a nested dictionary.
    Used to understand parent-child relationships (e.g., India -> Maharashtra -> Mumbai).
    """
    return {
        "India": {
            "Maharashtra": ["Mumbai"],
        }
    }

def describe_hierarchy():
    """
    Returns a string representation of the region hierarchy for logging/display.
    """
    hierarchy = get_region_hierarchy()
    return "India → Maharashtra → Mumbai"

def reconcile_hierarchy(forecast_df):
    """
    Applies hierarchical reconciliation to ensure that forecasts at lower levels 
    (e.g., city, state) sum up correctly to forecasts at higher levels (e.g., national).
    
    Uses sktime's BottomUp (bu) method.
    """
    if Reconciler is None:
        raise ImportError(
            "sktime is required for hierarchical reconciliation. Please install the requirements."
        )
    
    print("Applying sktime hierarchical reconciliation (BottomUp)...")
    
    df = forecast_df.copy()
    
    # To use sktime, we need a MultiIndex with levels: (country, state, city, timestamp)
    # The convention in sktime is that total levels are omitted or marked in a specific way, 
    # but the easiest way to reconcile is to provide the bottom level and use Aggregator, 
    # OR provide all levels and use Reconciler.
    # We will format it as a MultiIndex and apply Reconciler.
    
    def get_levels(row):
        """
        Maps a region ID to its corresponding levels in the hierarchy.
        '__total' is used to denote that the region is an aggregate at that level.
        """
        if row["region_id"] == "Mumbai":
            # Bottom level: fully specified
            return "India", "Maharashtra", "Mumbai"
        elif row["region_id"] == "Maharashtra":
            # Middle level: city is aggregated
            return "India", "Maharashtra", "__total"
        elif row["region_id"] == "India":
            # Top level: state and city are aggregated
            return "India", "__total", "__total"
        return "__total", "__total", "__total"
    
    # Apply the level mapping
    levels = df.apply(get_levels, axis=1)
    df["country"] = [x[0] for x in levels]
    df["state"] = [x[1] for x in levels]
    df["city"] = [x[2] for x in levels]
    
    # Setup sktime multi-index for reconciliation
    df = df.set_index(["country", "state", "city", "timestamp"])
    
    # We only reconcile the target variable (forecasted demand)
    y = df[["forecast_demand_mw"]].sort_index()
    
    # Apply sktime Reconciler using the Bottom-Up method
    # This means higher-level forecasts are simply the sum of their constituent lower-level forecasts
    reconciler = Reconciler(method="bu") 
    reconciled_y = reconciler.fit_transform(y)
    
    # Convert back to standard dataframe
    reconciled_df = reconciled_y.reset_index()
    
    # Map the hierarchy levels back to the original region_id
    def get_region_id(row):
        if row["city"] != "__total":
            return row["city"]
        elif row["state"] != "__total":
            return row["state"]
        else:
            return row["country"]
            
    reconciled_df["region_id"] = reconciled_df.apply(get_region_id, axis=1)
    
    # Extract only the necessary columns for the final output
    final_df = reconciled_df[["timestamp", "region_id", "forecast_demand_mw"]].copy()
    
    # Merge region_type and parent_region back into the dataframe
    region_info = forecast_df[["region_id", "region_type", "parent_region"]].drop_duplicates()
    final_df = final_df.merge(region_info, on="region_id", how="left")
    
    return final_df
