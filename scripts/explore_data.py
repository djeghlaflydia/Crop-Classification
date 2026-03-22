import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

# Ensure config can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from scripts.api_access import get_stac_data
import stackstac

def explore_area(area_name, epsg):
    print(f"\nExploratory Analysis for {area_name}")
    bbox = config.STUDY_AREAS[area_name]["bbox"]
    
    # Fetch a smaller patch for exploration (1x1 km)
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2
    small_bbox = [center_lon - 0.005, center_lat - 0.005, center_lon + 0.005, center_lat + 0.005]

    # Fetch S2 and CDL
    s2_items = get_stac_data(area_name, config.S2_COLLECTION, small_bbox, config.DATE_RANGE)
    s2_items = [i for i in s2_items if i.properties["eo:cloud_cover"] < config.CLOUD_PERCENT]
    
    cdl_items = get_stac_data(area_name, config.CDL_COLLECTION, small_bbox, "2020-01-01/2020-12-31")

    # Stack
    s2_stack = stackstac.stack(s2_items, assets=["B04", "B08"], bounds_latlon=small_bbox, epsg=epsg, resolution=10)
    cdl_stack = stackstac.stack(cdl_items, assets=["cropland"], bounds_latlon=small_bbox, epsg=epsg, resolution=10)

    # Compute NDVI
    red = s2_stack.sel(band="B04")
    nir = s2_stack.sel(band="B08")
    ndvi = (nir - red) / (nir + red)
    
    # 1. Temporal NDVI Pattern (Mean of patch)
    plt.figure(figsize=(10, 5))
    ndvi_mean = ndvi.mean(dim=["x", "y"]).compute()
    ndvi_mean.plot(marker='o', linestyle='-', color='forestgreen')
    plt.title(f"Mean NDVI Time-Series - {area_name} (2020)")
    plt.ylabel("NDVI")
    plt.grid(True)
    plot_path = os.path.abspath(f"ndvi_timeseries_{area_name}.png")
    plt.savefig(plot_path)
    print(f"Saved NDVI plot to {plot_path}")

    # 2. Class Distribution (CDL)
    plt.figure(figsize=(10, 5))
    cdl_data = cdl_stack.isel(time=0, band=0).compute().values.flatten()
    unique, counts = np.unique(cdl_data, return_counts=True)
    # Filter out 0 (No Data/Background if applicable)
    mask = unique > 0
    unique, counts = unique[mask], counts[mask]
    
    plt.bar(unique.astype(str), counts, color='teal')
    plt.title(f"Crop Class Distribution (CDL) - {area_name}")
    plt.xlabel("CDL Class ID")
    plt.ylabel("Pixel Count")
    plt.xticks(rotation=45)
    dist_plot_path = os.path.abspath(f"class_dist_{area_name}.png")
    plt.savefig(dist_plot_path)
    print(f"Saved distribution plot to {dist_plot_path}")

def main():
    explore_area("California", 32611)
    explore_area("Arkansas", 32615)

if __name__ == "__main__":
    main()
